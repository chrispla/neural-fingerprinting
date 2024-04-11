import json
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from torch import nn
from tqdm import tqdm

from dataset import AudioDB
from model import VGGlike


def fingerprintBuilder(audio_dir, fp_dir, batch_size=16):
    """Do inference and save embeddings to fp_dir."""
    Path(fp_dir).mkdir(exist_ok=True)

    model = VGGlike()
    dataset = AudioDB(root=audio_dir, augmentations=False)
    dataloader = dataset.get_loader(batch_size=batch_size, shuffle=False, num_workers=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # keep a dict of file indices to names, for convenient retrieval later
    index_to_name = {}

    for i, batch in tqdm(
        enumerate(dataloader), total=len(dataloader), leave=True, ncols=80
    ):
        x, name = batch
        x = x.to(device)
        with torch.no_grad():
            y = model(x)
        for j, emb in enumerate(y):
            # save using emb index to be able to retrieve segment during query
            np.save(fp_dir / f"{i * batch_size + j}.npy", emb.cpu().numpy())
            index_to_name[i * batch_size + j] = name[j]

    with open(fp_dir / "idx_dict.json", "w") as f:
        json.dump(index_to_name, f)


def audioIdentification(query_dir, fp_dir, output_path):
    """Do identification of files in a query db, based on the extracted
    fingerprints in fp_dir. Save results to output_path."""

    # get mel spectrogram transform
    melspec = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=8000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=64,
            f_min=300,
            f_max=4000,
            power=1.0,
        ),
        torchaudio.transforms.AmplitudeToDB(),
    )

    # load all fps to RAM (yes, they're just ~100MB in total)
    fps = []
    for fp_path in Path(fp_dir).glob("*.npy"):
        fps.append(np.load(fp_path))

    # load index to name mapping
    with open(fp_dir / "idx_dict.json", "r") as f:
        index_to_name = json.load(f)

    # get all .wav files in the query directory
    query_paths = list(Path(query_dir).glob("*.wav"))

    # load the model
    model = VGGlike()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    checkpoint = torch.load("ckpt/model_72.pt", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    for qp in tqdm(query_paths):
        y, sr = librosa.load(qp, sr=8000)
        # split the query into 1 second segments
        y = np.array_split(y, len(y) // 8000)
        # pad the last segment
        y[-1] = np.pad(y[-1], (0, 8000 - len(y[-1])), "constant")
        y = torch.from_numpy(y).unsqueeze(0).float().to(device)
        y = melspec(y)
        with torch.no_grad():
            y = model(y)
        y = y.cpu().numpy()

        # for each query fingerprint, find the 3 closest matches from fps
        # using the cosine similarity
        results = []
        for q in y[0]:
            sims = np.dot(fps, q) / (np.linalg.norm(fps, axis=1) * np.linalg.norm(q))
            top3 = sims.argsort()[-3:][::-1]
            results.append([(index_to_name[str(i)], sims[i]) for i in top3])

        with open(output_path, "a") as f:
            f.write(f"{Path(qp).name()}\t{"\t".join([f"{r[0]}\t{r[1]}" for r in results])}\n")
