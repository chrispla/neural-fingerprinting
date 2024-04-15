import json
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from sklearn.neighbors import NearestNeighbors
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
    checkpoint = torch.load("ckpt/model_90_loss_2.2881.pt", map_location=device)
    model.load_state_dict(checkpoint)
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
            np.save(Path(fp_dir) / f"{i * batch_size + j}.npy", emb.cpu().numpy())
            index_to_name[i * batch_size + j] = name[j]

    with open(Path(fp_dir) / "idx_dict.json", "w") as f:
        json.dump(index_to_name, f)


def audioIdentification(query_dir, fp_dir, output_path):
    """Do identification of files in a query db, based on the extracted
    fingerprints in fp_dir. Save results to output_path."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    ).to(device)

    # load index to name mapping
    with open(Path(fp_dir) / "idx_dict.json", "r") as f:
        index_to_name = json.load(f)

    # load all fps to RAM (yes, they're just ~100MB in total)
    fps = {}
    for fp_idx in index_to_name.keys():
        fps[fp_idx] = np.load(Path(fp_dir) / f"{fp_idx}.npy")

    # get all .wav files in the query directory
    query_paths = list(Path(query_dir).glob("*.wav"))

    # fit nearest neighbors model
    nn_model = NearestNeighbors(n_neighbors=3, metric="cosine")
    nn_model.fit(np.array(list(fps.values())))

    # load the model
    model = VGGlike()
    model.to(device)
    checkpoint = torch.load("ckpt/model_90_loss_2.2881.pt", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    top1_hits = 0
    top2_hits = 0
    top3_hits = 0
    for qp in tqdm(query_paths, leave=True, ncols=80):
        y, sr = librosa.load(qp, sr=8000)
        # split the query into exactly 1 second segments
        y = np.array([y[i : i + 8000] for i in range(0, len(y) - 8000, 8000)])
        y = torch.from_numpy(y).unsqueeze(1).float().to(device)
        y = melspec(y)
        with torch.no_grad():
            y = model(y)
        y = y.cpu().numpy()

        # for each query fingerprint, find the 3 closest matches from fps
        # using the cosine similarity
        distances, indices = nn_model.kneighbors(y, return_distance=True)
        # go from array of indices to array of filenames
        results = []
        for i in range(len(indices)):
            results_i = []
            for j in range(len(indices[i])):
                results_i.append(index_to_name[str(indices[i][j])])
            results.append(results_i)
        # flatten the results
        results = np.array(results).flatten()
        # get filename counts
        results = np.unique(results, return_counts=True)
        # sort by counts
        results = np.array(
            sorted(list(zip(*results)), key=lambda x: x[1], reverse=True)
        )
        # get the top 3
        results = results[:, 0][:3]
        # if there are less than 3 results, pad with unkown
        # ideally I'd return more neighbors in nn_models
        results = np.pad(results, (0, 3 - len(results)), constant_values="unknown")

        # check if the query file is in the top 3
        if qp.stem.split("-")[0] in results:
            top3_hits += 1
        if qp.stem.split("-")[0] in results[:2]:
            top2_hits += 1
        if qp.stem.split("-")[0] == results[0]:
            top1_hits += 1

        with open(output_path, "a") as f:
            f.write(
                f"{qp.name}\t{results[0]}\t{results[1]}\t{results[2]}\n",
            )

    print(f"Top 1 hit rate: {top1_hits / len(query_paths)}")
    print(f"Top 2 hit rate: {top2_hits / len(query_paths)}")
    print(f"Top 3 hit rate: {top3_hits / len(query_paths)}")


if __name__ == "__main__":
    fingerprintBuilder("data/database_recordings", "db/")
    audioIdentification("data/query_recordings", "db/", "results.txt")
