from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataset import AudioDB
from model import VGGlike


def fingerprintBuilder(audio_dir, fp_dir):
    """Do inference and save embeddings to dir."""
    Path(fp_dir).mkdir(exist_ok=True)

    batch_size = 16

    model = VGGlike()
    dataset = AudioDB(root=audio_dir, augmentations=False)
    dataloader = dataset.get_loader(batch_size=batch_size, shuffle=False, num_workers=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    for i, batch in tqdm(
        enumerate(dataloader), total=len(dataloader), leave=True, ncols=80
    ):
        x = batch
        x = x.to(device)
        with torch.no_grad():
            y = model(x)
        for j, emb in enumerate(y):
            # save using emb index to be able to retrieve segment during query
            np.save(fp_dir / f"{i * batch_size + j}.npy", emb.cpu().numpy())
