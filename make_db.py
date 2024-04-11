"""Do inference and save embeddings to a dir."""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataset import AudioDB
from model import VGGlike

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", "-c", type=str, required=True)
parser.add_argument("--in_dir", "-i", type=str, default="data/database_recordings")
parser.add_argument("--out_dir", "-o", type=str, default="db/")
args = parser.parse_args()

Path(args.out_dir).mkdir(exist_ok=True)

batch_size = 16

model = VGGlike()
dataset = AudioDB(root=args.in_dir, augmentations=False)
dataloader = dataset.get_loader(batch_size=batch_size, shuffle=False, num_workers=8)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

emb_dir = Path(args.out_dir)
emb_dir.mkdir(exist_ok=True)

for i, batch in tqdm(
    enumerate(dataloader), total=len(dataloader), leave=True, ncols=80
):
    x = batch
    x = x.to(device)
    with torch.no_grad():
        y = model(x)
    for j, emb in enumerate(y):
        # save using emb index to be able to retrieve segment during query
        np.save(emb_dir / f"{i * batch_size + j}.npy", emb.cpu().numpy())
