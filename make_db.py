"""Do inference and save embeddings to a dir."""

from pathlib import Path

import torch

from dataset import AudioDB
from model import VGGlike

model = VGGlike()
dataset = AudioDB(root="data/database_recordings", augmentations=False)
dataloader = dataset.get_loader(batch_size=16, shuffle=False, num_workers=8)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

emb_dir = Path("db")
emb_dir.mkdir(exist_ok=True)

for i, batch in enumerate(dataloader):
    x = batch
    x = x.to(device)
    with torch.no_grad():
        y = model(x)
    for j, emb in enumerate(y):
        torch.save(emb, emb_dir / f"{i}_{j}.pt")
