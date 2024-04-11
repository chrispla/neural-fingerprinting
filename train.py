"""Train the VGGish-like model using contrastive learning"""

from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

import wandb
from dataset import AudioDB
from model import VGGlike

run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Initialize a new wandb run
wandb.init(project="neuralfp", name=run_name)

# Define the model
model = VGGlike()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Log the model's architecture to wandb
wandb.watch(model)


# Define the contrastive loss function
def contrastive_loss(x1, x2, temperature=0.1):
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)

    # Compute similarity matrix
    sim_matrix = torch.einsum("ik,jk->ij", x1, x2) / temperature

    # Compute loss
    loss = F.cross_entropy(sim_matrix, torch.arange(len(x1)).to(sim_matrix.device))

    return loss


# Define the training loop
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    # Add a progress bar
    progress_bar = tqdm(
        dataloader, desc="Training", total=len(dataloader), leave=True, ncols=80
    )

    for batch in progress_bar:
        x1, x2 = batch
        x1, x2 = x1.to(device), x2.to(device)

        optimizer.zero_grad()

        # Forward pass
        y1, y2 = model(x1), model(x2)

        # Compute loss
        loss = contrastive_loss(y1, y2)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        # Update the progress bar
        progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

        # Log the loss to wandb
        wandb.log({"loss": total_loss / (progress_bar.n + 1)})

    return total_loss / len(dataloader)


# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

dataloader = AudioDB(root="data/database_recordings").get_loader(
    batch_size=128, num_workers=22, shuffle=True
)

Path("ckpt").mkdir(exist_ok=True)

for epoch in range(100):  # Number of epochs
    loss = train(model, dataloader, optimizer, device)
    print(f"Epoch {epoch + 1}, Loss: {loss}")

    # Log the epoch and loss to wandb
    wandb.log({"epoch": epoch + 1, "loss": loss})

    if (epoch + 1) % 2 == 0:
        torch.save(model.state_dict(), f"ckpt/model_{epoch + 1}_loss_{loss:.4f}.pt")
