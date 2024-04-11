import torch
import torch.nn.functional as F
from torch import optim

from dataset import AudioDB
from model import VGGlike

# Define the model
model = VGGlike()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


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

    for batch in dataloader:
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

    return total_loss / len(dataloader)


# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

dataloader = AudioDB(root="data/database_recordings").get_loader(
    batch_size=32, num_workers=16
)

for epoch in range(100):  # Number of epochs
    loss = train(model, dataloader, optimizer, device)
    print(f"Epoch {epoch + 1}, Loss: {loss}")
