from dataset import AudioDB
from model import VGG

dataset = AudioDB(
    root="data/database_recordings", sr=8000, input_rep="chromagram", fp_len=0.5
)
train_loader = dataset.get_loader(batch_size=32, num_workers=4, shuffle=True)

model = VGG()

for X1, X2 in train_loader:
    print(X1.shape, X2.shape)
    break
