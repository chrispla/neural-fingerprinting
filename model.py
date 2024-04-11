"""VGGish encoder, adapted from:
https://github.com/minzwon/sota-music-tagging-models"""

import torch
from torch import nn


class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            shape,
            stride=stride,
            padding=shape // 2,
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class VGG(nn.Module):

    def __init__(
        self,
        n_channels=128,
    ):
        super(VGG, self).__init__()

        # CNN
        self.layer1 = Conv_2d(1, n_channels, pooling=2)
        self.layer2 = Conv_2d(n_channels, n_channels, pooling=2)
        self.layer3 = Conv_2d(n_channels, n_channels * 2, pooling=2)
        self.layer4 = Conv_2d(n_channels * 2, n_channels * 2, pooling=2)
        self.layer5 = Conv_2d(n_channels * 2, n_channels * 2, pooling=2)
        self.layer6 = Conv_2d(n_channels * 2, n_channels * 2, pooling=2)
        # self.layer7 = Conv_2d(n_channels * 2, n_channels * 4, pooling=2)

        # projection layer for contrastive learning
        self.dense1 = nn.Linear(n_channels * 4, 128)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        # x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x


if __name__ == "__main__":
    from dataset import AudioDB

    model = VGG()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset = AudioDB(root="data/database_recordings", input_rep="mel")
    dataloader = dataset.get_loader(batch_size=32, num_workers=4, shuffle=True)

    for X1, X2 in dataloader:
        model(X1.to(device))
