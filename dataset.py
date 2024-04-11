"""Dataset method for the training database."""

from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ChromaSTFT(nn.Module):
    """Wrap chroma_stft from librosa as PyTorch module
    for convenience."""

    def __init__(self, sr, n_fft, hop_length, n_chroma):
        super(ChromaSTFT, self).__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chroma = n_chroma

    def forward(self, y):
        y = y.numpy()
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma,
        )
        return torch.from_numpy(chroma)


class AudioDB(Dataset):
    """Dataset class for the training database. The files provided
    are between 29.992 and 30.488 seconds long, sampled at 22050.
    We'll assume they are exactly 30 seconds long and trim/pad
    accordingly, to simplify the indexing of individual fingerprints.
    """

    def __init__(
        self,
        root,
        # audio
        filetype: str = "wav",
        target_duration: float = 30.0,  # in seconds
        # representation
        sr: int = 8000,
        input_rep: str = "mel",
        input_rep_cfg: dict = dict(),
        # fingerprint
        fp_len: float = 1,  # in seconds
    ):
        self.root = Path(root)
        self.sr = sr
        self.filetype = filetype
        self.input_rep = input_rep
        self.input_rep_cfg = input_rep_cfg
        self.fp_len = fp_len
        self.fp_hop = self.fp_len / 2  # fixed hop size
        self.target_duration = target_duration

        # get filepaths as keys, indexed by names
        self.filepaths = {
            fp.stem: str(fp) for fp in list(self.root.glob(f"*.{self.filetype}"))
        }
        self.names = list(self.filepaths.keys())

        # Each file has multiple fingerprints. We'll use an index that maps
        # each fingerprint to the corresponding file and position within it.
        # We'll pad the start and end of each file to ensure all parts of the
        # file are seen in 2 fingerprints. It should also help with performance
        # at the start and end of the queries where there might be silence.
        self.fp_per_file = int(target_duration / self.fp_hop)

        if self.input_rep == "mel":
            self.represent = nn.Sequential(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sr,
                    n_fft=self.input_rep_cfg.get("n_fft", 1024),
                    win_length=self.input_rep_cfg.get("win_length", 1024),
                    hop_length=self.input_rep_cfg.get("hop_length", 256),
                    n_mels=self.input_rep_cfg.get("n_mels", 64),
                    f_min=self.input_rep_cfg.get("f_min", 300),
                    f_max=self.input_rep_cfg.get("f_max", 4000),
                    power=self.input_rep_cfg.get("power", 1.0),
                ),
                torchaudio.transforms.AmplitudeToDB(),
            )
        elif self.input_rep == "chromagram":
            self.represent = ChromaSTFT(
                sr=self.sr,
                n_fft=self.input_rep_cfg.get("n_fft", 1024),
                hop_length=self.input_rep_cfg.get("hop_length", 256),
                n_chroma=self.input_rep_cfg.get("n_chroma", 36),
            )

    def __len__(self):
        return len(self.names * self.fp_per_file)

    def augment(self, y, prob=1):
        """Data augmentation function.
        Probability controls if augmentation is applied. This
        gives us the option to have anchors that are not augmented."""
        return y

    def __getitem__(self, idx):
        name = self.names[idx // self.fp_per_file]
        idx_in_file = idx % self.fp_per_file

        start_sec = idx_in_file * self.fp_hop - self.fp_hop  # remember padding

        waveform, sample_rate = torchaudio.load(self.filepaths[name])

        # Resample the audio if the sample rate is not the same as self.sr
        if sample_rate != self.sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sr
            )
            waveform = resampler(waveform)

        # Trim or pad the audio
        start_sample = int(start_sec * self.sr)
        end_sample = start_sample + int(self.fp_len * self.sr)
        if start_sample < 0:
            padding = torch.zeros((waveform.shape[0], -start_sample))
            waveform = torch.cat((padding, waveform), dim=-1)
            start_sample = 0
        if end_sample > waveform.shape[-1]:
            padding = torch.zeros((waveform.shape[0], end_sample - waveform.shape[-1]))
            waveform = torch.cat((waveform, padding), dim=-1)

        y = waveform[:, start_sample:end_sample]

        # run through augmentation chain
        y1_aug = self.augment(y, prob=0.5)
        y2_aug = self.augment(y, prob=1)

        # compute the desired representations
        X1 = self.represent(y1_aug)
        X2 = self.represent(y2_aug)

        return X1, X2

    def get_loader(self, batch_size=32, shuffle=True, num_workers=0):
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
