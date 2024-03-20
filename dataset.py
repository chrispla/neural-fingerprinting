"""Dataset method for the training database."""

from pathlib import Path

import librosa
import numpy as np
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset


class AudioDB(Dataset):
    """Dataset class for the training database. The files provided
    are between 29.992 and 30.488 seconds long, sampled at 22050.
    We'll assume they are exactly 15 seconds long and trim/pad
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
        input_rep_cfg: dict = None,
        # fingerprint
        fp_len: float = 1.0,  # in seconds
    ):
        self.root = Path(root)
        self.sr = sr
        self.filetype = filetype
        self.input_rep = input_rep
        self.input_rep_cfg = input_rep_cfg
        self.fp_len = fp_len
        self.fp_hop = self.fp_len / 2  # fixed hop size

        # get filepaths as keys, indexed by names
        self.filepaths = {
            fp.stem: fp for fp in list(self.root.glob(f"*.{self.filetype}"))
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
                    n_mels=self.input_rep_cfg.get("n_mels", 256),
                    f_min=self.input_rep_cfg.get("f_min", 300),
                    f_max=self.input_rep_cfg.get("f_max", 4000),
                    power=self.input_rep_cfg.get("power", 1.0),
                ),
                torchaudio.transforms.AmplitudeToDB(),
            )
        elif self.input_rep == "chromagram":
            self.represent = nn.Sequential(
                torchaudio.transforms.ChromaSTFT(
                    sample_rate=self.sr,
                    n_fft=self.input_rep_cfg.get("n_fft", 1024),
                    hop_length=self.input_rep_cfg.get("hop_length", 256),
                    n_chroma=self.input_rep_cfg.get("n_chroma", 12),
                ),
                torchaudio.transforms.AmplitudeToDB(),
            )

    def __len__(self):
        return len(self.names * self.fp_per_file)

    def augment(self, y):
        return y

    def __getitem__(self, idx):
        name = self.names[idx // self.fp_per_file]
        idx_in_file = idx % self.fp_per_file

        start_sec = idx_in_file * self.fp_hop - self.fp.hop  # remember padding

        # librosa allows us to load a specific part of a wav file.
        # It will handle negative seconds in the offset and durations
        # longer than the audio duration by loading just the audio
        # that does exist. We'll handle padding after.
        y, _ = librosa.load(
            self.filepaths[name],
            sr=self.sr,
            offset=start_sec,
            duration=self.fp_len,
        )

        if start_sec < 0:
            y = np.concatenate([np.zeros(self.fp_hop * self.sr), y])
        elif start_sec + self.fp_len > self.target_duration:
            y = np.concatenate([y, np.zeros(self.fp_hop * self.sr)])

        # run through augmentation chain
        y_aug = self.augment(y)

        # compute the desired representation
        X = self.represent(y)
        X_aug = self.represent(y_aug)

        return X, X_aug
