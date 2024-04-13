"""Dataset method for the training database."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from audiomentations import AddGaussianSNR, ClippingDistortion, Gain
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
        bg_dir="./bg",
        augmentations=True,
        # audio
        filetype: str = "wav",
        target_duration: float = 30.0,  # in seconds
        # representation
        sr: int = 8000,
        input_rep: str = "mel",
        input_rep_cfg: dict = dict(),
        # fingerprint
        fp_len: float = 1,  # in seconds
        debug=False,
    ):
        self.root = Path(root)
        self.sr = sr
        self.debug = debug  # True returns audio files
        self.filetype = filetype
        self.input_rep = input_rep
        self.input_rep_cfg = input_rep_cfg
        self.fp_len = fp_len
        self.fp_len_samples = self.fp_len * self.sr
        self.fp_hop = self.fp_len / 2  # fixed hop size
        self.target_duration = target_duration
        self.augmentations = augmentations
        try:
            self.bg_paths = list(
                Path(bg_dir).glob("*.wav")
            )  # get all files for background mix
        except:
            self.bg_paths = None

        # get filepaths as keys, indexed by names
        self.filepaths = {
            fp.stem: str(fp) for fp in list(self.root.glob(f"*.{self.filetype}"))
        }
        self.names = list(self.filepaths.keys())

        """Each file has multiple fingerprints. We'll use an index that maps
        each fingerprint to the corresponding file and position within it.
        We'll pad the start and end of each file to ensure all parts of the
        file are seen in 2 fingerprints. It should also help with performance
        at the start and end of the queries where there might be silence."""
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

    def waveform_augment(self, y, sr, prob=1):
        """Data augmentation function.
        Probability controls if each augmentation is applied. This
        gives us the option to have anchors that are not augmented."""
        transform = AddGaussianSNR(min_snr_in_db=0, max_snr_in_db=20, p=prob)
        y = transform(y, sr)
        transform = ClippingDistortion(max_percentile_threshold=10, p=prob)
        y = transform(y, sr)
        transform = Gain(min_gain_in_db=-15, max_gain_in_db=15, p=prob)
        # background mix
        if np.random.rand() < prob:
            y_bg, og_sr = sf.read(self.filepaths[np.random.choice(self.names)])
            y_bg = librosa.resample(y_bg, orig_sr=og_sr, target_sr=sr)
            # get a random crop of length fp_len
            start = np.random.randint(0, len(y_bg) - len(y))
            # if less than fp_len, pad with zeros
            if start + len(y) > len(y_bg):
                y_bg = np.pad(y_bg, (0, len(y) - len(y_bg) + start))
            # get random mix percentage over 0.5
            mix = np.random.rand() * 0.5 + 0.5
            y = y * mix + y_bg[start : start + len(y)] * (1 - mix)

        return y

    def spect_augment(self, X, prob=1):
        """Data augmentation function for spectrograms.
        Probability controls if augmentation is applied. This
        gives us the option to have anchors that are not augmented."""
        # set a random rectangle of area between 10% and 50% to zero
        if np.random.rand() < prob:
            mask = np.random.rand(*X.shape) < 0.5
            X[mask] = 0
        return X

    def __getitem__(self, idx):
        name = self.names[idx // self.fp_per_file]
        idx_in_file = idx % self.fp_per_file

        start_sec = idx_in_file * self.fp_hop - self.fp_hop  # remember padding

        y, sr = sf.read(self.filepaths[name])

        start_sr = int(start_sec * sr)
        start_sr = max(0, start_sr)  # to account for padding
        y = y[start_sr : int((start_sec + self.fp_len) * sr)]
        if sr != self.sr:
            # resample
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)

        if start_sec < 0:
            y = np.concatenate([np.zeros(int(self.fp_hop * self.sr)), y])
        elif start_sec + self.fp_len > self.target_duration:
            y = np.concatenate([y, np.zeros(int(self.fp_hop * self.sr))])

        y = y.astype(np.float32)

        if len(y.shape) == 1:
            y = np.expand_dims(y, 0)

        if self.augmentations:
            # run through augmentation chain
            y1_aug = self.waveform_augment(y, sr, prob=0.75)
            y2_aug = self.waveform_augment(y, sr, prob=1)

            # pad again just in case, augmentations can mess length up
            if y1_aug.shape[1] != self.fp_len_samples:
                y1_aug = np.pad(
                    y1_aug,
                    ((0, 0), (0, self.fp_len_samples - y1_aug.shape[1])),
                    mode="constant",
                )
                y1_aug = y1_aug[:, : self.fp_len_samples]
            if y2_aug.shape[1] != self.fp_len_samples:
                y2_aug = np.pad(
                    y2_aug,
                    ((0, 0), (0, self.fp_len_samples - y2_aug.shape[1])),
                    mode="constant",
                )
                y2_aug = y2_aug[:, : self.fp_len_samples]

            # compute the desired representations
            X1 = self.represent(torch.from_numpy(y1_aug).float())
            X2 = self.represent(torch.from_numpy(y2_aug).float())

            # compute spect-based augmentations
            X1_aug = self.spect_augment(X1, prob=0.5)
            X2_aug = self.spect_augment(X2, prob=1)

            if not self.debug:
                return (
                    X1_aug,
                    X2_aug,
                )
            else:
                return X1_aug, X2_aug, y1_aug, y2_aug

        if not self.debug:
            return self.represent(torch.from_numpy(y).float())
        else:
            return self.represent(torch.from_numpy(y).float()), y

    def get_loader(self, batch_size=32, shuffle=True, num_workers=0):
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )


if __name__ == "__main__":
    # test dataloader
    dataset = AudioDB(root="data/database_recordings", debug=True)
    dataloader = dataset.get_loader(batch_size=128, num_workers=22, shuffle=True)
    for i, (X1, X2, y1, y2) in enumerate(dataloader):
        # save the two audio files
        sf.write(f"y1_{i}.wav", y1[0][0].numpy(), 8000)
        sf.write(f"y2_{i}.wav", y2[0][0].numpy(), 8000)
        if i >= 3:
            break
