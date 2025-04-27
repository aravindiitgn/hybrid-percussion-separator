# dataset.py
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import musdb

SAMPLE_RATE = 44100

class MUSDB18Dataset(Dataset):
    def __init__(self, subset="train", duration=3.0, download=False):
        """
        Args:
            subset (str): "train" or "test"
            duration (float): Duration (in seconds) for each segment.
            download (bool): Whether to download MUSDB18 if not present.
        """
        self.mus = musdb.DB(subsets=[subset], download=download)
        self.tracks = list(self.mus)
        self.duration = duration
        self.sample_rate = SAMPLE_RATE
        self.num_samples = int(duration * self.sample_rate)

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]
        # Get mixture (shape: (nb_samples, 2)) and drums (shape: (nb_samples, 2))
        mixture = track.audio  
        drums = track.stems[1]  # drums stem is at index 1.
        
        # Resample if necessary.
        if track.rate != self.sample_rate:
            mixture_tensor = torch.tensor(mixture.T, dtype=torch.float32)
            drums_tensor = torch.tensor(drums.T, dtype=torch.float32)
            resampler = torchaudio.transforms.Resample(orig_freq=track.rate, new_freq=self.sample_rate)
            mixture = resampler(mixture_tensor).numpy().T
            drums = resampler(drums_tensor).numpy().T
        
        # Convert to mono by averaging channels if stereo.
        if mixture.ndim == 2 and mixture.shape[1] > 1:
            mixture = mixture.mean(axis=1, keepdims=True)
        if drums.ndim == 2 and drums.shape[1] > 1:
            drums = drums.mean(axis=1, keepdims=True)
        
        total_samples = mixture.shape[0]
        if total_samples > self.num_samples:
            start = random.randint(0, total_samples - self.num_samples)
            mixture = mixture[start:start+self.num_samples, :]
            drums = drums[start:start+self.num_samples, :]
        else:
            pad_length = self.num_samples - total_samples
            mixture = np.pad(mixture, ((0, pad_length), (0, 0)), mode='constant')
            drums = np.pad(drums, ((0, pad_length), (0, 0)), mode='constant')
        
        mixture = torch.tensor(mixture.T, dtype=torch.float32)  # shape: (1, num_samples)
        drums = torch.tensor(drums.T, dtype=torch.float32)        # shape: (1, num_samples)
        return mixture, drums
