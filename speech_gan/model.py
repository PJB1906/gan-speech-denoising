"""Small conditional waveform GAN and the additional spectrogram objective."""

import torch
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    """Four-level 1D U-Net that predicts an additive waveform correction."""

    def __init__(self):
        super().__init__()
        channels = (1, 16, 24, 40, 64)
        self.encoder = nn.ModuleList(
            nn.Conv1d(source, target, 8, stride=4, padding=2)
            for source, target in zip(channels, channels[1:])
        )
        self.context = nn.Sequential(
            nn.Conv1d(64, 64, 3, dilation=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 3, dilation=4, padding=4),
        )
        self.decoder = nn.ModuleList(
            nn.ConvTranspose1d(source, target, 8, stride=4, padding=2)
            for source, target in ((64, 40), (40, 24), (24, 16), (16, 8))
        )
        self.fuse = nn.ModuleList(nn.Conv1d(width * 2, width, 5, padding=2) for width in (40, 24, 16))
        self.correction = nn.Conv1d(8, 1, 7, padding=3)
        nn.init.normal_(self.correction.weight, std=0.001)
        nn.init.zeros_(self.correction.bias)

    def forward(self, noisy):
        length = noisy.shape[-1]
        features = F.pad(noisy, (0, -length % 256))
        skips = []
        for down in self.encoder:
            features = F.leaky_relu(down(features), 0.2)
            skips.append(features)
        features = F.leaky_relu(features + self.context(features), 0.2)
        for up, fuse, skip in zip(self.decoder, self.fuse, reversed(skips[:-1])):
            features = F.leaky_relu(up(features), 0.2)
            features = F.leaky_relu(fuse(torch.cat((features, skip), dim=1)), 0.2)
        features = F.leaky_relu(self.decoder[-1](features), 0.2)
        return noisy + self.correction(features)[..., :length]


class Discriminator(nn.Module):
    """Score local waveform patches conditioned on the original noisy clip."""

    def __init__(self):
        super().__init__()
        layers = []
        channels = (2, 16, 32, 48, 64)
        for source, target in zip(channels, channels[1:]):
            layers.extend((nn.Conv1d(source, target, 8, stride=4, padding=2), nn.LeakyReLU(0.2)))
        self.network = nn.Sequential(*layers, nn.Conv1d(64, 1, 3, padding=1))

    def forward(self, noisy, candidate):
        return self.network(torch.cat((noisy, candidate), dim=1))


class MultiResolutionSTFT(nn.Module):
    """Average spectral convergence and log-magnitude L1 across FFT sizes."""

    FFT_SIZES = (256, 512, 1024)

    def __init__(self):
        super().__init__()
        for size in self.FFT_SIZES:
            self.register_buffer(f"window_{size}", torch.hann_window(size), persistent=False)

    def forward(self, estimate, clean):
        batch = torch.cat((estimate, clean), dim=0).squeeze(1).float()
        losses = []
        for size in self.FFT_SIZES:
            spectrum = torch.stft(
                batch, n_fft=size, hop_length=size // 4, window=getattr(self, f"window_{size}"),
                return_complex=True, pad_mode="constant",
            )
            estimated, target = spectrum.abs().clamp_min(1e-5).chunk(2)
            convergence = (
                torch.linalg.vector_norm(estimated - target, dim=(-2, -1))
                / torch.linalg.vector_norm(target, dim=(-2, -1)).clamp_min(1e-5)
            )
            losses.append(convergence.mean() + F.l1_loss(estimated.log(), target.log()))
        return torch.stack(losses).mean()
