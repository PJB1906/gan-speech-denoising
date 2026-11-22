"""Matched Transformer and Conformer-style complex spectral masking models."""

import math

import torch
from torch import nn
from torch.nn import functional as F

ATTENTION_MODELS = ("transformer", "conformer")
FFT_SIZE = 512
HOP = 128
CONTEXT_FRAMES = 256
COMPRESSION = 0.3
WAVEFORM_WEIGHT = 0.35


class TemporalConvolution(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, width * 2)
        self.depthwise = nn.Conv1d(width, width, 9, padding=4, groups=width)
        self.project = nn.Linear(width, width)

    def forward(self, sequence):
        gated = F.glu(self.gate(self.norm(sequence)), dim=-1)
        local = self.depthwise(gated.transpose(1, 2)).transpose(1, 2)
        return self.project(F.silu(local))


class AttentionBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, num_heads=4, batch_first=True)
        self.convolution = None
        self.feedforward = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)
        )

    def forward(self, sequence):
        query = self.attention_norm(sequence)
        context, _ = self.attention(query, query, query, need_weights=False)
        sequence = sequence + context
        if self.convolution is not None:
            sequence = sequence + self.convolution(sequence)
        return sequence + self.feedforward(sequence)


class SpectralEnhancer(nn.Module):
    """Estimate gain and phase corrections with bounded temporal attention."""

    def __init__(self, use_convolution=False):
        super().__init__()
        widths = (3, 12, 20, 28)
        self.encoder = nn.ModuleList(
            nn.Conv2d(source, target, 3, stride=(2, 1), padding=1)
            for source, target in zip(widths, widths[1:])
        )
        self.bottleneck_bins = (FFT_SIZE // 2) // 8 + 1
        self.bottleneck_channels = widths[-1]
        features = self.bottleneck_channels * self.bottleneck_bins
        width = 80
        self.to_sequence = nn.Linear(features, width)
        self.context = nn.Sequential(AttentionBlock(width), AttentionBlock(width), nn.LayerNorm(width))
        self.from_sequence = nn.Linear(width, features)
        self.decoder = nn.ModuleList(
            nn.ConvTranspose2d(source, target, 3, stride=(2, 1), padding=1)
            for source, target in ((28, 20), (20, 12), (12, 8))
        )
        self.mask = nn.Conv2d(8, 2, 3, padding=1)
        nn.init.zeros_(self.mask.weight)
        nn.init.zeros_(self.mask.bias)
        positions = torch.arange(CONTEXT_FRAMES, dtype=torch.float32)[:, None]
        frequencies = torch.exp(torch.arange(0, width, 2) * (-math.log(10_000) / width))
        angles = positions * frequencies
        encoding = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
        self.register_buffer("positions", encoding, persistent=False)
        self.register_buffer("window", torch.hann_window(FFT_SIZE), persistent=False)
        if use_convolution:
            # Build every common weight before allocating variant-specific weights.
            for block in self.context[:2]:
                block.convolution = TemporalConvolution(width)

    def analyze(self, waveform):
        return torch.stft(
            waveform.squeeze(1).float(), FFT_SIZE, HOP, window=self.window,
            pad_mode="constant", return_complex=True,
        )

    def forward(self, noisy, *, return_spectrum=False):
        with torch.autocast(noisy.device.type, enabled=False):
            scale = noisy.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(0.005)
            spectrum = self.analyze(noisy / scale)
            magnitude = spectrum.abs()
            phase = spectrum / magnitude.clamp_min(1e-4)
            features = torch.stack((magnitude.log1p(), phase.real, phase.imag), dim=1)
        skips = []
        for down in self.encoder:
            features = F.silu(down(features))
            skips.append(features)
        sequence = self.to_sequence(features.flatten(1, 2).transpose(1, 2))
        chunks = [
            self.context(chunk + 0.1 * self.positions[:chunk.shape[1]].to(chunk.dtype))
            for chunk in sequence.split(CONTEXT_FRAMES, dim=1)
        ]
        frames = self.from_sequence(torch.cat(chunks, dim=1)).transpose(1, 2)
        context = frames.reshape(-1, self.bottleneck_channels, self.bottleneck_bins, frames.shape[-1])
        features = features + context
        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            features = F.silu(up(features) + skip)
        mask = self.mask(F.silu(self.decoder[-1](features))).float()
        with torch.autocast(noisy.device.type, enabled=False):
            gain = 2 * mask[:, 0].sigmoid()
            phase_shift = (math.pi / 2) * mask[:, 1].tanh()
            estimate = spectrum * torch.polar(gain, phase_shift)
            waveform = torch.istft(estimate, FFT_SIZE, HOP, window=self.window, length=noisy.shape[-1])
            waveform = waveform.unsqueeze(1) * scale
        if return_spectrum:
            return waveform, estimate, scale
        return waveform


def enhancement_loss(model, noisy, clean):
    waveform, estimate, scale = model(noisy, return_spectrum=True)
    with torch.autocast(noisy.device.type, enabled=False):
        target = model.analyze(clean / scale)
        magnitude = estimate.abs().clamp_min(1e-4)
        target_magnitude = target.abs().clamp_min(1e-4)
        magnitude_loss = F.l1_loss(magnitude.pow(COMPRESSION), target_magnitude.pow(COMPRESSION))
        complex_loss = (
            estimate * magnitude.pow(COMPRESSION - 1) - target * target_magnitude.pow(COMPRESSION - 1)
        ).abs().mean()
        waveform_loss = ((waveform - clean) / scale).abs().mean()
        return 0.7 * magnitude_loss + 0.3 * complex_loss + WAVEFORM_WEIGHT * waveform_loss
