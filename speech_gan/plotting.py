"""Shared-scale spectrogram previews for both demo interfaces."""

import numpy as np
from matplotlib.figure import Figure
from scipy.signal import stft

from . import SAMPLE_RATE


def plot_spectrograms(clips):
    columns = 2 if len(clips) <= 4 else 3
    figure = Figure(figsize=(5 * columns, 5.6), layout="constrained")
    axes = figure.subplots(2, columns, sharex=True, sharey=True).ravel()
    duration = max(len(audio) for _, audio in clips if audio is not None) / SAMPLE_RATE
    for axis, (label, audio) in zip(axes, clips):
        axis.set_title(label, loc="left")
        axis.set_xlim(0, duration)
        axis.set_ylim(0, SAMPLE_RATE / 2)
        if audio is None:
            axis.text(0.5, 0.5, "No audio in this preview", transform=axis.transAxes, ha="center")
            continue
        if len(audio) < 512:
            audio = np.pad(audio, (0, 512 - len(audio)))
        frequency, time, magnitudes = stft(audio, fs=SAMPLE_RATE, nperseg=512, noverlap=384)
        db = 20 * np.log10(np.maximum(np.abs(magnitudes), 1e-8))
        image = axis.pcolormesh(time, frequency, db, shading="auto", vmin=-80, vmax=0, cmap="magma")
    figure.supxlabel("Time (s)")
    figure.supylabel("Frequency (Hz)")
    for axis in axes[len(clips):]:
        axis.set_axis_off()
    if len(clips) % columns:
        for axis in axes[len(clips) - columns:columns]:
            axis.tick_params(labelbottom=True)
    figure.colorbar(image, ax=axes[:len(clips)].tolist(), label="STFT magnitude (dB)")
    return figure
