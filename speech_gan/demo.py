"""Offline browser demo for comparing the two speech enhancement generators."""

import argparse
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np
import soundfile as sf
from matplotlib.figure import Figure
from scipy.signal import resample_poly, stft

from . import SAMPLE_RATE
from .data import write_wav
from .metrics import enhance
from .train import load_generator, resolve_device

MAX_DURATION = 60


def load_note(path):
    with sf.SoundFile(path) as source:
        if not 0 < source.frames <= MAX_DURATION * source.samplerate:
            raise ValueError("Choose a nonempty WAV, FLAC, OGG or MP3 up to 60 seconds long.")
        rate = source.samplerate
        audio = source.read(dtype="float32", always_2d=True).mean(axis=1)
    if not np.isfinite(audio).all():
        raise ValueError("The audio contains non-finite samples.")
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor)
    return audio.clip(-1, 1)


def plot_spectrograms(clips):
    figure = Figure(figsize=(10, 5.6), layout="constrained")
    axes = figure.subplots(2, 2, sharex=True, sharey=True).ravel()
    duration = max(len(audio) for _, audio in clips if audio is not None) / SAMPLE_RATE
    for axis, (label, audio) in zip(axes, clips):
        axis.set_title(label, loc="left")
        axis.set_xlim(0, duration)
        axis.set_ylim(0, SAMPLE_RATE / 2)
        if audio is None:
            axis.text(0.5, 0.5, "No clean reference provided", transform=axis.transAxes, ha="center")
            continue
        if len(audio) < 512:
            audio = np.pad(audio, (0, 512 - len(audio)))
        frequency, time, magnitudes = stft(audio, fs=SAMPLE_RATE, nperseg=512, noverlap=384)
        db = 20 * np.log10(np.maximum(np.abs(magnitudes), 1e-8))
        image = axis.pcolormesh(time, frequency, db, shading="auto", vmin=-80, vmax=0, cmap="magma")
    figure.supxlabel("Time (s)")
    figure.supylabel("Frequency (Hz)")
    figure.colorbar(image, ax=axes.tolist(), label="STFT magnitude (dB)")
    return figure


def create_demo(baseline, spectral, device, temp_root):
    models = {}
    for variant, path in (("baseline", baseline), ("spectral", spectral)):
        model, checkpoint = load_generator(path, device)
        if checkpoint["variant"] != variant:
            raise ValueError(f"{path}: expected a {variant} checkpoint")
        models[variant] = model

    def compare(noisy_path, clean_path):
        if not noisy_path:
            raise gr.Error("Upload or record a noisy voice note first.")
        try:
            noisy = load_note(noisy_path)
            clean = load_note(clean_path) if clean_path else None
        except (ValueError, RuntimeError) as error:
            raise gr.Error(str(error)) from error
        baseline_audio, _ = enhance(models["baseline"], noisy, device)
        spectral_audio, _ = enhance(models["spectral"], noisy, device)
        clips = [("Noisy", noisy), ("Baseline GAN", baseline_audio),
                 ("MR-STFT GAN", spectral_audio), ("Clean reference", clean)]
        destination = Path(tempfile.mkdtemp(dir=temp_root))
        audio_paths = []
        for index, (_, audio) in enumerate(clips):
            path = destination / f"{index}.wav"
            if audio is not None:
                write_wav(path, audio)
            audio_paths.append(str(path) if audio is not None else None)
        return *audio_paths, plot_spectrograms(clips)

    with gr.Blocks(title="Voice Note Denoiser", delete_cache=(3600, 3600)) as app:
        gr.Markdown("# Voice Note Denoiser")
        with gr.Row():
            noisy = gr.Audio(label="Noisy voice note", type="filepath", sources=["upload", "microphone"])
            clean = gr.Audio(label="Clean reference (optional)", type="filepath", sources=["upload"])
        button = gr.Button("Denoise and compare", variant="primary")
        with gr.Row():
            noisy_result = gr.Audio(label="Noisy", interactive=False)
            baseline_result = gr.Audio(label="Baseline GAN", interactive=False)
            spectral_result = gr.Audio(label="MR-STFT GAN", interactive=False)
            clean_result = gr.Audio(label="Clean reference", interactive=False)
        plot = gr.Plot(label="Spectrograms")
        button.click(compare, [noisy, clean], [noisy_result, baseline_result, spectral_result, clean_result, plot])
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/baseline.pt"))
    parser.add_argument("--spectral", type=Path, default=Path("artifacts/spectral.pt"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    with tempfile.TemporaryDirectory(prefix="speech-gan-") as temp_root:
        app = create_demo(args.baseline, args.spectral, device, temp_root)
        app.launch(server_name="127.0.0.1", share=False, allowed_paths=[temp_root])


if __name__ == "__main__":
    main()
