"""Offline browser demo for comparing speech enhancement models."""

import argparse
import html
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import av
import gradio as gr
import numpy as np
from matplotlib.figure import Figure
from scipy.signal import stft

from . import SAMPLE_RATE
from .data import write_wav
from .recording import PREVIEW_SECONDS, render_audio, timecode
from .train import load_generator, resolve_device
from .transcription import MODEL_PATH, load_transcriber, transcribe_audio


def full_recordings(paths, duration):
    """Serve long files through native players that fetch audio as needed."""
    if duration <= PREVIEW_SECONDS:
        return ""
    players = []
    for label, path in paths.items():
        url = "/gradio_api/file=" + quote(path.resolve().as_posix(), safe="/:")
        name = html.escape(label)
        players.append(
            f'<div><strong>{name}</strong><br><audio controls preload="none" '
            f'src="{url}" aria-label="{name}"></audio><br>'
            f'<a href="{url}" download="{path.name}">Download {name}</a></div>'
        )
    return (
        f"<h3>Full recordings</h3><p>The waveform and spectrogram previews show "
        f"the first {PREVIEW_SECONDS} seconds.</p>" + "".join(players)
    )


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
            axis.text(0.5, 0.5, "No clean reference provided", transform=axis.transAxes, ha="center")
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


def create_demo(baseline, spectral, device, temp_root, conformer=None, whisper_model=MODEL_PATH, whisper_device=None):
    models = {}
    variants = (
        ("baseline", "Baseline GAN", baseline), ("spectral", "MR-STFT GAN", spectral),
        ("conformer", "Conformer", conformer),
    )
    for variant, label, path in variants:
        if path is None:
            continue
        model, checkpoint = load_generator(path, device)
        if checkpoint["variant"] != variant:
            raise ValueError(f"{path}: expected a {variant} checkpoint")
        models[label] = model
    transcript_source = "Conformer" if "Conformer" in models else "MR-STFT GAN"
    transcriber = None

    def compare(noisy_path, clean_path, recording_path=None, progress=gr.Progress()):
        nonlocal transcriber
        source = noisy_path or recording_path
        if not source:
            raise gr.Error("Upload an audio file or record a voice note first.")
        destination = Path(tempfile.mkdtemp(dir=temp_root))

        def update(stage, beginning, span):
            def report(done, total):
                ratio = min(1, done / total) if total else 0
                extent = f" / {timecode(total)}" if total else ""
                progress(beginning + span * ratio, desc=f"{stage} {timecode(done)}{extent}")
            return report

        try:
            paths, previews, duration = render_audio(
                source, "Noisy", models, device, destination, update("Enhancing audio", 0, 0.5)
            )
            if clean_path:
                clean_files, clean_preview, clean_duration = render_audio(
                    clean_path, "Clean reference", {}, device, destination, update("Decoding reference", 0.5, 0.05)
                )
                paths.update(clean_files)
                previews.update(clean_preview)
                duration = max(duration, clean_duration)
        except (av.error.FFmpegError, OSError, ValueError, RuntimeError) as error:
            shutil.rmtree(destination, ignore_errors=True)
            raise gr.Error(f"Could not process the recording. {error}") from error

        clips = [(label, previews.get(label)) for label in ["Noisy", *models, "Clean reference"]]
        audio_paths = []
        for index, (_, preview) in enumerate(clips):
            output = destination / f"preview-{index}.wav"
            if preview is not None:
                write_wav(output, preview)
            audio_paths.append(str(output) if preview is not None else None)
        outputs = (*audio_paths, plot_spectrograms(clips), full_recordings(paths, duration))
        progress(0.55, desc="Loading Whisper")
        yield *outputs, "Transcribing the enhanced audio…", None
        try:
            if transcriber is None:
                transcriber = load_transcriber(whisper_model, whisper_device or device.type)
            transcript = transcribe_audio(
                transcriber, paths[transcript_source], update("Transcribing audio", 0.55, 0.45)
            )
            if transcript:
                transcript_path = destination / "transcript.txt"
                transcript_path.write_text(transcript + "\n", encoding="utf-8")
                yield *outputs, transcript, str(transcript_path)
            else:
                yield *outputs, "No speech detected.", None
        except (OSError, ValueError, RuntimeError) as error:
            gr.Warning("Audio processing finished. Transcription could not be completed.")
            yield *outputs, f"Transcription error: {error}", None

    with gr.Blocks(title="Voice Note Denoiser") as app:
        gr.Markdown("# Voice Note Denoiser")
        with gr.Row():
            noisy = gr.File(label="Audio recording", type="filepath", file_types=["audio"])
            clean = gr.File(label="Clean reference (optional)", type="filepath", file_types=["audio"])
        with gr.Accordion("Record with microphone", open=False):
            recording = gr.Audio(label="Microphone", type="filepath", sources=["microphone"])
        button = gr.Button("Denoise and transcribe", variant="primary")
        with gr.Row():
            noisy_result = gr.Audio(label="Noisy", interactive=False)
            clean_result = gr.Audio(label="Clean reference", interactive=False)
        with gr.Row():
            denoised_results = [gr.Audio(label=label, interactive=False) for label in models]
        complete_audio = gr.HTML()
        transcript = gr.Textbox(
            label=f"Timestamped transcript · {transcript_source}", lines=12, max_lines=22,
            buttons=["copy"], interactive=False, autoscroll=False,
        )
        transcript_file = gr.File(label="Download transcript", interactive=False)
        plot = gr.Plot(label="Spectrograms")
        button.click(
            compare, [noisy, clean, recording],
            [noisy_result, *denoised_results, clean_result, plot, complete_audio, transcript, transcript_file],
        )
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/baseline.pt"))
    parser.add_argument("--spectral", type=Path, default=Path("artifacts/spectral.pt"))
    bundled_conformer = Path("artifacts/conformer.pt")
    parser.add_argument("--conformer", type=Path, default=bundled_conformer if bundled_conformer.is_file() else None)
    parser.add_argument("--port", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--whisper-model", type=Path, default=MODEL_PATH)
    parser.add_argument("--whisper-device", choices=("cuda", "cpu"), help="Defaults to the denoising device")
    args = parser.parse_args()
    device = resolve_device(args.device)
    with tempfile.TemporaryDirectory(prefix="speech-gan-") as temporary:
        temp_root = Path(temporary)
        output_dir = temp_root / "recordings"
        output_dir.mkdir()
        os.environ.setdefault("GRADIO_TEMP_DIR", str(temp_root / "gradio"))
        app = create_demo(
            args.baseline, args.spectral, device, output_dir,
            args.conformer, args.whisper_model, args.whisper_device,
        )
        app.launch(
            server_name="127.0.0.1", server_port=args.port, share=False,
            allowed_paths=[str(output_dir)], max_file_size=None,
        )


if __name__ == "__main__":
    main()
