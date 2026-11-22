"""Hosted frontend for the speech enhancement and transcription pipeline."""

import io
from pathlib import Path
import shutil
import sys
import tempfile
from threading import Lock

import av
from matplotlib.figure import Figure
import numpy as np
import soundfile as sf
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from speech_gan import SAMPLE_RATE
from speech_gan.plotting import plot_spectrograms
from speech_gan.recording import PREVIEW_SECONDS, render_audio, timecode
from speech_gan.train import load_generator, resolve_device
from speech_gan.transcription import MODEL_FILES, MODEL_PATH, download_model, load_transcriber, transcribe_audio

SUPPORTED_AUDIO = ["wav", "flac", "mp3", "m4a", "aac", "ogg", "opus", "webm", "aiff"]
MODEL_VARIANTS = {"Baseline GAN": "baseline", "MR-STFT GAN": "spectral", "Conformer": "conformer"}


@st.cache_resource
def shared_models():
    device = resolve_device("cpu")
    torch.set_num_threads(2)
    models = {}
    for label, variant in MODEL_VARIANTS.items():
        checkpoint_path = ROOT / "artifacts" / f"{variant}.pt"
        model, checkpoint = load_generator(checkpoint_path, device)
        if checkpoint["variant"] != variant:
            raise ValueError(f"Expected a {variant} checkpoint at {checkpoint_path}.")
        models[label] = model
    return device, models, Lock()


@st.cache_resource
def whisper_model():
    model_path = ROOT / MODEL_PATH
    if any(not (model_path / name).is_file() for name in MODEL_FILES):
        download_model(model_path)
    return load_transcriber(model_path, "cpu")


def save_upload(upload, destination):
    upload.seek(0)
    with destination.open("wb") as output:
        shutil.copyfileobj(upload, output, length=1024 * 1024)


def stage_recording(noisy, recording, clean):
    source = noisy or recording
    if source is None:
        st.warning("Upload an audio file or record a voice note first.")
        return
    storage = tempfile.TemporaryDirectory(prefix="voice-note-streamlit-")
    base = Path(storage.name)
    try:
        save_upload(source, base / "input.audio")
        if clean is not None:
            save_upload(clean, base / "reference.audio")
    except OSError:
        storage.cleanup()
        raise
    st.session_state.pending_recording = storage
    st.session_state.upload_generation = st.session_state.get("upload_generation", 0) + 1
    st.rerun()


def process_recording(storage):
    root = Path(storage.name)
    device, models, lock = shared_models()
    bar = st.progress(0.0, text="Waiting for the denoiser")

    def update(stage, beginning, span):
        def report(done, total):
            ratio = min(1, done / total) if total else 0
            extent = f" / {timecode(total)}" if total else ""
            bar.progress(beginning + span * ratio, text=f"{stage} {timecode(done)}{extent}")
        return report

    with lock:
        paths, _, duration = render_audio(
            root / "input.audio", "Noisy", models, device, root, update("Enhancing audio", 0, 0.5)
        )
        clean_path = root / "reference.audio"
        if clean_path.is_file():
            clean_files, _, clean_duration = render_audio(
                clean_path, "Clean reference", {}, device, root, update("Decoding reference", 0.5, 0.05)
            )
            paths.update(clean_files)
            duration = max(duration, clean_duration)
        bar.progress(0.55, text="Loading Whisper")
        transcript_error = None
        try:
            transcript = transcribe_audio(
                whisper_model(), paths["Conformer"], update("Transcribing audio", 0.55, 0.45)
            )
        except (OSError, ValueError, RuntimeError) as error:
            transcript = ""
            transcript_error = str(error)
    bar.empty()
    return {"paths": paths, "duration": duration, "transcript": transcript, "error": transcript_error}


def audio_preview(path, start):
    with sf.SoundFile(path) as source:
        sample = start * SAMPLE_RATE
        if sample >= source.frames:
            return None
        source.seek(sample)
        return source.read(PREVIEW_SECONDS * SAMPLE_RATE, dtype="float32")


def plot_waveforms(clips):
    figure = Figure(figsize=(12, 5.6), layout="constrained")
    axes = figure.subplots(3, 2, sharex=True, sharey=True).ravel()
    duration = max(len(audio) for _, audio in clips if audio is not None) / SAMPLE_RATE
    for axis, (label, audio) in zip(axes, clips):
        axis.set_title(label, loc="left")
        axis.set_xlim(0, duration)
        axis.set_ylim(-1, 1)
        if audio is None:
            axis.text(0.5, 0.5, "No audio in this preview", transform=axis.transAxes, ha="center")
        else:
            block_size = max(1, (len(audio) + 1999) // 2000)
            block_starts = np.arange(0, len(audio), block_size)
            lower = np.minimum.reduceat(audio, block_starts)
            upper = np.maximum.reduceat(audio, block_starts)
            axis.fill_between(block_starts / SAMPLE_RATE, lower, upper, color="#247aaf")
    axes[-1].set_axis_off()
    axes[-3].tick_params(labelbottom=True)
    figure.supxlabel("Time (s)")
    figure.supylabel("Amplitude")
    return figure


def show_audio_cards(paths, offset):
    clips = []
    for group in (("Noisy", "Clean reference"), tuple(MODEL_VARIANTS)):
        columns = st.columns(len(group), gap="small", vertical_alignment="top")
        for column, label in zip(columns, group):
            path = paths.get(label)
            preview = audio_preview(path, offset) if path is not None else None
            clips.append((label, preview))
            with column.container(border=True, key=f"audio-{len(clips)}"):
                st.markdown(f"**{label}**")
                if preview is not None:
                    wav = io.BytesIO()
                    sf.write(wav, preview, SAMPLE_RATE, format="WAV", subtype="PCM_16")
                    st.audio(wav.getvalue(), format="audio/wav")
                elif not paths:
                    st.caption("No audio yet")
                elif path is None:
                    st.caption("No reference uploaded")
                else:
                    st.caption("Recording ends before this point")
    return clips


def show_full_recordings(paths):
    options = list(paths)
    with st.container(border=True):
        st.markdown("**Full recordings**")
        selected = st.selectbox("Recording", options, index=options.index("Conformer"))
        path = paths[selected]
        player, download = st.columns(2, gap="small")
        load_audio = player.button("Load full audio player", width="stretch")
        download.download_button(
            "Download full audio", data=path.read_bytes, file_name=path.name, mime="audio/flac",
            on_click="ignore", width="stretch",
        )
        if load_audio:
            st.audio(str(path), format="audio/flac")


def show_comparison(result):
    paths = result["paths"] if result else {}
    offset = 0
    if result and result["duration"] > PREVIEW_SECONDS:
        offset = st.slider(
            "Preview start (seconds)", 0, max(1, int(result["duration"]) - 1),
            step=1, key="preview_start",
        )
        st.caption(f"Up to {PREVIEW_SECONDS} seconds from {timecode(offset)}.")

    clips = show_audio_cards(paths, offset)
    if result:
        show_full_recordings(paths)

    with st.container(border=True):
        transcript = result["transcript"] if result else ""
        st.text_area(
            "Timestamped transcript · Conformer", transcript, height=240, disabled=True,
            placeholder="Run denoising to see the transcript.",
        )
        if result and result["error"]:
            st.warning(f"Transcription could not be completed. {result['error']}")
        elif transcript:
            st.download_button(
                "Download transcript", transcript + "\n", file_name="transcript.txt",
                mime="text/plain", on_click="ignore",
            )
        elif result:
            st.caption("No speech detected.")

    with st.container(border=True):
        if result:
            spectrum, waveform = st.tabs(("Spectrograms", "Waveforms"))
            with spectrum:
                st.pyplot(plot_spectrograms(clips), width="stretch")
            with waveform:
                st.pyplot(plot_waveforms(clips), width="stretch")
        else:
            st.markdown("**Spectrograms**")
            st.caption("Enhance a recording to compare the models.")


st.set_page_config(page_title="Voice Note Denoiser", layout="wide")
st.html(Path(__file__).with_name("style.css"))
st.title("Voice Note Denoiser")

generation = st.session_state.get("upload_generation", 0)
with st.form("recording_form", border=False):
    left, right = st.columns(2, gap="small")
    with left.container(border=True):
        noisy = st.file_uploader("Audio recording", type=SUPPORTED_AUDIO, key=f"noisy_{generation}")
    with right.container(border=True):
        clean = st.file_uploader("Clean reference (optional)", type=SUPPORTED_AUDIO, key=f"clean_{generation}")
    with st.expander("Record with microphone"):
        recording = st.audio_input("Microphone", key=f"recording_{generation}")
    submitted = st.form_submit_button("Denoise and transcribe", type="primary", width="stretch")
if submitted:
    stage_recording(noisy, recording, clean)

pending = st.session_state.pop("pending_recording", None)
if pending is not None:
    try:
        result = process_recording(pending)
    except (av.error.FFmpegError, OSError, ValueError, RuntimeError) as error:
        pending.cleanup()
        st.error(f"Could not process the recording. {error}")
    else:
        previous = st.session_state.get("result_storage")
        st.session_state.result_storage = pending
        st.session_state.result = result
        st.session_state.pop("preview_start", None)
        if previous is not None:
            previous.cleanup()
show_comparison(st.session_state.get("result"))
