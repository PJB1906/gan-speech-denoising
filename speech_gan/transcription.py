"""Download the Whisper weights once and transcribe recordings locally."""

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import soundfile as sf
import torch

from . import SAMPLE_RATE
from .recording import timecode

MODEL_REPO = "Systran/faster-whisper-base"
MODEL_REVISION = "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
MODEL_PATH = Path("models/whisper-base")
MODEL_FILES = {
    "config.json": "56a6d8110d311f19c8f0471e562832c7527f146b567275bfca59fcf7c184da9a",
    "model.bin": "d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9",
    "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
    "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
}
BODY_SECONDS = 26
CONTEXT_SECONDS = 2


def load_transcriber(model_path, device):
    if any(not (model_path / name).is_file() for name in MODEL_FILES):
        raise FileNotFoundError(
            f"Whisper weights are missing from {model_path}. "
            "Run python -m speech_gan.transcription --download first."
        )
    if os.name == "nt" and device == "cuda":
        # The Windows CUDA build of PyTorch ships the cuBLAS and cuDNN DLLs
        # CTranslate2 loads at inference time.
        torch_lib = str(Path(torch.__file__).resolve().parent / "lib")
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    from faster_whisper import WhisperModel

    return WhisperModel(
        str(model_path.resolve()), device=device,
        compute_type="int8_float16" if device == "cuda" else "int8",
        cpu_threads=4, local_files_only=True,
    )


def transcribe_audio(model, path, progress=None):
    """Decode short Whisper windows while retaining recording-wide timestamps."""
    lines = []
    language = None
    position = 0
    context = CONTEXT_SECONDS * SAMPLE_RATE
    previous = np.empty(0, dtype=np.float32)
    with sf.SoundFile(path) as source:
        if source.samplerate != SAMPLE_RATE or source.channels != 1:
            raise ValueError("Transcription requires the demo's mono 16 kHz output.")
        duration = source.frames / SAMPLE_RATE
        blocks = source.blocks(blocksize=BODY_SECONDS * SAMPLE_RATE, dtype="float32")
        current = next(blocks, None)
        while current is not None:
            following = next(blocks, None)
            trailing = following[:context] if following is not None else previous[:0]
            window = np.concatenate((previous, current, trailing))
            start = len(previous) / SAMPLE_RATE
            end = start + len(current) / SAMPLE_RATE
            offset = position / SAMPLE_RATE - start
            segments, info = model.transcribe(
                window, language=language, beam_size=3, vad_filter=True,
                condition_on_previous_text=False, word_timestamps=True,
            )
            for segment in segments:
                words = [
                    word for word in segment.words or ()
                    if start <= (word.start + word.end) / 2 < end
                ]
                if words:
                    text = "".join(word.word for word in words).strip()
                    if text:
                        language = language or info.language
                        timestamp = timecode(max(position / SAMPLE_RATE, offset + words[0].start))
                        lines.append(f"[{timestamp}] {text}")
            position += len(current)
            if progress is not None:
                progress(position / SAMPLE_RATE, duration)
            previous = current[-context:].copy()
            current = following
    return "\n".join(lines)


def verify_file(path, checksum):
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest() == checksum


def download_model(directory):
    directory.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl")
    for name, checksum in MODEL_FILES.items():
        path = directory / name
        if verify_file(path, checksum):
            continue
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}"
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".partial", delete=False) as target:
            temporary = Path(target.name)
        try:
            if curl:
                subprocess.run(
                    [curl, "--fail", "--silent", "--show-error", "--location", "--connect-timeout", "15",
                     "--max-time", "240", "--retry", "1", "--output", str(temporary), url],
                    check=True,
                )
            else:
                with urlopen(url, timeout=60) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1 << 20)
            if not verify_file(temporary, checksum):
                raise OSError(f"Downloaded Whisper file failed its checksum: {name}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", required=True, action="store_true", help="Fetch the pinned model")
    parser.add_argument("--model-dir", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    download_model(args.model_dir)
    print(f"Whisper weights ready in {args.model_dir}")


if __name__ == "__main__":
    main()
