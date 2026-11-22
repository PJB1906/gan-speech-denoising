"""Decode and enhance recordings as bounded blocks of 16 kHz audio."""

from contextlib import ExitStack, contextmanager

import av
import numpy as np
import soundfile as sf

from . import SAMPLE_RATE
from .metrics import enhance

PREVIEW_SECONDS = 30
BLOCK_SECONDS = 16
OVERLAP_SECONDS = 1


@contextmanager
def decode_audio(path):
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("The selected file has no audio track.")
        stream = container.streams.audio[0]
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None and stream.time_base is not None
            else (container.duration or 0) / av.time_base
        )
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)

        def samples():
            for frame in container.decode(stream):
                frame.pts = None
                for converted in resampler.resample(frame):
                    yield converted.to_ndarray()[0]
            for converted in resampler.resample(None):
                yield converted.to_ndarray()[0]

        yield samples(), duration


def sample_blocks(parts, size):
    buffer = np.empty(size, dtype=np.float32)
    filled = 0
    for part in parts:
        if not np.isfinite(part).all():
            raise ValueError("The audio contains non-finite samples.")
        offset = 0
        while offset < len(part):
            count = min(size - filled, len(part) - offset)
            buffer[filled:filled + count] = part[offset:offset + count]
            offset += count
            filled += count
            if filled == size:
                yield buffer.clip(-1, 1)
                filled = 0
    if filled:
        yield buffer[:filled].clip(-1, 1)


def timecode(seconds):
    minutes, second = divmod(max(0, int(seconds)), 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{second:02d}"


def render_audio(source, source_label, models, device, destination, progress=None):
    """Write lossless full recordings and retain just the preview in RAM."""
    labels = [source_label, *models]
    paths = {label: destination / f"{label.lower().replace(' ', '-')}.flac" for label in labels}
    previews = {label: [] for label in labels}
    previous = {}
    emitted = 0
    with decode_audio(source) as (parts, duration), ExitStack() as stack:
        chunks = sample_blocks(parts, BLOCK_SECONDS * SAMPLE_RATE)
        current = next(chunks, None)
        if current is None:
            raise ValueError("The selected recording is empty.")
        writers = {
            label: stack.enter_context(sf.SoundFile(
                path, "w", samplerate=SAMPLE_RATE, channels=1, format="FLAC", subtype="PCM_16"
            ))
            for label, path in paths.items()
        }
        while current is not None:
            following = next(chunks, None)
            keep = min(len(current), max(0, PREVIEW_SECONDS * SAMPLE_RATE - emitted))
            window = current
            if following is not None and models:
                window = np.concatenate((current, following[:OVERLAP_SECONDS * SAMPLE_RATE]))
            writers[source_label].write(np.minimum(current, 32767 / 32768))
            if keep:
                previews[source_label].append(current[:keep].copy())
            for label, model in models.items():
                enhanced, _ = enhance(model, window, device)
                output = enhanced[:len(current)]
                if label in previous:
                    tail = previous[label]
                    fade = np.linspace(0, 1, len(tail), endpoint=False, dtype=np.float32)
                    output[:len(tail)] = tail * (1 - fade) + output[:len(tail)] * fade
                previous[label] = enhanced[len(current):].copy()
                writers[label].write(np.minimum(output, 32767 / 32768))
                if keep:
                    previews[label].append(output[:keep].copy())
            emitted += len(current)
            if progress is not None:
                progress(emitted / SAMPLE_RATE, duration)
            current = following
    return paths, {label: np.concatenate(parts) for label, parts in previews.items()}, emitted / SAMPLE_RATE
