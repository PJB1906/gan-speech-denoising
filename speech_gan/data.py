"""Verify the original shards and prepare a memory-mapped PCM cache."""

import argparse
import hashlib
import io
import json
import random
import tempfile
import wave
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from . import SAMPLE_RATE

REVISION = "4497db342d7312978c45690591fda86117831940"
TRAIN_SHARDS = {
    "train-00000-of-00005.parquet":
        (2315, 444302320, "d8195194a27e8d007502d6eacf5e0fac71adb9fe343ee228b78bae3559464319"),
    "train-00001-of-00005.parquet":
        (2315, 416366498, "d8a5c932846399a7021b544157b25bfce736122f66a27239c13bbb9262aa98bc"),
    "train-00002-of-00005.parquet":
        (2314, 430855381, "e486e0782f9c978cd2ea08fb25dbe1694cc4c87efaecde8b04308e683db8a91c"),
    "train-00003-of-00005.parquet":
        (2314, 444005802, "6cf3cc86af272546cb94c1e8d26088b3a15feef1f662b145cea86a7c33387b6f"),
    "train-00004-of-00005.parquet":
        (2314, 415510617, "2afbf89fc0b9e651f7990ff17d247ab18803d463842281a67c4a6b7c25abc7b2"),
}
TEST_SHARDS = {
    "test-00000-of-00001.parquet":
        (824, 132200447, "08b43dbea4a15a5f3067233f5cb2de3e5c9a600a3ae5b969661740778f642c87"),
}
SPLIT_SEED = 42
VALIDATION_SPEAKERS = 4


def open_shard(data_dir, name, expected):
    path = data_dir / name
    rows, size, _ = expected
    if path.stat().st_size != size:
        raise ValueError(f"{name}: unexpected file size")
    shard = pq.ParquetFile(path)
    if shard.metadata.num_rows != rows or set(shard.schema_arrow.names) != {"id", "noisy", "clean"}:
        raise ValueError(f"{name}: unexpected row count or schema")
    return shard


def verify(data_dir):
    for name, expected in (TRAIN_SHARDS | TEST_SHARDS).items():
        with open_shard(data_dir, name, expected):
            pass
        sha256 = hashlib.sha256()
        with (data_dir / name).open("rb") as file:
            for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
                sha256.update(chunk)
        if sha256.hexdigest() != expected[2]:
            raise ValueError(f"{name}: SHA-256 differs from dataset revision {REVISION}")
        print(f"{name}: {expected[0]:,} pairs, SHA-256 OK")


def pcm_bytes(wav):
    with wave.open(io.BytesIO(wav), "rb") as reader:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError("Expected mono 16-bit PCM at 16 kHz")
        samples = reader.getnframes()
        pcm = reader.readframes(samples)
        if samples == 0 or len(pcm) != samples * 2:
            raise ValueError("Empty or truncated WAV")
        return pcm, samples


def cache_path(data_dir, partition):
    return data_dir / ".cache" / f"{REVISION[:10]}-v1" / partition


def prepare(data_dir, partition):
    output = cache_path(data_dir, partition)
    if output.exists():
        return output
    shards = TRAIN_SHARDS if partition == "development" else TEST_SHARDS
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / partition
        staging.mkdir()
        records = []
        seen = set()
        with (staging / "pairs.pcm").open("wb") as pcm:
            for name, expected in shards.items():
                with open_shard(data_dir, name, expected) as shard:
                    for batch in shard.iter_batches(batch_size=32):
                        for row in batch.to_pylist():
                            clip = row["id"]
                            speaker, separator, _ = clip.partition("_")
                            if clip in seen or not speaker or not separator:
                                raise ValueError(f"Invalid or repeated clip ID: {clip}")
                            seen.add(clip)
                            noisy, samples = pcm_bytes(row["noisy"]["bytes"])
                            clean, clean_samples = pcm_bytes(row["clean"]["bytes"])
                            if samples != clean_samples:
                                raise ValueError(f"{clip}: unequal noisy and clean durations")
                            records.append({
                                "id": clip, "speaker": speaker, "offset": pcm.tell() // 2,
                                "samples": samples, "shard": name,
                            })
                            pcm.write(noisy)
                            pcm.write(clean)
                print(f"Decoded and checked {name}", flush=True)
        speakers = sorted({record["speaker"] for record in records})
        held_out = (
            sorted(random.Random(SPLIT_SEED).sample(speakers, VALIDATION_SPEAKERS))
            if partition == "development" else []
        )
        index = {"revision": REVISION, "validation_speakers": held_out, "records": records}
        (staging / "index.json").write_text(json.dumps(index), encoding="utf-8")
        try:
            staging.rename(output)
        except OSError:
            if not output.is_dir():
                raise
    return output


class Pairs:
    def __init__(self, data_dir, split):
        self.split = split
        self.path = prepare(data_dir, "test" if split == "test" else "development")
        index = json.loads((self.path / "index.json").read_text(encoding="utf-8"))
        self.validation_speakers = index["validation_speakers"]
        self.records = [
            record for record in index["records"]
            if split == "test" or ((record["speaker"] in self.validation_speakers) == (split == "validation"))
        ]
        self.pcm = None

    def __len__(self):
        return len(self.records)

    def read(self, index, crop_samples=None, rng=None):
        if self.pcm is None:
            self.pcm = np.memmap(self.path / "pairs.pcm", dtype="<i2", mode="r")
        record = self.records[index]
        noisy_start, size = record["offset"], record["samples"]
        start = 0
        if crop_samples is not None and size > crop_samples:
            start = int(rng.integers(size - crop_samples + 1))
        length = min(size, crop_samples) if crop_samples is not None else size
        noisy = self.pcm[noisy_start + start:noisy_start + start + length].astype(np.float32)
        clean_start = noisy_start + size + start
        clean = self.pcm[clean_start:clean_start + length].astype(np.float32)
        noisy /= 32768
        clean /= 32768
        if crop_samples is not None and crop_samples > length:
            pad = (0, crop_samples - length)
            noisy, clean = np.pad(noisy, pad), np.pad(clean, pad)
        return noisy, clean


def write_wav(path, audio):
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        writer.writeframes(np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "prepare", "examples"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("examples"))
    parser.add_argument("--play", action="store_true", help="Play the exported pairs on Windows")
    args = parser.parse_args()
    if args.action == "verify":
        verify(args.data_dir)
        return
    train = Pairs(args.data_dir, "train")
    validation = Pairs(args.data_dir, "validation")
    print(f"Train {len(train):,} | validation {len(validation):,} | holdout {validation.validation_speakers}")
    if args.action == "examples":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for index in np.linspace(0, len(train) - 1, 3, dtype=int):
            clip = train.records[index]["id"]
            noisy, clean = train.read(index)
            for label, audio in (("noisy", noisy), ("clean", clean)):
                path = args.output_dir / f"{clip}_{label}.wav"
                write_wav(path, audio)
                print(path, flush=True)
                if args.play:
                    import winsound
                    winsound.PlaySound(str(path.resolve()), winsound.SND_FILENAME)


if __name__ == "__main__":
    main()
