"""Evaluate saved generators on full utterances from validation or official test."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from . import SAMPLE_RATE
from .data import Pairs, REVISION
from .metrics import enhance, score_pairs, summarize
from .train import load_generator, resolve_device

COMPARISONS = (
    ("baseline", "noisy"), ("spectral", "noisy"), ("spectral", "baseline"),
    ("transformer", "noisy"), ("transformer", "baseline"), ("transformer", "spectral"),
    ("conformer", "noisy"), ("conformer", "baseline"), ("conformer", "spectral"),
    ("conformer", "transformer"),
)


def paired_delta(rows, first, second, metric):
    by_method = {
        method: {row["id"]: row[metric] for row in rows if row["method"] == method}
        for method in (first, second)
    }
    differences = [
        score - by_method[second][clip]
        for clip, score in by_method[first].items()
        if score is not None and by_method[second][clip] is not None
    ]
    return {
        "count": len(differences),
        "mean": float(np.mean(differences)) if differences else None,
    }


def run(args):
    device = resolve_device(args.device)
    models = {"noisy": None}
    checkpoints = {}
    for variant in ("baseline", "spectral", "transformer", "conformer"):
        path = getattr(args, variant)
        if path is None:
            continue
        model, state = load_generator(path, device)
        if state["variant"] != variant or state["config"]["dataset_revision"] != REVISION:
            raise ValueError(f"{path}: checkpoint variant or dataset revision mismatch")
        models[variant], checkpoints[variant] = model, state
    if args.split == "test":
        if "spectral" not in checkpoints or any(state["config"]["smoke"] for state in checkpoints.values()):
            raise ValueError("Official test evaluation requires both GAN checkpoints and excludes smoke runs")
        if ("transformer" in checkpoints) != ("conformer" in checkpoints):
            raise ValueError("Supply both attention checkpoints for official test evaluation")
    for baseline, modified in (("baseline", "spectral"), ("transformer", "conformer")):
        if baseline in checkpoints and modified in checkpoints:
            if checkpoints[baseline]["config"] != checkpoints[modified]["config"]:
                raise ValueError(f"{baseline} and {modified} must have the same training configuration")

    pairs = Pairs(args.data_dir, args.split)
    speakers = sorted({row["speaker"] for row in pairs.records})
    if args.split == "test":
        development = Pairs(args.data_dir, "train")
        development_speakers = {row["speaker"] for row in development.records} | set(development.validation_speakers)
        if development_speakers.intersection(speakers):
            raise ValueError("Development and test speakers overlap")
    print(f"Scoring {len(pairs):,} full {args.split} pairs on {device}", flush=True)
    for model in models.values():
        if model is not None:
            for _ in range(5):
                enhance(model, np.zeros(SAMPLE_RATE, dtype=np.float32), device)
    rows = score_pairs(pairs, range(len(pairs)), models, device, warmup_each=True)
    summary = {
        "split": args.split,
        "speakers": speakers,
        "dataset_revision": REVISION,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": str(torch.__version__),
        "inference_timing": "FP32 forward and clamp, one warm-up forward per clip and model, alternating model order",
        "checkpoint_epochs": {label: state["epoch"] for label, state in checkpoints.items()},
        "methods": {
            name: summarize([row for row in rows if row["method"] == name]) for name in models
        },
        "paired_deltas": {
            f"{a}_minus_{b}": {
                metric: paired_delta(rows, a, b, metric) for metric in ("stoi", "si_sdr_db")
            }
            for a, b in COMPARISONS
            if a in models and b in models
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(summary, indent=2) + "\n"
    (args.output_dir / "summary.json").write_text(encoded, encoding="utf-8")
    with (args.output_dir / "utterances.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--spectral", type=Path)
    parser.add_argument("--transformer", type=Path)
    parser.add_argument("--conformer", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
