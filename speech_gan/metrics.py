"""Full-utterance STOI, SI-SDR and synchronized generator timing."""

import math
import time
import warnings

import numpy as np
import torch
from pystoi import stoi

from . import SAMPLE_RATE


def measure(clean, estimate):
    if not np.isfinite(estimate).all():
        raise ValueError("The generator produced non-finite audio")
    with warnings.catch_warnings(record=True) as caught:
        intelligibility = float(stoi(clean, estimate, SAMPLE_RATE, extended=False))
    # pystoi returns a sentinel for utterances with too few active frames.
    if any("Not enough STFT frames" in str(warning.message) for warning in caught):
        intelligibility = None
    else:
        for warning in caught:
            warnings.warn(warning.message, warning.category)
        if not math.isfinite(intelligibility):
            raise ValueError("STOI is not finite")

    reference = clean.astype(np.float64) - clean.mean(dtype=np.float64)
    signal = estimate.astype(np.float64) - estimate.mean(dtype=np.float64)
    energy = np.sum(reference ** 2)
    if energy < 1e-12:
        return intelligibility, None
    target = np.sum(reference * signal) / energy * reference
    residual = signal - target
    sisdr = 10 * math.log10((np.sum(target ** 2) + 1e-12) / (np.sum(residual ** 2) + 1e-12))
    return intelligibility, sisdr


@torch.inference_mode()
def enhance(model, noisy, device):
    tensor = torch.from_numpy(np.ascontiguousarray(noisy, dtype=np.float32))[None, None, :].to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    prediction = model(tensor).clamp(-1, 1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return prediction[0, 0].cpu().numpy(), elapsed


def summarize(rows):
    summary = {"clips": len(rows)}
    for metric in ("stoi", "si_sdr_db"):
        valid = [row[metric] for row in rows if row[metric] is not None]
        summary[metric] = float(np.mean(valid)) if valid else None
        summary[f"{metric}_count"] = len(valid)
    timed = [row for row in rows if row["inference_s"] is not None]
    if timed:
        summary["median_inference_ms"] = float(np.median([row["inference_s"] for row in timed])) * 1000
        summary["real_time_factor"] = sum(row["inference_s"] for row in timed) / sum(
            row["duration_s"] for row in timed
        )
    return summary


def score_pairs(pairs, indices, models, device, *, warmup_each=False):
    rows = []
    generators = [(label, model) for label, model in models.items() if model is not None]
    for position, index in enumerate(indices):
        noisy, clean = pairs.read(index)
        record = pairs.records[index]
        ordered = generators if position % 2 == 0 else generators[::-1]
        if warmup_each:
            for _, model in ordered:
                enhance(model, noisy, device)
        estimates = {label: enhance(model, noisy, device) for label, model in ordered}
        for label in models:
            estimate, elapsed = estimates.get(label, (noisy, None))
            stoi_score, sisdr = measure(clean, estimate)
            rows.append({
                "id": record["id"], "speaker": record["speaker"], "method": label,
                "duration_s": len(clean) / SAMPLE_RATE, "stoi": stoi_score, "si_sdr_db": sisdr,
                "inference_s": elapsed,
            })
    return rows
