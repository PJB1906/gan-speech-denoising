"""Train the baseline or spectral-loss GAN using identical crops and schedules."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from . import SAMPLE_RATE
from .data import Pairs, REVISION, SPLIT_SEED
from .metrics import score_pairs, summarize
from .model import Discriminator, Generator, MultiResolutionSTFT

TRAIN_SEED = 1234
L1_WEIGHT = 50.0
SPECTRAL_WEIGHT = 1.0
LEARNING_RATE = 0.0002


class Crops(Dataset):
    def __init__(self, pairs, epoch):
        self.pairs = pairs
        self.epoch = epoch

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        rng = np.random.default_rng([TRAIN_SEED, self.epoch, index])
        noisy, clean = self.pairs.read(index, SAMPLE_RATE, rng)
        return torch.from_numpy(noisy)[None, :], torch.from_numpy(clean)[None, :]


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install CUDA PyTorch or use --device cpu for a smoke test.")
    torch.set_num_threads(4)
    return torch.device(name)


def validation_indices(validation, limit):
    if limit >= len(validation):
        return list(range(len(validation)))
    return sorted(random.Random(TRAIN_SEED).sample(range(len(validation)), limit))


def atomic_save(state, path):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_generator(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Generator().to(device)
    model.load_state_dict(checkpoint["generator"])
    return model.eval(), checkpoint


def train_epoch(generator, discriminator, optimizer_g, optimizer_d, spectral, crops, config, device):
    generator.train()
    loader = DataLoader(
        crops, batch_size=config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(TRAIN_SEED + crops.epoch),
        pin_memory=device.type == "cuda",
    )
    totals = torch.zeros(4, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    steps = 0
    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        optimizer_d.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config["bf16"]):
            generated = generator(noisy)
            real_score = discriminator(noisy, clean)
            fake_score = discriminator(noisy, generated.detach())
            loss_d = 0.5 * ((real_score.float() - 1).square().mean() + fake_score.float().square().mean())
        loss_d.backward()
        optimizer_d.step()

        optimizer_g.zero_grad(set_to_none=True)
        discriminator.requires_grad_(False)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config["bf16"]):
            fool_score = discriminator(noisy, generated)
        waveform = F.l1_loss(generated, clean)
        loss_spectral = spectral(generated, clean) if spectral is not None else waveform.new_zeros(())
        loss_g = 0.5 * (fool_score.float() - 1).square().mean() + L1_WEIGHT * waveform
        loss_g = loss_g + SPECTRAL_WEIGHT * loss_spectral
        loss_g.backward()
        optimizer_g.step()
        discriminator.requires_grad_(True)
        totals += torch.stack((loss_d.detach(), loss_g.detach(), waveform.detach(), loss_spectral.detach()))
        steps += 1
        if steps % 100 == 0:
            print(f"  step {steps}/{len(loader)}, {time.perf_counter() - start:.1f} s", flush=True)
        if config["smoke"] and steps == 4:
            break
    losses = (totals / steps).tolist()
    if not all(math.isfinite(value) for value in losses):
        raise FloatingPointError(f"Non-finite training losses: {losses}")
    return {
        **dict(zip(("loss_d", "loss_g", "waveform_l1", "mr_stft"), losses)),
        "steps": steps,
        "train_seconds": time.perf_counter() - start,
    }


def run(args):
    device = resolve_device(args.device)
    torch.manual_seed(TRAIN_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train = Pairs(args.data_dir, "train")
    val = Pairs(args.data_dir, "validation")
    max_val = 8 if args.smoke else args.validation_limit
    indices = validation_indices(val, max_val)
    config = {
        "dataset_revision": REVISION, "holdout": val.validation_speakers, "split_seed": SPLIT_SEED,
        "seed": TRAIN_SEED, "epochs": 1 if args.smoke else args.epochs,
        "batch_size": args.batch_size, "crop_samples": SAMPLE_RATE,
        "validation_limit": max_val, "l1_weight": L1_WEIGHT, "learning_rate": LEARNING_RATE,
        "adam_betas": [0.5, 0.9], "smoke": args.smoke, "device": str(device),
        "bf16": device.type == "cuda" and torch.cuda.is_bf16_supported(),
    }
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    spectral = MultiResolutionSTFT().to(device) if args.variant == "spectral" else None
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.run_dir / "last.pt"
    best_path = args.run_dir / "best.pt"
    history = []
    if last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if state["variant"] != args.variant or state["config"] != config:
            raise ValueError("Run directory has a different configuration. Choose an unused --run-dir.")
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        optimizer_g.load_state_dict(state["optimizer_g"])
        optimizer_d.load_state_dict(state["optimizer_d"])
        history = state["history"]
    if len(history) == config["epochs"]:
        print(f"{args.variant}: training already completed in {args.run_dir}")
        return
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"{args.variant} on {hardware}, PyTorch {torch.__version__}, BF16 {config['bf16']}", flush=True)
    print(f"{len(train):,} train pairs | {len(indices):,} scored validation pairs", flush=True)
    noisy_summary = summarize(score_pairs(val, indices, {"noisy": None}, device))
    print(f"Noisy validation STOI {noisy_summary['stoi']:.5f}", flush=True)
    best = max((row["stoi"] for row in history), default=-math.inf)
    for epoch in range(len(history) + 1, config["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        print(f"Epoch {epoch}/{config['epochs']}", flush=True)
        measurement = train_epoch(
            generator, discriminator, optimizer_g, optimizer_d, spectral, Crops(train, epoch), config, device
        )
        if device.type == "cuda":
            measurement["peak_cuda_mib"] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        rows = score_pairs(val, indices, {args.variant: generator.eval()}, device)
        validation = summarize(rows)
        if validation["stoi"] is None or validation["stoi_count"] != noisy_summary["stoi_count"]:
            raise ValueError("STOI could not score the same validation clips for noisy and enhanced audio")
        result = {
            "epoch": epoch, **measurement, **validation,
            "stoi_delta_vs_noisy": validation["stoi"] - noisy_summary["stoi"],
        }
        history.append(result)
        state = {
            "variant": args.variant, "config": config, "hardware": hardware,
            "spectral_weight": SPECTRAL_WEIGHT if spectral is not None else 0.0,
            "torch_version": str(torch.__version__),
            "generator": generator.state_dict(), "noisy_validation": noisy_summary,
            "epoch": epoch, "history": history,
        }
        if validation["stoi"] > best:
            atomic_save(state, best_path)
            best = validation["stoi"]
        atomic_save({
            **state, "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(),
        }, last_path)
        (args.run_dir / "history.json").write_text(
            json.dumps({"config": config, "noisy_validation": noisy_summary, "epochs": history}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"  STOI {validation['stoi']:.5f} ({result['stoi_delta_vs_noisy']:+.5f} vs noisy) | "
            f"SI-SDR {validation['si_sdr_db']:.2f} dB | {measurement['train_seconds']:.1f} s training",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("baseline", "spectral"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-limit", type=int, default=256)
    parser.add_argument("--smoke", action="store_true", help="Four training batches and eight validation clips")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.validation_limit) < 1:
        parser.error("epochs, batch size and validation limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
