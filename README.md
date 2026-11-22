# Preserving Speech Intelligibility in GAN-Based Noise Removal

An offline voice-note denoiser and a controlled comparison of two conditional
waveform GANs. Both use the same 1D U-Net generator and patch discriminator.
The baseline uses adversarial and waveform L1 losses. The second model also
uses multi-resolution STFT loss.

See [the project report](REPORT.md) for the architecture comparison, proposed
modification, benchmark protocol, and results.

## Setup on Windows

Install 64-bit Python 3.10. For NVIDIA training, install an NVIDIA driver that
supports the CUDA version of the selected PyTorch wheel. The experiment uses
the PyTorch 2.11.0 CUDA 12.8 build on an RTX 4050 Laptop GPU. [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/).

```powershell
py -3.10 -m venv .venv
$projectPython = ".\.venv\Scripts\python.exe"
& $projectPython -m pip install --upgrade pip
& $projectPython -m pip install "torch==2.11.0+cu128" --extra-index-url https://download.pytorch.org/whl/cu128
& $projectPython -m pip install -r requirements.txt
& $projectPython -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')"
```

For CPU inference or a CPU smoke test, install `torch==2.11.0+cpu` using
`https://download.pytorch.org/whl/cpu` as the extra index in the PyTorch command.
The tools and demo work offline after installation. FFmpeg can help Gradio
handle uploads that need format conversion.

## Verify and prepare the data

Put the five training Parquet shards and one official test shard in `data/`.
The verifier checks SHA-256, size, schema and row count against the pinned
16 kHz mirror. The cache decodes and checks aligned mono PCM pairs. Preparing
development data uses roughly 2.2 GB of additional disk space.
[Pinned VoiceBank–DEMAND mirror](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k/tree/4497db342d7312978c45690591fda86117831940/data).

```powershell
& $projectPython -m speech_gan.data verify
& $projectPython -m speech_gan.data examples --play
```

Playback uses Windows audio. Omit `--play` to export only. The second command
also prepares the 9,964 training and 1,608 validation clips. The official 824
test clips are decoded when final evaluation runs.

## Smoke test and train

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
& $projectPython -m speech_gan.train baseline --smoke --batch-size 4 --device cuda --run-dir runs/smoke/baseline
& $projectPython -m speech_gan.train spectral --smoke --batch-size 4 --device cuda --run-dir runs/smoke/spectral

& $projectPython -m speech_gan.train baseline --device cuda --run-dir runs/baseline
& $projectPython -m speech_gan.evaluate --baseline runs/baseline/best.pt --split validation --device cuda --output-dir runs/baseline-validation
& $projectPython -m speech_gan.train spectral --device cuda --run-dir runs/spectral

& $projectPython -m speech_gan.evaluate --baseline runs/baseline/best.pt --spectral runs/spectral/best.pt --split test --device cuda --output-dir runs/official-test
```

Use the same epoch count, batch size, and validation limit in both full runs.
The defaults are eight epochs, batches of 16 one-second crops, and 256 validation
clips for checkpoint selection. Each run writes its best validation-STOI
checkpoint, most recent resumable checkpoint, and per-epoch history. Rerun a
training command with the same options to resume from the last complete epoch.
Evaluation writes a summary and per-utterance measurements.

## Run the offline demo

The trained generator checkpoints in `artifacts/` let you run the demo
without downloading VoiceBank–DEMAND.

```powershell
& $projectPython -m speech_gan.demo --device cpu
```

Open the localhost URL printed by Gradio. Upload or record a noisy voice note
up to 60 seconds long. Add the matching clean recording if you have it, then
select **Denoise and compare**. The four audio players keep a common gain and
the plots share the same time and frequency limits and dB scale.
Use `--baseline` and `--spectral` to select checkpoints from a new run.

## Data credit

The experiment uses Cassia Valentini-Botinhao’s noisy speech database via
JacobLinCool’s 16 kHz VoiceBank–DEMAND mirror. The original dataset lists a
CC BY 4.0 license. [University of Edinburgh DataShare](https://doi.org/10.7488/ds/2117).
