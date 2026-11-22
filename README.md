# Preserving Speech Intelligibility in GAN-Based Noise Removal

An offline voice-note denoiser and a controlled comparison of two conditional
waveform GANs. Both use the same 1D U-Net generator and patch discriminator.
The baseline uses adversarial and waveform L1 losses. The second model also
uses multi-resolution STFT loss.
The project also compares a compact Transformer with a matched Conformer
that adds local temporal convolution to the attention blocks.

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
The tools and demo work offline after installation and the one-time Whisper
model download below. PyAV provides audio decoding and resampling.

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

## Train and compare the Conformer

Train the Transformer and Conformer with the same settings and compare them
alongside the two published GAN checkpoints. Each attention model uses the
same complex-mask decoder and supervised loss.

```powershell
& $projectPython -m speech_gan.train transformer --device cuda --run-dir runs/transformer
& $projectPython -m speech_gan.train conformer --device cuda --run-dir runs/conformer
& $projectPython -m speech_gan.evaluate --baseline artifacts/baseline.pt --spectral artifacts/spectral.pt --transformer runs/transformer/best.pt --conformer runs/conformer/best.pt --split test --device cuda --output-dir runs/conformer-comparison
```

Add `--smoke --batch-size 4` and use separate run directories for a four-batch
check on either model. The official test command expects both complete
attention checkpoints. To reproduce the comparison with the supplied
checkpoints, use `artifacts/transformer.pt` and `artifacts/conformer.pt`.
[Follow-up benchmark results](REPORT.md#follow-up-test-results).

## Run the offline demo

The trained generator checkpoints in `artifacts/` let you run the demo
without downloading VoiceBank–DEMAND.

```powershell
& $projectPython -m speech_gan.transcription --download
& $projectPython -m speech_gan.demo --device auto
```

Open the localhost URL printed by Gradio. Upload an audio file or record
with the microphone. You can also upload a clean reference. Select
**Denoise and transcribe** to compare the enhancements and see a timestamped
transcript of the Conformer output. The demo uses the MR-STFT GAN output
for transcription when a Conformer checkpoint is unavailable.

Audio decoding, enhancement and transcription use bounded windows. The
waveform and spectrogram previews show up to the first 30 seconds.
Longer recordings also have full-length FLAC players and download links.
Save the transcript using the text download. Results are held in temporary
storage while the local app runs, so download the files you want to keep
before closing it.

Recordings and transcripts stay on the machine. The download command
fetches the pinned multilingual Whisper Base weights. Use
`--whisper-device cpu` if CUDA transcription is unavailable. Model options
`--baseline`, `--spectral`, and `--conformer` load alternate denoising checkpoints.
[Whisper inference engine](https://github.com/SYSTRAN/faster-whisper).

## Host on Streamlit Community Cloud

Deploy the `main` branch with `streamlit_app/app.py` as the entry point
and Python 3.12. In Advanced settings, set the app's Secrets to
`VOICE_NOTE_PASSWORD = "your-access-password"`. Share this password
separately from the app URL.

The hosted app compares all three denoisers with audio previews, waveforms,
and spectrograms from any part of the recording. It also provides full-length
FLAC playback and downloads, and a timestamped Conformer transcript. It uses
the bundled GAN and Conformer checkpoints and downloads the pinned Whisper
weights on first use.

Community Cloud can sleep between visits. Available memory and CPU
determine how large a recording the hosted app can process. Use the offline
demo for sensitive recordings or recordings that exceed the cloud resources.
[Streamlit Community Cloud deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).

## Data credit

The experiment uses Cassia Valentini-Botinhao’s noisy speech database via
JacobLinCool’s 16 kHz VoiceBank–DEMAND mirror. The original dataset lists a
CC BY 4.0 license. [University of Edinburgh DataShare](https://doi.org/10.7488/ds/2117).
