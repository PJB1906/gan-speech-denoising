# Preserving Speech Intelligibility in GAN-Based Noise Removal

The project tests whether multi-resolution spectrogram loss helps a small GAN
remove background noise while preserving the speech in short voice notes.
The two models share an architecture and training budget. The comparison
measures intelligibility, signal distortion, and inference time.
On the official test, the MR-STFT model scored slightly below the baseline
on both STOI and SI-SDR. Both models improved SI-SDR over noisy input while
lowering STOI.

## Phase 1 · Architecture comparison

The comparison describes typical speech enhancement designs. The controlled
experiment benchmarks the two GAN objectives on the same hardware and data.

| Family | Training stability | Output quality | Generation speed | Latent space interpretability |
| --- | --- | --- | --- | --- |
| Transformer | A supervised reconstruction objective can simplify optimization. Attention memory grows with sequence length unless the model chunks the input or uses a restricted attention pattern. | Local and global context can help distinguish speech from noise. Reconstruction quality depends on the training objective and noise coverage. | A masking model such as TSTNN can process segments in parallel in one pass. Attention adds cost on long inputs. | Attention maps show context relationships. Internal features usually lack a direct interpretation as individual acoustic factors. |
| GAN | A generator and discriminator must learn at compatible rates. A strong discriminator or unstable adversarial signal can hurt the generator. | Adversarial feedback can encourage realistic detail. A plausible waveform can still distort quiet consonants or leave residual noise. | The generator produces an estimate in one pass. The discriminator is only needed during training. | This conditional GAN has a learned bottleneck but no prescribed latent axes. The noisy input controls its output. |
| VAE | The evidence lower bound balances reconstruction against a latent prior. A strong decoder can reduce use of the latents. | A learned clean-speech prior can help recover speech under noise. Reconstruction can smooth fine detail, and enhancement quality depends on the inference procedure. | A decoder pass can be fast. A recurrent VAE with iterative posterior inference needs additional work at test time. | A prior gives the latent space statistical structure. Individual dimensions need additional constraints to correspond to specific factors. |
| Diffusion | Score matching avoids the alternating generator–discriminator game. The noise schedule and sampling solver affect results. | A model can refine speech structure over several steps and handle reverberation. Quality depends on the conditioning, sampler and number of steps. | Reverse diffusion repeats the denoising network. Fewer steps trade refinement for lower latency. | The noise schedule defines a progression from corrupted to clean audio. The sample state does not explicitly isolate speakers or phonemes. |

The comparison draws on [TSTNN](https://arxiv.org/abs/2103.09963),
[SEGAN](https://arxiv.org/abs/1703.09452),
[Recurrent VAE for speech enhancement](https://arxiv.org/abs/1910.10942),
[Speech enhancement with diffusion](https://arxiv.org/abs/2208.05830).

## Phase 2 · Bottleneck and proposal

Weak speech cues can contribute little to overall waveform error. A
discriminator can also accept a waveform that sounds speech-like even when it
changes details in the reference. The practical risk is that noise reduction
changes cues needed to distinguish words in a noisy voice note.

The 113,009-parameter generator estimates an additive correction to the input. Four stride-4
encoder stages, dilated bottleneck convolutions, and three decoder skip
connections give it local detail and wider context. The 41,569-parameter discriminator sees
the noisy waveform paired with either the clean target or the generated
waveform and scores local patches. Least-squares losses train the two networks.
[SEGAN](https://arxiv.org/abs/1703.09452).

For an estimate ŷ and clean target y, the baseline generator loss is
`0.5 × mean((D(x, ŷ) − 1)²) + 50 × mean(abs(ŷ − y))`. The modified objective
adds the mean of two terms across three STFT resolutions with a weight of 1.

- Spectral convergence, `||M(ŷ) − M(y)||₂ / ||M(y)||₂`.
- Log-magnitude error, `mean(abs(log M(ŷ) − log M(y)))`.

Here `M` is the magnitude of a Hann-window STFT with a floor of `10⁻⁵`. FFT
sizes 256, 512, and 1024 correspond to 16, 32, and 64 ms at 16 kHz. Each hop is
one quarter of the window size. Short windows expose brief transitions and
longer windows resolve narrower frequency bands. The loss uses magnitude and
has no separate phase target. The waveform term continues to penalize
sample-level errors. [Parallel WaveGAN](https://arxiv.org/abs/1910.11480).

## Phase 3 · Implementation and benchmark

The six Parquet shards matched the published SHA-256 hashes. Across the five
development shards, the four validation speakers (p226, p230, p274, p278)
account for 1,608 pairs. The other 24 speakers contribute 9,964 training pairs.
The official test shard has 824 pairs from p232 and p257.
[Dataset mirror](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k/tree/4497db342d7312978c45690591fda86117831940/data).

Both runs use the same seeded initialization, deterministic crop and shuffle
schedule, and budget of eight epochs. Each epoch uses 623 batches of up to
16 paired one-second crops. Both optimizers use Adam with a learning rate of
0.0002 and β values of 0.5 and 0.9.
Checkpoint selection uses the highest mean STOI over a fixed 256-utterance
validation sample. The full validation split checks the baseline against noisy
input before the spectral run. Final scoring uses full utterances and reports
means over the same test clips for each method.

STOI estimates intelligibility relative to clean speech. SI-SDR measures
distortion after aligning the estimate and reference in scale. Inference timing
measures single-utterance FP32 generation and clipping with synchronized GPU
execution. The timing pass warms up both models for each utterance and
alternates their run order. Audio decoding, transfers, metric computation and
the Gradio interface fall outside the timed region. The real-time factor
divides total inference seconds by total audio seconds.
[STOI implementation and references](https://github.com/mpariente/pystoi),
[SI-SDR](https://arxiv.org/abs/1811.02508).

### Baseline validation

Both variants passed four-batch smoke tests on CPU and on the RTX 4050.
On all 1,608 validation utterances, the selected baseline (epoch 4) scored
0.85581 STOI and 9.93 dB SI-SDR. The noisy input scored 0.85685 STOI and
6.11 dB SI-SDR. The baseline gained 3.82 dB in SI-SDR while losing 0.00103
in STOI. On the 256-utterance checkpoint-selection sample, the best baseline
scored 0.85630 STOI and the best MR-STFT model (epoch 5) scored 0.85923.
The noisy input scored 0.85724 on that sample.
[Baseline validation measurements](artifacts/baseline-validation.json).

### Official test

| Method | Mean STOI ↑ | Mean SI-SDR (dB) ↑ | Median inference (ms) ↓ | Real-time factor ↓ |
| --- | ---: | ---: | ---: | ---: |
| Noisy input | 0.92101 | 8.45 | — | — |
| Baseline GAN | 0.91466 | 14.82 | 2.575 | 0.001409 |
| MR-STFT GAN | 0.91253 | 14.57 | 2.582 | 0.001411 |

Across 824 test utterances, the baseline gained 6.37 dB in mean SI-SDR
relative to noisy input and the MR-STFT GAN gained 6.12 dB. Mean STOI fell by
0.00636 for the baseline and 0.00848 for the MR-STFT GAN. The MR-STFT
model's advantage on the validation sample reversed on the official test.
Adding the spectral loss reduced test STOI by 0.00213 and SI-SDR by 0.24 dB
relative to the baseline. Inference latency was nearly identical.
[Test summary](artifacts/test-summary.json), [Per-utterance scores](artifacts/test-utterances.csv).

| Training run | Generator updates | Training time | Peak allocated CUDA memory |
| --- | ---: | ---: | ---: |
| Baseline | 4,984 | 141.75 s | 55.6 MiB |
| MR-STFT GAN | 4,984 | 124.21 s | 148.6 MiB |

Training used PyTorch 2.11.0 with CUDA 12.8 and BF16 autocast on the RTX 4050
Laptop GPU. Times sum the training loops and exclude validation. Peak memory
is PyTorch's CUDA allocation. Both runs lost validation STOI in late epochs,
making the checkpoint-selection rule material to the comparison.
[Baseline history](artifacts/baseline-training.json),
[MR-STFT history](artifacts/spectral-training.json).

The test covers two speakers and this experiment uses one initialization seed.
STOI measures an intelligibility proxy on simulated mixtures. Word recognition
on real voice notes remains unmeasured.

## Phase 4 · Offline voice-note demo

The local Gradio interface accepts a recording or uploaded voice note up to
60 seconds, converts it to mono at 16 kHz, and runs both generators. The noisy
input, two estimates, and optional clean reference can be played individually.
Spectrograms use 512-sample Hann windows, 128-sample hops, and a shared −80 to
0 dB magnitude scale. Upload tests with the published checkpoints returned
the expected audio players and spectrograms with and without a clean reference.
Setup, training, and demo commands are in [README.md](README.md).
