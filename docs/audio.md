# Speech Recognition (Qwen3-ASR-1.7B)

Orinfer runs **Qwen3-ASR-1.7B** natively on Jetson AGX Orin 64GB with Rust audio decoding/resampling and a TileLang execution package. It supports automatic language detection or explicit language hints, one recording per request.

---

## 1. Quick Start

### Managed CLI Transcription

```bash
orinfer import qwen3-asr-1.7b /path/to/prepared-asr-model
orinfer transcribe qwen3-asr-1.7b ./sample.wav --language zh --prompt "Optional domain vocabulary"
```

### OpenAI-Compatible File Endpoint

```bash
curl http://127.0.0.1:8088/v1/audio/transcriptions \
  -F model=qwen3-asr-1.7b \
  -F file=@sample.wav \
  -F language=zh \
  -F response_format=json
```

- **Formats & Limits**: Supports WAV, FLAC, MP3, AAC/MP4, Ogg/Vorbis (all enabled Symphonia codecs). Max file size **24 MiB**, max duration **1,200 seconds** (20 minutes).
- **Parameters**: `response_format` (`json` → `{"text":"..."}`, or `text`), optional `language`, `prompt`, and `temperature`.
- **Chat & Batch APIs**: `/v1/chat/completions` and `orinfer batch` also accept `input_audio` content parts (base64 data + format hint), preserving the checkpoint's `language ...<asr_text>...` protocol and supporting token streaming.
- **Scope**: Provides transcription text; word-level timestamps require a separate forced-aligner model.

---

## 2. Architecture & Precision

- **Source**: Pinned `Qwen/Qwen3-ASR-1.7B-hf` (1.72B Qwen3 full-attention decoder: 28 layers, 16/8 GQA heads, head dim 128, full RoPE, ordinary RMSNorm, 65,536 max context + ~0.32B audio encoder/projector).
- **CPU Audio Frontend**: Symphonia decode → channel downmix → Rubato fixed-ratio FFT resampler to 16 kHz → NEON RealFFT 128-bin log-mel features (parallelized across frame groups with Rayon).
- **Quantization**:
  - Text decoder & shared embedding/head: two-stage E8P Q4A8 (`[4,1]`, signed-block128, beam8).
  - Decoder layers 0–3 Q/K/V & FFN and all attention output projections: W8.
  - Audio encoder & projector: FP16; KV cache: group-64 INT8 + FP16 scales.
  - Native weight payload: **1.665 GB** (~4.71 bits/param text, ~6.53 bits/param overall).
- **GPU Execution**: NHWC bounded CNN + 104-token bidirectional window encoder; text decoder reuses the shared Qwen dynamic-row executor with address-table packet boundaries (347 kernel nodes across 1–128 active requests) and request-isolated audio/KV/prefix state.

---

## 3. Validation & Performance (Jetson AGX Orin 64GB)

Measured 2026-10-21 (`decode_only`, greedy, seed `20261002`, MTP off, prefix cache off; median of 3 runs after 1 warmup). vLLM baseline: `vrfai/Qwen3-ASR-1.7B-fp8` (Marlin FP8 weights, FP16 activations/KV). TTFT includes file decode, resampling, log-mel, audio encoding, and text prefill.

### Single-Request Latency & Decode (`latency` policy)

| Recording | Prompt Tokens | vLLM TTFT (ms) | Orinfer TTFT (ms) | vLLM Decode (tok/s) | Orinfer Decode (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chinese (4.20 s) | 70 | 96.6 | **34.1** | 74.7 | **115.2** |
| English (15.05 s) | 211 | 113.4 | **59.6** | 73.1 | **116.9** |
| Independent English (7.06 s) | 107 | 99.4 | **38.9** | 73.9 | **117.0** |
| Silence (2.00 s) | 41 | 92.3 | **29.1** | 76.2 | **108.9** |
| Concatenated (19.46 s) | 268 | 125.7 | **75.8** | 73.0 | **116.8** |
| Repeated (122.01 s) | 1,602 | 479.7 | **377.1** | 68.0 | **105.2** |

### Concurrent Cohort Throughput (`--scheduler-policy throughput`, up to 128 slots)

| Workload | Concurrency | vLLM (tok/s) | Orinfer (tok/s) |
| --- | ---: | ---: | ---: |
| Mixed short recordings | 8 / 32 / 64 / 128 | 192.4 / 403.9 / 512.6 / 577.8 | **279.4 / 455.7 / 513.8** / 522.0 |
| English 15.05 s each | 8 / 32 / 64 / 128 | 303.7 / 544.0 / 511.9 / 527.4 | **395.0 / 574.2 / 623.9 / 624.0** *(fresh cohort: 661.3)* |

- **Startup**: Median **0.95 s** to `/health`, **1.21 s** to first content, **1.30 s** to complete first Chinese request (warm page cache).
- **Quality**: 9 frozen-history cases (312 target tokens) achieve weighted ΔNLL **0.0396 nats/token** vs. BF16, with 100% top-1 choices inside BF16 top-3. 1,200 s silent FLAC transcribes in **5.44 s** (15,615 prompt tokens).
- **Building**: Built via [Qwen3.5 Dense Builder](../tools/model/qwen35/README.md) with `--context 65536`.
