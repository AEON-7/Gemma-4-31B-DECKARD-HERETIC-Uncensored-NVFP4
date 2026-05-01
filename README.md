# Gemma 4 31B DECKARD HERETIC Uncensored NVFP4


[![☕ Tips](https://img.shields.io/badge/%E2%98%95_Tips-Support_the_work-ff5e5b?style=flat)](https://github.com/AEON-7/AEON-7#-support-the-work)
NVFP4-quantized version of [DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking](https://huggingface.co/DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking) — a dense 31B uncensored Gemma 4 model with thinking/reasoning capabilities. Quantized using NVIDIA ModelOpt 0.42.0 on a native B200 GPU (Blackwell SM 12.0) with two advanced quantization techniques for maximum fidelity at 4-bit precision.

Optimized for deployment on **NVIDIA DGX Spark** (GB10, SM 12.1) and other Blackwell-architecture GPUs.

## Model Downloads

Two quantization variants are available on HuggingFace:

| Variant | Technique | Size | Quality | Speed | HuggingFace |
|---|---|---|---|---|---|
| **AWQ_FULL** | Exhaustive grid search + clipping optimization | 20.45 GB | Excellent | Faster | [Download](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4) |
| **SVDQuant** | SVD decomposition + low-rank BF16 residual | 20.94 GB | Potentially higher | Slightly slower | [Download](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4-SVDQuant) |

**Recommended**: Start with AWQ_FULL for the best balance of quality and throughput.

## Pre-Built Container Image

A pre-built vLLM container compiled for NVIDIA DGX Spark (GB10, SM 12.1) is available with all required patches:

```bash
docker pull ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest
```

**Image contents:**
- vLLM 0.19.1rc1 compiled for SM 12.1 (Blackwell GB10)
- PyTorch 2.12.0 + CUDA 13.0
- transformers 5.5.0
- FlashInfer 0.6.7
- **Patched `modelopt.py`** — fixes FP8 NaN in weight scales, adds NVFP4_AWQ support, AWQ pre_quant_scale handling
- Built from [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) with `--tf5` flag

> [AWQ Container on GHCR](https://github.com/users/AEON-7/packages/container/package/vllm-spark-gemma4-nvfp4-awq) | For AWQ_FULL quantized models
>
> The original non-AWQ container is also available: `docker pull ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4:latest`

### Container Tags

| Tag | Description |
|---|---|
| `latest` | Current patched build (v2) with NVFP4_AWQ fixes |
| `v2` | Same as latest — explicit version tag |

## Critical Fix: FP8 NaN in Weight Scales

The NVFP4 checkpoint produced by ModelOpt 0.42.0 contains **scattered FP8 NaN values** (0x7F / 0xFF) in the per-block `weight_scale` tensors. These are a quantizer bug — 60 NaN values across 39 of the 60 layers, primarily in `down_proj` and `o_proj` weight scales.

**Impact**: A single FP8 NaN scale causes the entire 16-element weight block to dequantize to NaN, which propagates through the model and produces empty/garbage output.

**Fix**: The patched container automatically scrubs FP8 NaN values to zero at model load time, zeroing out the affected blocks (60 out of ~280 million scale elements — negligible quality impact).

If you're **not** using the pre-built container, you must apply the [`modelopt_patched.py`](modelopt_patched.py) patch. See [Manual Patching](#manual-patching) below.

## Performance (DGX Spark GB10)

Benchmarked with `ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest` on NVIDIA DGX Spark (GB10, SM 12.1, 128 GB unified memory). Native FP4 via FLASHINFER_CUTLASS backend. **Zero failures** across all concurrency levels.

| Concurrent | Aggregate tok/s | Per-Request tok/s | Avg Latency (200 tok) |
|---:|---:|---:|---:|
| 1 | 11 | 11 | 18.8s |
| 2 | 22 | 11 | 18.1s |
| 4 | 43 | 11 | 18.4s |
| 8 | 84 | 11 | 19.0s |
| 16 | 161 | 10 | 19.8s |
| 32 | 162 | 8 | 29.6s |

Throughput scales linearly up to 16 concurrent requests (161 aggregate tok/s), saturating at 32.

### With Speculative Decoding (EAGLE Drafter)

Using the [DECKARD E4B NVFP4 drafter](https://huggingface.co/AEON-7/Gemma-4-E4B-DECKARD-HERETIC-Uncensored-NVFP4) (9.6 GB) with 5 speculative tokens. Benchmarked with 300 max tokens per request on DGX Spark.

| Concurrent | Aggregate tok/s | Per-Request tok/s | Avg Latency (300 tok) |
|---:|---:|---:|---:|
| 1 | 7.6 | 8.9 | 39.4s |
| 2 | 21.7 | 10.8 | 27.7s |
| 4 | 42.7 | 10.7 | 28.1s |

> **Note**: On the DGX Spark, the target and drafter models share the same GPU. The drafter's overhead means per-request tok/s is slightly lower than without speculative decoding. The primary benefit of spec decode is realized on multi-GPU systems where the drafter runs on a separate device, or with smaller target models where the drafter acceptance rate is higher.

## Model Details

| Property | Value |
|---|---|
| **Architecture** | Gemma 4 (Dense, 31B parameters) |
| **Layers** | 60 (50 sliding-window + 10 full-attention) |
| **Sliding Window** | 1024 tokens |
| **Max Context** | 262,144 tokens |
| **Hidden Size** | 5376 |
| **Intermediate Size** | 21,504 |
| **Attention Heads** | 32 (16 KV heads), head_dim=256, global_head_dim=512 |
| **Vision Encoder** | 27-layer ViT (1152 hidden) |
| **Vocabulary** | 262,144 tokens |
| **Quantization** | NVFP4 AWQ_FULL or SVDQuant (ModelOpt format) |

## Quick Start

### Docker Compose (DGX Spark)

```yaml
services:
  vllm:
    image: ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest
    container_name: vllm-deckard-31b
    restart: unless-stopped
    network_mode: host
    volumes:
      - /path/to/model:/models/deckard
    environment:
      - VLLM_TEST_FORCE_FP8_MARLIN=1
      - VLLM_MARLIN_USE_ATOMIC_ADD=1
      - VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    command:
      - bash
      - -c
      - |
        exec vllm serve /models/deckard \
          --served-model-name deckard-31b \
          --quantization modelopt \
          --dtype auto \
          --kv-cache-dtype auto \
          --tensor-parallel-size 1 \
          --max-model-len 65536 \
          --max-num-seqs 4 \
          --gpu-memory-utilization 0.85 \
          --trust-remote-code \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --enable-auto-tool-choice \
          --tool-call-parser gemma4 \
          --reasoning-parser gemma4
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Docker Compose (B200 / GPUs with native FP4)

```yaml
services:
  vllm:
    image: ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest
    container_name: vllm-deckard-31b
    restart: unless-stopped
    network_mode: host
    volumes:
      - /path/to/model:/models/deckard
    environment:
      - VLLM_TEST_FORCE_FP8_MARLIN=1
      - VLLM_MARLIN_USE_ATOMIC_ADD=1
      - VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    command:
      - bash
      - -c
      - |
        exec vllm serve /models/deckard \
          --served-model-name deckard-31b \
          --quantization modelopt \
          --dtype auto \
          --kv-cache-dtype fp8 \
          --tensor-parallel-size 1 \
          --max-model-len 131072 \
          --max-num-seqs 16 \
          --gpu-memory-utilization 0.85 \
          --trust-remote-code \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --enable-auto-tool-choice \
          --tool-call-parser gemma4 \
          --reasoning-parser gemma4
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

### Docker Compose with Speculative Decoding (EAGLE)

Uses the [DECKARD E4B drafter](https://huggingface.co/AEON-7/Gemma-4-E4B-DECKARD-HERETIC-Uncensored-NVFP4) for EAGLE-based speculative decoding. Requires three patched files: `modelopt_patched.py`, `serving_chat_patched.py`, and `eagle_patched.py`.

```yaml
services:
  vllm:
    image: ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest
    container_name: vllm-deckard-31b-spec
    restart: unless-stopped
    network_mode: host
    volumes:
      - /path/to/deckard-31b-awq:/models/deckard
      - /path/to/e4b-deckard-nvfp4:/models/e4b-drafter
      - ./modelopt_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py
      - ./serving_chat_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py
      - ./eagle_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py
    environment:
      - VLLM_TEST_FORCE_FP8_MARLIN=1
      - VLLM_MARLIN_USE_ATOMIC_ADD=1
      - VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
      - TORCH_MATMUL_PRECISION=high
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    command:
      - bash
      - -c
      - |
        exec vllm serve /models/deckard \
          --served-model-name deckard-31b \
          --quantization modelopt \
          --dtype auto \
          --kv-cache-dtype fp8 \
          --tensor-parallel-size 1 \
          --max-model-len 131072 \
          --max-num-seqs 4 \
          --gpu-memory-utilization 0.85 \
          --trust-remote-code \
          --host 0.0.0.0 --port 8000 \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --enable-auto-tool-choice \
          --tool-call-parser gemma4 \
          --reasoning-parser gemma4 \
          --speculative-config '{"method":"draft_model","model":"/models/e4b-drafter","num_speculative_tokens":5,"quantization":"modelopt"}'
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Key Deployment Flags

| Flag | Purpose |
|---|---|
| `--quantization modelopt` | **Required** — tells vLLM to use NVIDIA ModelOpt NVFP4 format |
| `--kv-cache-dtype auto` | Let vLLM choose KV cache dtype (use `fp8` on B200 for 2x compression) |
| `--max-model-len 65536` | 64K context — conservative default for DGX Spark. Model supports up to 256K; increase with fewer concurrent sequences (use 131072 on B200) |
| `--reasoning-parser gemma4` | Extracts `<think>` blocks for thinking/reasoning display |
| `--tool-call-parser gemma4` | Enables native Gemma 4 function calling |
| `--enable-chunked-prefill` | Processes long prompts in chunks to avoid OOM |
| `--enable-prefix-caching` | Caches common prompt prefixes for faster responses |

### NVFP4 Backend Selection

vLLM auto-selects the best FP4 backend for your hardware. On DGX Spark, it selects **FLASHINFER_CUTLASS** which uses native FP4 GEMM kernels. You do **not** need to set `VLLM_NVFP4_GEMM_BACKEND` — auto-selection works correctly.

If your system lacks a supported backend, vLLM falls back to the EMULATION backend. The patched `modelopt.py` provides a W4A16 bypass for emulation (dequantizes weights to BF16, keeps activations at full precision) rather than the default W4A4 emulation. If using emulation, add `--enforce-eager` and set `VLLM_NVFP4_GEMM_BACKEND=emulation`.

## Manual Patching

If you're using your own vLLM installation (not the pre-built container), you need to patch `modelopt.py`:

```bash
# Download the patched file
curl -L -o modelopt_patched.py \
  https://raw.githubusercontent.com/AEON-7/Gemma-4-31B-DECKARD-HERETIC-Uncensored-NVFP4/main/modelopt_patched.py

# Replace the original (adjust path for your Python environment)
cp modelopt_patched.py \
  $(python3 -c "import vllm; print(vllm.__path__[0])")/model_executor/layers/quantization/modelopt.py
```

### What the Patch Fixes

1. **FP8 NaN Scrubbing** — Detects and replaces FP8 E4M3 NaN values (0x7F/0xFF) in `weight_scale` tensors at load time. ModelOpt 0.42.0 produces ~60 NaN values across 39 layers.

2. **NVFP4_AWQ Quant Algo** — Adds `NVFP4_AWQ` to the recognized `quant_algo` list (the upstream code only handles `NVFP4`).

3. **AWQ Pre-Quant Scale** — Registers, loads, and applies per-channel `pre_quant_scale` tensors that AWQ uses to redistribute weight magnitudes before quantization. Without this, activations are not properly scaled for the quantized weights.

4. **W4A16 Emulation Fallback** — On the EMULATION backend only, bypasses input FP4 quantization and performs a W4A16 matmul (dequant weights to BF16, keep activations at full precision). All other backends use the standard hardware-accelerated path.

## Speculative Decoding (EAGLE) Patches

Speculative decoding with Gemma 4 requires three patches to vLLM 0.19.1. The patched files are included in this repository.

### `eagle_patched.py` — Three fixes for Gemma 4 compatibility

1. **Multimodal guard removal** — vLLM 0.19.1 blocks ALL multimodal targets from speculative decoding, even when the drafter is text-only. The patch removes the overly conservative `_raise_if_multimodal()` check since the downstream code already handles text-only drafters with multimodal targets correctly.

2. **Gemma4 model whitelist** — Adds `Gemma4ForConditionalGeneration` to the model whitelist for `image_token_id` → `image_token_index` mapping, since Gemma 4 uses `image_token_id` (258880) but not `image_token_index`.

3. **Multi-group KV cache support** — Gemma 4 uses heterogeneous attention: `head_dim=256` for sliding-window layers and `head_dim=512` for global attention layers, creating two separate KV cache groups. The spec decode framework assumed a single uniform group. The patch rewrites `initialize_attn_backend` to build a `layer_to_group` map and key attention groups by `(backend_class, kv_cache_group_id)` instead of just `backend_class`.

### `serving_chat_patched.py` — Non-streaming reasoning parser fix

The Gemma 4 reasoning parser relies on `<|channel>` (token 100) and `<channel|>` (token 101) delimiters to extract thinking content. With `skip_special_tokens=True` (the default), these delimiters are stripped from the decoded text, causing `extract_reasoning()` to return `None`. The patch re-decodes from `token_ids` with `skip_special_tokens=False` when text-based extraction fails.

### `modelopt_patched.py` — NVFP4 AWQ support + FP8 NaN fix

See [Critical Fix: FP8 NaN in Weight Scales](#critical-fix-fp8-nan-in-weight-scales) and [What the Patch Fixes](#what-the-patch-fixes) below.

### Applying the Patches

Mount all three patched files into the container as volume binds (shown in the [Docker Compose with Speculative Decoding](#docker-compose-with-speculative-decoding-eagle) example above), or copy them manually:

```bash
VLLM_PATH=$(python3 -c "import vllm; print(vllm.__path__[0])")
cp eagle_patched.py    $VLLM_PATH/v1/spec_decode/eagle.py
cp serving_chat_patched.py $VLLM_PATH/entrypoints/openai/chat_completion/serving.py
cp modelopt_patched.py $VLLM_PATH/model_executor/layers/quantization/modelopt.py
```

## Quantization Pipeline

Three quantization variants were produced and benchmarked on B200:

```
Gemma 4 31B DECKARD HERETIC (BF16, ~62 GB)
    |
    +-- [NVFP4 AWQ_LITE]     --> 20.45 GB  (10.6 min)  -- baseline
    |
    +-- [NVFP4 AWQ_FULL]     --> 20.45 GB  (74.4 min)  -- exhaustive optimization
    |
    +-- [NVFP4 SVDQuant]     --> 20.94 GB  (69.1 min)  -- SVD decomposition
```

### AWQ_FULL (Recommended)

- **Algorithm**: `NVFP4_AWQ_FULL_CFG` — exhaustive grid search with `alpha_step=0.1` across 10 scaling factors per layer + `awq_clip` clipping ratio optimization
- **Calibration**: 2048 samples from CNN DailyMail, 1024 token sequence length
- **Key advantage**: Mathematically optimal per-channel scaling at the cost of longer quantization time

### SVDQuant

- **Algorithm**: `NVFP4_SVDQUANT_DEFAULT_CFG` — SVD decomposition separates outlier weight channels into a low-rank BF16 residual (rank=32), then quantizes cleaned weights to NVFP4
- **Calibration**: 2048 samples from CNN DailyMail, 1024 token sequence length
- **Key advantage**: Preserves outlier channels at full BF16 precision, potentially higher quality

### Native B200 Calibration

All variants were quantized on NVIDIA B200 with native FP4 hardware instructions (SM 12.0). The calibration measures actual FP4 rounding behavior on real Blackwell hardware rather than simulating it, producing more accurate scale factors than calibrating on non-FP4 GPUs.

### Dense (31B) vs MoE (26B) Comparison

| Metric | This Model (31B Dense) | [MoE 26B-A4B](https://github.com/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) |
|---|---|---|
| Active params/token | **31.3B** | ~4B |
| NVFP4 model size | 20.45 GB | 15.3 GB |
| Aggregate tok/s @ 16 conc | 161 | 368 |
| Per-request tok/s | ~11 | ~23 |
| Quality | Higher (full dense) | Good (MoE routing) |
| Best for | Quality-critical tasks | Speed, concurrency |

## NVFP4 Weight Format

Each quantized layer stores:
- `weight` (uint8) — packed FP4 E2M1 pairs (16-element blocks)
- `weight_scale` (float8_e4m3fn) — per-block scale (1 per 16 elements)
- `weight_scale_2` (float32) — per-tensor global scale
- `pre_quant_scale` (bfloat16) — AWQ per-channel pre-scaling factors
- `input_scale` (float32) — static activation scale from calibration

**Quantized components**: All attention projections (Q/K/V/O) + all MLP layers (gate/up/down)
**Preserved in BF16**: Vision tower (27 layers), embedding projection, layer norms, lm_head

## Related Projects

### Models

| Model | Type | Size | tok/s (1 req) | Link |
|---|---|---|---|---|
| **Gemma 4 31B DECKARD AWQ_FULL** | Dense NVFP4 | 20.5 GB | ~11 | [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4) |
| **Gemma 4 31B DECKARD SVDQuant** | Dense NVFP4 | 20.9 GB | ~10 | [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4-SVDQuant) |
| **DECKARD E4B Drafter** | EAGLE NVFP4 | 9.6 GB | — | [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-E4B-DECKARD-HERETIC-Uncensored-NVFP4) \| [GitHub](https://github.com/AEON-7/Gemma-4-E4B-DECKARD-HERETIC-Uncensored-NVFP4) |
| **Gemma 4 26B MoE Uncensored** | MoE NVFP4 | 15.3 GB | ~50 | [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) \| [GitHub](https://github.com/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) |
| **DFlash Qwen3.5-27B Uncensored** | Dense BF16 | 52 GB | — | [HuggingFace](https://huggingface.co/AEON-7/DFlash-Qwen3.5-27B-Uncensored) |
| **DFlash Qwen3.5-27B Uncensored NVFP4** | Dense NVFP4 | 18.8 GB | — | [HuggingFace](https://huggingface.co/AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4) |

### Infrastructure

| Resource | Description | Link |
|---|---|---|
| **vLLM AWQ Container** | Patched for NVFP4_AWQ (FP8 NaN fix + pre_quant_scale) | [GHCR](https://github.com/users/AEON-7/packages/container/package/vllm-spark-gemma4-nvfp4-awq) |
| **vLLM Base Container** | Pre-built for DGX Spark SM 12.1 (non-AWQ models) | [GHCR](https://github.com/users/AEON-7/packages/container/package/vllm-spark-gemma4-nvfp4) |
| **Build System** | spark-vllm-docker (compile vLLM from source) | [GitHub](https://github.com/eugr/spark-vllm-docker) |
| **Base Model** | DECKARD HERETIC (BF16) | [HuggingFace](https://huggingface.co/DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking) |

## Hardware Requirements

- **Minimum**: Any NVIDIA GPU with >= 24 GB VRAM
- **Recommended**: NVIDIA DGX Spark (GB10), RTX 5090, B200, or any Blackwell/Ada GPU
- **Optimal**: DGX Spark for native FP4 hardware acceleration

## Building from Source

If you're not on a DGX Spark, compile vLLM for your GPU architecture:

```bash
# Set your GPU's SM version
export TORCH_CUDA_ARCH_LIST="your_sm_version"  # e.g., 8.9 for RTX 4090, 12.0 for B200
export FLASHINFER_CUDA_ARCH_LIST="your_sm_version"

# Clone and build vLLM
git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout v0.19.1
pip install -e . --no-build-isolation

# Install dependencies
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.6/
pip install "transformers>=5.4.0"

# Apply the NVFP4_AWQ patch (required for AWQ_FULL models)
curl -L -o $(python3 -c "import vllm; print(vllm.__path__[0])")/model_executor/layers/quantization/modelopt.py \
  https://raw.githubusercontent.com/AEON-7/Gemma-4-31B-DECKARD-HERETIC-Uncensored-NVFP4/main/modelopt_patched.py
```

### DGX Spark users

Use the [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) build system:

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
./build.sh --tf5  # --tf5 installs transformers v5 (required for Gemma 4)
```

## Disclaimer

**THIS IS AN UNCENSORED MODEL.** By downloading, accessing, or using this model, you expressly acknowledge and agree to the following:

You assume full and sole responsibility for all outputs generated, all actions taken based on outputs, and compliance with applicable laws. The authors are not responsible for any harmful, illegal, or objectionable content produced by the model.

These tools serve legitimate purposes including security research, red-teaming, content analysis, and creative work. The absence of safety guardrails demands a correspondingly higher standard of care from the operator. You must implement your own safeguards appropriate to your use case and jurisdiction.

## License

This model inherits the [Gemma license](https://ai.google.dev/gemma/terms) from Google.

---

## ☕ Support the work

If this release has been useful, tips are deeply appreciated — they go directly toward more compute, more models, and more open releases.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <strong>₿ Bitcoin (BTC)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/btc.png" alt="BTC QR" width="200"/><br/>
      <sub><code>bc1q09xmzn00q4z3c5raene0f3pzn9d9pvawfm0py4</code></sub>
    </td>
    <td align="center" width="50%">
      <strong>Ξ Ethereum (ETH)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/eth.png" alt="ETH QR" width="200"/><br/>
      <sub><code>0x1512667F6D61454ad531d2E45C0a5d1fd82D0500</code></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>◎ Solana (SOL)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/sol.png" alt="SOL QR" width="200"/><br/>
      <sub><code>DgQsjHdAnT5PNLQTNpJdpLS3tYGpVcsHQCkpoiAKsw8t</code></sub>
    </td>
    <td align="center" width="50%">
      <strong>ⓜ Monero (XMR)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/xmr.png" alt="XMR QR" width="200"/><br/>
      <sub><code>836XrSKw4R76vNi3QPJ5Fa9ugcyvE2cWmKSPv3AhpTNNKvqP8v5ba9JRL4Vh7UnFNjDz3E2GXZDVVenu3rkZaNdUFhjAvgd</code></sub>
    </td>
  </tr>
</table>

> **Ethereum L2s (Base, Arbitrum, Optimism, Polygon, etc.) and EVM-compatible tokens** can be sent to the same Ethereum address.
