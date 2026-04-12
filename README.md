# Gemma 4 31B DECKARD HERETIC Uncensored NVFP4

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
docker pull ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4:latest
```

**Image contents:**
- vLLM 0.19.1rc1 compiled for SM 12.1 (Blackwell GB10)
- PyTorch 2.12.0 + CUDA 13.0
- transformers 5.5.0
- FlashInfer 0.6.7
- **Patched `modelopt.py`** — fixes FP8 NaN in weight scales + adds NVFP4_AWQ support + W4A16 emulation bypass
- Built from [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) with `--tf5` flag

> [Container on GHCR](https://github.com/users/AEON-7/packages/container/package/vllm-spark-gemma4-nvfp4) | Compatible with both the MoE and Dense DECKARD models

### Container Tags

| Tag | Description |
|---|---|
| `latest` | Current patched build (v2) with NVFP4_AWQ fixes |
| `v2-nvfp4-awq` | Same as latest — explicit version tag |

## Critical Fix: FP8 NaN in Weight Scales

The NVFP4 checkpoint produced by ModelOpt 0.42.0 contains **scattered FP8 NaN values** (0x7F / 0xFF) in the per-block `weight_scale` tensors. These are a quantizer bug — 60 NaN values across 39 of the 60 layers, primarily in `down_proj` and `o_proj` weight scales.

**Impact**: A single FP8 NaN scale causes the entire 16-element weight block to dequantize to NaN, which propagates through the model and produces empty/garbage output.

**Fix**: The patched container automatically scrubs FP8 NaN values to zero at model load time, zeroing out the affected blocks (60 out of ~280 million scale elements — negligible quality impact).

If you're **not** using the pre-built container, you must apply the [`modelopt_patched.py`](modelopt_patched.py) patch. See [Manual Patching](#manual-patching) below.

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
    image: ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4:latest
    container_name: vllm-deckard-31b
    restart: unless-stopped
    network_mode: host
    volumes:
      - /path/to/model:/models/deckard
    environment:
      - VLLM_NVFP4_GEMM_BACKEND=emulation
      - VLLM_TEST_FORCE_FP8_MARLIN=1
      - VLLM_MARLIN_USE_ATOMIC_ADD=1
      - VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
      - TORCH_MATMUL_PRECISION=high
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - NVIDIA_FORWARD_COMPAT=1
    command:
      - bash
      - -c
      - |
        exec vllm serve /models/deckard \
          --served-model-name deckard \
          --quantization modelopt \
          --dtype auto \
          --kv-cache-dtype auto \
          --tensor-parallel-size 1 \
          --max-model-len 4096 \
          --max-num-seqs 16 \
          --gpu-memory-utilization 0.85 \
          --trust-remote-code \
          --enforce-eager \
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
    image: ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4:latest
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
          --served-model-name deckard \
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

### Key Deployment Flags

| Flag | Purpose | When Required |
|---|---|---|
| `--quantization modelopt` | Use NVIDIA ModelOpt NVFP4 format | Always |
| `--enforce-eager` | Disable torch.compile (avoids dynamo errors in emulation) | DGX Spark / emulation backend |
| `--kv-cache-dtype auto` | Let vLLM choose KV cache dtype | DGX Spark (use `fp8` on B200) |
| `--max-model-len 4096` | Conservative context for 128 GB unified memory | DGX Spark (use 131072 on B200) |
| `--reasoning-parser gemma4` | Extracts `<think>` blocks for reasoning display | Optional |
| `--tool-call-parser gemma4` | Enables native Gemma 4 function calling | Optional |
| `--enable-chunked-prefill` | Processes long prompts in chunks to avoid OOM | Recommended |
| `--enable-prefix-caching` | Caches common prompt prefixes for faster responses | Recommended |

### Environment Variables

| Variable | Value | Purpose |
|---|---|---|
| `VLLM_NVFP4_GEMM_BACKEND` | `emulation` | **Required on DGX Spark** — uses W4A16 dequant bypass (no fbgemm_gpu) |
| `VLLM_TEST_FORCE_FP8_MARLIN` | `1` | Force FP8 Marlin kernel selection |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | `1` | Enable atomic operations for Marlin |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | Allow context lengths above default |
| `TORCH_MATMUL_PRECISION` | `high` | Higher precision matrix multiply |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Better CUDA memory allocation |

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

Or mount it as a volume:

```bash
docker run ... \
  -v ./modelopt_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py \
  ...
```

### What the Patch Fixes

1. **FP8 NaN Scrubbing** — Detects and replaces FP8 E4M3 NaN values (0x7F/0xFF) in `weight_scale` tensors at load time. ModelOpt 0.42.0 produces ~60 NaN values across 39 layers.

2. **NVFP4_AWQ Quant Algo** — Adds `NVFP4_AWQ` to the recognized `quant_algo` list (the upstream code only handles `NVFP4`).

3. **AWQ Pre-Quant Scale** — Registers, loads, and applies per-channel `pre_quant_scale` tensors that AWQ uses to redistribute weight magnitudes before quantization. Without this, activations are not properly scaled for the quantized weights.

4. **W4A16 Emulation Bypass** — On the EMULATION backend (DGX Spark), the standard code path quantizes *both* inputs and weights to FP4 (W4A4). The patch bypasses input quantization and only dequantizes weights, performing the matmul in BF16 (W4A16). This avoids the aggressive quality loss from FP4 input quantization.

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

## Performance Expectations

### DGX Spark (W4A16 Emulation)

| Configuration | Estimated tok/s |
|---|---|
| BF16 (no quantization) | ~3-5 |
| NVFP4 AWQ_FULL (W4A16 bypass) | ~8-12 |
| NVFP4 SVDQuant (W4A16 bypass) | ~7-10 |

> Note: DGX Spark uses W4A16 emulation (weights dequantized to BF16, activations stay BF16). This is slower than native FP4 hardware but produces higher quality output than full W4A4 emulation.

### B200 / Native FP4 GPUs

| Configuration | Estimated tok/s |
|---|---|
| NVFP4 AWQ_FULL (native) | ~12-14 |
| NVFP4 SVDQuant (native) | ~10-13 |

### Dense (31B) vs MoE (26B) Comparison

| Metric | This Model (31B Dense) | [MoE 26B-A4B](https://github.com/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) |
|---|---|---|
| Active params/token | **31.3B** | ~4B |
| NVFP4 model size | 20.45 GB | 15.3 GB |
| Expected tok/s (DGX Spark) | ~8-12 | ~43-50 |
| Quality | Higher (full dense) | Good (MoE routing) |
| Best for | Quality-critical tasks | Speed, concurrency |

The dense model reads all 31.3B parameters per token vs ~4B active for MoE, making it slower but providing higher quality from full parameter utilization.

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

| Model | Type | Size | tok/s | Link |
|---|---|---|---|---|
| **Gemma 4 31B DECKARD AWQ_FULL** | Dense NVFP4 | 20.5 GB | ~8-12 | [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4) |
| **Gemma 4 31B DECKARD SVDQuant** | Dense NVFP4 | 20.9 GB | ~7-10 | [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4-SVDQuant) |
| **Gemma 4 26B MoE Uncensored** | MoE NVFP4 | 15.3 GB | ~43-50 | [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) \| [GitHub](https://github.com/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) |
| **DFlash Qwen3.5-27B Uncensored** | Dense BF16 | 52 GB | — | [HuggingFace](https://huggingface.co/AEON-7/DFlash-Qwen3.5-27B-Uncensored) |
| **DFlash Qwen3.5-27B Uncensored NVFP4** | Dense NVFP4 | 18.8 GB | ~15-18 | [HuggingFace](https://huggingface.co/AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4) |

### Infrastructure

| Resource | Description | Link |
|---|---|---|
| **vLLM Container** | Pre-built for DGX Spark SM 12.1, patched for NVFP4_AWQ | [GHCR](https://github.com/users/AEON-7/packages/container/package/vllm-spark-gemma4-nvfp4) |
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
