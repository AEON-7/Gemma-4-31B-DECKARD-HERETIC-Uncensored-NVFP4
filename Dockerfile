FROM ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4:latest

# Patch modelopt.py with NVFP4_AWQ support + FP8 NaN scrubbing fix
# Fixes: ModelOpt quantizer bug that produces FP8 E4M3 NaN (0x7F/0xFF)
#        in weight_scale tensors, causing NaN propagation during inference.
# Also adds: NVFP4_AWQ quant_algo support, AWQ pre_quant_scale handling,
#            W4A16 bypass for EMULATION backend (dequant weights only).
COPY modelopt_patched.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py

# Patch non-streaming reasoning parser: re-decode with skip_special_tokens=False
# when <|channel>/<channel|> delimiters are stripped from output.text
COPY serving_chat_patched.py /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py

LABEL org.opencontainers.image.description="vLLM 0.19.1rc1 for DGX Spark (SM 12.1) with NVFP4_AWQ patch + reasoning parser fix"
LABEL org.opencontainers.image.source="https://github.com/AEON-7/Gemma-4-31B-DECKARD-HERETIC-Uncensored-NVFP4"
