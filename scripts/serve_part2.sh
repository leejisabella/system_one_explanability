#!/usr/bin/env bash
# Serve one Part 2 model with vLLM on the GPU machine (research_docs/14, "Serving and evaluation").
#
# Usage: scripts/serve_part2.sh <checkpoint dir or Hub ID> <served name> [port] [extra vllm args...]
#   scripts/serve_part2.sh checkpoints/qwen3-14b-base               qwen3-14b-base
#   scripts/serve_part2.sh checkpoints/qwen3-14b-sft-review-merged  qwen3-14b-sft-review
#   scripts/serve_part2.sh checkpoints/qwen3-14b-sft-plain-merged   qwen3-14b-sft-plain
#   scripts/serve_part2.sh Qwen/Qwen3-32B qwen3-32b-teacher 8000 --quantization fp8   # E.105's teacher
#
# The served name must match the alias's model in configs/models.yaml (base_local, sft_review, ...).
# Then, from wherever the harness runs (the same machine, or through an SSH tunnel to the port):
#   export LOCAL_LLM_BASE_URL=http://localhost:${PORT}/v1
#   uv run python scripts/preflight_part2.py --local base_local
#   uv run python -m sysone.experiments.e106 --run-id e106-v1-<date> --reviewers base_local
#
# Qwen3 thinks by default. --reasoning-parser qwen3 splits its <think> block into the reply's
# reasoning field, which the harness saves as the trace; without it the reasoning stays in the
# answer and replies don't parse. --generation-config auto uses the checkpoint's own sampling
# settings (Qwen3's recommended temperature 0.6, top-p 0.95, top-k 20 for thinking; greedy
# decoding makes it repeat itself). Every saved checkpoint carries the same generation_config.json.
# For a gpt-oss checkpoint, set REASONING_PARSER= (empty) and GENERATION_CONFIG=vllm.
#
# Qwen3-32B in bf16 leaves little KV cache on an 80 GB GPU; --quantization fp8 frees room for the
# teacher. The reviewers stay bf16, so the base and the trained ones run at the same precision.
#
# vLLM is installed separately from the harness (its CUDA build must match the machine), e.g.
#   uv tool install vllm   (or: pip install vllm in its own environment)
set -euo pipefail
CKPT=${1:?checkpoint dir or Hub ID}
NAME=${2:?served model name}
PORT=${3:-8000}
shift $(( $# < 3 ? $# : 3 ))
REASONING_PARSER=${REASONING_PARSER-qwen3}
GENERATION_CONFIG=${GENERATION_CONFIG:-auto}
exec vllm serve "$CKPT" \
  --served-model-name "$NAME" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --generation-config "$GENERATION_CONFIG" \
  ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"} \
  "$@"
