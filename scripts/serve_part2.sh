#!/usr/bin/env bash
# Serve one Part 2 reviewer with vLLM on the GPU machine (research_docs/14, "Serving and evaluation").
#
# Usage: scripts/serve_part2.sh <checkpoint dir> <served name> [port]
#   scripts/serve_part2.sh checkpoints/gpt-oss-20b-base               gpt-oss-20b-base
#   scripts/serve_part2.sh checkpoints/gpt-oss-20b-sft-review-merged  gpt-oss-20b-sft-review
#   scripts/serve_part2.sh checkpoints/gpt-oss-20b-sft-plain-merged   gpt-oss-20b-sft-plain
#
# The served name must match the alias's model in configs/models.yaml (base_local, sft_review, ...).
# Then, from wherever the harness runs (the same machine, or through an SSH tunnel to the port):
#   export LOCAL_LLM_BASE_URL=http://localhost:${PORT}/v1
#   uv run python scripts/preflight_part2.py --local base_local
#   uv run python -m sysone.experiments.e106 --run-id e106-v1-<date> --reviewers base_local
#
# vLLM is installed separately from the harness (its CUDA build must match the machine), e.g.
#   uv tool install vllm   (or: pip install vllm in its own environment)
# Every model, the base included, is a bf16 checkpoint, so they all run at the same precision.
set -euo pipefail
CKPT=${1:?checkpoint dir}
NAME=${2:?served model name}
PORT=${3:-8000}
exec vllm serve "$CKPT" \
  --served-model-name "$NAME" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm
