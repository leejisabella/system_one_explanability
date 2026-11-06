#!/usr/bin/env bash
# E.107 on the GPU machine (research_docs/14): DPO from the merged SFT treatment checkpoint, merge,
# evaluate as E.106 in the same run, so it is compared with the base and the SFT arms on the same stack.
# Needs E.106's checkpoints (gpt-oss-20b-sft-review-merged). Resumable like scripts/run_e106.sh.
#
# Usage: scripts/run_e107.sh <E.106 run id>     e.g. scripts/run_e107.sh e106-v1-2026-11-04
set -uo pipefail
RUN_ID=${1:?run id}
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" UV_PYTHON=${UV_PYTHON:-3.12} PYTORCH_ALLOC_CONF=expandable_segments:True
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1
CK=checkpoints
MARKS=$HOME/.e106_done/$RUN_ID
mkdir -p "$MARKS"

stage() { echo "STAGE $* $(date -u +%FT%TZ)"; }
fail() { stage "$1 FAILED"; pkill -f "[v]llm serve"; exit 1; }

[ -f $CK/gpt-oss-20b-sft-review-merged/train_info.json ] || fail "missing the SFT treatment checkpoint (run E.106 first)"
if [ ! -f $CK/gpt-oss-20b-dpo-review/train_info.json ]; then
  stage "dpo START"; rm -rf $CK/gpt-oss-20b-dpo-review
  uv run --group train python -m sysone.posttrain.dpo --config configs/posttrain/dpo_review.yaml || fail dpo
  stage "dpo OK"
fi
if [ ! -f $CK/gpt-oss-20b-dpo-review-merged/train_info.json ]; then
  stage "merge dpo START"; rm -rf $CK/gpt-oss-20b-dpo-review-merged
  uv run --group train python -m sysone.posttrain.dpo --config configs/posttrain/dpo_review.yaml --merge || fail "merge dpo"
  stage "merge dpo OK"
fi
if [ ! -f "$MARKS/eval_dpo_review" ]; then
  stage "eval dpo_review START"
  pkill -f "[v]llm serve"; sleep 5
  scripts/serve_part2.sh $CK/gpt-oss-20b-dpo-review-merged gpt-oss-20b-dpo-review 8000 > "$HOME/vllm_dpo_review.log" 2>&1 &
  for _ in $(seq 1 90); do curl -sf localhost:8000/v1/models > /dev/null && break; sleep 10; done
  curl -sf localhost:8000/v1/models > /dev/null || fail "serve dpo_review"
  uv run --group train python -m sysone.experiments.e106 --run-id "$RUN_ID" --reviewers dpo_review || fail "eval dpo_review"
  pkill -f "[v]llm serve"; sleep 5
  touch "$MARKS/eval_dpo_review"; stage "eval dpo_review OK"
fi
stage "E107 DONE"
