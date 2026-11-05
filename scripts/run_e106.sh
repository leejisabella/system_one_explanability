#!/usr/bin/env bash
# E.106 end to end on the GPU machine (research_docs/14, runbook): save the bf16 base, evaluate it,
# SFT and merge both arms, evaluate both. Resumable: every step that already finished is skipped
# (training and merging by their train_info.json, evaluations by a marker), and E.106 itself resumes
# per reviewer, so after a Spot preemption restart the VM and run this again with the same run id.
#
# Usage: scripts/run_e106.sh <run id>     e.g. scripts/run_e106.sh e106-v1-2026-11-04
# Progress lines start with "STAGE"; the full output goes to stdout.
set -uo pipefail
RUN_ID=${1:?run id}
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" UV_PYTHON=${UV_PYTHON:-3.12} PYTORCH_ALLOC_CONF=expandable_segments:True
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1
CK=checkpoints
MARKS=$HOME/.e106_done/$RUN_ID   # outside results/, which E.106 owns
mkdir -p "$MARKS"

stage() { echo "STAGE $* $(date -u +%FT%TZ)"; }
fail() { stage "$1 FAILED"; pkill -f "[v]llm serve"; exit 1; }

if [ ! -f $CK/gpt-oss-20b-base/train_info.json ]; then
  stage "save_base START"; rm -rf $CK/gpt-oss-20b-base
  uv run --group train python -m sysone.posttrain.sft --save-base $CK/gpt-oss-20b-base || fail save_base
  stage "save_base OK"
fi

evaluate() {  # alias, checkpoint dir, served name
  [ -f "$MARKS/eval_$1" ] && return
  stage "eval $1 START"
  pkill -f "[v]llm serve"; sleep 5
  scripts/serve_part2.sh "$2" "$3" 8000 > "$HOME/vllm_$1.log" 2>&1 &
  for _ in $(seq 1 90); do curl -sf localhost:8000/v1/models > /dev/null && break; sleep 10; done
  curl -sf localhost:8000/v1/models > /dev/null || fail "serve $1"
  uv run --group train python -m sysone.experiments.e106 --run-id "$RUN_ID" --reviewers "$1" || fail "eval $1"
  pkill -f "[v]llm serve"; sleep 5
  touch "$MARKS/eval_$1"; stage "eval $1 OK"
}

train() {  # config name, run name
  if [ ! -f $CK/$2/train_info.json ]; then
    stage "sft $1 START"; rm -rf $CK/$2
    uv run --group train python -m sysone.posttrain.sft --config configs/posttrain/$1.yaml || fail "sft $1"
    stage "sft $1 OK"
  fi
  if [ ! -f $CK/$2-merged/train_info.json ]; then
    stage "merge $1 START"; rm -rf $CK/$2-merged
    uv run --group train python -m sysone.posttrain.sft --config configs/posttrain/$1.yaml --merge || fail "merge $1"
    stage "merge $1 OK"
  fi
}

evaluate base_local $CK/gpt-oss-20b-base gpt-oss-20b-base
train sft_review gpt-oss-20b-sft-review
train sft_plain gpt-oss-20b-sft-plain
evaluate sft_review $CK/gpt-oss-20b-sft-review-merged gpt-oss-20b-sft-review
evaluate sft_plain $CK/gpt-oss-20b-sft-plain-merged gpt-oss-20b-sft-plain
stage "E106 DONE"
