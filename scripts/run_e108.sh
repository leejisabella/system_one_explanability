#!/usr/bin/env bash
# E.108 on the GPU machine (research_docs/15): SFT both round-2 arms from the base, merge, and evaluate
# each as E.106 in the same run, so they're compared with the base and round 1 on the same stack.
# Needs the bf16 base (checkpoints/gpt-oss-20b-base, from E.106 or `sft --save-base`), E.106's run
# folder with the base's evaluation, and data/posttrain/part2_round2_v1 from sysone.experiments.e108.
# Resumable like scripts/run_e106.sh: rerun after a Spot preemption and it skips finished steps.
#
# Usage: scripts/run_e108.sh <E.106 run id>     e.g. scripts/run_e108.sh e106-v1-2026-11-04
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

[ -f "results/runs/$RUN_ID/reviews.jsonl" ] || fail "missing results/runs/$RUN_ID (copy E.106's run, with the base's evaluation)"
for arm in consist_review balanced_review; do
  cfg=configs/posttrain/$arm.yaml
  name=gpt-oss-20b-${arm//_/-}
  uv run python -m sysone.posttrain.sft --config $cfg --check || fail "check $arm"
  if [ ! -f $CK/$name/train_info.json ]; then
    stage "sft $arm START"; rm -rf $CK/$name
    uv run --group train python -m sysone.posttrain.sft --config $cfg || fail "sft $arm"
    stage "sft $arm OK"
  fi
  if [ ! -f $CK/$name-merged/train_info.json ]; then
    stage "merge $arm START"; rm -rf $CK/$name-merged
    uv run --group train python -m sysone.posttrain.sft --config $cfg --merge || fail "merge $arm"
    stage "merge $arm OK"
  fi
  if [ ! -f "$MARKS/eval_$arm" ]; then
    stage "eval $arm START"
    pkill -f "[v]llm serve"; sleep 5
    scripts/serve_part2.sh $CK/$name-merged $name 8000 > "$HOME/vllm_$arm.log" 2>&1 &
    for _ in $(seq 1 90); do curl -sf localhost:8000/v1/models > /dev/null && break; sleep 10; done
    curl -sf localhost:8000/v1/models > /dev/null || fail "serve $arm"
    uv run --group train python -m sysone.experiments.e106 --run-id "$RUN_ID" --reviewers $arm || fail "eval $arm"
    pkill -f "[v]llm serve"; sleep 5
    touch "$MARKS/eval_$arm"; stage "eval $arm OK"
  fi
done
stage "E108 DONE"
