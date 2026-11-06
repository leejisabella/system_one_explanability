#!/usr/bin/env bash
# E.110 on the GPU machine (research_docs/15): activation steering on the base, then measurement 3 on
# every trained checkpoint that exists so far. The base steps can run in E.108's GPU session; run it
# again after E.108 and E.109 to add their models (finished steps are skipped).
#
# Needs: the E.110 run built on the laptop, where E.105's traces are
#   (uv run python -m sysone.interp.steering build --run-id <id>), copied to results/runs/<id>;
#   data/cascade/ (the item texts); checkpoints/gpt-oss-20b-base (sysone.posttrain.sft --save-base);
#   results/runs/e106-v1-2026-11-04/reviews.jsonl for the report's vLLM reference (optional).
# Everything runs in transformers with forward hooks, not vLLM, so any vLLM server is stopped first.
#
# Usage: scripts/run_e110.sh <E.110 run id>     e.g. scripts/run_e110.sh e110-v1-2026-11-06
set -uo pipefail
RUN_ID=${1:?run id}
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" UV_PYTHON=${UV_PYTHON:-3.12} PYTORCH_ALLOC_CONF=expandable_segments:True
CK=checkpoints
BASE=$CK/gpt-oss-20b-base
MARKS=$HOME/.e110_done/$RUN_ID
mkdir -p "$MARKS"
# Always --group train: plain `uv run` re-syncs and removes torch (research_docs/14).
E110=(uv run --group train python -m sysone.interp.steering)

stage() { echo "STAGE $* $(date -u +%FT%TZ)"; }
fail() { stage "$1 FAILED"; exit 1; }
step() {  # step <name> <command...>: run once; the mark file makes a rerun skip it
  local name=$1; shift
  [ -f "$MARKS/$name" ] && return 0
  stage "$name START"
  "$@" || fail "$name"
  touch "$MARKS/$name"; stage "$name OK"
}

[ -f "results/runs/$RUN_ID/selection.jsonl" ] || fail "missing the build (run build on the laptop and copy results/runs/$RUN_ID)"
[ -f "$BASE/config.json" ] || fail "missing the bf16 base (sysone.posttrain.sft --save-base $BASE)"
pkill -f "[v]llm serve" && sleep 5

step capture_base "${E110[@]}" capture --run-id "$RUN_ID" --model "$BASE"
step directions "${E110[@]}" directions --run-id "$RUN_ID"
step select_alpha "${E110[@]}" select-alpha --run-id "$RUN_ID" --model "$BASE"
# The unsteered pass first: it's the paired baseline on the same stack.
for d in none steer random proposal; do
  step "steer_$d" "${E110[@]}" steer --run-id "$RUN_ID" --model "$BASE" --directions "$d"
done
for name in consist_review balanced_review grpo_review; do
  dir=$CK/gpt-oss-20b-${name//_/-}-merged
  if [ -f "$dir/train_info.json" ]; then
    step "compare_$name" "${E110[@]}" compare --run-id "$RUN_ID" --model "$dir" --name "$name"
  else
    stage "compare $name SKIPPED (no $dir yet)"
  fi
done
"${E110[@]}" report --run-id "$RUN_ID" > /dev/null || fail report
stage "E110 DONE (results/runs/$RUN_ID/report.md)"
