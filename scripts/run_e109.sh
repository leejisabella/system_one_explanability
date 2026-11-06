#!/usr/bin/env bash
# E.109 on the GPU machine (research_docs/15): GRPO in review mode, then evaluate as E.106 in the same run.
#   1. pre-pass: serve the starting model (init_from) with scripts/serve_part2.sh, sample 8 answers per
#      candidate prompt, write the training set (1-7 of 8 correct) and the held-out prompts; stop the server
#   2. train: `trl vllm-serve` on GPU 1 generates, GRPO trains on GPU 0 (vllm_mode: colocate uses GPU 0 only)
#   3. merge, then serve the merged model and evaluate it: e106 --reviewers grpo_review
# Needs 2 GPUs in server mode, init_from set in the config (not the placeholder), E.108's prompt files in
# data/posttrain/, and vLLM importable from the train environment (the trainer syncs weights through it).
# Resumable like scripts/run_e107.sh: the pre-pass resumes per prompt, training from its last checkpoint
# (every 25 updates), and finished steps are skipped. GPU hours are counted by the trainer (stop rule).
#
# Usage: scripts/run_e109.sh <E.106 run id> [config]     e.g. scripts/run_e109.sh e106-v1-2026-11-04
#   For the 20-update smoke run, pass a copy of the config with max_steps: 20 and another name;
#   only the main config (name gpt-oss-20b-grpo-review) is evaluated.
set -uo pipefail
RUN_ID=${1:?run id}
CFG=${2:-configs/posttrain/grpo_review.yaml}
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" UV_PYTHON=${UV_PYTHON:-3.12} PYTORCH_ALLOC_CONF=expandable_segments:True
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1
UVR=${UVR:-"uv run --group train"}
MARKS=$HOME/.e106_done/$RUN_ID
mkdir -p "$MARKS"
TRL_PID=

stage() { echo "STAGE $* $(date -u +%FT%TZ)"; }
stop_servers() {  # the bracket keeps pkill from matching this shell; [- ] covers `vllm serve` and `trl vllm-serve`
  [ -n "$TRL_PID" ] && kill -- -"$TRL_PID" 2>/dev/null
  pkill -f "[v]llm[- ]serve"; sleep 5
}
fail() { stage "$1 FAILED"; stop_servers; exit 1; }
wait_for() {  # url, what
  for _ in $(seq 1 90); do curl -sf "$1" > /dev/null && return 0; sleep 10; done
  fail "serve $2"
}

$UVR python -m sysone.posttrain.grpo --config "$CFG" --check || fail check
eval "$($UVR python -m sysone.posttrain.grpo --config "$CFG" --paths)" || fail paths

if [ ! -f "$PREPASS_DIR/summary.json" ]; then
  stage "prepass START"
  stop_servers
  CUDA_VISIBLE_DEVICES=0 scripts/serve_part2.sh "$INIT_FROM" "$PREPASS_NAME" 8000 > "$HOME/vllm_grpo_start.log" 2>&1 &
  wait_for localhost:8000/v1/models "$PREPASS_NAME"
  $UVR python -m sysone.posttrain.grpo --config "$CFG" --prepass || fail prepass
  stop_servers
  stage "prepass OK"
fi

if [ ! -f "$OUT_DIR/train_info.json" ]; then
  stage "grpo START"
  stop_servers
  if [ "$VLLM_MODE" = server ]; then
    # Its own process group, so its vLLM workers are stopped with it.
    CUDA_VISIBLE_DEVICES=1 setsid $UVR trl vllm-serve --model "$INIT_FROM" --port "$TRL_PORT" --dtype bfloat16 \
      --max_model_len 8192 --gpu_memory_utilization 0.90 > "$HOME/vllm_grpo_trl.log" 2>&1 &
    TRL_PID=$!
    wait_for "localhost:$TRL_PORT/health/" "trl vllm-serve"
  fi
  CUDA_VISIBLE_DEVICES=0 $UVR python -m sysone.posttrain.grpo --config "$CFG" || fail grpo
  stop_servers; TRL_PID=
  stage "grpo OK"
fi

if [ ! -f "$MERGED_DIR/train_info.json" ]; then
  stage "merge grpo START"; rm -rf "$MERGED_DIR"
  $UVR python -m sysone.posttrain.grpo --config "$CFG" --merge || fail "merge grpo"
  stage "merge grpo OK"
fi

if [ "$SERVED_NAME" != gpt-oss-20b-grpo-review ]; then
  stage "eval grpo_review SKIPPED (not the main config)"
elif [ ! -f "$MARKS/eval_grpo_review" ]; then
  stage "eval grpo_review START"
  stop_servers
  scripts/serve_part2.sh "$MERGED_DIR" "$SERVED_NAME" 8000 > "$HOME/vllm_grpo_review.log" 2>&1 &
  wait_for localhost:8000/v1/models grpo_review
  $UVR python -m sysone.experiments.e106 --run-id "$RUN_ID" --reviewers grpo_review || fail "eval grpo_review"
  stop_servers
  touch "$MARKS/eval_grpo_review"; stage "eval grpo_review OK"
fi
stage "E109 DONE"
