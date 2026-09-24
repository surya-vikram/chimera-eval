#!/usr/bin/env bash
set -euo pipefail
# EDIT THIS BLOCK. Environment overrides are useful for containers/automation.
export EVAL_ROOT="${EVAL_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export DATA_DIR="${DATA_DIR:-$EVAL_ROOT/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-$EVAL_ROOT/outputs}"
export RUN_NAME="${RUN_NAME:-evaluation}"
export SPLIT="${SPLIT:-main_test}"
export MODEL_URL="${MODEL_URL:-http://127.0.0.1:8000/v1}"
export MODEL_NAME="${MODEL_NAME:-eval-smoke}"
export MODEL_CONCURRENCY="${MODEL_CONCURRENCY:-4}"
export MODEL_CONTEXT="${MODEL_CONTEXT:-8192}"
export MODEL_TEMPERATURE="${MODEL_TEMPERATURE:-0.6}"
export MODEL_TOP_P="${MODEL_TOP_P:-0.95}"
export MODEL_TOP_K="${MODEL_TOP_K:--1}"
export MODEL_REPETITION_PENALTY="${MODEL_REPETITION_PENALTY:-1.0}"
export MODEL_MIN_P="${MODEL_MIN_P:-0.0}"
export MODEL_PRESENCE_PENALTY="${MODEL_PRESENCE_PENALTY:-0.0}"
export MODEL_FREQUENCY_PENALTY="${MODEL_FREQUENCY_PENALTY:-0.0}"
export MODEL_CHAT_TEMPLATE_KWARGS="${MODEL_CHAT_TEMPLATE_KWARGS:-}"
export MODEL_STOP_JSON="${MODEL_STOP_JSON:-[]}"
export JUDGE_URL="${JUDGE_URL:-$MODEL_URL}"
export JUDGE_NAME="${JUDGE_NAME:-$MODEL_NAME}"
export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-2}"
export JUDGE_CONTEXT="${JUDGE_CONTEXT:-32768}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-8192}"
export JUDGE_MAX_RETRY_TOKENS="${JUDGE_MAX_RETRY_TOKENS:-16384}"
export JUDGE_ATTEMPTS="${JUDGE_ATTEMPTS:-3}"
export JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
export JUDGE_TOP_P="${JUDGE_TOP_P:-1.0}"
export JUDGE_TOP_K="${JUDGE_TOP_K:--1}"
export JUDGE_REPETITION_PENALTY="${JUDGE_REPETITION_PENALTY:-1.0}"
export JUDGE_MIN_P="${JUDGE_MIN_P:-0.0}"
export JUDGE_PRESENCE_PENALTY="${JUDGE_PRESENCE_PENALTY:-0.0}"
export JUDGE_FREQUENCY_PENALTY="${JUDGE_FREQUENCY_PENALTY:-0.0}"
export JUDGE_STOP_JSON="${JUDGE_STOP_JSON:-[]}"
export JUDGE_CHAT_TEMPLATE_KWARGS="${JUDGE_CHAT_TEMPLATE_KWARGS:-}"
export SHARED_ENDPOINT_CONCURRENCY="${SHARED_ENDPOINT_CONCURRENCY:-4}"
export N_SAMPLES="${N_SAMPLES:-2}"
export PASS_K="${PASS_K:-2}"
export SEED="${SEED:-42}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
export REQUEST_RETRIES="${REQUEST_RETRIES:-2}"
export MAX_PENDING="${MAX_PENDING:-16}"
export LONG_CONTEXT_CONCURRENCY="${LONG_CONTEXT_CONCURRENCY:-8}" # Generation requests for >=16K cells.
export TEST_MAX_ITEMS="${TEST_MAX_ITEMS:-4000}"
export LIMIT_PER_TASK="${LIMIT_PER_TASK:-0}" # Nonzero => diagnostic, never full benchmark.
export TASKS="${TASKS:-}"                  # Comma-separated task IDs; blank = all.
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}" # 0 uses per-record caps; overrides are recorded.
export DOMAIN_MAX_TOKENS_JSON="${DOMAIN_MAX_TOKENS_JSON:-}"
export TASK_MAX_TOKENS_JSON="${TASK_MAX_TOKENS_JSON:-}" # e.g. {"gsm8k":8192,"math500":32768}
if [[ -z "$TASK_MAX_TOKENS_JSON" ]]; then
  export TASK_MAX_TOKENS_JSON='{"gsm8k":8192,"gsm8k_train":8192,"math500":32768,"nemotron_math":32768,"triviaqa":4096,"mcqa":4096,"arc":4096,"mmlu_pro":8192,"openqa":8192,"science":8192,"hotpot":4096,"hotpot_train":4096,"cascade_chat":8192,"cascade_lists":8192,"cascade_plans":16384,"biggen":16384,"reasoning_gym":16384,"calendar":16384,"apps":16384,"humanevalplus":16384}'
fi
# Set TASK_MAX_TOKENS_JSON='{}' to use only domain/frozen-row budgets.
# Budget precedence: MAX_NEW_TOKENS > task override > domain override > frozen record.
# Output caps include reasoning tokens where the server accounts for them.
# Pilot larger caps (e.g. {"math":32768}) before freezing a comparison protocol.
# Fixed-budget evaluation: finish_reason=length scores zero and is reported.
# Compare models at the same task caps; never silently shrink prompts.
export CODE_IMAGE="${CODE_IMAGE:-chimera-eval:0.1.0}"
export CODE_TIMEOUT="${CODE_TIMEOUT:-15}"
export CODE_CONCURRENCY="${CODE_CONCURRENCY:-2}"
export PREP_TOKENIZER="${PREP_TOKENIZER:-}" # Local tokenizer path for frozen preparation admission.
export LONG_CONTEXT_BUCKETS="${LONG_CONTEXT_BUCKETS:-4096,8192,16384,32768,65536,131072}"
export PREP_LONG_CONTEXT="${PREP_LONG_CONTEXT:-1}"
export LONG_RESPONSE_RESERVE="${LONG_RESPONSE_RESERVE:-8192}"
export REGRADES_SOURCE="${REGRADES_SOURCE:-}" # Existing run directory; regrade never generates target responses.
export REGRADES_IDS="${REGRADES_IDS:-}" # Optional comma-separated IDs for a focused regrade.
export SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-}" # revise-data source; DATA_DIR must be a new destination.
# Comma-separated token windows, e.g. 16384,32768; blank evaluates all fitting MODEL_CONTEXT.
# This selects long-context tasks only; regular-domain tasks remain enabled unless TASKS filters them.
export EVAL_CONTEXT_BUCKETS="${EVAL_CONTEXT_BUCKETS:-}"
export PYTHONPATH="$EVAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Main-test task counts: edit these, then prepare a NEW data directory to enlarge a frozen split.
# Evaluation can reduce these counts on an existing split; it cannot access unreserved rows.
# Pool comments are observed source/reserved counts, not guarantees after runtime context admission.
# Revised frozen defaults total 3,982. Increase allocations only within the 4,000 cap.
declare -A TASK_SAMPLES=(
  [gsm8k]=400 # Source pool: 1,319; frozen target: 400.
  [math500]=200 # Source pool: 500; frozen target: 200.
  [arc]=160 # Source pool: 1,172; frozen target: 160.
  [mmlu_pro]=280 # Source pool: 12,000; frozen target: 280.
  [triviaqa]=200 # Source pool: 17,944; frozen target: 200.
  [hotpot]=400 # Source pool: 7,396; frozen target: 400.
  [biggen]=390 # Source pool: 406; revised admitted test: 390.
  [ifeval]=200 # Source pool: 541; frozen target: 200.
  [ifbench]=200 # Source pool: 300; frozen target: 200.
  [multi_if]=200 # Source pool: 896; frozen target: 200.
  [multichallenge]=200 # Reserved pool: 262 after screening; frozen target: 200.
  [structeval]=194 # Source pool: 950; revised admitted test: 194.
  [humanevalplus]=163 # Source: 164; admitted: 163. HumanEval/32 quarantined after reference/oracle audit.
  [bbh_boolean_expressions]=30 # Source pool: 250; frozen target: 30.
  [bbh_date_understanding]=30 # Source pool: 250; frozen target: 30.
  [bbh_disambiguation_qa]=30 # Source pool: 250; frozen target: 30.
  [bbh_formal_fallacies]=30 # Source pool: 250; frozen target: 30.
  [bbh_logical_deduction_five_objects]=30 # Source pool: 250; frozen target: 30.
  [bbh_navigate]=30 # Source pool: 250; frozen target: 30.
  [bbh_object_counting]=30 # Source pool: 250; frozen target: 30.
  [bbh_penguins_in_a_table]=30 # Source pool: 146; frozen target: 30.
  [bbh_reasoning_about_colored_objects]=30 # Source pool: 250; frozen target: 30.
  [bbh_temporal_sequences]=30 # Source pool: 250; frozen target: 30.
  [bbh_tracking_shuffled_objects_five_objects]=30 # Source pool: 250; frozen target: 30.
  [bbh_web_of_lies]=30 # Source pool: 250; frozen target: 30.
  # 4K: evaluation only; 40-row admitted RULER pools (256 published rows scanned).
  [long_4096_niah_single_1]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_4096_niah_single_2]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_4096_niah_single_3]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_4096_niah_multikey_1]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_4096_niah_multivalue]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_4096_niah_multiquery]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_4096_vt]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_4096_cwe]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_4096_fwe]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_4096_qa_1]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_4096_rag]=16 # Reserved pool: 36; frozen target: 16; context admission still applies.
  [long_4096_icl]=17 # Reserved pool: 51; frozen target: 17; context admission still applies.
  # 8K: evaluation only; 40-row admitted RULER pools (256 published rows scanned).
  [long_8192_niah_single_1]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_8192_niah_single_2]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_8192_niah_single_3]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_8192_niah_multikey_1]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_8192_niah_multivalue]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_8192_niah_multiquery]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_8192_vt]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_8192_cwe]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_8192_fwe]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_8192_qa_1]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_8192_rag]=16 # Reserved pool: 42; frozen target: 16; context admission still applies.
  [long_8192_icl]=17 # Reserved pool: 51; frozen target: 17; context admission still applies.
  # 16K: evaluation only; 40-row admitted RULER pools (256 published rows scanned).
  [long_16384_niah_single_1]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_16384_niah_single_2]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_16384_niah_single_3]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_16384_niah_multikey_1]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_16384_niah_multivalue]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_16384_niah_multiquery]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_16384_vt]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_16384_cwe]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_16384_fwe]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_16384_qa_1]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_16384_rag]=16 # Reserved pool: 43; frozen target: 16; context admission still applies.
  [long_16384_icl]=17 # Reserved pool: 51; frozen target: 17; context admission still applies.
  # 32K: evaluation only; 40-row admitted RULER pools (256 published rows scanned).
  [long_32768_niah_single_1]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_32768_niah_single_2]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_32768_niah_single_3]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_32768_niah_multikey_1]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_32768_niah_multivalue]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_32768_niah_multiquery]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_32768_vt]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_32768_cwe]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_32768_fwe]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_32768_qa_1]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_32768_rag]=16 # Reserved pool: 43; frozen target: 16; context admission still applies.
  [long_32768_icl]=17 # Reserved pool: 51; frozen target: 17; context admission still applies.
  # 64K: evaluation only; 40-row admitted RULER pools (256 published rows scanned).
  [long_65536_niah_single_1]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_65536_niah_single_2]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_65536_niah_single_3]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_65536_niah_multikey_1]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_65536_niah_multivalue]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_65536_niah_multiquery]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_65536_vt]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_65536_cwe]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_65536_fwe]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_65536_qa_1]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_65536_rag]=16 # Reserved pool: 43; frozen target: 16; context admission still applies.
  [long_65536_icl]=16 # Reserved pool: 48; frozen target: 16; context admission still applies.
  # 128K: evaluation only; observed pools below (256 published rows scanned).
  [long_131072_niah_single_1]=4 # Reserved pool: 78; frozen target: 4; context admission still applies.
  [long_131072_niah_single_2]=4 # Reserved pool: 78; frozen target: 4; context admission still applies.
  [long_131072_niah_single_3]=4 # Reserved pool: 78; frozen target: 4; context admission still applies.
  [long_131072_niah_multikey_1]=4 # Reserved pool: 78; frozen target: 4; context admission still applies.
  [long_131072_niah_multivalue]=4 # Reserved pool: 78; frozen target: 4; context admission still applies.
  [long_131072_niah_multiquery]=4 # Reserved pool: 78; frozen target: 4; context admission still applies.
  [long_131072_vt]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_131072_cwe]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_131072_fwe]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_131072_qa_1]=4 # Reserved pool: 40; frozen target: 4; context admission still applies.
  [long_131072_rag]=15 # Reserved pool: 43; revised test: 15 after one context-overflow exclusion.
  [long_131072_icl]=16 # Reserved pool: 48; frozen target: 16; context admission still applies.
)
sample_args=()
for task in "${!TASK_SAMPLES[@]}"; do sample_args+=("$task=${TASK_SAMPLES[$task]}"); done
export TASK_SAMPLE_COUNTS_JSON="${TASK_SAMPLE_COUNTS_JSON:-$("$PYTHON_BIN" -c 'import json,sys; print(json.dumps({k:int(v) for k,v in (s.split("=",1) for s in sys.argv[1:])}))' "${sample_args[@]}")}"
if [[ $# -eq 0 ]]; then set -- evaluate; fi
exec "$PYTHON_BIN" -m eval_stack.cli "$@"
