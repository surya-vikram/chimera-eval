#!/usr/bin/env bash
# Offline Docker launcher. Edit this block, then: bash run_eval.sh
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"  # e.g. /nvme_zone3/home/ekamai1/chimera/eval/src/chimera-eval-main

# ======================= EDIT CONFIGURATION HERE =======================
IMAGE="${IMAGE:-suryavikram6/chimera-eval:0.1.1}"  # Must already be loaded.
CODE_IMAGE="${CODE_IMAGE:-$IMAGE}"                # Local Python grading image.
DATA_PATH="${DATA_PATH:-$REPO_ROOT/prepared-data}" # manifest.json + splits/*.jsonl
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/outputs}"
RUN_NAME="${RUN_NAME:-quick-all-$(date +%Y%m%d-%H%M%S)}"
# Use the same explicit RUN_NAME to resume unchanged code/data/model/settings.
DOCKER_SOCKET="${DOCKER_SOCKET:-/var/run/docker.sock}"

declare -A CONFIG=(
  # Models are already hosted. Use reachable internal URLs ending in /v1.
  # MODEL_PATH: the weights served at MODEL_URL. Recorded with the run for compare.py; not loaded here.
  [MODEL_PATH]="/nvme_zone3/home/ekamai1/Gemma/gemma4-31b"
  [MODEL_URL]="http://127.0.0.1:8025/v1"
  [MODEL_NAME]="judge"
  [MODEL_CONTEXT]=32768
  [JUDGE_URL]="http://127.0.0.1:8025/v1"
  [JUDGE_NAME]="judge"
  [JUDGE_CONTEXT]=131072

  # Independent token budgets. On one server, divide its capacity between roles.
  [MODEL_KV_CACHE_NUM_TOKENS]=10000000
  [JUDGE_KV_CACHE_NUM_TOKENS]=10000000
  [MAX_PENDING]=1024

  # Diagnostic run: up to 50 prompts per task, 4 responses each, pass@1 and pass@4.
  # Full inventory: LIMIT_PER_TASK=0. PASS_K values must be <= N_SAMPLES.
  [SPLIT]="main_test"
  [TASKS]=""                          # Blank = all; e.g. "gsm8k,math500,humanevalplus"
  [LIMIT_PER_TASK]=50
  [TASK_SAMPLE_COUNTS_JSON]='{}'        # All frozen counts; e.g. {"gsm8k":50}
  [N_SAMPLES]=4
  [PASS_K]="1,4"
  [SEED]=42
  [EVAL_CONTEXT_BUCKETS]=""            # Blank = all fitting; e.g. "4096,8192,16384"

  # Model sampling: the same as Chimera MixRL rollouts and training (slime mixrl/config.env
  # ROLLOUT_TEMPERATURE=1.0, ROLLOUT_TOP_P=0.95, top-k off). Chat template options: '{}' for Chimera.
  [MODEL_TEMPERATURE]=1.0
  [MODEL_TOP_P]=0.95
  [MODEL_TOP_K]=-1
  [MODEL_REPETITION_PENALTY]=1.0
  [MODEL_MIN_P]=0.0
  [MODEL_PRESENCE_PENALTY]=0.0
  [MODEL_FREQUENCY_PENALTY]=0.0
  [MODEL_CHAT_TEMPLATE_KWARGS]='{}'
  [MODEL_STOP_JSON]='[]'
  # Token ids that end the model's answer (never sent to the judge). Chimera: <EOS> 1 and
  # <end_of_turn> 3, which its generation_config does not list. Other models: '[]' (ids differ).
  [MODEL_STOP_TOKEN_IDS_JSON]='[1,3]'
  [JUDGE_TEMPERATURE]=1
  [JUDGE_TOP_P]=0.95
  [JUDGE_TOP_K]=20
  [JUDGE_REPETITION_PENALTY]=1.0
  [JUDGE_MIN_P]=0.0
  [JUDGE_PRESENCE_PENALTY]=0.0
  [JUDGE_FREQUENCY_PENALTY]=0.0
  [JUDGE_CHAT_TEMPLATE_KWARGS]='{"reasoning_strength":"high"}'
  [JUDGE_STOP_JSON]='[]'

  # Response ceilings: global > task > domain > frozen row. Never truncate inputs.
  [MAX_NEW_TOKENS]=0
  [TASK_MAX_TOKENS_JSON]='{"gsm8k":1024,"math500":2048,"arc":1024,"mmlu_pro":1024,"triviaqa":1024,"hotpot":1024,"biggen":2048,"humanevalplus":2048}'
  [DOMAIN_MAX_TOKENS_JSON]='{}'
  [JUDGE_MAX_TOKENS]=8192
  [JUDGE_MAX_RETRY_TOKENS]=16384
  [JUDGE_ATTEMPTS]=3
  [REQUEST_TIMEOUT]=1800
  [REQUEST_RETRIES]=2
  [CODE_TIMEOUT]=15
  [CODE_CONCURRENCY]=2
  [PROGRESS_SECONDS]=30               # One status line per interval, not one per sample.

  # Legacy limits, used only when the corresponding token budget is zero.
  [MODEL_CONCURRENCY]=16
  [JUDGE_CONCURRENCY]=8
  [SHARED_ENDPOINT_CONCURRENCY]=24
  [LONG_CONTEXT_CONCURRENCY]=8
)
# ===================== END EDITABLE CONFIGURATION =====================
# Explicit environment overrides are also accepted for every CONFIG field.
for key in "${!CONFIG[@]}"; do
  if [[ -v "$key" ]]; then CONFIG[$key]="${!key}"; fi
done

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
DRY_RUN=0
case "${1:-}" in
  '') ;;
  --dry-run) DRY_RUN=1 ;;
  *) die 'Usage: bash run_eval.sh [--dry-run]' ;;
esac
[[ $# -le 1 ]] || die 'Usage: bash run_eval.sh [--dry-run]'
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die 'RUN_NAME must contain only letters, digits, dots, underscores, or hyphens, starting with a letter/digit.'
[[ "${CONFIG[SPLIT]}" =~ ^(main_test|rl_val|rl_train)$ ]] || die 'SPLIT must be main_test, rl_val, or rl_train.'
for role in MODEL JUDGE; do
  key="${role}_URL"
  [[ "${CONFIG[$key]}" =~ ^https?://.+/v1/?$ ]] || die "$key must be an HTTP(S) base URL ending in /v1."
done
command -v docker >/dev/null || die 'Docker is not installed. Install it before transferring this offline package.'
docker info >/dev/null 2>&1 || die 'Cannot access the Docker daemon. Check that Docker is running and your account has access.'
[[ -S "$DOCKER_SOCKET" ]] || die "Docker socket missing: $DOCKER_SOCKET"
for local_image in "$IMAGE" "$CODE_IMAGE"; do
  docker image inspect "$local_image" >/dev/null 2>&1 || die "Required local image is missing: $local_image. Load the transferred image with docker load; this script never pulls images."
done
[[ -f "$REPO_ROOT/eval_entrypoint.sh" && -d "$REPO_ROOT/eval_stack" ]] || die 'Keep run_eval.sh in the chimera-eval repository.'
[[ -f "$DATA_PATH/manifest.json" ]] || die "Prepared manifest missing: $DATA_PATH/manifest.json. Transfer the matching prepared dataset first."
[[ -f "$DATA_PATH/splits/${CONFIG[SPLIT]}.jsonl" ]] || die "Prepared split missing: $DATA_PATH/splits/${CONFIG[SPLIT]}.jsonl"
DATA_PATH="$(cd -- "$DATA_PATH" && pwd -P)"
OUTPUT_PATH="$(realpath -m -- "$OUTPUT_PATH")"
LAUNCH_PATH="$OUTPUT_PATH/.launchers/$RUN_NAME"
CODE_PATH="$LAUNCH_PATH/code"
CONTAINER_NAME="chimera-eval-$RUN_NAME-$$"
LOG_FILE="$OUTPUT_PATH/$RUN_NAME.log"
for mount_path in "$DATA_PATH" "$OUTPUT_PATH" "$CODE_PATH" "$DOCKER_SOCKET"; do
  [[ "$mount_path" != *:* ]] || die 'Mount paths containing colons are unsupported.'
done

args=(run --rm --pull=never --init --network host --workdir /repo
  --name "$CONTAINER_NAME"
  -v "$CODE_PATH:/repo:ro" -v "$DATA_PATH:/data:ro" -v "$OUTPUT_PATH:/results"
  -v "$DOCKER_SOCKET:/var/run/docker.sock"
  --ulimit nofile=65536:65536
  -e EVAL_ROOT=/repo -e DATA_DIR=/data -e OUTPUT_DIR=/results -e PYTHON_BIN=python3
  -e "RUN_NAME=$RUN_NAME" -e "CODE_IMAGE=$CODE_IMAGE"
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)
for key in "${!CONFIG[@]}"; do args+=(-e "$key=${CONFIG[$key]}"); done
args+=(--entrypoint bash "$IMAGE" /repo/eval_entrypoint.sh)
if (( DRY_RUN )); then
  printf 'Offline launch preview (no files created and no container started):\n'
  printf 'docker '; printf '%q ' "${args[@]}"; printf '\n'
  exit 0
fi

mkdir -p -- "$OUTPUT_PATH" "$LAUNCH_PATH"
# Freeze the evaluator for this run; edits to the repo cannot alter an active run
# or an unchanged resume. To use changed code, choose a new RUN_NAME.
if [[ ! -d "$CODE_PATH" ]]; then
  STAGING_PATH="$(mktemp -d "$LAUNCH_PATH/code.XXXXXX")"
  cp -a -- "$REPO_ROOT/eval_stack" "$REPO_ROOT/eval_entrypoint.sh" "$STAGING_PATH/"
  if [[ -f "$REPO_ROOT/source_revisions.json" ]]; then
    cp -- "$REPO_ROOT/source_revisions.json" "$STAGING_PATH/"
  fi
  mv -T -- "$STAGING_PATH" "$CODE_PATH"
fi
[[ -f "$CODE_PATH/eval_entrypoint.sh" && -d "$CODE_PATH/eval_stack" ]] || die "Incomplete code snapshot: $CODE_PATH. Choose a new RUN_NAME."

# Save how the run was launched next to its results: this script as edited, and the values
# actually used (environment overrides included). Written once; a resume must match it.
RUN_PATH="$OUTPUT_PATH/$RUN_NAME"
mkdir -p -- "$RUN_PATH"
quote() { printf "'%s'" "${1//\'/\'\\\'\'}"; }
launch_settings() {
  for key in IMAGE CODE_IMAGE DATA_PATH OUTPUT_PATH RUN_NAME; do printf '%s=%s\n' "$key" "$(quote "${!key}")"; done
  for key in $(printf '%s\n' "${!CONFIG[@]}" | sort); do printf '%s=%s\n' "$key" "$(quote "${CONFIG[$key]}")"; done
}
if [[ -f "$RUN_PATH/launch_config.env" ]]; then
  recorded="$(grep '^MODEL_PATH=' "$RUN_PATH/launch_config.env" || true)"
  [[ "$recorded" == "MODEL_PATH=$(quote "${CONFIG[MODEL_PATH]}")" ]] || die "$RUN_NAME was launched with $recorded; resuming with another MODEL_PATH would mix models. Choose a new RUN_NAME."
else
  cp -- "${BASH_SOURCE[0]}" "$RUN_PATH/run_eval.sh"
  { printf '# Settings used for run %s, launched %s\n' "$RUN_NAME" "$(date -Is)"; launch_settings; } > "$RUN_PATH/launch_config.env"
fi

CONTROL_PATH="$(mktemp -d "$LAUNCH_PATH/control.XXXXXX")"
CID_FILE="$CONTROL_PATH/container.id"
cleanup() {
  # Only stop a container whose ID this invocation actually created.
  if [[ -f "$CID_FILE" ]]; then
    cid="$(cat -- "$CID_FILE")"
    if [[ "$cid" =~ ^[a-f0-9]{64}$ ]]; then
      docker stop --time 10 "$cid" >/dev/null 2>&1 || true
    fi
    rm -f -- "$CID_FILE"
  fi
  rmdir -- "$CONTROL_PATH" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
args=("${args[@]:0:1}" --cidfile "$CID_FILE" "${args[@]:1}")
printf 'Run: %s\nResults: %s/%s\nLog: %s\n' "$RUN_NAME" "$OUTPUT_PATH" "$RUN_NAME" "$LOG_FILE" | tee -a "$LOG_FILE"
set +e
docker "${args[@]}" 2>&1 | tee -a "$LOG_FILE"
statuses=("${PIPESTATUS[@]}")
set -e
status="${statuses[0]}"
if (( status == 0 && statuses[1] != 0 )); then status="${statuses[1]}"; fi
printf 'Exit code: %s\nResults: %s/%s/\nAudit: %s/%s/audit/report.md\n' \
  "$status" "$OUTPUT_PATH" "$RUN_NAME" "$OUTPUT_PATH" "$RUN_NAME"
exit "$status"
