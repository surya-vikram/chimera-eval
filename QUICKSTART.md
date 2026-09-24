# Run your own evaluation

Focused local validation is complete: all-domain live checks, resume, deliberately
capped outputs, Python execution, and separate 16K/32K/64K/128K input-profile checks.
See VALIDATION.md for exact counts and limitations. A full capability benchmark is
not required to repeat those checks and has not been claimed.

Status: runnable fixed-budget evaluation tooling. The user accepted automated data
screening and approximate judge quality; judge accuracy is not independently certified.
No full revised strong-model benchmark has completed. This is evaluation
only, not RL training. Models must already be hosted; no API keys are needed.

## On this workstation

```bash
cd /home/surya/workspace/repos/chimera-eval
```

For your first run, paste these settings into your shell and replace the endpoint/model
placeholders. For subsequent runs, put these values into the existing defaults in
`run_eval.sh` so that one script is your entrypoint:

```bash
export PYTHON_BIN="/home/surya/workspace/eval-env/bin/python"
export DATA_DIR="/home/surya/workspace/eval-data-quality-v2"
export OUTPUT_DIR="/home/surya/workspace/eval-results"
export RUN_NAME="my-model-main-pass2-01"
export MODEL_URL="http://YOUR_TARGET_HOST:8000/v1"
export MODEL_NAME="YOUR_SERVED_TARGET_NAME"
export JUDGE_URL="http://YOUR_JUDGE_HOST:8000/v1"
export JUDGE_NAME="YOUR_SERVED_JUDGE_NAME"
export MODEL_CONTEXT=131072  # Replace with actual supported/served limit.
export JUDGE_CONTEXT=131072  # Replace with actual supported/served limit.
export MODEL_CONCURRENCY=16
export JUDGE_CONCURRENCY=8
export SHARED_ENDPOINT_CONCURRENCY=24
export MAX_PENDING=48
export REQUEST_TIMEOUT=1800
export N_SAMPLES=2
export PASS_K=2
export SEED=42
export SPLIT=main_test
export TASK_SAMPLE_COUNTS_JSON='{}'
```

The script honours these environment settings. The last setting selects all rows of each selected task in the revised frozen split
(3,982 total before context/task selection), rather than the old 3,999-row allocation.
To reduce tasks individually, replace `{}` with e.g. `{"gsm8k":50,"math500":30}`:
unspecified tasks keep their frozen counts. Alternatively remove that override and
edit the per-task `TASK_SAMPLES` block, but reduce counts to the actual revised
inventory first (BiGGen 390, StructEval 194, and one quarantined long-context row).
Actual full task counts are in `DATA_DIR/manifest.json` under `task_counts.main_test`.

Sampling fields: `MODEL_TEMPERATURE`, `MODEL_TOP_P`, `MODEL_TOP_K`, and
`MODEL_REPETITION_PENALTY`. Keep judge sampling fixed across model comparisons.
`MODEL_CHAT_TEMPLATE_KWARGS` and `JUDGE_CHAT_TEMPLATE_KWARGS` are model-specific;
Glimmer uses `{"reasoning_strength":"high"}`, which is not a universal vLLM option.

Output-budget precedence: `MAX_NEW_TOKENS` > `TASK_MAX_TOKENS_JSON` >
`DOMAIN_MAX_TOKENS_JSON` > frozen row budget. Example task override:
`export TASK_MAX_TOKENS_JSON='{"gsm8k":8192,"math500":32768}'`.
These are configurable ceilings, not validated guarantees. Hard math truncated even
at 32K in the Glimmer pilot. Prompt plus output must fit the configured model context;
an 8K model cannot run this 32K setting. The runner rejects overflow, not truncates inputs.

Select long-context profiles with `EVAL_CONTEXT_BUCKETS=16384,32768,65536,131072`.
Other domains remain enabled. Long-profile labels describe nominal input profiles;
the server still needs output headroom. Not all published 128K inputs fit a 128K
total window. Frozen preparation changes and actual token counts are recorded.

First run a small diagnostic:

```bash
RUN_NAME=my-model-pilot-01 LIMIT_PER_TASK=1 bash run_eval.sh
```

Then evaluate the selected full inventory:

```bash
bash run_eval.sh
```

If you replace an environment-default assignment with a literal assignment in the
script, external overrides will no longer override that field. Preserve the existing
`${NAME:-default}` pattern when you want command-line overrides for pilot runs.

## On another Linux host: one CPU evaluator image

Copy this repo and the **entire prepared data directory** to that host. The evaluation
container does not host the models and needs no GPU. Docker is needed for isolated
Python execution. Build the image, then mount the repo so script edits take effect:

```bash
docker build -t chimera-eval:0.1.0 .
docker run --rm --network host \
  -v "$PWD:/repo:ro" \
  -v /absolute/path/to/prepared-data:/data:ro \
  -v /absolute/path/to/results:/results \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e EVAL_ROOT=/repo -e DATA_DIR=/data -e OUTPUT_DIR=/results \
  -e PYTHON_BIN=python3 \
  --entrypoint bash chimera-eval:0.1.0 /repo/run_eval.sh
```

Use environment-default assignments in the script so these mounted paths override
workstation defaults. The Docker socket gives the trusted evaluator host-level Docker
control; run on an appropriate dedicated host. Generated-code containers do not get
the socket, network, model credentials, or host mounts. See RUNBOOK.md for details.
Both model endpoints must provide `/v1/chat/completions` and vLLM's `/tokenize`.

## Read the result

Look under `OUTPUT_DIR/RUN_NAME/`:

- `metrics.json`: task/domain scores, binary pass@2, context-profile scores, errors,
  and truncation counts. The primary aggregate is withheld when validity gates fail.
- `reliability.json`: judge-attempt statistics and diagnostic score/length/repetition checks.
- `samples/`: saved generated answers and grades.
- `judge_attempts/`: auditable judge requests, outputs, and retries.
- `config.json`: exact configuration and data/code fingerprint.

Exit 0 means execution completed without grading incompleteness;
it does NOT mean judge accuracy is certified. Exit 2 indicates grading/incomplete-result
errors. Candidate truncation prints a warning and scores zero; it does not fail the run.
The current revised inventory was accepted by the user on September 25. The headline
is available only for a complete configured main-test run with all ten domains and
no unresolved grading errors. Subsets remain diagnostic. The scoring policy is
`fixed_budget_v1`; comparisons must use the same task caps.

Resume with the same run name and unchanged configuration/code/data. For another
model, sampling setting, or budget, use a new name. Never delete saved outputs to
make a mismatched run appear resumable.
