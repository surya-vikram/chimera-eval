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

Edit the configuration block in `run_eval.sh`: set `DATA_PATH`, `OUTPUT_PATH`,
`RUN_NAME`, target/judge endpoint URLs and served model IDs, actual context limits,
request or token budgets, sample counts, and tasks. Then launch:

```bash
bash run_eval.sh
```

The default evaluates one prompt per task across all domains. To run the frozen
inventory, set `LIMIT_PER_TASK=0`; for two responses per prompt set `N_SAMPLES=2`
and `PASS_K="1,2"`. Leave `TASK_SAMPLE_COUNTS_JSON='{}'` to keep the frozen per-task
counts (3,982 prompts total before context selection), or set a smaller count such
as `{"gsm8k":50,"math500":30}`. Actual counts are in `DATA_PATH/manifest.json`.

Use `bash run_eval.sh --dry-run` to check local prerequisites and preview Docker's
command without launching a container. The offline launcher fails if either image,
the selected data manifest, or the split is missing. It does not download or pull.
For exact options, see [AIRGAPPED_RUN.md](AIRGAPPED_RUN.md).

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
Set `RUN_NAME=my-model-pilot-01` in the config block and keep `LIMIT_PER_TASK=1` for a pilot.
```

Then set `LIMIT_PER_TASK=0` and choose a new `RUN_NAME` for the full inventory.

## On another Linux host: one CPU evaluator image

Copy this repo and the **entire prepared data directory** to that host. The evaluation
container does not host the models and needs no GPU. Docker is needed for isolated
Python execution. Use `run_eval.sh` for normal runs. The following lower-level command invokes the Python entrypoint directly:

```bash
docker build -t suryavikram6/chimera-eval:0.1.1 .
docker run --rm --network host \
  -v "$PWD:/repo:ro" \
  -v /absolute/path/to/prepared-data:/data:ro \
  -v /absolute/path/to/results:/results \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e EVAL_ROOT=/repo -e DATA_DIR=/data -e OUTPUT_DIR=/results \
  -e PYTHON_BIN=python3 \
  --entrypoint bash suryavikram6/chimera-eval:0.1.1 /repo/eval_entrypoint.sh
```

Use environment-default assignments in the script so these mounted paths override
workstation defaults. The Docker socket gives the trusted evaluator host-level Docker
control; run on an appropriate dedicated host. Generated-code containers do not get
the socket, network, model credentials, or host mounts. See RUNBOOK.md for details.
Both model endpoints must provide `/v1/chat/completions` and vLLM's `/tokenize`.

## Read the result

Look under `OUTPUT_DIR/RUN_NAME/`:

- `metrics.json`: task/domain scores, binary pass@k, context-profile scores, errors,
  and truncation counts. The aggregate is always reported; `aggregate_complete` is
  false when domains are missing or have ungraded prompts.
- `reliability.json`: judge-attempt statistics and diagnostic score/length/repetition checks.
- `samples/`: saved generated answers with the prompt each turn answered, and grades.
- `judge_attempts/`: auditable judge requests, outputs, and retries.
- `config.json`: exact configuration and data/code fingerprint.
- `audit/report.md`: scores with every pass@k, zero-score reasons, coverage, and prompt/response token distributions by domain and task.
- `audit/report.json`: percentiles, histograms, judge usage, per-turn/trajectory statistics, and recomputed scores.
- `audit/samples.jsonl`: one audit entry for every expected sample, including missing/pending entries.

To audit or refresh an existing run while it is running (no endpoint calls):

```bash
python3 -m eval_stack.run_audit \
  --run-dir /absolute/path/to/results/my-run \
  --data-dir /absolute/path/to/prepared-data
```

The audit verifies the split hash and reproduces the saved run selection/fingerprint.
It scans every expected sample, never only a random subset. A live scan is explicitly
marked as a running snapshot and can see records at different moments. All-domain
frozen prompt distributions use preparation-tokenizer counts; actual prompt and
response distributions use server-reported usage for generated turns. Missing usage
is counted separately, never treated as zero. Multi-turn trajectory prompt totals
include repeated conversation history. Response counts follow server accounting,
including reasoning when the server includes it in completion usage.

Both prompt and response reports include count, missing count, total, min/max,
mean, standard deviation, p50/p90/p95/p99, and histogram buckets. Task reports
separate incorrect binary answers, continuous quality scores, grading errors,
truncated responses, missing samples, and incomplete prompts. There is no universal
binary pass/fail threshold for an entire task. `with_no_passes_when_complete`
identifies completed binary tasks with zero passing samples. Truncation is counted
both per turn and per sample/trajectory. The audit also reports judge retries,
judge truncation, token usage, latency, and any sample-integrity/policy violations.
Final saved headline/error/truncation counts are checked against recomputed metrics
after the run lock is released. Pass@1 accompanies requested pass@k for binary tasks.

The scan does not rerun native graders, execute candidate code, or independently
validate judge decisions. A clean audit establishes coverage/accounting consistency,
not semantic correctness of every grade or independence of a self-judge.

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
