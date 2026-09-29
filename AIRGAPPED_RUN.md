# Offline run guide

Use this guide on a Linux machine with Docker, the evaluator image and prepared
dataset already available locally. The target and judge models must already be
hosted at endpoints reachable from this machine.

## 1. Put the image and data on the evaluation machine

On a connected machine, save the evaluator image:

```bash
docker pull suryavikram6/chimera-eval:0.1.1
docker save -o chimera-eval-image.tar suryavikram6/chimera-eval:0.1.1
```

Download dataset v5 (`quality-v5-clean`, public) at revision
`9f204f733a762c3766407636e1fbb4f8aa41dc9e`. For a main evaluation the required
files are `manifest.json` and `splits/main_test.jsonl` (`main_test` is unchanged since v3). From
this repository folder, with the Hugging Face CLI installed:

```bash
hf download surya-vikram/chimera-eval-data manifest.json splits/main_test.jsonl \
  --repo-type dataset \
  --revision 9f204f733a762c3766407636e1fbb4f8aa41dc9e \
  --local-dir ./prepared-data
tar -czf chimera-eval-data.tar.gz -C prepared-data manifest.json splits
```

Name the files as arguments, as above. With `--include manifest.json splits/main_test.jsonl`
the CLI treats the second path as a filename, ignores `--include`, and skips
`manifest.json`. Check that both files are present before transferring.

Transfer the image archive, prepared data, and this repository to the evaluation
machine. Load the image and unpack the data into `prepared-data/` inside the repository, where
`run_eval.sh` looks by default (`DATA_PATH` points elsewhere if you prefer):

```bash
docker load -i chimera-eval-image.tar
mkdir -p prepared-data
tar -xzf chimera-eval-data.tar.gz -C prepared-data
```

The data directory should contain:

```text
prepared-data/manifest.json
prepared-data/splits/main_test.jsonl
```

## 2. Configure and run

Edit the configuration block near the top of [`run_eval.sh`](run_eval.sh). Set:

- `DATA_PATH` and `OUTPUT_PATH` (defaults: `prepared-data/` and `outputs/` in this folder).
- `MODEL_URL` / `MODEL_NAME` and `JUDGE_URL` / `JUDGE_NAME`.
- Actual `MODEL_CONTEXT` and `JUDGE_CONTEXT` limits.
- `MODEL_KV_CACHE_NUM_TOKENS` and `JUDGE_KV_CACHE_NUM_TOKENS`, or leave them at
  zero to use request-count limits.
- `LIMIT_PER_TASK`, `N_SAMPLES`, `PASS_K`, and optional `TASKS`.

Both endpoints need `/v1/models`, `/v1/chat/completions`, and vLLM's `/tokenize`
route. The judge must support JSON-object output. Ensure prompt plus output budget
fits each model's served context. For a model other than Glimmer, change the
default chat-template kwargs to options that model supports.

Then run:

```bash
bash run_eval.sh
```

The checked-in configuration is a diagnostic: up to 50 prompts per task across all
domains, four responses per prompt, pass@1 and pass@4. For the full frozen inventory,
set `LIMIT_PER_TASK=0`. `PASS_K` takes any values up to `N_SAMPLES`; the report
has one column per value. Leave `TASKS` blank to include all
tasks. Select a smaller subset with comma-separated task IDs or reduce individual
task counts in `TASK_SAMPLE_COUNTS_JSON`.

The launcher checks for Docker, the local evaluator and worker images, and the
selected data files. It never pulls images or downloads data. You can preview
the Docker command without launching it:

```bash
bash run_eval.sh --dry-run
```

## 3. Follow the run and inspect results

The terminal and `OUTPUT_PATH/RUN_NAME.log` show one progress line every
`PROGRESS_SECONDS` (default 30) with counts of passed, failed, truncated and errored
samples, plus each distinct grading error the first time it occurs. At the end the
aggregate score and the domain and task score tables are printed. Results are saved
under `OUTPUT_PATH/RUN_NAME/`:

- `metrics.json`: task/domain scores, pass@k, truncation, errors and token totals.
  `aggregate_score_0_100` is always reported once any domain has a fully graded
  prompt: the equal-weight mean of the scored domains. `aggregate_complete` is true
  only when all ten domains are present with every selected prompt graded;
  `aggregate_missing_domains` and `aggregate_partial_domains` name the gaps.
- `audit/report.md` and `audit/report.json`: scores with a column for every pass@k,
  why samples scored zero (truncated, wrong answer, code syntax error, and so on),
  sample coverage, and prompt/response token distributions by domain and task.
- `audit/samples.jsonl`: one entry for every expected sample, including missing ones.
- `samples/`: one file per response. Each turn holds the `prompt` messages it
  answered next to the `response`, then the grade.
- `judge_attempts/`: auditable judge calls.
- `config.json`: the frozen settings and run fingerprint.
- `run_eval.sh` and `launch_config.env`: the launcher as edited and the values it
  actually used (environment overrides included), with `MODEL_PATH`, so the run can
  be reproduced. A resume under the same `RUN_NAME` must use the same `MODEL_PATH`.

## 4. Compare runs

```bash
python3 compare.py outputs            # writes outputs/comparison.csv
```

One row per finished run, highest aggregate first: `model_path` (from the run's saved
launch settings), aggregate score and pass@k, the ten domain scores side by side, pass@k
for each domain, then model and judge names, selection and sampling settings, and the
run folder. Scores
are percentages. The terminal shows the same ranking with the domain scores.
`model_path` drops the `/nvme_zone3/home/ekamai1/chimera/data/exports/` prefix, so
`.../exports/zoro2_v2_full` shows as `zoro2_v2_full`; other paths stay whole. Change
it with `--exports-root`. Cells a
run does not have (a domain it skipped, a k it did not request, quality pass@k) are
empty. It needs only Python 3's standard library.

Exit 0 means grading completed without unresolved errors; it does not certify judge
accuracy. Exit 2 means grading or evaluation is incomplete. A diagnostic subset
does not produce a full benchmark headline.

To resume, keep the same `RUN_NAME`, endpoints/weights, data, and settings. The
launcher saves a code snapshot for that run; use a new name to run changed code or
configuration. Completed responses remain saved between interruptions.

The evaluator needs the host Docker socket to start isolated, network-disabled
Python grading workers. Run it on a machine where you trust the evaluator code.
Hugging Face and Transformers downloads are disabled during evaluation; internal
model endpoint traffic still requires network access.

For scoring details, truncation policy and context-length coverage, see
[RUNBOOK.md](RUNBOOK.md). For data revision/hash details, see
[METADATA_REPAIR.md](METADATA_REPAIR.md).
