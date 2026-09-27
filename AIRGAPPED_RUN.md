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

Download the private dataset v3 with an account that has access. Follow the
dataset download instructions in [METADATA_REPAIR.md](METADATA_REPAIR.md), using
revision `0c4b5e43d163f333422fa0855e6f1fb708acbc7a`. For a main evaluation, the
required files are `manifest.json` and `splits/main_test.jsonl`.

For example, after installing the Hugging Face CLI and signing in:

```bash
hf download surya-vikram/chimera-eval-data \
  --repo-type dataset \
  --revision 0c4b5e43d163f333422fa0855e6f1fb708acbc7a \
  --include manifest.json splits/main_test.jsonl \
  --local-dir ./prepared-data
tar -czf chimera-eval-data.tar.gz -C prepared-data manifest.json splits
```

Transfer the image archive, prepared data, and this repository to the evaluation
machine. Load the image and unpack data there:

```bash
docker load -i chimera-eval-image.tar
mkdir -p /data/chimera-eval
tar -xzf chimera-eval-data.tar.gz -C /data/chimera-eval
```

The data directory should contain:

```text
/data/chimera-eval/manifest.json
/data/chimera-eval/splits/main_test.jsonl
```

## 2. Configure and run

Edit the configuration block near the top of [`run_eval.sh`](run_eval.sh). Set:

- `DATA_PATH` and `OUTPUT_PATH`.
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

The default is a quick diagnostic: one prompt per task across all domains, one
response per prompt. For the full frozen inventory, set `LIMIT_PER_TASK=0`. For
pass@2, set `N_SAMPLES=2` and `PASS_K="1,2"`. Leave `TASKS` blank to include all
tasks. Select a smaller subset with comma-separated task IDs or reduce individual
task counts in `TASK_SAMPLE_COUNTS_JSON`.

The launcher checks for Docker, the local evaluator and worker images, and the
selected data files. It never pulls images or downloads data. You can preview
the Docker command without launching it:

```bash
bash run_eval.sh --dry-run
```

## 3. Follow the run and inspect results

Progress streams to the terminal and `OUTPUT_PATH/RUN_NAME.log`. Results are saved
under `OUTPUT_PATH/RUN_NAME/`:

- `metrics.json`: task/domain scores, pass@k, truncation, errors and token totals.
- `audit/report.md` and `audit/report.json`: sample coverage and prompt/response
  token distributions by domain and task.
- `audit/samples.jsonl`: one entry for every expected sample, including missing ones.
- `samples/` and `judge_attempts/`: auditable answers and judge calls.
- `config.json`: the frozen settings and run fingerprint.

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
