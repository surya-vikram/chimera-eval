# Air-gapped evaluation

Run the evaluator on a Linux AMD64 Docker host with target and judge models
already hosted on reachable internal vLLM endpoints. No API keys are required.
The evaluator needs no GPU; model serving is separate. Downloading the private
dataset on the connected machine does require Hugging Face authentication;
offline evaluation does not.

## 1. Prepare the transfer on an internet-connected machine

```bash
docker pull suryavikram6/chimera-eval:0.1.1
docker save -o chimera-eval-image.tar suryavikram6/chimera-eval:0.1.1
sha256sum chimera-eval-image.tar > chimera-eval-image.tar.sha256
```

Release 0.1.1 includes the MCQA metadata safeguard and corrected preparation adapter.
Use `docker image inspect` to record the pulled image digest in your run manifest.
The evaluator image is approximately 300 MiB locally; archive size can differ.
The image includes evaluation dependencies and grader resources, not datasets,
model weights, or model-serving software. Do not build or install dependencies
on the disconnected host.

### Download the frozen dataset

Dataset: [surya-vikram/chimera-eval-data](https://huggingface.co/datasets/surya-vikram/chimera-eval-data).
It is private: log in with an account/token that has read access. Never put a
token in this repository, a transfer archive, or the evaluation script.

On the internet-connected machine, install the download CLI in a separate environment:

```bash
python3 -m venv hf-download-env
source hf-download-env/bin/activate
python -m pip install 'huggingface_hub==0.36.0'
hf auth login
```

For evaluation only (about 82 MiB plus the manifest):

```bash
hf download surya-vikram/chimera-eval-data \
  --repo-type dataset \
  --revision 0c4b5e43d163f333422fa0855e6f1fb708acbc7a \
  --include manifest.json splits/main_test.jsonl \
  --local-dir ./prepared-data
```

Or download all three splits (about 544 MiB) for evaluation and future RL use:

```bash
hf download surya-vikram/chimera-eval-data \
  --repo-type dataset \
  --revision 0c4b5e43d163f333422fa0855e6f1fb708acbc7a \
  --include README.md manifest.json 'splits/*.jsonl' \
  --local-dir ./prepared-data
```

Both commands pin the same published snapshot. Choose one; no preparation or
conversion step is needed. Package only the data/manifest, excluding CLI caches
and authentication state:

```bash
tar -czf chimera-eval-data.tar.gz -C prepared-data manifest.json splits
sha256sum chimera-eval-data.tar.gz > chimera-eval-data.tar.gz.sha256
```

Transfer both archives (image and data), both checksum files, and this repository
using your approved transfer mechanism. Do not transfer the CLI environment or
Hugging Face login cache. The data archive contains this ready-to-use layout:
For main evaluation the required data layout is:

```text
prepared-data/
  manifest.json
  splits/
    main_test.jsonl
```

Keep the matching manifest unchanged: the runner checks the split content hash.
The all-splits download additionally includes `rl_train.jsonl` and `rl_val.jsonl`
under `splits/`. The Hugging Face snapshot contains the three splits and manifest,
not the original source caches or reserved pools; reproducing preparation/audits
that require those artifacts needs a separate full preparation archive.
Do not run dataset download/preparation commands offline. Model servers need
their own preloaded images, weights, tokenizers, and chat templates.

### Current frozen split sizes

Measured on 2026-09-25, revised quality-v3-mcqa-extraction dataset. Sizes are uncompressed JSONL;
they exclude source caches, reserved pools, and manifests. MiB = 1,048,576 bytes.

| Split | Prompts | Bytes | MiB |
|---|---:|---:|---:|
| `rl_train` | 86,647 | 482,209,114 | 459.87 |
| `rl_val` | 128 | 1,202,230 | 1.15 |
| `main_test` | 3,982 | 85,973,993 | 81.99 |
| Total | 90,757 | 569,385,337 | 543.01 |

V3 repairs MCQA extraction and excludes 169 invalid training records. The 3,982-row
`main_test` JSONL is byte-identical to v2; no full evaluation rerun is required for
this repair. Use the new manifest with the new snapshot, not a mixed data directory.
See [METADATA_REPAIR.md](METADATA_REPAIR.md) for validation and old-revision details.

Splits are checked for exact/family overlap; this is not a guarantee against all
semantic near-duplicates. Long-context tasks are evaluation-only. Data stays
outside GitHub and outside the evaluator image.

## 2. Load on the disconnected host

```bash
sha256sum -c chimera-eval-image.tar.sha256
docker load -i chimera-eval-image.tar
docker image inspect suryavikram6/chimera-eval:0.1.1
```

Choose a data directory on the offline host, replacing the example absolute path:

```bash
sha256sum -c chimera-eval-data.tar.gz.sha256
mkdir -p /absolute/path/to/prepared-data
tar -xzf chimera-eval-data.tar.gz -C /absolute/path/to/prepared-data
test -f /absolute/path/to/prepared-data/manifest.json
test -f /absolute/path/to/prepared-data/splits/main_test.jsonl
```

Use a fresh destination to avoid mixing dataset revisions. In the Docker command
in section 4, mount this exact directory at `/data`; `DATA_DIR=/data` then points
the runner to its manifest and splits. Keep `SPLIT=main_test` in `run_eval.sh` for
the main evaluation. Neither Hugging Face access nor its token is needed offline.

Docker Engine must already be installed. Use the fully qualified image name for
both the evaluator and Python workers; otherwise Docker may try to pull a missing
local tag. This guide assumes standard rootful Linux Docker with its socket at
`/var/run/docker.sock`.

## 3. Configure one script

Edit the existing defaults in `run_eval.sh`, preserving `${NAME:-default}` so
container and diagnostic overrides still work:

- `MODEL_URL`, `MODEL_NAME`: target endpoint ending in `/v1` and served model name.
- `JUDGE_URL`, `JUDGE_NAME`: separate judge endpoint/model; defaults otherwise reuse target.
- `MODEL_CONTEXT`, `JUDGE_CONTEXT`: actual served total context limits.
- `MODEL_CONCURRENCY`, `JUDGE_CONCURRENCY`, `MAX_PENDING`, `REQUEST_TIMEOUT`:
  match serving capacity. Start conservatively and measure.
- `MODEL_TEMPERATURE`, `MODEL_TOP_P`, `MODEL_TOP_K`,
  `MODEL_REPETITION_PENALTY`: target sampling. Keep judge settings fixed.
- `MODEL_CHAT_TEMPLATE_KWARGS`, `JUDGE_CHAT_TEMPLATE_KWARGS`: only options
  supported by each model's template. Glimmer's high reasoning setting is
  `{"reasoning_strength":"high"}`.
- `N_SAMPLES=2`, `PASS_K=2`, `SEED=42`: current sampling/evaluation defaults.
- `TASK_SAMPLES`: per-task counts; comments document available frozen counts.
- `EVAL_CONTEXT_BUCKETS`: e.g. `16384,32768,65536,131072`; other domains remain
  enabled. Nominal input profiles still need output headroom in the served window.
- `TASK_MAX_TOKENS_JSON`: per-task response ceilings. The default math ceiling
  can reach 32K; an 8K model cannot support that. Prompt plus output must fit;
  inputs are not silently shortened.
- `RUN_NAME`: a new name for each changed model or sampling configuration.

Both servers must support chat completions and the vLLM `/tokenize` endpoint
for token-budget checks. No external internet is needed for these internal calls.

## 4. Run

From the transferred repository, replace the two host paths below. Create the
results directory first so outputs persist outside the disposable container.

```bash
mkdir -p /absolute/path/to/results
docker run --rm --pull=never --network host \
  -v "$PWD:/repo:ro" \
  -v /absolute/path/to/prepared-data:/data:ro \
  -v /absolute/path/to/results:/results \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e EVAL_ROOT=/repo -e DATA_DIR=/data -e OUTPUT_DIR=/results \
  -e PYTHON_BIN=python3 \
  -e CODE_IMAGE=suryavikram6/chimera-eval:0.1.1 \
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  --entrypoint bash suryavikram6/chimera-eval:0.1.1 /repo/run_eval.sh
```

The repo mount supplies the script and evaluator code. Python execution workers
use the image's bundled worker code, so keep image/code versions compatible.
For a pilot, add `-e LIMIT_PER_TASK=1 -e RUN_NAME=offline-pilot-01` before
`--entrypoint`; remove those overrides and choose a fresh run name for the main run.
Subset/pilot runs intentionally do not publish a full benchmark headline.

The Docker socket gives the trusted evaluator host Docker privileges. Generated
Python executes in restricted, network-disabled worker containers using the same
preloaded image. Do not mount that socket into target/judge model containers.

## 5. Read results and handle failures

Inspect the run's `metrics.json` and saved response/grading records under the
results directory. Domain scores contribute equally to the full aggregate;
pass@k applies only to binary tasks. Long-context profiles have separate scores.

- Candidate `finish_reason=length`: score zero under `fixed_budget_v1`, retained
  in the denominator and reported as truncation. Ceilings do not guarantee completion.
- Judge truncation/invalid JSON: bounded retries (default three attempts, with
  a larger retry budget). Exhausted failures remain grading errors, never fake zeros.
- Context overflow or grading failures: investigate; errors can produce exit 2
  and withhold the full headline score.
- Resume an unchanged run with the same name/configuration. Changed comparisons
  need new names; preserve the manifest and configuration alongside the outputs.

## Validation boundary

Local container checks exercised live generation/judging, isolated Python,
resume, overflow, deliberate truncation, and separate long-context profiles
through 128K. See [VALIDATION.md](VALIDATION.md). A full end-to-end run with all
external internet egress blocked has **not** been performed; the first offline
pilot is still required. This guide does not claim a completed full-model benchmark
or independently certified judge quality.
