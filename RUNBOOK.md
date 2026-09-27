# Running Chimera evaluation

See [Scoring and future RL contract](SCORING_AND_RL_CONTRACT.md) for the required
truncation/error handling policies and the checklist for later Slime integration.

## Truncation and fair model comparisons

Pilot the output budget before freezing a comparison. Reasoning and final-answer
tokens may share the completion allowance; a large final-answer allowance alone
does not guarantee completion. Configure `DOMAIN_MAX_TOKENS_JSON` (for example
`{"math":32768}`) or `MAX_NEW_TOKENS` and check actual prompt token counts against
each hosted model's context limit. Increasing the configured context does not
extend the model's supported context. Long-input tasks need explicit output reserve.

The accepted policy is `fixed_budget_v1`: candidate `finish_reason=length` scores
zero and fails pass@k, even if the partial text contains a correct-looking answer.
It prints `TRUNCATION DETECTED`, records counts and `truncation_free=false`, but
does not invalidate an otherwise complete fixed-budget aggregate. Exit status 2
denotes grading/incomplete-result errors; judge failures are never replaced by zero.
For multi-turn evaluation, a capped turn fails that trajectory and stops subsequent
turn generation. Completed sibling samples for the same prompt remain separately scored.
Do not remove capped samples,
score only completed answers, or resample until an answer is correct: those approaches
bias comparison. If changing the budget, rerun all samples of the affected
task for every compared model in a new versioned run. Freeze prompts, sample count,
seed and scoring protocol; record reasoning/sampling settings for each model.

Start a math budget pilot at 32K and consider 64K only if context capacity permits
and cap hits persist. This is a proposed pilot schedule, not a validated guarantee.
If a model keeps reasoning/repeating to the maximum feasible budget, report that
termination failure in the fixed-budget score; do not call
it a truncation-free capability comparison. Judge truncation is separately retried
within configured limits and fails grading if no valid judgment is obtained.

## Interface and installation

Use Python 3.12 and Docker. For the offline launcher, keep the image already loaded locally. If rebuilding or publishing a new image, build it from this repository:

```bash
docker build -t suryavikram6/chimera-eval:0.1.1 .
```

The image contains preparation/native graders, not CUDA or vLLM. `CODE_IMAGE` names the
same image for isolated Python execution. The Docker socket gives the **trusted evaluator**
host-level Docker access; generated-code containers never receive this socket, host mounts,
network access, credentials, or GPU access. Use a dedicated evaluation host for hostile code;
ordinary Docker isolation is not a formal security boundary against kernel exploits.

Alternatively create a dedicated virtual environment, install `requirements.txt`, and
install `evalplus==0.3.1 --no-deps`, or install the exact `requirements.lock` with `--no-deps`. EvalPlus' unused model-generation/cloud-client dependencies
are deliberately omitted. Use `PYTHON_BIN=/path/to/venv/bin/python` when running the script.

## Prepare data

Mount a writable data directory and a local model tokenizer. Runtime scripts contain no
machine-specific paths. Example from the repository root:

```bash
docker run --rm --network host \
  -v "$PWD:/repo:ro" -v /datasets/chimera-eval:/data \
  -v /path/to/tokenizer:/tokenizer:ro \
  -e EVAL_ROOT=/repo -e DATA_DIR=/data -e PREP_TOKENIZER=/tokenizer \
  --entrypoint bash suryavikram6/chimera-eval:0.1.1 /repo/eval_entrypoint.sh prepare
```

Initial preparation downloads public source data, including the approximately 11 GB HELMET
archive; allow additional space for source caches and normalized records. Subsequent runs reuse
source locks/caches. Do not run two preparation processes into the same data directory.

Outputs:

- `splits/{rl_train,rl_val,main_test}.jsonl`: the only three production partitions.
- `pools/main_test.jsonl`: reserved source pool for future main-test subset changes; never RL data.
- `manifest.json`: actual counts, hashes, per-task shortfalls, overlap checks and review status.
- `source-locks/`: source revisions and available licensing metadata.
- `adapted/*.audit.json`: admission/exclusion counts and source errors.

The whole available main source pool is reserved before validation/training allocation.
Exact canonical families, initial conversation turns, and high-overlap lexical screening
prevent detected cross-split leakage. All three pairwise family-overlap checks must be zero.
This is not an embedding-based semantic or code-clone contamination certification. Source
licenses and semantic refusal/abstention exclusions still need human review before release.

Never pad shortfalls with repeated prompts. Training's initial 100K request currently has
genuine shortages in several eligible source routes. Read actual counts in the manifest;
the 128-prompt validation set is not expanded to hide those shortages.

## Configure and evaluate hosted endpoints

Edit the offline launcher configuration in `run_eval.sh` and run `bash run_eval.sh`. For direct in-container execution, use `eval_entrypoint.sh`. No API keys are required.

```bash
docker run --rm --network host \
  -v "$PWD:/repo:ro" -v /datasets/chimera-eval:/data:ro \
  -v /datasets/chimera-eval-results:/results \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e EVAL_ROOT=/repo -e DATA_DIR=/data -e OUTPUT_DIR=/results \
  --entrypoint bash suryavikram6/chimera-eval:0.1.1 /repo/eval_entrypoint.sh
```

Both model servers need `/v1/models`, `/v1/chat/completions`, and vLLM's `/tokenize` route.
The judge must support JSON-object constrained output. Set `MODEL_NAME`/`JUDGE_NAME` to
served model IDs, not an assumed checkpoint directory. Pin hosted weight revisions separately
when comparing runs; reusing a model ID after swapping weights is not a safe resume.

Main controls:

| Controls | Meaning |
|---|---|
| `MODEL_URL`, `MODEL_NAME`, `MODEL_CONCURRENCY` | Candidate server |
| `JUDGE_URL`, `JUDGE_NAME`, `JUDGE_CONCURRENCY` | Independently replaceable judge |
| `SHARED_ENDPOINT_CONCURRENCY` | Combined request bound when sharing one endpoint |
| `MODEL_CONTEXT`, `JUDGE_CONTEXT` | Verified total context capacities, not context-extension training |
| Role-prefixed temperature/top-p/top-k/repetition/min-p/presence/frequency | Independent sampling controls |
| Role-prefixed chat-template kwargs and stop JSON | Template/thinking and termination settings |
| `SEED` | Frozen data selection and per-prompt/sample/turn generation seeds |
| `N_SAMPLES`, `PASS_K` | Default: 2 responses and pass@2 only; increase both explicitly for a larger budget |
| `TASK_SAMPLES[task]` | Number of prompts, not number of generated answers |
| `TASKS`, `LIMIT_PER_TASK` | Explicit diagnostic subset; never a full benchmark |
| `MAX_NEW_TOKENS`, `DOMAIN_MAX_TOKENS_JSON` | Recorded overrides of per-record response ceilings |
| `LONG_CONTEXT_BUCKETS`, `EVAL_CONTEXT_BUCKETS` | Prepared and evaluated length buckets |
| `MAX_PENDING`, timeouts, retries, code limits | Bounded resource use |
| `LONG_CONTEXT_CONCURRENCY` | Separate generation limit for 16K+ cells; avoids flooding KV cache while regular concurrency stays high |
| `RUN_NAME`, `OUTPUT_DIR`, `SPLIT` | Output namespace and partition |

Sample-count comments distinguish source-pool size from the frozen allocation. On an existing
split, decreases use a deterministic prefix; increasing beyond its size fails. To enlarge the
frozen allocation, edit counts and prepare a new data directory/manifest. Keep the total main
test at or below 4,000. Do not compare changed subsets as if they were the same protocol.

`SEED=42` is the default. Fixed source locks, input order, code and seed reproduce selection.
Generated outputs can still vary across server versions, GPU kernels and scheduling despite
per-request seeds. Store model revisions/server versions with your experiment.

Responses are graded as they finish. No all-generation barrier is used. Individual generations
are saved before grading, so a failed judge request can be retried without resampling the model.
Rerun the **unchanged** command/name to resume. A changed configuration, dataset or code hash
requires a new run name. Exit code 2 means unresolved evaluation/infrastructure errors.

## Token-budget admission for mixed domains

Set two independent budgets in the configuration block in `run_eval.sh` or pass them to Docker:

```bash
export MODEL_KV_CACHE_NUM_TOKENS=262144
export JUDGE_KV_CACHE_NUM_TOKENS=131072
```

These are illustrative capacities, not measured recommendations for your GPU.
Zero preserves request-count scheduling for that role. A positive target budget
replaces `MODEL_CONCURRENCY` and `LONG_CONTEXT_CONCURRENCY`; a positive judge
budget replaces `JUDGE_CONCURRENCY`. `SHARED_ENDPOINT_CONCURRENCY` applies only
when both roles use legacy scheduling. `MAX_PENDING` remains a worker safety
ceiling and defaults to 128 when either token budget is enabled.
The old single `KV_CACHE_NUM_TOKENS` setting is rejected if nonzero; replace it
with the two role-specific settings.

For every target generation, conversation turn, and judge attempt:

1. Count the actual templated prompt with that endpoint's `/tokenize` route.
2. Reserve `prompt_tokens + max_tokens`, using that request's configured output
   ceiling. Reasoning tokens share this allowance. Domain output caps already
   distinguish short answers from long math/code responses.
3. Admit only when the sum of reservations fits the budget. Release on completion
   or failure, before requesting judgment or another conversation turn. HTTP
   retries retain the reservation; judge retries reserve their new output ceiling.
4. Round-robin the pending jobs across domains. Within each role, admit the
   oldest request that fits; after eight bypasses, stop backfilling past that
   waiter until it fits. This trades some utilization for progress on long inputs.

Target and judge always have separate reservations, including when they share a
server. With both budgets enabled on one server, the maximum combined reservation
is their **sum**; divide that server's available capacity between the two roles.
Unused target capacity is not borrowed by the judge or vice versa. With separate
servers, size each budget for its server. A role in legacy mode has no token bound;
use both budgets to bound combined reservations on a shared server. This process
cannot coordinate other evaluator processes or external clients. For example,
a 262144 target budget admits up to eight target requests of 32768 reserved tokens
or two of 131072, subject to the worker/server ceilings.

An individual request larger than the budget becomes an explicit evaluation error;
the evaluator never shrinks its prompt/output cap or silently exceeds the budget.
Queue waits use `REQUEST_TIMEOUT`, separately from each HTTP attempt's timeout.
Saved responses include `kv_admission` reservation/wait diagnostics; `metrics.json`
includes per-role endpoint, capacity, and peak reservations for the current invocation.

This is a conservative **client admission budget**, not a vLLM memory-allocation
flag or a GPU-memory guarantee. It reserves the full output ceiling because the
client is non-streaming. Prefix-cache sharing can reduce actual use, while block
rounding, speculative decoding, hybrid attention layouts, other traffic, and
requests still running after a client timeout can invalidate a literal mapping
to physical KV occupancy. Size it from the deployed server's reported KV capacity
with headroom, and monitor cache pressure/preemption and throughput. Keep vLLM's
server sequence limit as its own safety bound; do not infer KV capacity from GPU
VRAM alone. A capacity below the largest prompt-plus-output request cannot cover
the full suite. Changing scheduler configuration requires a new run name.

## Long context

Long context is **only** in main evaluation, never `rl_train` or `rl_val`. The 436-cell
budget is shared across 4K, 8K, 16K, 32K, 64K and 128K by default. It includes published
RULER instances, HELMET NQ retrieval, and BANKING77 ICL with published examples and
deterministic label remapping. No newly generated synthetic questions are used.

These are adapted HELMET profiles, not official leaderboard scores. Context buckets include
the response reserve. Actual reference token counts are saved; no runtime prompt truncation
or silent completion-budget shrinking is permitted. The raw published prefix scanned and
admission rules are recorded. Some published variants are repeated questions at different
evidence depths: only distinct query families count towards a per-length allocation.

`metrics.json` has `long_context_by_length` and `untested_length_buckets`. An 8K server
does not attempt 16K–128K and those cells are not scored zero. Missing required lengths
invalidate the full ten-domain aggregate, while completed domain/length results remain visible.
Separate scores are evidence of measured performance; a server's configured window is not
a measured effective-context guarantee. Larger-context live validation needs a capable server.

For example, set `EVAL_CONTEXT_BUCKETS=16384,32768` to evaluate only the 16K/32K
long-context cells alongside the regular domains. Set `MODEL_CONTEXT` to the actual
server capacity (at least 32768 for this example). Leave the bucket selector blank for
all prepared lengths fitting that capacity. Restrict `TASKS` as well if you want only
long-context tasks. Skipped lengths remain explicitly untested, not zero-scored.

## Scoring

The shared grader returns a normalized score, a binary pass where meaningful, and native
diagnostics. Evaluation averages prompt-level means, fixed within-domain task weights,
then ten equal domain weights (10% each). The capability index is not a probability of
correctness or an official combined benchmark. Quality stays continuous; no quality pass@k.

Binary pass@k is `1 - choose(N-correct,k)/choose(N,k)`, averaged across prompts. It measures
oracle availability, not an answer-selection algorithm. Task/domain pass@1 accompanies it.

| Task | Primary score | Diagnostics / important distinctions |
|---|---|---|
| Math | Native symbolic/numeric correctness | No fluency bonus |
| MCQA | Unambiguous final choice matches gold | Preserve actual option labels |
| Alias QA | Exact answer fast path, otherwise reference-equivalence judgment | EM and token F1 are diagnostics, **not verbosity penalties** |
| Grounded QA | Correct answer and context-supported material claims | Correct final phrase cannot rescue contradictory prose |
| Logic | Exact final answer, semantic correctness check when needed | Explanation length is irrelevant |
| Quality | Native per-row 1–5 rubric, normalized `(score-1)/4`, essential-check gate | Keep score anchors and references private; no pass@k |
| IFEval/IFBench/Multi-IF | All native required checks pass | Word counts matter only when explicitly requested |
| MultiChallenge | All published yes/no rubric checks satisfy their criteria | Invalid judge JSON is an error, never a valid NO |
| Structure | Parse/schema success AND content correctness | Format is part of this task; syntax alone is insufficient |
| Calendar | Events, durations, windows, constraints and non-overlap | Empty schedule cannot pass |
| Python | All admitted executable tests pass in sandbox | APPS profile is strict stdin/stdout, not the full official harness |
| RULER / ICL | Required targets / exact remapped class | Native target recall remains diagnostic |

Completion cap hits are reported separately; they do not automatically zero an otherwise
correct answer. Missing content or broken syntax still fails the relevant task check.
Infrastructure/judge parse/context errors are never silently converted to model failures.

HumanEval/32 is quarantined: the published canonical `find_zero` solution fails a plus-test
residual tolerance, and EvalPlus 0.3.1's special-oracle bookkeeping also rejects it. Other tests
are not removed or relaxed. HumanEval/56 and /61 are distinct bracket tasks: canonical family
hashes preserve operators/punctuation to avoid incorrectly merging them.

## Local validation

Unit and integration fixtures:

```bash
python -m unittest discover -s tests -v
```

Optional local GPU smoke server (production use does not require this):

```bash
python -m eval_stack.serve_smoke --model-dir /path/to/smoke-model
curl http://127.0.0.1:8010/v1/models
python -m eval_stack.smoke_data --data-dir /path/to/data --output /path/to/diagnostic-data
```

Use the same endpoint and `eval-smoke` for candidate/judge, both contexts 8192, and
`{"enable_thinking":false}` chat-template kwargs. For this diagnostic-only subset set
`TASK_SAMPLE_COUNTS_JSON='{}'`, `N_SAMPLES=4`, `PASS_K=1,4`, and a new run name.
The local test used Qwen3-0.6B with the pinned vLLM image in `serve_smoke.py`.
It completed a 48-sample/12-task run, then an 80-sample/20-task run from the evaluation
container covering all ten domains, including Python, live multi-turn, and 4K/8K context.
Rerunning the latter resumed with no new completion calls (only model-list preflight calls).
All 21 unit/integration/sandbox tests passed. All 163 admitted HumanEval+ references passed;
12 APPS references were smoke-tested, not the entire APPS training inventory.

```bash
python -m eval_stack.score_audit --url http://127.0.0.1:8010/v1 \
  --model eval-smoke --output /path/to/results/score-audit.json
python -m eval_stack.code_audit --input /path/to/data/adapted/humanevalplus.jsonl \
  --output /path/to/results/code-reference-audit.json
```

The tiny judge failed several correctness/contradiction cases. Replace it with a capable
endpoint and repeat calibration before trusting judged metrics. A successful HTTP run is
not judge validation. Full APPS runtime profiling, a semantic refusal/contamination review,
and large-context live runs remain release gates; do not call the current data audited production data.

Verify saved splits/pool after preparation:

```bash
python -m eval_stack.audit --data-dir /path/to/data
```

This checks hashes, duplicate IDs, known refusal/meta-prompt patterns, exclusion of long-context
training rows, and all pairwise/whole-pool family intersections. It cannot certify semantic absence.
