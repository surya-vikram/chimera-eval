# Chimera evaluation

Start with [QUICKSTART.md](QUICKSTART.md) for the prepared local data and Docker run commands.
For disconnected systems, use [AIRGAPPED_RUN.md](AIRGAPPED_RUN.md), including
image transfer, prepared split sizes, configuration, and the offline launch command.

Current image: `suryavikram6/chimera-eval:0.1.1`. Use dataset v3; see
[MCQA metadata repair](METADATA_REPAIR.md). Main-test prompts and scores are unaffected.

Endpoint-only evaluation and preparation of three family-disjoint partitions: `rl_train`,
`rl_val`, and `main_test`. No Slime training code lives here.

For a one-command offline Docker launch, edit the configuration block in
**[`run_eval.sh`](run_eval.sh)**, then run `bash run_eval.sh`. It checks local images/data,
starts the evaluator, streams and saves progress, and preserves results and a code
snapshot. It never pulls images or downloads data. Models must already be hosted.
Checked-in configuration: up to 50 prompts per task across all domains, 4 responses each
(pass@1, pass@4); set `LIMIT_PER_TASK=0` for the full inventory. `serve_gemma.sh` and
`serve_chimera.sh` host the judge and target on the air-gapped machine. See [AIRGAPPED_RUN.md](AIRGAPPED_RUN.md).

`eval_entrypoint.sh` is the lower-level entrypoint used inside Docker or for direct Python runs.

For token-budget admission instead of request-count tuning, set
`MODEL_KV_CACHE_NUM_TOKENS` and `JUDGE_KV_CACHE_NUM_TOKENS` in `run_eval.sh`
(0 keeps legacy limits for that role). The two budgets are independent, even when
target and judge use the same endpoint. See the token-budget section
in [RUNBOOK.md](RUNBOOK.md) for sizing and mixed-domain scheduling.

The orchestration uses Python's standard library. Native task graders and dataset preparation
use pinned dependencies in a separate CPU evaluation image. Models run in your existing vLLM servers.

The local 0.6B model is a plumbing fixture, not a qualified judge. The user accepted
the automatically screened 3,982-row revised test inventory and using Glimmer as a
judge for a different target model. This is not a certification of judge accuracy or
zero semantic contamination. Candidate cap hits score zero under `fixed_budget_v1`;
judge failures remain errors after bounded retries. No RL truncation policy is implemented here.
