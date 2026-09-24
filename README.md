# Chimera evaluation

Start with [QUICKSTART.md](QUICKSTART.md) for the prepared local data and Docker run commands.
For disconnected systems, use [AIRGAPPED_RUN.md](AIRGAPPED_RUN.md), including
image transfer, prepared split sizes, configuration, and the offline launch command.

Current image: `suryavikram6/chimera-eval:0.1.1`. Use dataset v3; see
[MCQA metadata repair](METADATA_REPAIR.md). Main-test prompts and scores are unaffected.

Endpoint-only evaluation and preparation of three family-disjoint partitions: `rl_train`,
`rl_val`, and `main_test`. No Slime training code lives here.

Edit **`run_eval.sh`**. It is the single configuration entrypoint: model/judge endpoints,
concurrency, sampling (including repetition penalty), context budgets, N/pass@k, and
one sample-count field per main-evaluation task. See [RUNBOOK.md](RUNBOOK.md).

The orchestration uses Python's standard library. Native task graders and dataset preparation
use pinned dependencies in a separate CPU evaluation image. Models run in your existing vLLM servers.

The local 0.6B model is a plumbing fixture, not a qualified judge. The user accepted
the automatically screened 3,982-row revised test inventory and using Glimmer as a
judge for a different target model. This is not a certification of judge accuracy or
zero semantic contamination. Candidate cap hits score zero under `fixed_budget_v1`;
judge failures remain errors after bounded retries. No RL truncation policy is implemented here.
