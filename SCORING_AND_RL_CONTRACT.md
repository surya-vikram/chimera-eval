# Scoring, truncation, and future RL integration contract

## Superseding decision — 2026-09-25

Evaluation now uses **fixed_budget_v1**: capped candidate trajectories receive zero,
fail pass@k, remain in the denominator, and are counted explicitly. Truncation alone
does not block the aggregate or cause exit 3. Judge parsing failures get up to three
attempts, with corrective schema instructions or larger output budgets where feasible;
exhausted errors invalidate the result rather than becoming zero. RUNBOOK.md is the
current operational policy. The older evaluation text below is historical and superseded.
RL policies remain under discussion and are not implemented; no RL work is required
to complete this evaluation release.

Status: evaluation safeguards implemented; Slime integration and training validation
are future work. This document is a required checklist for that integration, not a
claim that training is protected already. No RL training is part of the present task.

## Separate three outcomes

- A completed, incorrect answer is a valid task failure (normally reward/score 0).
- A candidate cut off by its output/context budget is **truncated**, not evidence
  that its eventual answer would be incorrect or correct.
- A failed or malformed judge response is a **grading error**, not a task failure.

Preserve original requests, responses, finish reasons, budgets, token usage, seeds,
references, and judge attempts. Never turn an infrastructure failure into reward 0.
Reasoning tokens can consume the completion allowance even when the final answer
is empty. No finite budget guarantees termination, and a configuration value cannot
extend a model's supported context length.

## Evaluation policy (implemented)

Check tokenized prompt length plus the requested output allowance before generation.
Never silently shorten prompts or shrink output budgets to fit. Long-context tasks
must reserve output space and disclose actual prompt lengths.

Superseded (2026-09-29): `aggregate_score_0_100` is always reported, as the mean of the
domains that have fully graded prompts; `aggregate_complete` says whether it covers all
ten domains with every selected prompt graded. Grading errors are never scored as zero.
Historical text: any candidate `finish_reason=length` invalidated the primary aggregate:
`comparison_budget_valid=false`, `valid_full_benchmark=false`, and
`aggregate_score_0_100=null`. Saved task/domain scores and pass@k are diagnostic only.
The CLI prints `TRUNCATION DETECTED` and exits 3 after saving reports. Grading or
incomplete-result errors exit 2. These exit checks apply to newly started processes;
older already-running processes retain their originally loaded implementation.

Do not drop capped answers and report a mean over successful completions. Do not
regenerate until an answer passes, select the best retry, or give only one model
extra attempts. Pilot budgets, then freeze a common per-task allowance and the
comparison protocol. If it is inadequate, rerun the entire affected task for both
models with a larger common allowance in a new versioned run. Keep pass@2/sample
counts fixed. Different tokenizers mean equal token caps are not equal compute;
report token counts and latency as well as reasoning/sampling settings.

A separate **fixed-budget** benchmark may deliberately count unfinished answers
as failures. That is a different objective: performance under a declared resource
budget, not a truncation-free capability comparison. It must be named and versioned
separately; no such alternative headline mode is implemented yet.

Judge output truncation uses bounded retries with a larger allowance where context
permits. Exhausted retries remain grading errors. Valid JSON alone does not establish
judgment accuracy; independent calibration remains necessary.

## Future Slime policy (required, NOT integrated)

Before computing group advantages or updating the actor:

1. Require the configured number of samples, unique sample indices, and one prompt
   identity for the group. Validate complete trajectories and all required judgments.
2. Check finish reasons for every turn. If any response is truncated, do not update
   on that group and do not replace its reward with zero.
3. Treat invalid judgments as errors. Retry judging the same saved response within
   bounded limits; never resample the candidate until a judgment/reward is favourable.
4. A truncation must trigger a visible pause in the initial implementation, with a
   persisted failed group and checkpoint/progress metadata. Investigate the budget,
   prompt length, repetition, and termination before resuming. Do not allow indefinite
   group replacement to silently bias training toward short/easy prompts.
5. Freeze and record reward protocol, judge model, reference version, and generation
   budgets. A deliberate termination/length penalty is a later objective change,
   not an implicit interpretation of missing answers.

`eval_stack/rl_rewards.py` currently checks basic group completeness and grade values.
It is NOT connected to Slime and does NOT yet enforce the truncation/pause policy
above. Do not use this helper alone as proof that the training path is safe.

Required integration tests: capped correct-looking partial answer; empty final answer
after reasoning; multi-turn cap hit; judge timeout/malformed/truncated response;
missing/duplicate sample; context overflow; and restart after a paused group. Assert
that no invalid group reaches advantage computation or an optimizer update. Log
per-domain cap rates, invalid judgments, retries, paused/rejected groups, length and
repetition distributions, and the dynamic-filter acceptance rate.

## Budget-validation evidence (2026-09-24)

Glimmer high-reasoning diagnostic, 13 Nemotron math prompts with two responses each:

- 16K allowance: 20/26 candidate responses reached the cap; 19 had no final answer.
- 32K pilot: all 26 candidate responses saved; 10 reached the cap and had empty final
  answers. At the time of this entry, judging was still finishing. Thus **32K is not
  validated as sufficient**, and these scores are not a clean capability comparison.

Raw runs remain outside the repo under the local workspace's `eval-results` directory:
`glimmer-high-calibration-pass2-v2` and `glimmer-high-math-pass2-32k-v3`.
Do not assume a larger allowance will fix persistent non-termination. A 64K pilot
would require additional GPU time and is not automatically authorized by this record.
