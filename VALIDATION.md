# Evaluation validation — 2026-09-25

This records software and focused live checks, not a full model benchmark or independent
judge-quality certification. See QUICKSTART.md to run evaluation.

## Release 0.1.1 / dataset v3 follow-up

The MCQA repair is documented in [METADATA_REPAIR.md](METADATA_REPAIR.md).
Current inventory is 86,647 rl_train, 128 rl_val, 3,982 main_test. The older
inventory below describes the preceding v2 release; main_test is unchanged.

- All **41 release tests passed inside the new image**, including Docker sandbox
  tests, with the evaluator container's network disabled. This is not a complete
  live endpoint evaluation with internet egress blocked.
- All 9,339 retained MCQA records passed correct/wrong-label and malformed-output
  checks; 4,652 records repaired; 169 invalid training rows quarantined.
- Five exact/family overlap checks remained zero. Main-test bytes/SHA-256 unchanged.
- Image built from a clean Git archive of `c166132`, excluding uncommitted MixRL
  service work. Image index digest:
  `sha256:af4dad578d9afc614f61139c134cf36166de7219369cecd33eb9db782e4296e7`.
- HF v3 revision: `0c4b5e43d163f333422fa0855e6f1fb708acbc7a`.
- No full model benchmark rerun or RL optimizer experiment is claimed for this fix.

## Accepted release scope

- Three frozen partitions: 86,816 rl_train, 128 rl_val, 3,982 main_test.
- The 17 flagged test rows are excluded; the 49,923-row original main pool remains
  reserved against both RL partitions. Five exact/family overlap checks are zero.
- The user accepted automated screening and approximate Glimmer judgment quality,
  with a different production target model. Semantic near-duplicate review, independent
  judge certification, and paired confidence intervals are not release requirements.
- No full revised benchmark rerun is required to validate software. Focused coverage
  does not provide a full capability score.
- No RL training or Slime integration was performed.

## Current scoring contract

Candidate truncation scores zero and fails pass@k under fixed_budget_v1. Every
scheduled sample stays in the denominator. Counts and warnings are explicit.
For multi-turn trajectories, a capped turn ends generation and fails the trajectory.
Completed sibling samples remain separately scored. Judge errors never become zero.

Judge attempts are bounded (default three). Malformed outputs trigger a corrective
schema instruction; truncated judgments can receive a larger budget within configured
limits. Bare JSON and one complete outer JSON fence are accepted. Types, required keys,
ranges, duplicate keys, and successful termination are validated. Exhausted errors
invalidate the corresponding result. Original responses and every attempt are saved.

## Local checks

Hardware: 8GB RTX PRO 1000 Blackwell Laptop GPU. vLLM 0.30.0 image digest is pinned
in the optional smoke server module and saved launch scripts.

| Check | Result |
|---|---|
| Unit, fake-endpoint integration, Docker sandbox tests | 35 passed |
| Script syntax and evaluator image build | Passed |
| Cached Qwen3-4B-FP8, 20 tasks across all ten domains, pass@2 | 40 valid trajectories / 44 turns; zero errors, caps, or empty answers |
| Scorer sanity through local 4B endpoint | 13/13 expected cases; limited sanity, not certification |
| Actual-server resume | Only model-discovery requests; no new generation/judging |
| Forced one-token output budget | Both samples scored zero, pass@2 zero, warning emitted |
| Deliberate context overflow | Rejected before generation; exit 2, not a fabricated zero score |
| Packaged Docker evaluator, four tasks including Python | 8 valid responses; zero errors/caps |
| Qwen3.5-2B 16K/32K/64K profiles | 24 valid responses across retrieval, QA, RAG, ICL |
| Qwen3.5-2B 128K profile after window fix | 8/8 valid responses, zero errors/caps/empty answers; actual prompts 114,389–136,196 tokens |
| Final Docker image, longest 128K-profile ICL case | 2/2 valid responses, zero errors/caps |

Diagnostic subsets are drawn deterministically from the frozen main_test partition.
They favour short prompts within a task to keep plumbing checks inexpensive and are
not representative capability evaluations or a fourth production split.

The 4B server uses FP8 weights/cache and eager execution. FP8 cache-scale warnings mean
its scores must not be treated as a quantization-quality certification. Startup initially
failed at 90% GPU allocation; 80% and 512-token prefill chunks succeeded. A legacy KV
scale CLI flag was unsupported; the saved successful launcher does not use it.
The original 0.6B server and failed startup containers were preserved, not deleted.

The hybrid Qwen3.5-2B model is revision
15852e8c16360a2fea060d615a32b45270f8a8fc. It runs text-only, BF16, eager, with a
147,456-token total serving window, locally on the same GPU.

## Long-context issue found and fixed

The preparation model's window was incorrectly treated as a runtime ceiling for v2
nominal input profiles. Different tokenizers changed the lengths; two 128K-profile
prompts were rejected despite fitting the larger served window.

The runner now uses the actual configured serving window for explicitly nominal v2
profiles; legacy total-window records retain their original ceiling. No frozen prompt
is shortened. Per-length reports include actual prompt-token minima and maxima, so a
nominal 128K label is not confused with exactly 131,072 tokens on every tokenizer.

The separate 4B 16K probe completed six responses; two ICL samples exceeded its real
16K serving capacity and were correctly rejected. Increasing a config alone cannot
extend a server/model context.

## Reproducible artifacts

Under the workspace eval-results directory:

- local4b-release-pass2-v4: all-domain live run and resume.
- local4b-forced-cap-v4 and local4b-context-overflow-v4: deliberate failure checks.
- container-release-local2b-v4: live packaged evaluator.
- local-long2b-profiles-pass2-v4: lower-profile successes and the original 128K rejection.
- local-long2b-128k-pass2-v5: corrected 128K rerun.
- container-long128k-local2b-v5: final packaged image with the window correction.
- local4b-validation-20260925/: launch/run scripts, logs and scorer sanity report.

Each evaluation saves config/code/data fingerprints, generations, metrics and judge
traces. Earlier results are preserved; never silently merge runs into a headline score.
The completed focused checks required no new remote GPU. A remote H100 is optional
for testing the actual production target/judge configuration, not necessary to repeat
these plumbing tests. That follow-up should reuse the small diagnostic subsets, not
automatically launch the complete 3,982-prompt benchmark.

## Earlier supporting audits

- All 163 admitted HumanEval+ reference solutions passed; HumanEval/32 was quarantined
  for a reference/oracle inconsistency. No tests or tolerances were weakened.
- APPS (RL-side data) passed 12 reference smoke cases; its full pool is not profiled.
- All 600 frozen main-test math references parsed with the native math verifier.
- Earlier Glimmer high-reasoning calibration scored 178 responses without grading
  errors; 20 candidate responses capped. Its 32K math pilot had 10/26 cap hits.
- The original 0.6B judge failed 5/13 sanity cases and is not a production judge.

No finite task cap guarantees termination. Current budgets define a resource-bounded
evaluation, not a promise of zero truncation. Real target/judge endpoints must still be
checked for model-specific configuration and serving compatibility.
