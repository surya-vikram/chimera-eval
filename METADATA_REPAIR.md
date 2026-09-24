# MCQA metadata repair — release 0.1.1 / dataset v3

Date: 2026-09-25. This fixes data preparation and verifier metadata, not model weights.

The source adapter incorrectly collapsed doubled backslashes in regex strings that
the JSON/Parquet reader had already decoded. In MCQA, this changed the literal
backslash in `\\boxed` into `\b` (word boundary), rejecting correct boxed answers
and potentially accepting malformed `oxed{...}` output. Preserve publisher regexes
verbatim. The pinned MCQA, OpenQA and science source patterns were inspected;
regression tests cover both MCQA formats and nine semantic-QA formats.

## Released data

HF repository: `surya-vikram/chimera-eval-data` (private).

- New v3 revision: `0c4b5e43d163f333422fa0855e6f1fb708acbc7a`.
- Previous v2: `acee68097ec2058c75cbdc06d9353405f84048ba`, retained in history.
- Repaired and retained: 4,648 training rows and four validation rows.
- Quarantined: 169 training rows with missing/invalid gold-option relationships,
  duplicate option labels, or fewer than two options. We do not guess their answers.
- Remaining counts: 86,647 `rl_train`, 128 `rl_val`, 3,982 `main_test`.
- Retained prompts, references, family identities and partition assignments did
  not change. Repaired rows have new IDs; the repair report maps old to new IDs.
- All five exact/family overlap checks are zero, including against the locally
  reserved main-test pool. This does not certify absence of semantic duplicates.
- All 9,339 retained MCQA rows passed correct-label and every wrong-label tests,
  malformed-output checks and truncation-zero checks. All 23,450 records with an
  extraction regex compile successfully; compilation alone does not validate their
  semantic references. No blanket semantic data-quality certification is claimed.

`main_test.jsonl` is byte-identical between v2 and v3. SHA-256:
`184bb0331bf55ee528422ad68113866272ceb15d7bdc7c8070162ae6971d8885`.
Existing main-test scores do not need a generation rerun for this change. Previous
MCQA training/validation scores using broken metadata are not comparable to v3.
New dataset/code fingerprints intentionally prevent silently resuming old runs
under changed grading semantics. Use a fresh run name for the new protocol.

The evaluator now raises a grading error on recognized broken MCQA metadata rather
than quietly assigning zero. The standalone scorer functions are unchanged for
valid main-test records. Unreleased MixRL service/adapter work is not part of this
evaluation release.

## Reproduce the repair

Use the evaluation environment or image, with the old prepared-data directory
mounted read-only and a fresh writable destination:

```bash
python3 -m eval_stack.repair_metadata \
  --source /data/v2 --destination /data/v3
```

The command verifies source hashes, preserves unchanged splits byte-for-byte,
audits MCQA labels/extraction, updates split counts/hashes, and writes
`metadata_repair_report.json` plus `quarantine.json`. If the reserved pool exists
locally, it is copied and cross-checked; it is not included in the HF download.
Never overwrite v2 in place or mix its manifest with the repaired files.

For normal users no repair command is needed: download the pinned v3 snapshot
using [AIRGAPPED_RUN.md](AIRGAPPED_RUN.md).
