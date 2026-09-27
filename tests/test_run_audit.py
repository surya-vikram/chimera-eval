import json
from pathlib import Path
import tempfile
import unittest

from eval_stack.common import digest, write_json, write_jsonl
from eval_stack.run_audit import audit_run, token_distribution, selected_rows


class RunAuditTests(unittest.TestCase):
    def fixture(self, root):
        rows = [{'id': f'id{i}', 'family_id': f'f{i}', 'task': task, 'domain': domain,
                 'messages': [{'role': 'user', 'content': 'q'}], 'verifier': 'exact',
                 'source': {'repo': 'fixture'}, 'binary': binary}
                for i, (task, domain, binary) in enumerate([('math', 'math', True),
                    ('code', 'python', True), ('quality', 'quality', False)])]
        rows[1]['turns'] = [{}, {}]
        config = {'MODEL_CONTEXT': 8192, 'EVAL_CONTEXT_BUCKETS': [], 'TASKS': '',
                  'LIMIT_PER_TASK': 0, 'TASK_SAMPLE_COUNTS': {}, 'SPLIT': 'main_test', 'N_SAMPLES': 2}
        write_jsonl(root/'data/splits/main_test.jsonl', rows)
        write_json(root/'run/config.json', {'config': config, 'selected_rows': 3, 'source_hash': 'source',
                   'fingerprint': digest([config, rows, 'source']),
                   'data_manifest': {'splits': {'main_test': {'hash': digest(rows)}}}})
        return rows

    def sample(self, root, row, sample, usage, grade=None, finish='stop', turns=1):
        r = {k: row[k] for k in ('id', 'task', 'domain')}
        r.update(sample=sample, turns=[{'response': {'usage': u, 'text': 'answer',
                 'finish_reason': finish, 'request': {'max_tokens': 100}, 'latency': .1}}
                 for u in (usage if isinstance(usage, list) else [usage]*turns)])
        if grade is not None:
            r['grade'] = grade
        write_json(root/'run/samples'/f'{row["id"]}.{sample}.json', r)

    def test_histogram_percentiles_and_missing_usage(self):
        s = token_distribution([0, 128, 256, 512, None, float('nan'), True])
        self.assertEqual(s['count'], 4)
        self.assertEqual(s['missing'], 3)
        self.assertEqual(s['total'], 896)
        self.assertEqual(s['p50'], 192)
        self.assertAlmostEqual(s['p95'], 473.6)
        self.assertEqual(sum(s['histogram'].values()), 4)
        self.assertEqual(s['histogram']['[128,256)'], 1)
        self.assertIsNone(token_distribution([])['mean'])

    def test_every_expected_sample_partial_errors_and_multiturn(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); rows = self.fixture(root)
            valid = {'status': 'valid', 'score': 1., 'passed': True}
            self.sample(root, rows[0], 0, {'prompt_tokens': 10, 'completion_tokens': 20}, valid)
            self.sample(root, rows[0], 1, {'prompt_tokens': 30, 'completion_tokens': 100},
                        {'status': 'valid', 'score': 0., 'passed': False}, finish='length')
            self.sample(root, rows[1], 0, [{'prompt_tokens': 50, 'completion_tokens': 5},
                        {'prompt_tokens': 60}], {'status': 'error', 'error': 'judge failed'})
            self.sample(root, rows[2], 0, {'prompt_tokens': 20, 'completion_tokens': 10})
            report = audit_run(root/'run', root/'data', running=True)
            o = report['overall']
            self.assertFalse(report['coverage_complete'])
            self.assertEqual((o['expected_samples'], o['valid_samples'], o['missing_samples']), (6, 2, 2))
            self.assertEqual(o['grading_error_samples'], 1)
            self.assertEqual(o['pending_saved_samples'], 1)
            self.assertEqual(o['binary_failed_samples'], 1)
            self.assertEqual(o['truncated_samples'], 1)
            self.assertEqual(o['generated_turns'], 5)
            d = report['domains']['math']['per_turn']
            self.assertEqual(d['prompt_tokens']['p50'], 20)
            self.assertEqual(d['completion_tokens']['p50'], 60)
            trajectory = report['domains']['python']['per_generated_trajectory']
            self.assertEqual(trajectory['prompt_tokens']['total'], 110)
            self.assertEqual(trajectory['completion_tokens']['missing'], 1)
            self.assertEqual(report['task_summary']['with_grading_errors'], ['code'])
            self.assertEqual(report['integrity_issues'], [])
            catalogue = [json.loads(line) for line in (root/'run/audit/samples.jsonl').read_text().splitlines()]
            self.assertEqual(len(catalogue), 6)
            self.assertEqual(sum(x['status'] == 'missing' for x in catalogue), 2)

    def test_corrupt_extra_and_mismatched_samples_are_not_silently_counted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); rows = self.fixture(root)
            self.sample(root, rows[0], 5, {})
            (root/'run/samples/corrupt.json').write_text('{')
            self.sample(root, dict(rows[1], domain='math'), 0, {})
            report = audit_run(root/'run', root/'data')
            self.assertEqual(len(report['integrity_issues']), 3)
            self.assertEqual(report['overall']['missing_samples'], 6)

    def test_policy_and_dataset_mismatch_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); rows = self.fixture(root)
            self.sample(root, rows[0], 0, {}, {'status': 'valid', 'score': 1., 'passed': True}, finish='length')
            report = audit_run(root/'run', root/'data')
            self.assertEqual(len(report['policy_violations']), 1)
            rows[0]['messages'][0]['content'] = 'tampered'
            write_jsonl(root/'data/splits/main_test.jsonl', rows)
            with self.assertRaisesRegex(ValueError, 'hash'):
                audit_run(root/'run', root/'data')

    def test_selection_reproduces_context_task_and_count_filters(self):
        rows = [{'id': str(i), 'task': t, 'length_bucket': length} for i, (t, length) in
                enumerate([('a', 4096), ('a', 8192), ('a', 8192), ('b', 4096), ('b', 16384)])]
        c = {'MODEL_CONTEXT': 8192, 'EVAL_CONTEXT_BUCKETS': [], 'TASKS': 'a,b',
             'LIMIT_PER_TASK': 2, 'TASK_SAMPLE_COUNTS': {'a': 1}, 'SPLIT': 'main_test'}
        self.assertEqual([r['id'] for r in selected_rows(rows, c)], ['0', '3'])

    def test_final_metrics_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.fixture(root)
            write_json(root/'run/metrics.json', {'truncated_samples': 99, 'infrastructure_errors': 0,
                       'incomplete_prompts': 3, 'aggregate_score_0_100': None, 'valid_full_benchmark': False})
            report = audit_run(root/'run', root/'data', running=False)
            self.assertTrue(report['saved_metrics_consistency']['checked'])
            self.assertEqual(report['saved_metrics_consistency']['differences'],
                             [{'metric': 'truncated_samples', 'saved': 99, 'recomputed': 0}])
            report = audit_run(root/'run', root/'data', running=True)
            self.assertFalse(report['saved_metrics_consistency']['checked'])
