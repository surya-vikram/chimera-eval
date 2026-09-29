import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('compare', Path(__file__).resolve().parents[1] / 'compare.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def run(root, name, aggregate, ks, domains, model_path=None):
    path = root / name
    path.mkdir()
    (path / 'metrics.json').write_text(json.dumps({
        'aggregate_score_0_100': aggregate, 'aggregate_complete': False, 'pass_k': ks, 'n_samples': max(ks),
        'aggregate_pass_at_k': {str(k): aggregate for k in ks},
        'domains': {d: ({'score': v, 'pass': {str(k): None if d == 'quality' else v for k in ks}} if v is not None else None)
                    for d, v in domains.items()}}))
    (path / 'config.json').write_text(json.dumps({'config': {'MODEL_NAME': name, 'MODEL_SAMPLING': {'temperature': 1.0}}}))
    if model_path:
        quoted = "'" + model_path.replace("'", "'\\''") + "'"  # as run_eval.sh writes it
        (path / 'launch_config.env').write_text(f"# header\nMODEL_PATH={quoted}\nTASK_MAX_TOKENS_JSON='{{\"a\":1}}'\n")


class CompareTests(unittest.TestCase):
    def test_sorted_by_aggregate_with_empty_missing_values(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run(root, 'weak', 40., [1, 2], {'math': .4, 'quality': .4}, '/models/weak')
            run(root, 'strong', 90., [1, 4], {'math': .9}, "/models/it's strong")
            (root / 'unfinished').mkdir()
            rows, out = compare.compare(root)
            table = list(csv.DictReader(out.read_text().splitlines()))
            self.assertEqual([r['model_path'] for r in table], ["/models/it's strong", '/models/weak'])
            self.assertEqual(table[0]['model_path'], "/models/it's strong")
            self.assertEqual(table[0]['aggregate_score'], '90.0')
            self.assertEqual(table[0]['aggregate_pass@4'], '90.0')
            self.assertEqual(table[0]['aggregate_pass@2'], '')   # k not requested by this run
            self.assertEqual(table[1]['math_pass@2'], '40.0')
            self.assertEqual(table[1]['quality_pass@2'], '')     # quality has no pass@k
            self.assertEqual(table[0]['python_score'], '')       # domain not evaluated
            header = list(table[0])
            self.assertEqual(header[:4], ['model_path', 'aggregate_score', 'aggregate_pass@1', 'aggregate_pass@2'])
            self.assertEqual(header[5:15], [d + '_score' for d in compare.DOMAINS])
            self.assertEqual(header[-1], 'run')
            self.assertEqual(table[0]['math_score'], '90.0')

    def test_model_path_from_saved_run_script(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run(root, 'r', 50., [1], {'math': .5})
            (root / 'r' / 'run_eval.sh').write_text('  [MODEL_PATH]="/models/x"  # weights\n  [N_SAMPLES]=4\n')
            self.assertEqual(compare.compare(root)[0][0]['model_path'], '/models/x')
            (root / 'r' / 'run_eval.sh').unlink()
            self.assertEqual(compare.compare(root)[0][0]['model_path'], 'r')  # falls back to the model name


if __name__ == '__main__':
    unittest.main()
