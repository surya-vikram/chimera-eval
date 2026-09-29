import contextlib
import io
import unittest
from unittest.mock import patch

from eval_stack.graders import candidate_code, code_diagnostics
from eval_stack.metrics import aggregate
from eval_stack.progress import Progress, outcome
from eval_stack.run_audit import failure_reason, score_tables


def graded(passed, score=None, finish='stop', **components):
    return {'task': 't', 'grade': {'status': 'valid', 'passed': passed, 'score': float(passed) if score is None else score,
                                   'components': components},
            'turns': [{'finish_reason': finish, 'response': {'finish_reason': finish}}]}


class CodeExtractionTests(unittest.TestCase):
    def test_last_block_defining_the_function(self):
        text = "```python\ndef f(x):\n    return x\n```\nUsage:\n```python\nprint(f(3))\n```"
        self.assertEqual(candidate_code(text, 'f'), ('def f(x):\n    return x\n', 'block 1 of 2'))
        revised = "```python\ndef f(x): return 0\n```\nFixed:\n```python\ndef f(x): return 1\n```"
        self.assertEqual(candidate_code(revised, 'f')[0], 'def f(x): return 1\n')
        # Without an entry point (stdin programs), the last block is the answer.
        self.assertEqual(candidate_code(text)[0], 'print(f(3))\n')

    def test_fence_labels_and_other_languages(self):
        for label in ('python3', 'Python', 'py', ''):
            text = f"```{label}\ndef f(): return 1\n```\nOutput:\n```text\n1\n```"
            self.assertEqual(candidate_code(text, 'f')[0], 'def f(): return 1\n', label)

    def test_unfenced_and_unclosed(self):
        self.assertEqual(candidate_code('def f(): return 1', 'f'), ('def f(): return 1', 'whole_response'))
        self.assertEqual(candidate_code('```python\ndef f(): return 1\n', 'f'), ('def f(): return 1\n', 'unclosed_block'))

    def test_diagnostics(self):
        self.assertIn('syntax_error', code_diagnostics('def f(:', 'f'))
        self.assertEqual(code_diagnostics('def g(): pass', 'f'), {'defines_entry_point': False})
        self.assertEqual(code_diagnostics('f = lambda: 1', 'f'), {'defines_entry_point': True})


class ReportTests(unittest.TestCase):
    def scores(self):
        rows = [{'id': f'{d}{i}', 'task': d, 'domain': d, 'binary': True} for d in ('math', 'python') for i in range(2)]
        records = [{'id': r['id'], 'sample': s, 'grade': {'status': 'valid', 'score': float(s == 0), 'passed': s == 0},
                    'turns': [{'response': {'finish_reason': 'stop'}}]} for r in rows for s in range(4)]
        return aggregate(rows, records[:-1], 4, [1, 4], False)

    def test_pass_at_every_requested_k_and_partial_aggregate(self):
        text = '\n'.join(score_tables(self.scores()))
        self.assertIn('Pass@1', text)
        self.assertIn('Pass@4', text)
        self.assertIn('100.0%', text)  # pass@4: one of four samples passes
        self.assertIn('Aggregate score: 25.0 / 100 | Pass@1: 25.0% | Pass@4: 100.0% (partial:', text)
        self.assertRegex(text, r'\| aggregate +\| +3 / 4 \| +25\.0%\* \| +25\.0% \| +100\.0% \|')
        self.assertIn('25.0%*', text)  # python has an incomplete prompt

    def test_tables_are_aligned(self):
        lines = score_tables(self.scores())
        start = next(i for i, line in enumerate(lines) if line.startswith('| Domain'))
        domain_table = lines[start:lines.index('', start)]
        self.assertEqual(len({len(line) for line in domain_table}), 1)

    def test_failure_reasons(self):
        self.assertEqual(failure_reason(graded(False, finish='length')), 'truncated')
        self.assertEqual(failure_reason(graded(False, execution_status='fail', syntax_error='line 1')), 'code: syntax error')
        self.assertEqual(failure_reason(graded(False, execution_status='fail', defines_entry_point=False)),
                         'code: requested function not defined')
        self.assertEqual(failure_reason(graded(False, failure='empty_response')), 'empty response')
        self.assertEqual(failure_reason(graded(False)), 'wrong answer')


class ProgressTests(unittest.TestCase):
    def test_outcomes(self):
        self.assertEqual(outcome(graded(True)), 'passed')
        self.assertEqual(outcome(graded(False)), 'failed')
        self.assertEqual(outcome(graded(None, .5)), 'scored')
        self.assertEqual(outcome(graded(False, finish='length')), 'truncated')
        self.assertEqual(outcome({'grade': {'status': 'error', 'error': 'x'}}), 'error')

    def test_throttled_line_and_each_error_once(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), patch('eval_stack.progress.time.monotonic', return_value=0.):
            p = Progress(5, [graded(True)], interval=30)
            for record in [graded(False), {'task': 'humanevalplus', 'grade': {'status': 'error', 'error': 'no docker'}},
                           {'task': 'apps', 'grade': {'status': 'error', 'error': 'no docker'}}]:
                p.update(record)
            p.update(graded(True))
        lines = out.getvalue().splitlines()
        self.assertIn('1 already graded, 4 to run', lines[0])
        self.assertEqual(sum('[error] humanevalplus: no docker' in line for line in lines), 1)
        self.assertEqual(len(lines), 3)  # header, one error, final progress line
        self.assertIn('5/5 (100%)', lines[-1])
        self.assertIn('passed 2 · failed 1 · error 2', lines[-1])


if __name__ == '__main__':
    unittest.main()
