"""judge_use() must match what grade() actually does, for every verifier."""
import json
import unittest
from unittest.mock import patch

from eval_stack.graders import JUDGE_USES, Grader, GradingError, judge_use, result


class RecordingJudge:
    model = 'recording'

    def __init__(self):
        self.calls = 0

    def token_count(self, messages):
        return 10

    def complete(self, messages, max_tokens, **kwargs):
        self.calls += 1
        if 'mandatory_pass' in messages[0]['content']:
            value = {'score': 4, 'mandatory_pass': True, 'reason': 'fixture'}
        else:
            value = {'verdict': True, 'reason': 'fixture'}
        return {'text': json.dumps(value), 'finish_reason': 'stop', 'usage': {}}


def row(kind, domain='math', **meta):
    return {'id': 'a', 'task': 't', 'domain': domain, 'verifier': kind, 'verification': meta,
            'messages': [{'role': 'user', 'content': 'Question'}], 'binary': kind != 'quality'}


CALENDAR = {'1': {'duration': 30, 'min_time': '10:00', 'max_time': '16:00', 'constraint': 'before 11am'}}
GOOD_EVENT = json.dumps([{'event_id': 1, 'event_name': 'm', 'duration': 30, 'start_time': '10:30'}])

# (row, deterministic pass, deterministic misses). Misses include wrong and malformed answers.
CASES = {
    'math': (row('math', answer='6'), r'\boxed{6}', [r'\boxed{7}', 'I am not sure.']),
    'choice': (row('choice', answer='B', labels=['A', 'B']), 'B', ['A', 'Answer: C', 'maybe']),
    'instruction': (row('instruction', ids=['length_constraints:number_words'],
                        kwargs=[{'num_words': 3, 'relation': 'less than'}]),
                    'Hello world', ['Hello world this is too long']),
    'calendar': (row('calendar', calendar=CALENDAR), GOOD_EVENT, ['[]', 'not json']),
    'retrieval': (row('retrieval', targets=['alpha']), 'alpha', ['beta']),
    'exact': (row('exact', answer='on'), 'on', ['off']),
    'exact/logic': (row('exact', domain='logic', answer='on'), 'on', ['off']),
    'alias': (row('alias', aliases=['Paris']), 'Paris', ['Lyon', 'Paris, I think']),
    'structure': (row('structure', format='json', schema={'type': 'object', 'required': ['a']}),
                  '{"a": 1}', ['{bad', '{"b": 1}']),
    'grounded': (row('grounded', aliases=['Paris']), None, ['Paris', 'Lyon']),
    'equivalence': (row('equivalence', answer='42'), None, ['Answer: 42', 'Answer: 41']),
    'quality': (row('quality', rubric='Be helpful.'), None, ['A helpful answer.']),
    'rubric': (row('rubric', checks=[{'content': 'Is it polite?'}]), None, ['Hello there.']),
    'apps': (row('apps'), None, ['print(1)']),
    'humanevalplus': (row('humanevalplus'), None, ['def f(): pass']),
}


def judge_calls(case_row, text):
    judge = RecordingJudge()
    code_result = result(1, True)
    with patch.object(Grader, 'execute_code', return_value=code_result):
        Grader(judge).grade(case_row, {'text': text, 'finish_reason': 'stop'})
    return judge.calls


class JudgeUseTests(unittest.TestCase):
    def test_every_verifier_is_classified(self):
        for name, (case_row, _, _) in CASES.items():
            self.assertIn(judge_use(case_row), JUDGE_USES, name)
        with self.assertRaises(GradingError):
            judge_use(row('unknown'))

    def test_classification_matches_grading(self):
        for name, (case_row, passing, misses) in CASES.items():
            use = judge_use(case_row)
            with self.subTest(verifier=name, use=use):
                pass_calls = None if passing is None else judge_calls(case_row, passing)
                miss_calls = [judge_calls(case_row, text) for text in misses]
                if use == 'none':
                    self.assertEqual(pass_calls or 0, 0)
                    self.assertEqual(miss_calls, [0] * len(misses))
                elif use == 'on_miss':
                    self.assertEqual(pass_calls, 0)
                    self.assertTrue(all(miss_calls))
                elif use == 'to_pass':
                    self.assertGreater(pass_calls, 0)
                    self.assertIn(0, miss_calls)
                else:
                    self.assertTrue(all(miss_calls))

    def test_expected_training_classes(self):
        expected = {'math': 'none', 'choice': 'none', 'instruction': 'none', 'calendar': 'none',
                    'apps': 'none', 'exact': 'none', 'exact/logic': 'on_miss', 'structure': 'to_pass',
                    'grounded': 'always', 'equivalence': 'always', 'quality': 'always', 'rubric': 'always'}
        for name, use in expected.items():
            self.assertEqual(judge_use(CASES[name][0]), use, name)

    def test_judge_free_grader_refuses_judge_paths(self):
        for name, (case_row, _, misses) in CASES.items():
            if judge_use(case_row) in ('always', 'on_miss'):
                with self.subTest(verifier=name), self.assertRaisesRegex(GradingError, 'Judge endpoint'):
                    Grader(None).grade(case_row, {'text': misses[-1], 'finish_reason': 'stop'})


if __name__ == '__main__':
    unittest.main()
