import copy
import unittest

from eval_stack.repair_metadata import BROKEN_BOX, CORRECT_BOX, ANSWER, repair_row, audit_mcqa
from eval_stack.sources import SPECS, adapt, Rejected
from eval_stack.graders import Grader, GradingError, extract_final


class MetadataRepairTests(unittest.TestCase):
    def source(self, pattern):
        return {'expected_answer': 'C', 'options': [{'A': 'one'}, {'B': 'two'}, {'C': 'three'}],
                'prompt': 'Choose the correct option. Answer in the requested format.',
                'template_metadata': {'output_regex': pattern}}

    def test_pinned_mcqa_source_regex_is_already_decoded(self):
        spec = next(s for s in SPECS if s[0] == 'mcqa')
        for pattern in (CORRECT_BOX, ANSWER):
            row = adapt(spec, self.source(pattern), 0, 'fixture')
            self.assertEqual(row['verification']['output_regex'], pattern)
            audit_mcqa(row)

    def test_repair_is_narrow_and_preserves_prompt_family_gold(self):
        spec = next(s for s in SPECS if s[0] == 'mcqa')
        row = adapt(spec, self.source(CORRECT_BOX), 0, 'fixture')
        row['verification']['output_regex'] = BROKEN_BOX
        before = copy.deepcopy(row)
        fixed, changed = repair_row(row)
        self.assertTrue(changed)
        self.assertEqual(row, before)
        self.assertNotEqual(fixed['id'], row['id'])
        self.assertEqual(fixed['messages'], row['messages'])
        self.assertEqual(fixed['family_id'], row['family_id'])
        self.assertEqual(fixed['verification']['answer'], row['verification']['answer'])
        audit_mcqa(fixed)
        self.assertFalse(repair_row(fixed)[1])

    def test_old_bug_demonstrated_and_false_word_boundary_rejected(self):
        row = adapt(next(s for s in SPECS if s[0] == 'mcqa'), self.source(CORRECT_BOX), 0, 'fixture')
        broken = copy.deepcopy(row)
        broken['verification']['output_regex'] = BROKEN_BOX
        self.assertEqual(extract_final('Answer: \\boxed{C}', broken['verification']), '')
        self.assertEqual(extract_final('oxed{C}', broken['verification']), 'C')
        with self.assertRaises(GradingError):
            Grader().grade(broken, {'text': 'Answer: \\boxed{C}', 'finish_reason': 'stop'})
        self.assertEqual(Grader().grade(row, {'text': 'oxed{C}', 'finish_reason': 'stop'})['score'], 0)

    def test_unknown_pattern_not_guessed(self):
        row = adapt(next(s for s in SPECS if s[0] == 'mcqa'), self.source('unknown'), 0, 'fixture')
        with self.assertRaises(ValueError):
            repair_row(row)

    def test_source_rejects_gold_not_in_options(self):
        raw = self.source(CORRECT_BOX)
        raw['expected_answer'] = '13'
        with self.assertRaises(Rejected):
            adapt(next(s for s in SPECS if s[0] == 'mcqa'), raw, 0, 'fixture')
        raw['expected_answer'] = 'C'
        raw['options'] = [{'C': 'only option'}]
        with self.assertRaises(Rejected):
            adapt(next(s for s in SPECS if s[0] == 'mcqa'), raw, 0, 'fixture')

    def test_other_source_regexes_preserved(self):
        patterns = [r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}',
                    r'Final Answer:\s*\|\|(.*?)\|\|', r'\(Answer:\s*(.+?)\)',
                    r'\(\((.*?)\)\)', r'<final_answer>\s*(.+?)\s*</final_answer>',
                    r'\[Answer:\s*(.+?)\]', r'Answer is\s*\[(.+?)\]', r'\*\*(.*?)\*\*', r'<<(.*?)>>']
        examples = [r'\boxed{value}', 'Final Answer: ||value||', '(Answer: value)',
                    '((value))', '<final_answer>value</final_answer>', '[Answer: value]',
                    'Answer is [value]', '**value**', '<<value>>']
        for task in ('openqa', 'science', 'nemotron_math'):
            for pattern, text in zip(patterns, examples):
                row = adapt(next(s for s in SPECS if s[0] == task), self.source(pattern), 0, 'fixture')
                self.assertEqual(row['verification']['output_regex'], pattern)
                self.assertEqual(extract_final(text, row['verification']), 'value')


if __name__ == '__main__':
    unittest.main()
