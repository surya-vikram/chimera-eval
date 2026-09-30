"""Format policy: presentation and extra text never cost reward; a missing requested format does.

Decided 2026-09-29 after auditing every rl_train, rl_val and main_test row: markdown, lists and
near-miss "Answer:" wording pass; extra text around a correctly formatted answer passes; a
response without the format the prompt explicitly asked for fails.
"""
import unittest

from eval_stack.graders import BOXED, Grader, candidate_code, extract_final, last_boxed
from test_answer_policies import Judge

FINAL = "\nEnd with 'Final answer: <short answer>'."


def row(kind, prompt='Question', domain='knowledge', **meta):
    return {'id': 'a', 'task': 't', 'domain': domain, 'verifier': kind, 'verification': meta,
            'messages': [{'role': 'user', 'content': prompt}], 'binary': True}


def grade(r, text, judge=None):
    return Grader(judge or Judge()).grade(r, {'text': text, 'finish_reason': 'stop'})


class FinalAnswerLineTests(unittest.TestCase):
    def test_presentation_and_extra_text_pass(self):
        trivia = row('alias', 'Who?' + FINAL, aliases=['Oh So Sharp'])
        for text in ['Reasoning.\nFinal answer: Oh So Sharp', 'Reasoning.\n**Final answer:** Oh So Sharp',
                     'Final answer: **Oh So Sharp**', 'Answer: Oh So Sharp',
                     'Final answer: Oh So Sharp\n\nShe won the fillies triple crown in 1985.']:
            self.assertTrue(grade(trivia, text)['passed'], text)

    def test_missing_line_fails_before_the_judge(self):
        judge = Judge()
        for kind, meta in [('alias', {'aliases': ['Paris']}), ('grounded', {'aliases': ['Paris']}),
                           ('exact', {'answer': 'Paris'})]:
            out = grade(row(kind, 'Capital?' + FINAL, domain='logic', **meta), 'The capital is Paris.', judge)
            self.assertEqual((out['passed'], out['components']['failure']), (False, 'format missing: Final answer line'))
        self.assertEqual(judge.payloads, [])


class ChoiceTests(unittest.TestCase):
    def test_labels(self):
        arc = row('choice', "Q\nYou may reason. End with 'Final answer: X' where X is one option label.",
                  answer='D', labels=['A', 'B', 'C', 'D'])
        for text in ['Final answer: D', '**Final answer: D**', 'Final answer: **D**', 'Final answer: D) repeat it',
                     'Final answer: D. It fits.', 'Final answer: (D)\n\nThe others contradict the data.']:
            self.assertTrue(grade(arc, text)['passed'], text)
        for text in ['Final answer: D or B', 'Final answer: D/B', 'The correct option is D.']:
            self.assertFalse(grade(arc, text)['passed'], text)
        numeric = row('choice', arc['messages'][0]['content'], answer='2', labels=['1', '2', '3', '4'])
        self.assertTrue(grade(numeric, 'Final answer: 2')['passed'])
        self.assertFalse(grade(numeric, 'Final answer: 2 or 3')['passed'])

    def test_mcqa_bold_letter_and_trailing_text(self):
        mcqa = dict(row('choice', 'Q', answer='C', labels=['A', 'B', 'C', 'D'],
                        output_regex=r'Answer\s*:\s*(?!Answer)\s*([A-Za-z0-9])\s*'), task='mcqa')
        self.assertTrue(grade(mcqa, 'Answer: **C**')['passed'])
        self.assertTrue(grade(mcqa, 'Answer: C\n\nBecause of the above.')['passed'])


class ExtractionTests(unittest.TestCase):
    def test_label_anywhere_and_bold(self):
        icl = r'label:\s*(\d+)\s*$'
        self.assertEqual(extract_final('label: 5\nIt is about cards.', {'output_regex': icl}), '5')
        self.assertEqual(extract_final('**label: 7**', {'output_regex': icl}), '7')
        self.assertEqual(extract_final('The category is 4.', {'output_regex': icl}), '')

    def test_boxed_matches_braces_to_any_depth(self):
        deep = r'\frac{\sqrt{\dfrac{5+\sqrt{5}}{2}}}{3}'
        self.assertEqual(extract_final('so \\boxed{' + deep + '} done', {'output_regex': BOXED}), deep)
        self.assertEqual(last_boxed(r'\boxed{\left\{ x \right.}'), r'\left\{ x \right.')
        self.assertEqual(last_boxed(r'\boxed{\begin{array}{l} a \\ {b} \end{array}}'), r'\begin{array}{l} a \\ {b} \end{array}')
        self.assertEqual(last_boxed(r'\boxed{x**2 + 1}'), 'x**2 + 1')  # content is never unemphasized

    def test_bracketed_formats_keep_inner_brackets(self):
        self.assertEqual(extract_final(r'(Answer: \(y^2=4x\) (i.e. \(x\)).)', {'output_regex': r'\(Answer:\s*(.+?)\)'}),
                         r'\(y^2=4x\) (i.e. \(x\)).')
        self.assertEqual(extract_final('((iron(III) chloride))', {'output_regex': r'\(\((.*?)\)\)'}), 'iron(III) chloride')
        self.assertEqual(extract_final('Answer is [f[x]]', {'output_regex': r'Answer is\s*\[(.+?)\]'}), 'f[x]')

    def test_retrieval_reads_a_list_under_the_answer_line(self):
        niah = row('retrieval', targets=['111', '222'], any_alias=False)
        self.assertTrue(grade(niah, 'Final answer:\n- 111\n- 222')['passed'])
        self.assertTrue(grade(niah, '**Answer:**\n1. **111**\n2. **222**')['passed'])


class RequestedFormatTests(unittest.TestCase):
    def test_box_required_when_asked(self):
        judge = Judge()
        asked = row('equivalence', 'Solve. Put your final answer inside \\boxed{}.', answer='x^2')
        self.assertEqual(grade(asked, 'The answer is x^2.', judge)['components']['failure'], 'format missing: \\boxed{}')
        self.assertTrue(grade(asked, 'So \\boxed{x^2}\n\nHope this helps.', judge)['passed'])
        self.assertEqual(judge.payloads[-1]['response'], 'x^2')

    def test_multiline_final_answer_reaches_the_judge_whole(self):
        judge = Judge()
        free = row('equivalence', 'Solve.', answer='a')
        grade(free, 'Work.\nFinal answer:\n\\[\n x = 1\n\\]', judge)
        self.assertEqual(judge.payloads[-1]['response'], '\\[\n x = 1\n\\]')

    def test_code_needs_a_block_extra_blocks_are_fine(self):
        apps = row('apps', 'Return a complete Python program using stdin/stdout in one python code block.',
                   domain='python', tests={'inputs': ['1\n'], 'outputs': ['1\n']})
        self.assertEqual(grade(apps, 'print(input())')['components']['failure'], 'format missing: python code block')
        solution = 'from sys import stdin\nprint(stdin.readline().strip())'
        text = f'```python\n{solution}\n```\n\nExample:\n```python\nprint("demo")\n```'
        self.assertEqual(candidate_code(text)[0].strip(), solution)

    def test_prompt_helpers_are_provided_imports_are_not(self):
        from eval_stack.graders import with_prompt_helpers
        problem = {'entry_point': 'make_palindrome',
                   'prompt': 'from typing import List\n\ndef is_palindrome(s):\n    return s == s[::-1]\n\n'
                             'def make_palindrome(s):\n    """doc"""\n'}
        answer = 'def make_palindrome(s):\n    return s if is_palindrome(s) else s + s[::-1]'
        code = with_prompt_helpers(problem, answer)
        self.assertIn('def is_palindrome', code)
        self.assertNotIn('from typing', code)  # imports stay the answer's job, as the prompt asks
        own = 'def is_palindrome(s):\n    return True\n\n' + answer
        self.assertEqual(with_prompt_helpers(problem, own), own)  # the answer's own helper wins

    def test_bare_yaml_scalar_is_not_data(self):
        yaml_row = row('structure', 'Convert to YAML.', domain='structure', format='yaml', requirements=[])
        out = grade(yaml_row, 'I could not find the requested information.')
        self.assertFalse(out['passed'])
        self.assertTrue(grade(yaml_row, 'name: a\nvalue: 1')['passed'])


if __name__ == '__main__':
    unittest.main()
