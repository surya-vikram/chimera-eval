"""What earns reward: reasoning tags, MCQA answer formats, structured data inside prose."""
import json
import unittest

from eval_stack.graders import (MCQA_FORMATS, OTHER_FORMAT_CREDIT, Grader, answer_text,
                                quarantine_reason, rubric_question)


class Judge:
    model = 'recording'

    def __init__(self, verdict=True):
        self.verdict, self.payloads = verdict, []

    def token_count(self, messages):
        return 10

    def complete(self, messages, max_tokens, **kwargs):
        self.payloads.append(json.loads(messages[-1]['content']))
        return {'text': json.dumps({'verdict': self.verdict, 'reason': 'fixture'}), 'finish_reason': 'stop', 'usage': {}}


def row(kind, domain='math', task='t', **meta):
    return {'id': 'a', 'task': task, 'domain': domain, 'verifier': kind, 'verification': meta,
            'messages': [{'role': 'user', 'content': 'Question'}], 'binary': True}


def grade(r, text, judge=None):
    return Grader(judge).grade(r, {'text': text, 'finish_reason': 'stop'})


class ReasoningTagTests(unittest.TestCase):
    def test_answer_text(self):
        self.assertEqual(answer_text('<think>try \\boxed{5}</think>\nFinal: \\boxed{6}'), '\nFinal: \\boxed{6}')
        self.assertEqual(answer_text('plain \\boxed{6}'), 'plain \\boxed{6}')
        # No answer outside the reasoning: unfinished.
        self.assertIsNone(answer_text('<think>only reasoning \\boxed{6}'))
        self.assertIsNone(answer_text('<think>\\boxed{6}</think>  '))
        self.assertIsNone(answer_text('<think>a</think> draft <think>more'))

    def test_tags_never_cost_reward(self):
        gsm = row('math', answer='6', answer_policy='explicit_final_box_v2')
        tagged = grade(gsm, '<think>maybe \\boxed{5}</think>\nSo the answer is \\boxed{6}.')
        self.assertEqual((tagged['score'], tagged['components']['reasoning_tags']), (1, True))
        unfinished = grade(gsm, '<think>The answer is \\boxed{6}.')
        self.assertEqual((unfinished['score'], unfinished['components']['failure'], unfinished['components']['incomplete']),
                         (0, 'unfinished_reasoning', True))
        truncated = Grader().grade(gsm, {'text': '<think>still thinking', 'finish_reason': 'length'})
        self.assertEqual(truncated['components']['failure'], 'candidate_truncated')
        self.assertNotIn('reasoning_tags', grade(gsm, 'The answer is \\boxed{6}.')['components'])
        calendar = row('calendar', calendar={'1': {'duration': 30, 'min_time': '10:00', 'max_time': '16:00'}})
        event = json.dumps([{'event_id': 1, 'event_name': 'm', 'duration': 30, 'start_time': '10:30'}])
        self.assertEqual(grade(calendar, '<think>fits at 10:30</think>\n' + event)['score'], 1)
        self.assertEqual(grade(calendar, 'Here is the updated calendar:\n```json\n' + event + '\n```')['score'], 1)
        self.assertEqual(grade(calendar, 'Draft: ' + event + '\nFinal: []')['score'], 0)

    def test_judge_reads_the_answer_not_the_reasoning(self):
        judge = Judge()
        grade(row('grounded', aliases=['Paris']), '<think>maybe Lyon</think>\nFinal answer: Paris', judge)
        self.assertEqual(judge.payloads[0]['response'], '\nFinal answer: Paris')


class McqaFormatTests(unittest.TestCase):
    def mcqa(self, fmt):
        return row('choice', domain='knowledge', task='mcqa', answer='B', labels=list('ABCD'), output_regex=fmt)

    def test_requested_format_full_other_format_partial(self):
        answer_row, boxed_row = (self.mcqa(f) for f in MCQA_FORMATS)
        cases = [(answer_row, 'Reasoning...\nAnswer: B', 1, True), (answer_row, 'Reasoning... \\boxed{B}', OTHER_FORMAT_CREDIT, False),
                 (answer_row, 'Reasoning... \\boxed{C}', 0, False), (boxed_row, 'Thus \\boxed{B}', 1, True),
                 (boxed_row, 'Answer: B', OTHER_FORMAT_CREDIT, False), (boxed_row, 'Answer: A', 0, False),
                 (answer_row, 'Answer: The answer is B', 0, False), (answer_row, 'I think B', 0, False)]
        for r, text, score, passed in cases:
            with self.subTest(text=text):
                graded = grade(r, text)
                self.assertEqual((graded['score'], graded['passed']), (score, passed))


class StructuredTests(unittest.TestCase):
    schema = {'type': 'object', 'properties': {'a': {'type': 'integer'}}, 'required': ['a'], 'additionalProperties': False}

    def test_data_inside_prose_gets_full_reward(self):
        r = row('structure', format='json', schema=self.schema)
        for text in ('{"a": 1}', 'Here is the JSON:\n```json\n{"a": 1}\n```\nHope this helps.',
                     'Sure: {"a": 1} is the object.', '```json\n{"b": 2}\n```\nFixed:\n```json\n{"a": 1}\n```'):
            with self.subTest(text=text):
                judge = Judge()
                graded = grade(r, text, judge)
                self.assertEqual(graded['score'], 1)
                self.assertEqual(judge.payloads[0]['response'].strip(), '{"a": 1}')
        yaml_row = row('structure', format='yaml', schema=self.schema)
        self.assertEqual(grade(yaml_row, 'Here you go:\n```yaml\na: 1\n```', Judge())['score'], 1)

    def test_the_last_answer_counts(self):
        r = row('structure', format='json', schema=self.schema)
        judge = Judge()
        graded = grade(r, '```json\n{"a": 1}\n```\nOr maybe:\n```json\n{"b": 2}\n```', judge)
        self.assertEqual((graded['score'], graded['components']['schema'], judge.payloads), (0, False, []))
        # Permissive formats: a prose line must not be read as the CSV header when a block exists.
        item = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name'],
                'additionalProperties': False}
        table = row('structure', format='csv', schema={'type': 'array', 'items': item})
        self.assertEqual(grade(table, 'Here is the CSV:\n```csv\nname\nLopaphus\n```', Judge())['score'], 1)

    def test_invalid_data_still_fails_without_the_judge(self):
        r = row('structure', format='json', schema=self.schema)
        for text, key in (('No data here at all.', 'syntax'), ('Here: {"a": "one"}', 'schema'), ('```\n```', 'syntax')):
            with self.subTest(text=text):
                judge = Judge()
                graded = grade(r, text, judge)
                self.assertEqual((graded['score'], graded['components'][key], judge.payloads), (0, False, []))

    def test_csv_cells_follow_schema_types(self):
        item = {'type': 'object', 'properties': {'name': {'type': 'string'}, 'year': {'type': 'integer'},
                                                 'extinct': {'type': ['boolean', 'null']}},
                'required': ['name', 'year'], 'additionalProperties': False}
        table = row('structure', format='csv', schema={'type': 'array', 'items': item})
        self.assertEqual(grade(table, 'name,year,extinct\nLopaphus,1908,false\nX,2000,', Judge())['score'], 1)
        single = row('structure', format='csv', schema=item)
        self.assertEqual(grade(single, 'The record:\n```csv\nname,year,extinct\nLopaphus,1908,true\n```', Judge())['score'], 1)
        self.assertEqual(grade(table, 'name,year\nLopaphus,nineteen', Judge())['components']['schema'], False)

    def test_quarantine_only_rows_no_response_can_pass(self):
        nested = {'type': 'object', 'properties': {'authority': {'type': 'object'}}}
        flat = {'type': 'array', 'items': {'type': 'object', 'properties': {'n': {'type': 'integer'}}}}
        self.assertIn('CSV cannot express', quarantine_reason(row('structure', format='csv', schema=nested)))
        self.assertIsNone(quarantine_reason(row('structure', format='csv', schema=flat)))
        self.assertIn('TOML top level', quarantine_reason(row('structure', format='toml', schema={'type': 'array'})))
        self.assertIsNone(quarantine_reason(row('structure', format='json', schema=nested)))
        self.assertIsNone(quarantine_reason(row('math', answer='1')))



class RubricTests(unittest.TestCase):
    def test_judge_reads_the_completed_conversation_once(self):
        content = ("Given the following conversation:\n\n[USER]: Hi\n\n[ASSISTANT]: Hello\n\n[USER]: List birds.\n\n"
                   "Does the model's final response satisfy this criterion?\n\nCriterion: Did it list birds?\n\n"
                   "Expected answer: YES")
        self.assertEqual(rubric_question(content),
                         "Does the model's final response satisfy this criterion?\n\nCriterion: Did it list birds?")
        self.assertEqual(rubric_question('Did the model keep the name?'), 'Did the model keep the name?')
        r = row('rubric', checks=[{'content': content, 'pass_criteria': 'YES'}])
        r['messages'] = [{'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello'},
                         {'role': 'user', 'content': 'List birds.'}]
        judge = Judge()
        self.assertEqual(grade(r, 'Robin, Wren', judge)['score'], 1)
        payload = judge.payloads[0]
        self.assertEqual(set(payload), {'question', 'conversation'})
        self.assertEqual(payload['conversation'][-1], {'role': 'assistant', 'content': 'Robin, Wren'})
        self.assertNotIn('Expected answer', payload['question'])
        self.assertNotIn('[USER]', payload['question'])


if __name__ == '__main__':
    unittest.main()
