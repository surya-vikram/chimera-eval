import unittest
from eval_stack.runner import response_budget


class BudgetTests(unittest.TestCase):
    def test_explicit_caps_do_not_require_row_defaults(self):
        row = {'task': 'temporary', 'domain': 'math'}
        c = {'MAX_NEW_TOKENS': 0, 'TASK_MAX_TOKENS': {'temporary': 8192}, 'DOMAIN_MAX_TOKENS': {}}
        self.assertEqual(response_budget(row, c), 8192)
        c['TASK_MAX_TOKENS'] = {}
        c['DOMAIN_MAX_TOKENS'] = {'math': 4096}
        self.assertEqual(response_budget(row, c), 4096)
        c['MAX_NEW_TOKENS'] = 128
        self.assertEqual(response_budget(row, c), 128)

    def test_missing_and_invalid_caps_fail(self):
        row = {'task': 'temporary', 'domain': 'math'}
        c = {'MAX_NEW_TOKENS': 0, 'TASK_MAX_TOKENS': {}, 'DOMAIN_MAX_TOKENS': {}}
        with self.assertRaises(ValueError):
            response_budget(row, c)
        self.assertEqual(response_budget(dict(row, max_new_tokens=64), c), 64)
        with self.assertRaises(ValueError):
            response_budget(dict(row, max_new_tokens=-1), c)
