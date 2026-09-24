"""Opt-in Docker integration tests: RUN_SANDBOX_TESTS=1 python -m unittest discover -s tests."""
import os
import unittest
from eval_stack.graders import Grader


@unittest.skipUnless(os.environ.get('RUN_SANDBOX_TESTS') == '1', 'requires the built Docker image')
class SandboxTests(unittest.TestCase):
    def test_apps_good_wrong_and_network(self):
        row = {'verifier':'apps','verification':{'tests':{'inputs':['2 3\n'],'outputs':['5\n']}}}
        g = Grader(code_timeout=3)
        self.assertTrue(g.execute_code(row,'print(sum(map(int,input().split())))')['passed'])
        self.assertFalse(g.execute_code(row,'print(6)')['passed'])
        # Network is unavailable; no host datasets or Docker socket enter the child sandbox.
        code = "import os,socket\nassert not os.path.exists('/var/run/docker.sock')\ntry:\n socket.create_connection(('1.1.1.1',443),timeout=1)\n print(0)\nexcept OSError:\n print(5)"
        self.assertTrue(g.execute_code(row,code)['passed'])

    def test_timeout_is_failed_candidate(self):
        row = {'verifier':'apps','verification':{'tests':{'inputs':[''],'outputs':['0']}}}
        out = Grader(code_timeout=1).execute_code(row,'while True: pass')
        self.assertFalse(out['passed'])
        self.assertEqual(out['components']['execution_status'],'timeout')
