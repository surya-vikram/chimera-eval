import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'run_eval.sh'


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root/'data with spaces'
        (self.data/'splits').mkdir(parents=True)
        (self.data/'manifest.json').write_text('{}')
        (self.data/'splits/main_test.jsonl').write_text('{}\n')
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.bind(str(self.root/'docker.sock'))
        self.calls = self.root/'calls.jsonl'
        binary = self.root/'docker'
        binary.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['DOCKER_CALLS'], 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
args = sys.argv[1:]
if args[:2] == ['image', 'inspect'] and os.environ.get('MISSING_IMAGE') == args[2]:
    sys.exit(1)
if args[:1] == ['run']:
    print('{"completed":1,"total":1,"task":"fixture","status":"valid"}')
    sys.exit(int(os.environ.get('EVAL_EXIT', '0')))
sys.exit(0)
''')
        binary.chmod(0o755)
        self.env = {'PATH': str(self.root) + os.pathsep + os.environ['PATH'],
                    'DOCKER_CALLS': str(self.calls), 'DOCKER_SOCKET': str(self.root/'docker.sock'),
                    'DATA_PATH': str(self.data), 'OUTPUT_PATH': str(self.root/'results with spaces'),
                    'RUN_NAME': 'fixture-run', 'IMAGE': 'fixture:local'}

    def tearDown(self):
        self.sock.close()
        self.temp.cleanup()

    def run_script(self, *args, **env):
        return subprocess.run(['bash', str(SCRIPT), *args], env={**self.env, **env},
                              cwd=self.root, capture_output=True, text=True, timeout=10)

    def docker_calls(self):
        return [json.loads(x) for x in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_missing_image_fails_without_pull_or_container(self):
        p = self.run_script(MISSING_IMAGE='fixture:local')
        self.assertEqual(p.returncode, 1)
        self.assertIn('Required local image is missing', p.stderr)
        self.assertFalse(any(c[0] in ('run', 'pull', 'build') for c in self.docker_calls()))

    def test_missing_worker_image_also_fails(self):
        p = self.run_script(CODE_IMAGE='missing:worker', MISSING_IMAGE='missing:worker')
        self.assertEqual(p.returncode, 1)
        self.assertNotIn('run', [c[0] for c in self.docker_calls()])

    def test_missing_dataset_does_not_download(self):
        (self.data/'manifest.json').unlink()
        p = self.run_script()
        self.assertEqual(p.returncode, 1)
        self.assertIn('Prepared manifest missing', p.stderr)
        self.assertNotIn('run', [c[0] for c in self.docker_calls()])

    def test_dry_run_is_read_only(self):
        p = self.run_script('--dry-run')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('--pull=never', p.stdout)
        self.assertFalse(Path(self.env['OUTPUT_PATH']).exists())
        self.assertNotIn('run', [c[0] for c in self.docker_calls()])

    def test_config_quoting_offline_flags_logging_and_exit_status(self):
        p = self.run_script(EVAL_EXIT='2', TASKS='gsm8k,math500',
                            MODEL_CHAT_TEMPLATE_KWARGS='{"reasoning_strength":"high"}')
        self.assertEqual(p.returncode, 2, p.stderr)
        runs = [c for c in self.docker_calls() if c[0] == 'run']
        self.assertEqual(len(runs), 1)
        args = runs[0]
        self.assertIn('--pull=never', args)
        self.assertIn('HF_HUB_OFFLINE=1', args)
        self.assertIn('HF_DATASETS_OFFLINE=1', args)
        self.assertIn('TRANSFORMERS_OFFLINE=1', args)
        self.assertIn('TASKS=gsm8k,math500', args)
        self.assertIn('MODEL_CHAT_TEMPLATE_KWARGS={"reasoning_strength":"high"}', args)
        self.assertIn(str(self.data) + ':/data:ro', args)
        log = Path(self.env['OUTPUT_PATH'])/'fixture-run.log'
        self.assertIn('"completed":1', log.read_text())
        # The launch script and the values actually used are saved with the results.
        run_dir = Path(self.env['OUTPUT_PATH'])/'fixture-run'
        self.assertTrue((run_dir/'run_eval.sh').exists())
        saved = (run_dir/'launch_config.env').read_text()
        self.assertIn("TASKS='gsm8k,math500'", saved)
        self.assertIn('MODEL_CHAT_TEMPLATE_KWARGS=\'{"reasoning_strength":"high"}\'', saved)
        p = self.run_script(MODEL_PATH='/other/weights')
        self.assertEqual(p.returncode, 1)
        self.assertIn('would mix models', p.stderr)
        code = Path(self.env['OUTPUT_PATH'])/'.launchers/fixture-run/code'
        self.assertTrue((code/'eval_stack/run_audit.py').exists())
        self.assertFalse(list(code.parent.glob('control.*')))

    def test_resume_keeps_frozen_code(self):
        self.assertEqual(self.run_script().returncode, 0)
        code = Path(self.env['OUTPUT_PATH'])/'.launchers/fixture-run/code'
        marker = code/'marker'
        marker.write_text('keep')
        stamp = (code/'eval_entrypoint.sh').stat().st_mtime_ns
        self.assertEqual(self.run_script().returncode, 0)
        self.assertEqual(marker.read_text(), 'keep')
        self.assertEqual((code/'eval_entrypoint.sh').stat().st_mtime_ns, stamp)

    def test_invalid_run_name_cannot_escape_output_directory(self):
        p = self.run_script(RUN_NAME='../escape')
        self.assertEqual(p.returncode, 1)
        self.assertEqual(self.docker_calls(), [])
