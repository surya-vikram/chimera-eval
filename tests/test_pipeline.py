import contextlib
import http.server
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from eval_stack.client import Client
from eval_stack.common import digest, family, write_json, write_jsonl
from eval_stack.runner import evaluate
from eval_stack.prepare import build_splits


class Handler(http.server.BaseHTTPRequestHandler):
    requests = []
    def log_message(self, *args): pass
    def send(self, result):
        body = json.dumps(result).encode()
        self.send_response(200); self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self): self.send({'data':[{'id':'fixture'}]})
    def do_POST(self):
        obj = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.requests.append((self.path, obj))
        if self.path == '/tokenize': self.send({'count':10}); return
        text = 'Final answer: 42'
        if obj.get('response_format'): text = '{"score":4,"mandatory_pass":true,"reason":"fixture"}'
        self.send({'choices':[{'message':{'content':text},'finish_reason':'stop'}], 'usage':{'completion_tokens':5}})


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1',0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        Handler.requests = []
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1'
    def tearDown(self):
        self.server.shutdown(); self.thread.join(); self.server.server_close()

    def test_vllm_sampling_wire_format(self):
        settings = {'temperature':.7,'top_p':.8,'top_k':20,'repetition_penalty':1.1}
        c = Client(self.url,'fixture',settings)
        c.complete([{'role':'user','content':'Question'}],128,seed=12)
        body = Handler.requests[-1][1]
        for k,v in settings.items(): self.assertEqual(body[k],v)
        self.assertNotIn('extra_body',body)
        self.assertEqual(body['max_tokens'],128)
        self.assertEqual(body['seed'],12)

    def test_full_pipeline_resume_and_config_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = []
            for task, kind, binary, meta in [('exact','exact',True,{'answer':'42'}),('quality','quality',False,{'rubric':{}})]:
                rows.append({'id':digest(task),'family_id':family(task),'family_text':task, 'task':task,
                             'domain':'quality' if kind=='quality' else 'math','messages':[{'role':'user','content':'Question'}],
                             'verifier':kind,'verification':meta,'binary':binary,'max_new_tokens':128,'source':{'repo':'fixture'}})
            write_jsonl(root/'splits/main_test.jsonl',rows)
            write_json(root/'manifest.json',{'complete_inventory':False})
            cfg = {'DATA_DIR':td,'OUTPUT_DIR':str(root/'out'),'MODEL_URL':self.url,'JUDGE_URL':self.url,
                   'MODEL_NAME':'fixture','JUDGE_NAME':'fixture','N_SAMPLES':'4','PASS_K':'1,4','MAX_PENDING':'2'}
            with patch.dict(os.environ,cfg,clear=True), contextlib.redirect_stdout(io.StringIO()):
                a = evaluate()
                first = len([x for x in Handler.requests if x[0].endswith('completions')])
                b = evaluate()
                self.assertEqual(first,len([x for x in Handler.requests if x[0].endswith('completions')]))
                self.assertEqual(a['tasks'],b['tasks'])
                self.assertEqual(a['tasks']['exact']['pass']['4'],1.)
                self.assertIsNone(a['tasks']['quality']['pass']['4'])
                os.environ['MODEL_TEMPERATURE'] = '.9'
                with self.assertRaises(ValueError): evaluate()
            # No private gold in candidate requests.
            for route, body in Handler.requests:
                if route.endswith('completions') and not body.get('response_format'):
                    self.assertNotIn('verification',body)

    def test_heldouts_reserved_before_train(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def item(i, task):
                return {'id':digest([task,i]),'family_id':family(str(i)), 'family_text':str(i),
                        'stratum':'default','task':task,'domain':'math'}
            write_jsonl(root/'adapted/test.jsonl',[item(1,'test'),item(2,'test')])
            write_jsonl(root/'adapted/train.jsonl',[item(i,'train') for i in range(1,8)])
            specs = [('train','fixture',None,'train','exact','math',3,1,0),('test','fixture',None,'test','exact','math',0,0,1)]
            with contextlib.redirect_stdout(io.StringIO()): report = build_splits(root,specs)
            self.assertTrue(all(v==0 for v in report['exact_family_overlap'].values()))
            self.assertEqual(report['splits']['rl_train']['rows'],3)


if __name__ == '__main__': unittest.main()
