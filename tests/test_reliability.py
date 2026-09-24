import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch
from eval_stack.runner import settings, context_limit
from eval_stack.judging import parse_judgment
from eval_stack.graders import Grader, GradingError
from eval_stack.rl_rewards import validated_group, InvalidRewardGroup
from eval_stack.data_revision import issues, revise_row
from eval_stack.metrics import aggregate
from eval_stack.common import DOMAINS
from eval_stack.cli import check_report

class ReliabilityTests(unittest.TestCase):
    def test_nominal_profile_uses_actual_served_window(self):
        r={'context_window':131072,'bucket_semantics':'nominal published prompt-length profile; actual prompt tokens reported separately'}
        self.assertEqual(context_limit(r,147456),147456)
        self.assertEqual(context_limit(r,8192),8192)
        self.assertEqual(context_limit({'context_window':8192},16384),8192)

    def test_long_context_scores_separate_buckets_and_cap_failures(self):
        rows=[];records=[]
        for length in (16384,32768,65536,131072):
            ident=str(length)
            rows.append({'id':ident,'task':f'long_{length}_rag','domain':'long_context','length_bucket':length,'binary':True})
            for sample in range(2):
                records.append({'id':ident,'sample':sample,'grade':{'status':'valid','score':1,'passed':True},
                                'turns':[{'response':{'finish_reason':'length' if length==131072 and sample==0 else 'stop'}}]})
        report=aggregate(rows,records,2,[2])
        self.assertEqual(set(report['long_context_by_length']),{'16384','32768','65536','131072'})
        self.assertEqual(report['long_context_by_length']['131072']['score'],.5)
        self.assertEqual(report['long_context_by_length']['131072']['pass']['2'],1.)

    def test_exhausted_parser_retries_are_errors_not_zero(self):
        class Judge:
            calls=0
            def token_count(self,m):return 10
            def complete(self,*a,**kw):
                self.calls+=1
                return {'text':'not JSON','finish_reason':'stop'}
        j=Judge()
        with self.assertRaises(GradingError):Grader(j,100,1000).judge_json({'question':'fixture'})
        self.assertEqual(j.calls,3)

    def test_task_budget_configuration(self):
        with patch.dict('os.environ', {'TASK_MAX_TOKENS_JSON':'{"math500":32768}'}):
            self.assertEqual(settings()['TASK_MAX_TOKENS']['math500'],32768)
        for value in ('[]','{"math500":0}','{"math500":true}'):
            with patch.dict('os.environ', {'TASK_MAX_TOKENS_JSON':value}), self.assertRaises(ValueError):
                settings()

    def test_truncation_has_distinct_exit_and_warning(self):
        report={'truncated_samples':1,'infrastructure_errors':0,'incomplete_prompts':0}
        output=io.StringIO()
        with contextlib.redirect_stderr(output):
            check_report(report)
        self.assertIn('TRUNCATION DETECTED',output.getvalue())
        check_report(dict(report,truncated_samples=0))

    def test_truncation_blocks_comparison_without_dropping_samples(self):
        rows=[{'id':d,'task':d,'domain':d,'binary':True} for d in DOMAINS if d != 'long_context']
        rows.append({'id':'long_context','task':'long_4096_rag','domain':'long_context','binary':True})
        records=[{'id':r['id'],'sample':0,'grade':{'status':'valid','score':1.,'passed':True},
                  'turns':[{'response':{'finish_reason':'stop'}}]} for r in rows]
        self.assertTrue(aggregate(rows,records,1,[1],True)['valid_full_benchmark'])
        records[0]['turns'][0]['response']['finish_reason']='length'
        report=aggregate(rows,records,1,[1],True)
        self.assertTrue(report['valid_full_benchmark'])
        self.assertTrue(report['comparison_budget_valid'])
        self.assertFalse(report['truncation_free'])
        self.assertEqual(report['aggregate_score_0_100'],90.)
        self.assertEqual(report['tasks'][rows[0]['task']]['pass']['1'],0.)
        self.assertEqual(report['truncated_samples'],1)
        self.assertEqual(report['incomplete_prompts'],0)
        self.assertEqual(report['tasks'][rows[0]['task']]['prompts'],1)

    def test_truncation_zero_without_calling_judge(self):
        row={'verifier':'quality','binary':False}
        grade=Grader().grade(row,{'text':'A seemingly excellent partial answer','finish_reason':'length'})
        self.assertEqual(grade['score'],0)
        self.assertEqual(grade['components']['failure'],'candidate_truncated')

    def test_parser_retry_changes_instruction_and_remains_bounded(self):
        class Judge:
            def __init__(self):self.messages=[]
            def token_count(self,m):return 10
            def complete(self,m,budget,**kw):
                self.messages.append(m)
                return {'text':'invalid' if len(self.messages)==1 else '{"verdict":true,"reason":"ok"}', 'finish_reason':'stop'}
        j=Judge();g=Grader(j,100,1000)
        self.assertTrue(g.judge_json({'question':'fixture'})['verdict'])
        self.assertNotEqual(j.messages[0],j.messages[1])

    def test_strict_json_boundaries(self):
        good='{"verdict":true,"reason":"supported"}'
        for text in (good,'```json\n'+good+'\n```'):
            self.assertTrue(parse_judgment({'text':text,'finish_reason':'stop'})[0]['verdict'])
        for text in ('Explanation '+good,good+good,'```json\n'+good+'\n``` extra',
                     '{"verdict":true,"verdict":false,"reason":"x"}',
                     '{"verdict":"true","reason":"x"}',
                     '{"verdict":true}', '{"verdict":true,"reason":"x","extra":0}'):
            with self.assertRaises(ValueError):parse_judgment({'text':text,'finish_reason':'stop'})
        with self.assertRaises(ValueError):parse_judgment({'text':good,'finish_reason':'length'})
        with self.assertRaises(ValueError):parse_judgment({'text':'','reasoning':good,'finish_reason':'stop'})

    def test_quality_types_and_ranges(self):
        for score in (True,0,6,4.0,'4',float('nan')):
            with self.assertRaises(ValueError):
                parse_judgment({'text':json.dumps({'score':score,'mandatory_pass':True,'reason':'x'}),'finish_reason':'stop'},True)

    def test_retry_budget_and_durable_trace(self):
        class Judge:
            def __init__(self):self.budgets=[]
            def token_count(self,m):return 10
            def complete(self,m,budget,**kw):
                self.budgets.append(budget)
                return {'text':'{"verdict":true,"reason":"ok"}','finish_reason':'length' if len(self.budgets)==1 else 'stop'}
        with tempfile.TemporaryDirectory() as td:
            j=Judge();g=Grader(j,100,1000,judge_audit_dir=td,judge_max_retry_tokens=200)
            out=g.judge_json({'question':'fixture'})
            self.assertEqual(j.budgets,[100,200]);self.assertEqual(len(out['audit_ids']),2)
            traces=[json.loads(p.read_text()) for p in Path(td).glob('*.json')]
            self.assertEqual(sorted(t['status'] for t in traces),['error','valid'])

    def test_failed_judgment_never_reward_zero(self):
        rows=[{'id':'a','sample':i,'grade':{'status':'valid','score':i%2,'passed':bool(i%2)}} for i in range(4)]
        self.assertEqual(validated_group(rows,4),[0,1,0,1])
        rows[1]['grade']={'status':'error'}
        with self.assertRaises(InvalidRewardGroup):validated_group(rows,4)
        with self.assertRaises(InvalidRewardGroup):validated_group(rows[:3],4)
        rows[1]['grade']={'status':'valid','score':.8,'passed':False}
        with self.assertRaises(InvalidRewardGroup):validated_group(rows,4)

    def test_conservative_source_quarantine(self):
        self.assertTrue(issues({'task':'biggen','stratum':'planning/reward_modeling'}))
        self.assertTrue(issues({'task':'structeval','messages':[{'content':'Given a description, extract names.'}]}))
        self.assertFalse(issues({'task':'structeval','messages':[{'content':'Create a JSON example of three fictional names.'}]}))

    def test_revision_preserves_original(self):
        r={'id':'a','family_id':'f','domain':'math','verifier':'math','verification':{},'messages':[{'content':'2+2?'}]}
        v,reason=revise_row(r,None,131072)
        self.assertIsNone(reason);self.assertNotEqual(v['id'],r['id']);self.assertEqual(r['messages'][0]['content'],'2+2?')
        self.assertEqual(v['family_id'],r['family_id']);self.assertEqual(v['max_new_tokens'],16384)
