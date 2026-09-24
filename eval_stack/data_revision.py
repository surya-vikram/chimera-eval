"""Explicit new dataset version: quarantine known defects; never overwrite old splits."""
import argparse
import collections
import copy
import json
import re
from pathlib import Path
from .client import Client
from .common import digest, env, read_jsonl, write_json, write_jsonl
from .prepare import family_keys

POLICY='quality-first-v2'
CAPS={'math':16384,'knowledge':8192,'logic':16384,'grounding':8192,'quality':16384,
      'instruction':8192,'multiturn':8192,'structure':8192,'python':16384}


def issues(row):
    if row.get('task')=='biggen' and row.get('stratum')=='planning/reward_modeling':
        return ['reference_and_environment_consistency_requires_review']
    if row.get('task')=='structeval':
        text='\n'.join(m['content'] for m in row['messages'])
        if re.search(r'\bextract\w*\b',text,re.I) and re.search(r'\b(?:given|provided|input|description)\b',text,re.I):
            # Conservative quarantine. Do not pretend a keyword detector proves absence
            # of a source; individually reviewed approvals can restore valid rows later.
            return ['extraction_source_presence_requires_review']
    return []


def revise_row(original,client,window):
    row=copy.deepcopy(original)
    row['source_record_id']=original['id'];row['data_policy']=POLICY
    if row['verifier']=='math':
        row['messages'][-1]['content']+='\nEnd with exactly one unambiguous final answer in \\boxed{...}. Do not give alternative final answers.'
        row['verification']['answer_policy']='explicit_final_box_v2'
    if row['domain']=='long_context':
        reserve=env('LONG_RESPONSE_RESERVE',8192,int)
        cap=min(4096,reserve) if row['verifier']=='retrieval' else reserve
        if cap<1:raise ValueError('Response reserve must be positive')
        row['max_new_tokens']=cap;row['context_window']=window
        row['bucket_semantics']='nominal published prompt-length profile; actual prompt tokens reported separately'
        count=client.token_count(row['messages'])
        # ICL preparation may remove whole demonstrations, never the query or part of one.
        # This is a versioned prompt change, not runtime truncation.
        removed=0
        if count+cap>window and row['stratum']=='icl':
            text=row['messages'][0]['content'];header_end=text.find('\n\n')+2
            body=text[header_end:];demos=[];offset=0
            for match in re.finditer(r'\nlabel: \d+\n\n',body):
                demos.append(body[offset:match.end()-2]);offset=match.end()
            if not demos or not body[offset:].strip():
                return None,'ICL_demonstration_boundaries_require_review'
            parts=[text[:header_end-2],body[offset:]]
            low,high,best=1,len(demos),None
            while low<=high:
                mid=(low+high)//2
                row['messages'][0]['content']='\n\n'.join([parts[0],*demos[:mid],parts[-1]])
                candidate_count=client.token_count(row['messages'])
                if candidate_count+cap<=window:
                    best=(mid,row['messages'][0]['content'],candidate_count);low=mid+1
                else:high=mid-1
            if best is not None:
                kept,row['messages'][0]['content'],count=best
                removed=len(demos)-kept
        row['preparation_removed_whole_demonstrations']=removed
        row['reference_prompt_tokens']=count
        if count+cap>window: return None,'high_reasoning_budget_exceeds_native_window'
    else:
        row['max_new_tokens']=CAPS[row['domain']]
        row['context_window']=window
    row['id']=digest([original['id'],POLICY,row['messages'],row['max_new_tokens'],window])
    return row,None


def revise(source,destination,url,model,window=131072):
    source=Path(source).resolve();dest=Path(destination).resolve()
    if source==dest or dest.exists(): raise ValueError('Use a new, nonexistent destination directory')
    original=json.loads((source/'manifest.json').read_text())
    client=Client(url,model,{'chat_template_kwargs':{'reasoning_strength':'high'}},timeout=120)
    output={};quarantine=[];counts={}
    for split in ('rl_train','rl_val','main_test'):
        output[split]=[]
        for row in read_jsonl(source/'splits'/(split+'.jsonl')):
            reasons=issues(row)
            if reasons:
                quarantine.append({'id':row['id'],'split':split,'task':row['task'],'reasons':reasons});continue
            revised,reason=revise_row(row,client,window)
            if reason:
                quarantine.append({'id':row['id'],'split':split,'task':row['task'],'reasons':[reason]});continue
            output[split].append(revised)
        counts[split]=dict(collections.Counter(r['task'] for r in output[split]))
        print(json.dumps({'split':split,'rows':len(output[split]),'quarantined':sum(q['split']==split for q in quarantine)}),flush=True)
    # Preserve the entire reserved source pool, including quarantined families.
    # Excluding a benchmark record never makes its family eligible for RL training.
    pool=list(read_jsonl(source/'pools/main_test.jsonl'))
    sets={k:set().union(*(family_keys(r) for r in v)) for k,v in output.items()}
    pool_keys=set().union(*(family_keys(r) for r in pool))
    overlaps={a+'/'+b:len(sets[a]&sets[b]) for a,b in [('rl_train','rl_val'),('rl_train','main_test'),('rl_val','main_test')]}
    overlaps.update({s+'/main_pool':len(sets[s]&pool_keys) for s in ('rl_train','rl_val')})
    if any(overlaps.values()): raise ValueError('Revision introduced cross-split overlap')
    for split,rows in output.items():write_jsonl(dest/'splits'/(split+'.jsonl'),rows)
    write_jsonl(dest/'pools/main_test.jsonl',pool)
    manifest={'schema_version':2,'data_policy':POLICY,'source_manifest_hash':digest(original),
              'source_directory':str(source),'seed':original.get('seed',42),'review_status':'automated_and_AI_review_pending',
              'semantic_refusal_free_certified':False,'main_test_inventory_complete':False,
              'complete_inventory':False,'test_is_selection_set':True,'main_pool':original['main_pool'],
              'splits':{s:{'rows':len(v),'hash':digest(v),'domains':dict(collections.Counter(r['domain'] for r in v))} for s,v in output.items()},
              'task_counts':counts,'exact_family_overlap':overlaps,'quarantine_count':len(quarantine),
              'native_server_context':window,'tokenizer_model':model,
              'note':'Quotas and review must be accepted before release; no silent replacement or full-benchmark certification.'}
    write_json(dest/'manifest.json',manifest);write_json(dest/'quarantine.json',quarantine)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--url',required=True);p.add_argument('--model',required=True);p.add_argument('--context',type=int,default=131072)
    a=p.parse_args();revise(a.source,a.destination,a.url,a.model,a.context)
