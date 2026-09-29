"""Versioned regrading of existing generations. Never invokes the target endpoint."""
import concurrent.futures as futures
import copy
import fcntl
import json
from pathlib import Path
from .client import Client
from .common import digest, env, read_jsonl, write_json
from .graders import Grader, result
from .judging import protocol_id
from .metrics import aggregate
from .monitoring import audit_records
from .progress import Progress
from .run_audit import score_tables
from .runner import settings
from .token_budget import TokenBudget


def regrade():
    source=Path(env('REGRADES_SOURCE','')).resolve()
    if not (source/'config.json').is_file(): raise ValueError('REGRADES_SOURCE must name an existing run')
    out=(Path(env('OUTPUT_DIR','outputs'))/env('RUN_NAME','regrade')).resolve()
    if source==out: raise ValueError('Regrade must never overwrite source run')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _regrade(source,out)


def _regrade(source,out):
    c=settings();original=json.loads((source/'config.json').read_text())
    split=original['config']['SPLIT']
    rows=list(read_jsonl(Path(env('DATA_DIR','data'))/'splits'/(split+'.jsonl')))
    expected=original.get('data_manifest',{}).get('splits',{}).get(split,{}).get('hash')
    if expected and digest(rows)!=expected: raise ValueError('Use the original frozen source data for regrading')
    rowmap={r['id']:r for r in rows}
    paths=sorted((source/'samples').glob('*.json'))
    tasks=set(c['TASKS'].split(',')) if c['TASKS'] else None
    paths=[p for p in paths if p.name.split('.')[0] in rowmap and (not tasks or rowmap[p.name.split('.')[0]]['task'] in tasks)]
    if c['REGRADES_IDS']:
        ids=set(c['REGRADES_IDS'].split(','));paths=[p for p in paths if p.name.split('.')[0] in ids]
    if c['LIMIT_PER_TASK']:
        selected=set();counts={}
        for row in rows:
            if counts.get(row['task'],0)<c['LIMIT_PER_TASK']:
                selected.add(row['id']);counts[row['task']]=counts.get(row['task'],0)+1
        paths=[p for p in paths if p.name.split('.')[0] in selected]
    source_hash=digest({p.name:p.read_text() for p in Path(__file__).parent.glob('*.py')})
    fingerprint=digest([c,original['fingerprint'],source_hash,[(p.name,digest(json.loads(p.read_text()))) for p in paths]])
    config_path=out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())['fingerprint']!=fingerprint:
        raise ValueError('Regrade fingerprint changed; choose new RUN_NAME')
    write_json(config_path,{'fingerprint':fingerprint,'config':c,'source_fingerprint':original['fingerprint'],
                          'source_run':str(source),'source_hash':source_hash,'judge_protocol':protocol_id(),
                          'target_generation_config':original['config'],'no_new_target_generations':True})
    capacity=c['JUDGE_KV_CACHE_NUM_TOKENS']
    judge=Client(c['JUDGE_URL'],c['JUDGE_NAME'],c['JUDGE_SAMPLING'],c['MAX_PENDING'] if capacity else c['JUDGE_CONCURRENCY'],
                 timeout=c['REQUEST_TIMEOUT'],retries=c['REQUEST_RETRIES'],
                 token_budget=TokenBudget(capacity) if capacity else None)
    grader=Grader(judge,c['JUDGE_MAX_TOKENS'],c['JUDGE_CONTEXT'],c['CODE_IMAGE'],c['CODE_TIMEOUT'],c['CODE_CONCURRENCY'],
                  out/'judge_attempts',c['JUDGE_ATTEMPTS'],c['JUDGE_MAX_RETRY_TOKENS'])
    def work(path):
        dest=out/'samples'/path.name
        if dest.exists():
            cached=json.loads(dest.read_text())
            if cached.get('grade',{}).get('status')=='valid': return cached
        record=json.loads(path.read_text());row=rowmap[record['id']]
        record['regrade_provenance']={'source_run':str(source),'source_fingerprint':original['fingerprint']}
        messages=copy.deepcopy(row['messages']);grades=[]
        try:
            turns=row.get('turns',[None])
            saved=record.get('turns',[])
            capped_end=bool(saved and saved[-1]['response'].get('finish_reason')=='length')
            if len(saved)!=len(turns) and not (capped_end and len(saved)<len(turns)):
                raise ValueError('Incomplete saved trajectory; no target regeneration allowed')
            for i,(turn,item) in enumerate(zip(turns,record['turns'])):
                current=copy.deepcopy(row)
                if turn is not None:
                    if i: messages.append(turn['message'])
                    current['verification']={'ids':turn['ids'],'kwargs':turn['kwargs']}
                current['messages']=copy.deepcopy(messages)
                item['grade']=grader.grade(current,item['response']);grades.append(item['grade'])
                if item['response'].get('finish_reason')=='length':break
                messages.append({'role':'assistant','content':item['response']['text']})
            record['grade']=grades[0] if len(grades)==1 else result(all(g['passed'] for g in grades),all(g['passed'] for g in grades),turn_fraction=sum(g['score'] for g in grades)/len(grades))
        except Exception as e: record['grade']={'status':'error','error':f'{type(e).__name__}: {e}'}
        write_json(dest,record);return record
    records=[];progress=Progress(len(paths))
    with futures.ThreadPoolExecutor(max_workers=c['MAX_PENDING']) as pool:
        for record in pool.map(work,paths):
            records.append(record);progress.update(record)
    chosen=[r for r in rows if r['id'] in {x['id'] for x in records}]
    n=original['config']['N_SAMPLES'];ks=original['config']['PASS_K']
    report=aggregate(chosen,records,n,ks,False)
    report.update(regrade=True,no_new_target_generations=True,source_run=str(source),
                  source_target_sampling=original['config'].get('MODEL_SAMPLING'),judge_protocol=protocol_id(),
                  judge_validation='requires independent calibration; no automatic certification')
    write_json(out/'metrics.json',report);write_json(out/'reliability.json',audit_records(records,out/'judge_attempts'))
    print('\n'.join(['',*score_tables(report,tasks=False)]),flush=True)
    return report
