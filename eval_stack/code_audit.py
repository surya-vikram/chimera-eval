"""Check admitted public Python reference solutions in the same isolated grader."""
import argparse
import concurrent.futures
import json
from .common import read_jsonl, write_json
from .graders import Grader


def audit(path, output, limit=0):
    rows = list(read_jsonl(path))
    quarantined = [r for r in rows if r.get('verification',{}).get('task_id') == 'HumanEval/32']
    rows = [r for r in rows if r not in quarantined]
    if limit: rows = rows[:limit]
    grader = Grader(code_timeout=60, code_concurrency=2)
    def check(row):
        p = row['verification']
        candidates = [p['prompt'] + p['canonical_solution']] if row['verifier']=='humanevalplus' else p['reference_solutions'][:3]
        attempts = []
        for code in candidates:
            try: grade = grader.execute_code(row, code)
            except Exception as e: grade = {'status':'error','error':str(e)}
            attempts.append(grade)
            if grade.get('passed'): break
        result = {'id':row['id'],'source_id':row['source']['row_id'],'attempts':attempts,
                  'passed':any(g.get('passed') for g in attempts)}
        print(json.dumps({'source_id':result['source_id'],'passed':result['passed']}),flush=True)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(check, rows))
    report = {'total':len(rows),'passed':sum(r['passed'] for r in results),'results':results,
              'quarantined':[r['verification']['task_id'] for r in quarantined]}
    write_json(output,report)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();audit(a.input,a.output,a.limit)
