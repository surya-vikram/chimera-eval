"""Verify published split hashes, identities, refusal-target screening, and pool isolation."""
import argparse
import json
from pathlib import Path
from .admission import rejection_reason
from .common import digest, read_jsonl, write_json, write_jsonl
from .prepare import family_keys


def audit(root, sanitize_pool=False, output_path=None):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    keys, ids, report = {}, {}, {'splits':{}, 'errors':[]}
    for split in ('rl_train','rl_val','main_test'):
        rows = list(read_jsonl(root/'splits'/(split+'.jsonl')))
        ids[split] = {r['id'] for r in rows}
        keys[split] = set().union(*(family_keys(r) for r in rows)) if rows else set()
        bad = [r['id'] for r in rows if rejection_reason(r)]
        if digest(rows) != manifest['splits'][split]['hash']: report['errors'].append(split+': hash mismatch')
        if len(ids[split]) != len(rows): report['errors'].append(split+': duplicate IDs')
        if bad: report['errors'].append(f'{split}: {len(bad)} refusal/meta-prompt candidates')
        if split != 'main_test' and any(r['domain']=='long_context' for r in rows):
            report['errors'].append(split+': long-context training/validation leakage')
        report['splits'][split] = {'rows':len(rows),'unique_ids':len(ids[split]),'screening_candidates':len(bad)}
    pool = list(read_jsonl(root/'pools/main_test.jsonl'))
    if digest(pool) != manifest['main_pool']['hash']:
        raise ValueError('Reserved pool does not match its manifest')
    rejected = [(r,rejection_reason(r) or ('quarantined_reference' if r.get('verification',{}).get('task_id')=='HumanEval/32' else None)) for r in pool]
    excluded = [(r,why) for r,why in rejected if why]
    if sanitize_pool and excluded:
        # Narrowing an already reserved pool cannot introduce cross-split overlap.
        pool = [r for r,why in rejected if not why]
        from collections import Counter
        write_jsonl(root/'pools/main_test.jsonl',pool)
        manifest['main_pool'].update(rows=len(pool),tasks=dict(Counter(r['task'] for r in pool)),hash=digest(pool))
        manifest['main_pool']['final_screen_exclusions'] = [{'id':r['id'],'reason':why} for r,why in excluded]
        write_json(root/'manifest.json',manifest)
        excluded = []
    if excluded: report['errors'].append(f'main_pool: {len(excluded)} screening/quarantine candidates')
    keys['main_pool'] = set().union(*(family_keys(r) for r in pool)) if pool else set()
    report['overlaps'] = {a+'/'+b: len(keys[a]&keys[b]) for a,b in
                          [('rl_train','rl_val'),('rl_train','main_test'),('rl_val','main_test'),
                           ('rl_train','main_pool'),('rl_val','main_pool')]}
    if any(report['overlaps'].values()): report['errors'].append('Cross-split family overlap')
    report['passed'] = not report['errors']
    report['semantic_certification'] = False
    write_json(Path(output_path) if output_path else root/'split-audit.json',report)
    print(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True)
    p.add_argument('--sanitize-pool',action='store_true',help='Remove screened records from an older reserved pool; never changes the three splits')
    p.add_argument('--output', help='Write the report outside the prepared dataset')
    a=p.parse_args()
    if not audit(a.data_dir,a.sanitize_pool,a.output)['passed']: raise SystemExit(2)
