"""Blinded review export and fail-closed calibration reporting using rl_val only."""
import argparse
import collections
import json
import math
from pathlib import Path
from .common import digest, read_jsonl, write_json, write_jsonl


def prepare_inputs(source,output,window=131072):
    from .data_revision import issues,revise_row
    source,out=Path(source),Path(output)
    if out.exists():raise ValueError('Use a new calibration input directory')
    rows=[]
    for original in read_jsonl(source/'splits/rl_val.jsonl'):
        if issues(original):continue
        if original['verifier'] not in ('quality','rubric','alias','grounded','equivalence','structure'):continue
        row,reason=revise_row(original,None,window)
        if reason:raise ValueError(reason)
        rows.append(row)
    write_jsonl(out/'splits/rl_val.jsonl',rows)
    write_json(out/'manifest.json',{'diagnostic':True,'calibration_only':True,'main_test_used':False,
        'source':str(source),'complete_inventory':False,
        'splits':{'rl_val':{'rows':len(rows),'hash':digest(rows)}},
        'note':'Derived subset of existing rl_val, not a fourth production partition.'})


def export_review(data_dir,run_dir,output,limit=400):
    data,run,out=Path(data_dir),Path(run_dir),Path(output)
    if out.exists():raise ValueError('Use a new review directory')
    config=json.loads((run/'config.json').read_text())
    if config['config']['SPLIT']!='rl_val':raise ValueError('Calibration must not tune against main_test')
    rows=list(read_jsonl(data/'splits/rl_val.jsonl'));mapping={r['id']:r for r in rows}
    expected=config.get('data_manifest',{}).get('splits',{}).get('rl_val',{}).get('hash')
    if expected and expected!=digest(rows):raise ValueError('Review data hash mismatch')
    groups=collections.defaultdict(list)
    for path in (run/'samples').glob('*.json'):
        r=json.loads(path.read_text());row=mapping.get(r['id'])
        if not row or row['verifier'] not in ('quality','rubric','alias','grounded','equivalence','structure'):continue
        if not r.get('turns'):continue
        key=digest([r['id'],r['sample'],r['turns'][-1]['response']['text']])
        role='heldout' if int(digest(row['family_id'])[:8],16)%5==0 else 'development'
        case={'case_id':key,'family_id':row['family_id'],'domain':row['domain'],'task':row['task'],
              'calibration_role':role,'messages':row['messages'],'verification':row['verification'],
              'verifier':row['verifier'],'response':r['turns'][-1]['response']['text'],
              'finish_reason':r['turns'][-1]['response']['finish_reason']}
        groups[row['domain']].append((case,r.get('grade',{})))
    for values in groups.values():values.sort(key=lambda v:digest(v[0]['case_id']))
    chosen=[]
    for i in range(max((len(v) for v in groups.values()),default=0)):
        for domain in sorted(groups):
            if i<len(groups[domain]) and len(chosen)<limit:chosen.append(groups[domain][i])
    if not chosen:raise ValueError('No saved judge-route responses to review')
    write_jsonl(out/'blinded_cases.jsonl',[c for c,g in chosen])
    write_jsonl(out/'private_predictions.jsonl',[{'case_id':c['case_id'],'grade':g} for c,g in chosen])
    write_jsonl(out/'reviews.jsonl',[{'case_id':c['case_id'],'status':'pending','reviewer_type':'independent_AI',
          'reviewer_id':None,'expected_pass':None,'expected_score':None,'reference_issue':None,'rationale':None} for c,g in chosen])
    write_json(out/'manifest.json',{'source_run':str(run),'source_fingerprint':config['fingerprint'],
        'cases':len(chosen),'case_hash':digest([c for c,g in chosen]),'main_test_used':False,
        'roles':'family-grouped development/heldout for calibration only; not additional RL data partitions'})


def wilson_upper(errors,n):
    if not n:return None
    z=1.96;p=errors/n
    return (p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def report(directory):
    root=Path(directory);cases=list(read_jsonl(root/'blinded_cases.jsonl'))
    manifest=json.loads((root/'manifest.json').read_text())
    if digest(cases)!=manifest['case_hash']:raise ValueError('Blinded cases changed after export')
    predictions={r['case_id']:r['grade'] for r in read_jsonl(root/'private_predictions.jsonl')}
    labels=list(read_jsonl(root/'reviews.jsonl'))
    if len({r['case_id'] for r in labels})!=len(labels):raise ValueError('Duplicate review labels')
    review={r['case_id']:r for r in labels};domains=collections.defaultdict(list);pending=0
    for case in cases:
        label=review.get(case['case_id'],{})
        if label.get('status')!='reviewed' or not label.get('reviewer_id') or not label.get('rationale'):
            pending+=1;continue
        if label.get('reviewer_type') not in ('human','independent_AI'):raise ValueError('Invalid reviewer type')
        domains[case['domain']].append((case,predictions[case['case_id']],label))
    summary={'pending':pending,'domains':{},'approved_for_rl':False,
      'note':'Independent AI review is not human certification. Passing metrics is necessary, not proof against adaptive reward exploitation.'}
    for domain,items in domains.items():
        d={}
        for role in ('development','heldout'):
            part=[(c,p,l) for c,p,l in items if c['calibration_role']==role]
            invalid=sum(p.get('status')!='valid' for c,p,l in part)
            binary=[(p,l) for c,p,l in part if type(l.get('expected_pass')) is bool]
            negatives=[(p,l) for p,l in binary if not l['expected_pass']]
            positives=[(p,l) for p,l in binary if l['expected_pass']]
            fa=sum(p.get('passed') is True for p,l in negatives)
            fr=sum(p.get('passed') is False for p,l in positives)
            diffs=[abs(p['score']-l['expected_score']) for c,p,l in part if p.get('status')=='valid' and type(l.get('expected_score')) in (int,float)]
            d[role]={'reviewed':len(part),'invalid':invalid,'negatives':len(negatives),'false_accepts':fa,
                     'false_accept_rate':fa/len(negatives) if negatives else None,
                     'false_accept_95pct_upper':wilson_upper(fa,len(negatives)),
                     'positives':len(positives),'false_rejects':fr,
                     'mean_absolute_score_error':sum(diffs)/len(diffs) if diffs else None,
                     'reference_issues':sum(bool(l.get('reference_issue')) for c,p,l in part)}
        summary['domains'][domain]=d
    # This command intentionally cannot self-certify deployment from model-generated labels.
    # A signed release decision must specify domains, protocol/model hashes and outcome tests.
    summary['release_gates']=['per-domain heldout accuracy and false-accept confidence bounds',
        'reviewer/protocol/model provenance checked','reward-vs-objective divergence pilot',
        'explicit deployment decision; never inferred from aggregate agreement']
    write_json(root/'calibration_report.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export');e.add_argument('--data-dir',required=True);e.add_argument('--run-dir',required=True)
    e.add_argument('--output',required=True);e.add_argument('--limit',type=int,default=400)
    r=sub.add_parser('report');r.add_argument('--directory',required=True)
    i=sub.add_parser('prepare-inputs');i.add_argument('--source',required=True);i.add_argument('--output',required=True)
    i.add_argument('--context',type=int,default=131072)
    a=p.parse_args()
    if a.command=='export':export_review(a.data_dir,a.run_dir,a.output,a.limit)
    elif a.command=='prepare-inputs':prepare_inputs(a.source,a.output,a.context)
    else:print(json.dumps(report(a.directory),indent=2))
