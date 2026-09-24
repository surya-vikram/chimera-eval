"""Diagnostic correlations and audit queues, not a reward or a judge-certification score."""
import collections
import json
import math
from pathlib import Path


def correlation(pairs):
    if len(pairs)<3: return None
    x,y=zip(*pairs);mx=sum(x)/len(x);my=sum(y)/len(y)
    a=sum((v-mx)**2 for v in x);b=sum((v-my)**2 for v in y)
    return sum((v-mx)*(w-my) for v,w in pairs)/math.sqrt(a*b) if a*b else None


def audit_records(records, judge_dir=None):
    domains=collections.defaultdict(list); high=[]
    for r in records:
        text='\n'.join(t.get('response',{}).get('text','') for t in r.get('turns',[]))
        words=text.split();grams=[tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
        repetition=1-len(set(grams))/len(grams) if grams else 0
        tokens=sum(t.get('response',{}).get('usage',{}).get('completion_tokens',0) for t in r.get('turns',[]))
        g=r.get('grade',{});valid=g.get('status')=='valid'
        entry={'id':r['id'],'sample':r['sample'],'valid':valid,'score':g.get('score'),
               'completion_tokens':tokens,'repeated_4gram_fraction':repetition,
               'truncated':any(t.get('response',{}).get('finish_reason')=='length' for t in r.get('turns',[]))}
        domains[r['domain']].append(entry)
        if valid and g['score']>=.75: high.append({**entry,'domain':r['domain']})
    report={'domains':{},'high_reward_audit_queue':sorted(high,key=lambda x:(-x['repeated_4gram_fraction'],-x['completion_tokens']))[:100],
            'warning':'Correlations are diagnostics, not proof of bias; audit within comparable tasks.'}
    for domain,items in domains.items():
        valid=[r for r in items if r['valid']]
        report['domains'][domain]={'samples':len(items),'errors':len(items)-len(valid),
          'cap_hits':sum(r['truncated'] for r in items),
          'score_histogram':dict(collections.Counter(str(r['score']) for r in valid)),
          'score_length_correlation':correlation([(r['score'],r['completion_tokens']) for r in valid]),
          'score_repetition_correlation':correlation([(r['score'],r['repeated_4gram_fraction']) for r in valid])}
    attempts=[]
    if judge_dir:
        attempts=[json.loads(p.read_text()) for p in Path(judge_dir).glob('*.json')]
    report['judge']={'attempts':len(attempts),'failed_attempts':sum(r['status']!='valid' for r in attempts),
      'retries':sum(r['attempt']>0 for r in attempts),'fenced_json':sum(r.get('fenced',False) for r in attempts),
      'completion_tokens':sum(r.get('response',{}).get('usage',{}).get('completion_tokens',0) for r in attempts),
      'sum_request_seconds':sum(r.get('response',{}).get('latency',0) for r in attempts)}
    return report
