"""Prompt-first means, equal domain weights, and unbiased binary pass@k."""
import collections
import math
import statistics

from .common import DOMAINS


def pass_at_k(n, c, k):
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("pass@k requires 0 <= correct <= n and 1 <= k <= n")
    return 1.0 if n - c < k else 1 - math.comb(n - c, k) / math.comb(n, k)


def aggregate(rows, records, n, ks, complete_scope=False):
    ks = sorted({1, *ks})  # Always report pass@1 alongside requested pass@k.
    def capped(r):
        return any(t.get('response', {}).get('finish_reason') == 'length' for t in r.get('turns', []))
    by_id = collections.defaultdict(dict)
    for r in records:
        by_id[r['id']][r['sample']] = r
    truncated_samples = sum(
        any(t.get('response', {}).get('finish_reason') == 'length' for t in r.get('turns', []))
        for row in rows for r in by_id[row['id']].values())
    tasks, domains, errors, incomplete = {}, {}, 0, 0
    prompt_values = {}
    for row in rows:
        samples = list(by_id[row['id']].values())
        good = [r for r in samples if r.get('grade', {}).get('status') == 'valid']
        errors += len(samples) - len(good)
        if len(good) != n:
            incomplete += 1
            continue  # never silently treat infrastructure errors as model failures
        t = tasks.setdefault(row['task'], {'domain': row['domain'], 'prompt_scores': [],
                                          'pass': {str(k): [] for k in ks}, 'truncated': 0})
        t['prompt_scores'].append(statistics.mean(0.0 if capped(r) else r['grade']['score'] for r in good))
        prompt_values[row['id']] = t['prompt_scores'][-1]
        t['truncated'] += sum(any(x['response']['finish_reason'] == 'length' for x in r.get('turns', [])) for r in good)
        if row.get('binary', True):
            c = sum(r['grade']['passed'] is True and not capped(r) for r in good)
            for k in ks:
                t['pass'][str(k)].append(pass_at_k(n, c, k))
    for task, t in tasks.items():
        scores = t.pop('prompt_scores')
        t.update(prompts=len(scores), score=statistics.mean(scores),
                 prompt_standard_error=statistics.stdev(scores) / math.sqrt(len(scores)) if len(scores) > 1 else None)
        t['pass'] = {k: statistics.mean(v) if v else None for k, v in t['pass'].items()}
    # Fixed protocol weights, independent of sample count or successful request count.
    fixed = {'gsm8k': .5, 'math500': .5, 'arc': .25, 'mmlu_pro': .25, 'triviaqa': .5,
             'ifeval': .5, 'ifbench': .5, 'multi_if': .5, 'multichallenge': .5}
    planned = collections.Counter(r['task'] for r in rows)
    for domain in DOMAINS:
        names = [t for t in planned if next(r for r in rows if r['task'] == t)['domain'] == domain]
        # Score a domain from its tasks with at least one fully graded prompt; 'complete' says
        # whether every planned prompt of every task counted. Errors are never scored as zero.
        scored = [t for t in names if t in tasks]
        if not scored:
            domains[domain] = None
            continue
        weights = {t: fixed.get(t, 1.) for t in scored}
        if domain == 'long_context':
            for t in scored:
                subtype = t.split('_', 2)[-1]
                weights[t] = .3 if subtype in ('rag', 'icl') else .4 / 10
        total = sum(weights.values())
        domains[domain] = {'score': sum(tasks[t]['score'] * weights[t] for t in scored) / total,
                           'complete': all(t in tasks and tasks[t]['prompts'] == planned[t] for t in names),
                           'task_weights': {t: weights[t]/total for t in scored},
                           'tasks': names, 'prompts': sum(planned[t] for t in names),
                           'scored_prompts': sum(tasks[t]['prompts'] for t in scored),
                           'pass': {str(k): sum(tasks[t]['pass'][str(k)] * weights[t] for t in scored) / total
                                    if all(tasks[t]['pass'][str(k)] is not None for t in scored) else None for k in ks}}
    all_complete = all(d and d['complete'] for d in domains.values())
    full = complete_scope and not incomplete and all_complete
    lengths = {}
    for length in sorted({r['length_bucket'] for r in rows if r.get('length_bucket')}):
        subset = [r for r in rows if r.get('length_bucket') == length]
        names = {r['task'] for r in subset}
        valid = all(t in tasks and tasks[t]['prompts'] == sum(r['task'] == t for r in subset) for t in names)
        weights = {t: .3 if t.split('_',2)[2] in ('rag','icl') else .4/10 for t in names}
        lengths[str(length)] = {'prompts': len(subset), 'score': sum(tasks[t]['score'] * weights[t] for t in names) / sum(weights.values()) if valid else None,
                                'pass': {str(k): sum(tasks[t]['pass'][str(k)] * weights[t] for t in names) / sum(weights.values()) if valid else None for k in ks}}
        actual_counts = [t['response'].get('usage', {}).get('prompt_tokens')
                         for row in subset for r in by_id[row['id']].values() for t in r.get('turns', [])]
        actual_counts = [x for x in actual_counts if isinstance(x, int)]
        lengths[str(length)].update(actual_prompt_tokens_min=min(actual_counts) if actual_counts else None,
                                    actual_prompt_tokens_max=max(actual_counts) if actual_counts else None)
    strata = {}
    for r in rows:
        if r['id'] in prompt_values:
            strata.setdefault(r['task'] + '/' + r.get('stratum','default'), []).append(prompt_values[r['id']])
    strata = {k: {'prompts':len(v), 'score': statistics.mean(v)} for k,v in strata.items()}
    quality_groups = {'writing':[], 'chat':[], 'planning':[], 'roleplay':[]}
    for name, value in strata.items():
        if not name.startswith('biggen/'): continue
        _, capability, task = name.split('/',2)
        group = ('roleplay' if task == 'role_playing' else 'planning' if capability == 'planning' or task == 'replanning'
                 else 'writing' if capability == 'refinement' or task in ('writing_a_speech','education_content_creation','instruction_data_creation') else 'chat')
        quality_groups[group].append(value['score'])
    quality_weights = {'writing':.3,'chat':.3,'planning':.3,'roleplay':.1}
    if domains['quality'] and all(quality_groups.values()):
        domains['quality']['score'] = sum(quality_weights[k]*statistics.mean(v) for k,v in quality_groups.items())
        domains['quality']['subdomain_weights'] = quality_weights
    elif complete_scope and any(quality_groups.values()):
        full = False
    # Always report an aggregate: the equal-weight mean of the domains that have scores.
    # It is the protocol aggregate only when every domain is present and complete.
    scored_domains = [d for d in domains.values() if d]
    aggregate_complete = not incomplete and all_complete
    return {'schema_version': 2, 'valid_full_benchmark': bool(full), 'long_context_by_length': lengths,
            'truncated_samples': truncated_samples,
            'scoring_policy': 'fixed_budget_v1',
            'truncation_free': not bool(truncated_samples),
            'comparison_budget_valid': not bool(incomplete or errors),
            'score_interpretation': 'Fixed-budget performance: capped trajectories score zero and fail pass@k; judge errors are not model failures.',
            'aggregate_score_0_100': 100 * statistics.mean(d['score'] for d in scored_domains) if scored_domains else None,
            # Pass@k over the same domains; quality is continuous and has none.
            'aggregate_pass_at_k': {str(k): 100 * statistics.mean(v) if v else None for k, v in
                                    ((k, [d['pass'][str(k)] for d in scored_domains if d['pass'][str(k)] is not None])
                                     for k in ks)},
            'aggregate_complete': aggregate_complete,
            'aggregate_missing_domains': [k for k, d in domains.items() if not d],
            'aggregate_partial_domains': [k for k, d in domains.items() if d and not d['complete']],
            'aggregation': 'prompt mean -> fixed protocol task weights -> equal weights over scored domains (ten when complete)', 'strata': strata,
            'domains': domains, 'tasks': tasks, 'infrastructure_errors': errors,
            'incomplete_prompts': incomplete, 'n_samples': n, 'pass_k': ks,
            'note': 'Quality has no pass@k. main_test is a model/sampling selection set, not an untouched test.'}
