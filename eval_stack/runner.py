"""Bounded concurrent endpoint evaluation with durable per-sample resume."""
import concurrent.futures as futures
import copy
import contextlib
import fcntl
import json
import os
from pathlib import Path
import threading
import time

from .client import Client
from .common import canonical, digest, env, read_jsonl, validate_record, write_json
from .graders import Grader, result
from .metrics import aggregate
from .progress import Progress
from .token_budget import TokenBudget, interleave_domains


def context_limit(row, model_context):
    # Versioned v2 long-profile labels describe inputs, not a total runtime window.
    # Tokenizer changes can exceed the preparation model's token count. Respect the
    # actual served limit and never alter the frozen prompt to make it fit.
    if row.get('bucket_semantics', '').startswith('nominal published prompt-length profile'):
        return model_context
    return min(model_context, row.get('context_window', model_context))


def response_budget(row, config):
    # Explicit task/domain caps must not eagerly read an absent row default.
    cap = (config['MAX_NEW_TOKENS'] or config['TASK_MAX_TOKENS'].get(row['task'])
           or config['DOMAIN_MAX_TOKENS'].get(row['domain']) or row.get('max_new_tokens'))
    if type(cap) is not int or cap < 1:
        raise ValueError('No positive response budget for task: ' + row['task'])
    return cap


def settings():
    if env('KV_CACHE_NUM_TOKENS', 0, int):
        raise ValueError('Use MODEL_KV_CACHE_NUM_TOKENS and JUDGE_KV_CACHE_NUM_TOKENS instead of KV_CACHE_NUM_TOKENS')
    token_mode = any(env(role + '_KV_CACHE_NUM_TOKENS', 0, int) > 0 for role in ('MODEL', 'JUDGE'))
    ints = {'MODEL_CONCURRENCY': 4, 'MODEL_CONTEXT': 8192, 'JUDGE_CONCURRENCY': 2,
            'JUDGE_CONTEXT': 32768, 'JUDGE_MAX_TOKENS': 8192, 'SHARED_ENDPOINT_CONCURRENCY': 4,
            'JUDGE_MAX_RETRY_TOKENS': 16384, 'JUDGE_ATTEMPTS': 3,
            'N_SAMPLES': 2, 'SEED': 42, 'REQUEST_TIMEOUT': 180, 'REQUEST_RETRIES': 2,
            'MAX_PENDING': 128 if token_mode else 16,
            'MODEL_KV_CACHE_NUM_TOKENS': 0, 'JUDGE_KV_CACHE_NUM_TOKENS': 0, 'LIMIT_PER_TASK': 0, 'MAX_NEW_TOKENS': 0,
            'CODE_TIMEOUT': 15, 'CODE_CONCURRENCY': 2, 'LONG_CONTEXT_CONCURRENCY': 8}
    c = {k: env(k, v, int) for k, v in ints.items()}
    for role in ('MODEL', 'JUDGE'):
        c[role + '_URL'] = env(role + '_URL', 'http://127.0.0.1:8000/v1')
        c[role + '_NAME'] = env(role + '_NAME', 'eval-smoke')
        sampling = {k.lower(): env(role + '_' + k, default, typ) for k, default, typ in (
            ('TEMPERATURE', .6 if role == 'MODEL' else 0, float), ('TOP_P', .95 if role == 'MODEL' else 1., float),
            ('TOP_K', -1, int), ('REPETITION_PENALTY', 1., float), ('MIN_P', 0., float),
            ('PRESENCE_PENALTY', 0., float), ('FREQUENCY_PENALTY', 0., float))}
        kwargs = env(role + '_CHAT_TEMPLATE_KWARGS', '')
        if kwargs:
            sampling['chat_template_kwargs'] = json.loads(kwargs)
        stop = json.loads(env(role + '_STOP_JSON', '[]'))
        if stop:
            sampling['stop'] = stop
        stop_ids = json.loads(env('MODEL_STOP_TOKEN_IDS_JSON', '[]')) if role == 'MODEL' else []
        if not isinstance(stop_ids, list) or any(type(i) is not int or i < 0 for i in stop_ids):
            raise ValueError('MODEL_STOP_TOKEN_IDS_JSON must be a JSON list of token ids')
        if stop_ids:
            # Model under test only. vLLM/SGLang extension: end generation at any of these token ids,
            # left out of the text. Stop strings cannot, since special tokens are removed before matching.
            sampling['stop_token_ids'] = stop_ids
        c[role + '_SAMPLING'] = sampling
        if role == 'JUDGE':
            c[role + '_SAMPLING']['seed'] = c['SEED']
    c.update(SPLIT=env('SPLIT', 'main_test'), TASKS=env('TASKS', ''), CODE_IMAGE=env('CODE_IMAGE', 'suryavikram6/chimera-eval:0.1.1'),
             REGRADES_IDS=env('REGRADES_IDS',''),
             TASK_SAMPLE_COUNTS=json.loads(env('TASK_SAMPLE_COUNTS_JSON', '{}')),
             EVAL_CONTEXT_BUCKETS=[int(x) for x in env('EVAL_CONTEXT_BUCKETS', '').split(',') if x],
             PASS_K=[int(k) for k in env('PASS_K', '2').split(',')],
             TASK_MAX_TOKENS=json.loads(env('TASK_MAX_TOKENS_JSON', '') or '{}'),
             DOMAIN_MAX_TOKENS=json.loads(env('DOMAIN_MAX_TOKENS_JSON', '') or '{}'))
    for name in ('TASK_MAX_TOKENS', 'DOMAIN_MAX_TOKENS'):
        if not isinstance(c[name], dict) or any(type(v) is not int or v < 1 for v in c[name].values()):
            raise ValueError(name + ' must map names to positive integer token budgets')
    if c['MAX_NEW_TOKENS'] < 0:
        raise ValueError('MAX_NEW_TOKENS must be nonnegative')
    for role in ('MODEL', 'JUDGE'):
        if c[role + '_KV_CACHE_NUM_TOKENS'] < 0:
            raise ValueError(role + '_KV_CACHE_NUM_TOKENS must be nonnegative')
    if c['N_SAMPLES'] < 1 or any(k < 1 or k > c['N_SAMPLES'] for k in c['PASS_K']):
        raise ValueError('PASS_K must be between 1 and N_SAMPLES')
    if any(c[k] < 1 for k in ('MODEL_CONCURRENCY', 'JUDGE_CONCURRENCY', 'MAX_PENDING', 'SHARED_ENDPOINT_CONCURRENCY', 'LONG_CONTEXT_CONCURRENCY')):
        raise ValueError('Concurrency must be positive')
    return c


def evaluate():
    out = Path(env('OUTPUT_DIR', 'outputs')) / env('RUN_NAME', 'evaluation')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _evaluate()


def _evaluate():
    c = settings()
    root = Path(env('DATA_DIR', 'data'))
    rows = [validate_record(r) for r in read_jsonl(root / 'splits' / (c['SPLIT'] + '.jsonl'))]
    if c['SPLIT'] == 'main_test' and len(rows) > min(4000, env('TEST_MAX_ITEMS',4000,int)):
        raise ValueError('Frozen main_test exceeds the configured/hard 4,000-item cap')
    manifest = json.loads((root / 'manifest.json').read_text())
    expected_hash = manifest.get('splits',{}).get(c['SPLIT'],{}).get('hash')
    if expected_hash and digest(rows) != expected_hash:
        raise ValueError('Split content does not match manifest hash; do not evaluate partially published data')
    skipped = [r for r in rows if r.get('length_bucket', 0) > c['MODEL_CONTEXT'] or
               (r.get('length_bucket') and c['EVAL_CONTEXT_BUCKETS'] and r['length_bucket'] not in c['EVAL_CONTEXT_BUCKETS'])]
    skip_ids = {r['id'] for r in skipped}
    rows = [r for r in rows if r['id'] not in skip_ids]
    if c['TASKS']:
        chosen = set(c['TASKS'].split(','))
        if chosen - {r['task'] for r in rows}:
            raise ValueError('Requested task missing from split')
        rows = [r for r in rows if r['task'] in chosen]
    if c['LIMIT_PER_TASK']:
        counts, limited = {}, []
        for r in rows:
            counts[r['task']] = counts.get(r['task'], 0) + 1
            if counts[r['task']] <= c['LIMIT_PER_TASK']:
                limited.append(r)
        rows = limited
    subset_by_count = False
    if c['SPLIT'] == 'main_test' and c['TASK_SAMPLE_COUNTS']:
        limited = []
        for task in sorted({r['task'] for r in rows}):
            group = [r for r in rows if r['task'] == task]
            want = c['TASK_SAMPLE_COUNTS'].get(task, len(group))
            if c['LIMIT_PER_TASK'] and type(want) is int:
                want = min(want, c['LIMIT_PER_TASK'])
            if type(want) is not int or not 0 <= want <= len(group):
                raise ValueError(f'{task}: requested {want}, frozen split has {len(group)}. Reduce count or prepare a new manifest.')
            # Preserve frozen ordering: prefixes are nested regardless of scheduling/concurrency.
            limited.extend(group[:want])
            subset_by_count |= want < len(group)
        rows = limited
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty selection or duplicate record IDs')
    source_hash = digest({p.name: p.read_text() for p in Path(__file__).parent.glob('*.py')})
    fingerprint = digest([c, rows, source_hash])
    out = Path(env('OUTPUT_DIR', 'outputs')) / env('RUN_NAME', 'evaluation')
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Run configuration/data/code changed. Choose a new RUN_NAME; refusing unsafe resume.')
    write_json(config_path, {'fingerprint': fingerprint, 'config': c, 'source_hash': source_hash,
                             'selected_rows': len(rows), 'data_manifest': manifest})
    token_mode = any(c[role + '_KV_CACHE_NUM_TOKENS'] > 0 for role in ('MODEL', 'JUDGE'))
    shared = threading.BoundedSemaphore(c['MAX_PENDING'] if token_mode else c['SHARED_ENDPOINT_CONCURRENCY'])
    budgets = {}
    long_slots = threading.BoundedSemaphore(c['LONG_CONTEXT_CONCURRENCY'])
    def client(role):
        capacity = c[role + '_KV_CACHE_NUM_TOKENS']
        if capacity:
            budgets[role] = TokenBudget(capacity)
        return Client(c[role+'_URL'], c[role+'_NAME'], c[role+'_SAMPLING'],
                      c['MAX_PENDING'] if capacity else c[role+'_CONCURRENCY'],
                      shared=shared if c['MODEL_URL'].rstrip('/') == c['JUDGE_URL'].rstrip('/') else None,
                      timeout=c['REQUEST_TIMEOUT'], retries=c['REQUEST_RETRIES'], token_budget=budgets.get(role))
    model, judge = client('MODEL'), client('JUDGE')
    for endpoint in (model, judge):
        models = endpoint.request('/models')
        if endpoint.model not in [m['id'] for m in models['data']]:
            raise ValueError('Requested model not served: ' + endpoint.model)
    grader = Grader(judge, c['JUDGE_MAX_TOKENS'], c['JUDGE_CONTEXT'], c['CODE_IMAGE'], c['CODE_TIMEOUT'], c['CODE_CONCURRENCY'])
    grader.judge_audit_dir = out/'judge_attempts'
    grader.judge_attempts = c['JUDGE_ATTEMPTS']
    grader.judge_max_retry_tokens = c['JUDGE_MAX_RETRY_TOKENS']
    if c['JUDGE_ATTEMPTS'] < 1 or c['JUDGE_MAX_RETRY_TOKENS'] < c['JUDGE_MAX_TOKENS']:
        raise ValueError('Invalid judge retry configuration')
    samples_dir = out / 'samples'
    samples_dir.mkdir(exist_ok=True)
    records, jobs = [], []
    for row in rows:
        for sample in range(c['N_SAMPLES']):
            p = samples_dir / f"{row['id']}.{sample}.json"
            old = json.loads(p.read_text()) if p.exists() else None
            if old and old.get('grade', {}).get('status') == 'valid':
                records.append(old)
            else:
                jobs.append((row, sample, p, old))

    def work(job):
        row, sample, path, old = job
        record = old or {'id': row['id'], 'task': row['task'], 'domain': row['domain'], 'sample': sample, 'turns': []}
        messages = copy.deepcopy(row['messages'])
        turns = row.get('turns', [None])
        try:
            grades = []
            for turn_index, turn in enumerate(turns):
                current = copy.deepcopy(row)
                if turn is not None:
                    if turn_index:
                        messages.append(turn['message'])
                    current['verification'] = {'ids': turn['ids'], 'kwargs': turn['kwargs']}
                current['messages'] = copy.deepcopy(messages)
                # The messages this turn answers, saved beside its response.
                prompt = copy.deepcopy(messages if not turn_index else [turn['message']])
                cap = response_budget(row, c)
                # Do not silently truncate prompts or reduce the requested completion budget.
                context = context_limit(row, c['MODEL_CONTEXT'])
                if model.token_count(messages) + cap > context:
                    raise ValueError('Candidate context overflow; adjust declared context/cap, never truncate silently')
                if turn_index < len(record['turns']):
                    item = record['turns'][turn_index]
                    if 'prompt' not in item:
                        item = record['turns'][turn_index] = {'prompt': prompt, **item}
                else:
                    seed = (c['SEED'] + int(row['id'][:8], 16) + sample * 1009 + turn_index * 97) % (2**31)
                    # Long prefills share a separate bound so short-task concurrency
                    # can stay high without flooding KV cache with 128K requests.
                    with long_slots if not c['MODEL_KV_CACHE_NUM_TOKENS'] and row.get('length_bucket', 0) >= 16384 else contextlib.nullcontext():
                        response = model.complete(messages, cap, seed=seed)
                    item = {'prompt': prompt, 'response': response}
                    record['turns'].append(item)
                    write_json(path, record)  # persist generation before a judge failure
                if item.get('grade', {}).get('status') != 'valid':
                    grading_started = time.monotonic()
                    item['grade'] = grader.grade(current, item['response'])
                    item['grading_seconds'] = time.monotonic() - grading_started
                    write_json(path, record)
                grades.append(item['grade'])
                if item['response'].get('finish_reason') == 'length':
                    record['termination'] = 'candidate_budget_exhausted'
                    break  # Whole multi-turn trajectory has failed; do not spend on later turns.
                messages.append({'role': 'assistant', 'content': item['response']['text']})
            record['grade'] = grades[0] if len(grades) == 1 else result(
                all(g['passed'] for g in grades), all(g['passed'] for g in grades),
                turn_fraction=sum(g['score'] for g in grades) / len(grades))
        except Exception as e:
            record['grade'] = {'status': 'error', 'error': f'{type(e).__name__}: {e}'}
        write_json(path, record)
        return record

    # Each worker grades its completion immediately. No generation-wide barrier.
    progress = Progress(len(rows) * c['N_SAMPLES'], records)
    with futures.ThreadPoolExecutor(max_workers=c['MAX_PENDING']) as pool:
        iterator = interleave_domains(jobs) if token_mode else iter(jobs)
        pending = {pool.submit(work, job) for job in [next(iterator, None) for _ in range(c['MAX_PENDING'])] if job is not None}
        while pending:
            done, pending = futures.wait(pending, timeout=progress.interval, return_when=futures.FIRST_COMPLETED)
            for f in done:
                record = f.result()
                records.append(record)
                progress.update(record)
                job = next(iterator, None)
                if job is not None:
                    pending.add(pool.submit(work, job))
            progress.tick()
    complete = (not subset_by_count and not skipped and not c['TASKS'] and not c['LIMIT_PER_TASK'] and c['SPLIT'] == 'main_test'
                and manifest.get('main_test_inventory_complete', manifest.get('complete_inventory', False)))
    report = aggregate(rows, records, c['N_SAMPLES'], c['PASS_K'], complete)
    report['kv_token_budgets'] = {role: {'endpoint': c[role + '_URL'].rstrip('/'),
                                               'capacity': b.capacity, 'peak_reserved_tokens': b.peak}
                                  for role, b in budgets.items()}
    report.update(fingerprint=fingerprint, target=c['MODEL_NAME'], judge=c['JUDGE_NAME'],
                  judge_validation='not_certified_by_runner; requires independent calibration',
                  untested_length_buckets=sorted({r['length_bucket'] for r in skipped}),
                  same_model_judge=c['MODEL_NAME'] == c['JUDGE_NAME'],
                  candidate_requests=model.calls, judge_requests=judge.calls)
    generations = [t for r in records for t in r.get('turns', [])]
    report['operations'] = {
        'candidate_turns': len(generations),
        'candidate_prompt_tokens': sum(t['response'].get('usage', {}).get('prompt_tokens', 0) for t in generations),
        'candidate_completion_tokens': sum(t['response'].get('usage', {}).get('completion_tokens', 0) for t in generations),
        'sum_request_seconds': sum(t['response'].get('latency', 0) for t in generations),
        'sum_grading_seconds': sum(t.get('grading_seconds', 0) for t in generations),
        'cap_hit_turns': sum(t['response'].get('finish_reason') == 'length' for t in generations),
        'empty_turns': sum(not t['response']['text'].strip() for t in generations)}
    write_json(out / 'metrics.json', report)
    from .monitoring import audit_records
    write_json(out / 'reliability.json', audit_records(records, out/'judge_attempts'))
    from .run_audit import audit_run, score_tables
    audit = audit_run(out, root, running=False)
    print('\n'.join(['', f"Results: {c['MODEL_NAME']} judged by {c['JUDGE_NAME']} · {len(rows)} prompts × {c['N_SAMPLES']} samples",
                     *score_tables(report, audit, tasks=False),
                     'Per-task scores and why samples scored zero: audit/report.md'
                     + (f" · untested context lengths: {', '.join(map(str, report['untested_length_buckets']))}"
                        if report['untested_length_buckets'] and (not c['TASKS'] or 'long_' in c['TASKS']) else '')]), flush=True)
    return report
