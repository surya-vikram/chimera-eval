"""Evaluation-only published HELMET/RULER artifacts; no synthetic data generation."""
import itertools
import csv
import io
import json
import os
from pathlib import Path
import random
import tarfile
import urllib.request

from .common import digest, env, family, read_jsonl, write_json, write_jsonl

REVISION = 'dddb209d03e38f1f0faf76d6d05ef4ccf96240ee'
RULER = ('niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1',
         'niah_multivalue', 'niah_multiquery', 'vt', 'cwe', 'fwe', 'qa_1')


def specs():
    lengths = [int(x) for x in env('LONG_CONTEXT_BUCKETS', '4096,8192,16384,32768,65536,131072').split(',')]
    if len(set(lengths)) != len(lengths) or any(n not in (4096,8192,16384,32768,65536,131072) for n in lengths):
        raise ValueError('Invalid/duplicate long-context buckets')
    output = []
    for i, length in enumerate(lengths):
        budget = 436 // len(lengths) + (i < 436 % len(lengths))
        each = int(budget * .6) // len(RULER)
        quotas = [each] * len(RULER) + [(budget - each * len(RULER)) // 2,
                                       (budget - each * len(RULER) + 1) // 2]
        for name, count in zip((*RULER, 'rag', 'icl'), quotas):
            output.append((f'long_{length}_{name}', 'princeton-nlp/HELMET', str(length),
                           'test', 'long', 'long_context', 0, 0, count))
    return output


def extract_published(root):
    """Read selected regular-file members; never extract archive paths/symlinks."""
    root = Path(root)
    directory = root / 'sources' / 'helmet'
    done = directory / 'extracted-v2.json'
    if done.exists():
        return
    archive = directory / 'data.tar.gz'
    if not archive.exists():
        from huggingface_hub import hf_hub_download
        archive = Path(hf_hub_download('princeton-nlp/HELMET', 'data.tar.gz', repo_type='dataset', revision=REVISION))
    keep = {f'data/ruler/{name}/validation_{length}.jsonl' for name in RULER
            for length in (4096,8192,16384,32768,65536,131072)}
    keep |= {f'data/kilt/nq-dev-multikilt_1000_k{k}_dep6.jsonl' for k in (20,50,105,220,440,1000)}
    if (directory / 'extracted-v1.json').exists():
        keep = {'data/kilt/nq-dev-multikilt_1000_k20_dep6.jsonl'}
    written = []
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if member.name not in keep or not member.isfile():
                continue
            destination = directory / (member.name.replace('/', '--'))
            with tar.extractfile(member) as f:
                limit = 1024 if member.name == 'data/kilt/nq-dev-multikilt_1000_k20_dep6.jsonl' else 256
                rows = [json.loads(line) for line in itertools.islice(f, limit)]
            write_jsonl(destination, rows)
            written.append(member.name)
    if set(written) != keep:
        raise ValueError('Missing HELMET members: ' + str(keep - set(written)))
    write_json(done, {'revision': REVISION, 'members': written, 'source_prefix_limit': 256,
                     'nq_k20_prefix_limit':1024})


def ruler_prompt(name, r):
    if name.startswith('niah'):
        plural = name in ('niah_multivalue', 'niah_multiquery')
        return (f"Some special magic {r['type_needle_v']} are hidden within the following text. Make sure to memorize it.\n"
                + r['context'] + f"\nWhat {'are all' if plural else 'is'} the special magic {r['type_needle_v']} for {r['query']} mentioned in the provided text?")
    if name == 'vt':
        return r.get('example', '') + '\nMemorize and track the chains of variable assignment.\n' + r['context'] + f"\nFind all variables assigned the value {r['query']}."
    if name == 'cwe':
        return r.get('example', '') + '\n' + r['context'] + '\nWhat are the 10 most common words in the above list?'
    if name == 'fwe':
        return r['context'] + "\nIgnore the dots. What are the three most frequently appeared coded words? Give only the words."
    return 'Answer based only on the given documents. Give only the short answer.\n\n' + r['context'] + '\nQuestion: ' + r['question']


def prepare_long(root, tokenizer, only=None):
    if tokenizer is None:
        raise ValueError('PREP_TOKENIZER required for honest long-context bucket admission')
    root = Path(root)
    extract_published(root)
    # HELMET ICL profile: published BANKING77 examples; private deterministic label remapping.
    lock = root / 'source-locks' / 'PolyAI--banking77.json'
    revision = '57ec275d8078af65b7731c2a98be812d844a6d6b'
    write_json(lock, {'repo': 'PolyAI-LDN/task-specific-datasets', 'revision': revision, 'license':'CC-BY-4.0'})
    def banking(split):
        cache = root / 'sources' / 'helmet' / f'banking77-{split}.jsonl'
        if cache.exists(): return list(read_jsonl(cache))
        url = f'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{revision}/banking_data/{split}.csv'
        with urllib.request.urlopen(url, timeout=60) as response:
            rows = list(csv.DictReader(io.StringIO(response.read().decode())))
        write_jsonl(cache, rows)
        return rows
    train, test = banking('train'), banking('test')
    label_ids = {name:i for i,name in enumerate(sorted({r['category'] for r in train}))}
    train = [{'text':r['text'], 'label':label_ids[r['category']]} for r in train]
    test = [{'text':r['text'], 'label':label_ids[r['category']]} for r in test]
    test_groups = {}
    for r in test:
        test_groups.setdefault(r['label'], []).append(r)
    for group in test_groups.values():
        group.sort(key=lambda r: digest([42,r['text']]))
    test = [test_groups[k][i] for i in range(max(len(v) for v in test_groups.values()))
            for k in sorted(test_groups) if i < len(test_groups[k])]
    groups = {}
    for r in train:
        groups.setdefault(r['label'], []).append(r)
    directory = root / 'sources' / 'helmet'
    for spec in specs():
        task, _, length, *_ = spec
        length = int(length)
        name = task.split('_', 2)[2]
        if only and name not in only:
            continue
        path = root / 'adapted' / (task + '.jsonl')
        rows, drops, seen = [], {}, set()
        if name in RULER:
            raw = list(read_jsonl(directory / f'data--ruler--{name}--validation_{length}.jsonl'))
        elif name == 'rag':
            k = {4096:20,8192:50,16384:105,32768:220,65536:440,131072:1000}[length]
            raw = list(read_jsonl(directory / f'data--kilt--nq-dev-multikilt_1000_k{k}_dep6.jsonl'))
        else:
            raw = test[:256]
        for i, r in enumerate(raw):
            cap = 256
            if name in RULER:
                prompt = ruler_prompt(name, r)
                targets = r.get('answer', r.get('outputs'))
                verification = {'targets': targets if isinstance(targets, list) else [targets], 'any_alias': name.startswith('qa')}
                family_text = r['question'] if name.startswith('qa') else f"RULER/{name}/{r.get('index', i)}"
            elif name == 'rag':
                # Select whole passages, never truncate a passage or insert a gold answer.
                question = r['question']
                prompt_start = 'Use the supplied documents to answer the question. Give only the short answer.\n\n'
                ending = '\nQuestion: ' + question
                contexts, supported = [], False
                positive = {c['text'] for c in r['positive_ctxs']}
                size = len(tokenizer.encode(prompt_start + ending)) + cap + 64
                for ctx in r['ctxs']:
                    passage = f"Document (Title: {ctx['title']}): {ctx['text']}"
                    size += len(tokenizer.encode(passage + '\n\n'))
                    if size > length:
                        break
                    contexts.append(passage)
                    supported |= ctx['text'] in positive
                if not supported:
                    drops['unanswerable_selected_context'] = drops.get('unanswerable_selected_context', 0) + 1
                    continue
                prompt = prompt_start + '\n\n'.join(contexts) + ending
                verification = {'aliases': r['answers'], 'question': r['question']}
                family_text = question
            else:
                question = r['text']
                rand = random.Random(int(digest(question)[:8], 16) + 42)
                labels = list(range(77)); rand.shuffle(labels)
                pool = {k: rand.sample(v, len(v)) for k, v in groups.items()}
                examples = []
                # Balanced demonstration rounds, without replacement or generated examples.
                for round_index in range(max(len(v) for v in pool.values())):
                    order = list(pool); rand.shuffle(order)
                    examples.extend(pool[k][round_index] for k in order if round_index < len(pool[k]))
                start = 'Use the provided text-to-label mapping. Output only "label: <number>".\n\n'
                ending = '\n\n' + question
                accepted, size = [], len(tokenizer.encode(start + ending)) + cap + 64
                for example in examples:
                    if example['text'].strip().casefold() == question.strip().casefold():
                        continue
                    demo = example['text'] + '\nlabel: ' + str(labels[example['label']])
                    size += len(tokenizer.encode(demo + '\n\n'))
                    if size > length: break
                    accepted.append(demo)
                prompt = start + '\n\n'.join(accepted) + ending
                verification = {'answer': str(labels[r['label']]), 'output_regex': r'label:\s*(\d+)\s*$'}
                family_text = question
            msgs = [{'role': 'user', 'content': prompt}]
            if family(family_text) in seen:
                continue
            count = len(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False))
            if count + cap > length or count < .65 * length:
                drops['outside_bucket'] = drops.get('outside_bucket', 0) + 1
                continue
            row = {'id': digest([task, i, REVISION, prompt]), 'family_id': family(family_text), 'family_text': family_text,
                   'task': task, 'domain': 'long_context', 'stratum': name, 'messages': msgs,
                   'verifier': 'exact' if name == 'icl' else 'alias' if name == 'rag' else 'retrieval', 'verification': verification,
                   'binary': True, 'max_new_tokens': cap, 'context_window': length, 'length_bucket': length,
                   'reference_prompt_tokens': count, 'source': {'repo': 'PolyAI/banking77' if name == 'icl' else 'princeton-nlp/HELMET',
                   'revision': revision if name == 'icl' else REVISION, 'row_id': i, 'profile': 'chimera-adapted-published-helmet-v1'}}
            rows.append(row)
            seen.add(row['family_id'])
            if len(rows) >= max(spec[8] * 3, 40):
                break
        write_jsonl(path, rows)
        write_json(path.with_suffix('.audit.json'), {'task': task, 'eligible': len(rows), 'rejections': drops,
                                                    'status': 'ok', 'evaluation_only': True})
        print(json.dumps({'task': task, 'eligible': len(rows), 'rejections': drops}), flush=True)
