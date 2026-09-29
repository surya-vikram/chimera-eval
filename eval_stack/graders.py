"""One task-aware grader for reward and evaluation; native metrics stay explicit."""
from __future__ import annotations

import collections
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid

from .common import env, write_json
from .judging import parse_judgment, judge_messages, protocol_id


class GradingError(RuntimeError):
    pass


def result(score, passed, **components):
    return {"status": "valid", "score": float(score), "passed": passed,
            "components": components, "verifier_version": "chimera-eval/0.2.0"}


def norm_answer(s):
    s = re.sub(r"[^\w\s]", "", s.casefold())
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def answer_metrics(text, aliases):
    pred = norm_answer(text)
    em = int(any(pred == norm_answer(a) for a in aliases))
    f1 = 0.0
    for a in aliases:
        gold = norm_answer(a)
        if pred in ("yes", "no", "noanswer") or gold in ("yes", "no", "noanswer"):
            value = float(pred == gold)
        else:
            overlap = sum((collections.Counter(pred.split()) & collections.Counter(gold.split())).values())
            value = 2 * overlap / (len(pred.split()) + len(gold.split())) if pred or gold else 1.0
        f1 = max(f1, value)
    return em, f1


def extract_final(text, metadata):
    pattern = metadata.get("output_regex")
    if pattern:
        # Dataset regex is pinned trusted metadata, but bound input length.
        import regex
        matches = list(regex.finditer(pattern, text, flags=regex.DOTALL, timeout=1))
        if not matches:
            return ""
        m = matches[-1]
        return (next((g for g in m.groups() if g), m.group(0))).strip()
    matches = re.findall(r"(?:^|\n)\s*(?:Final answer|Answer)\s*:\s*(.+)", text, re.I)
    return matches[-1].strip() if matches else text.strip()


def instruction_checks(text, ids, kwargs, prompt):
    from ifbench import instructions_registry
    checks = []
    if not ids or len(ids) != len(kwargs):
        raise GradingError("Empty or mismatched instruction metadata")
    for ident, kw in zip(ids, kwargs):
        cls = instructions_registry.INSTRUCTION_DICT.get(ident)
        if cls is None:
            raise GradingError(f"Unsupported native instruction: {ident}")
        checker = cls(ident)
        checker.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
        if "prompt" in (checker.get_instruction_args() or []):
            checker.build_description(prompt=prompt)
        checks.append(bool(text.strip()) and bool(checker.check_following(text)))
    return checks


THINK_OPEN, THINK_CLOSE = '<think>', '</think>'


def answer_text(text):
    """The graded answer, or None when there is none. The policy is a non-thinking instruct model
    that still sometimes writes <think>...</think>; the tags must not cost reward. The answer is what
    follows the last closed block. Reasoning that never closes, or closes with nothing after it, is an
    unfinished response: like a truncated one it is masked in training, never rewarded or punished."""
    tail = text.rsplit(THINK_CLOSE, 1)[1] if THINK_CLOSE in text else text
    if THINK_OPEN in tail or (THINK_CLOSE in text and not tail.strip()):
        return None
    return tail


# The two final-answer formats MCQA prompts request (one per row). A correct answer in the
# other explicit format earns partial credit; the strict pass verdict stays False.
MCQA_FORMATS = (r'Answer\s*:\s*(?!Answer)\s*([A-Za-z0-9])\s*', r'\\boxed\{\s*([A-Za-z0-9])\s*\}')
OTHER_FORMAT_CREDIT = 0.5

SCALAR_TYPES = {'string', 'integer', 'number', 'boolean', 'null'}


def parse_structured(candidate, fmt, schema):
    """Parse one candidate in the requested format; CSV cells follow the schema's types."""
    if fmt == "json":
        return json.loads(candidate)
    if fmt == "yaml":
        import yaml
        return yaml.safe_load(candidate)
    if fmt == "xml":
        from defusedxml.ElementTree import fromstring
        return fromstring(candidate)
    if fmt == "toml":
        import tomllib
        return tomllib.loads(candidate)
    import csv
    import io
    rows = list(csv.DictReader(io.StringIO(candidate), strict=True))
    if not rows:
        raise ValueError("empty CSV")
    return csv_value(rows, schema) if schema else rows


def _scalar(prop):
    kinds = prop.get('type')
    kinds = set(kinds if isinstance(kinds, list) else [kinds]) - {None}
    return kinds <= SCALAR_TYPES and 'properties' not in prop and 'items' not in prop


def quarantine_reason(row):
    """Why no response can ever pass this row, or None. Such rows only burn compute and
    deflate eval, so they are excluded before training like other invalid metadata."""
    meta = row.get('verification', {})
    if row['verifier'] != 'structure' or not meta.get('schema'):
        return None
    fmt, schema = meta['format'].lower(), meta['schema']
    if fmt == 'toml' and schema.get('type', 'object') != 'object':
        return 'TOML top level is always a table; schema requires ' + str(schema.get('type'))
    if fmt == 'csv':
        item = schema.get('items', {}) if schema.get('type') == 'array' else schema
        if item.get('type') != 'object' or not all(_scalar(p) for p in item.get('properties', {}).values()):
            return 'CSV cannot express nested or non-scalar schema fields'
    return None


def _parses_json(candidate):
    try:
        json.loads(candidate)
        return True
    except ValueError:
        return False


def csv_value(rows, schema):
    """CSV cells are strings: convert them to the schema's scalar types; one row may be the object."""
    item = schema.get('items', {}) if schema.get('type') == 'array' else schema
    props = item.get('properties', {})
    def convert(value, prop):
        kinds = prop.get('type')
        for kind in (kinds if isinstance(kinds, list) else [kinds]):
            try:
                if kind == 'integer':
                    return int(value)
                if kind == 'number':
                    return float(value)
            except ValueError:
                continue
            if kind == 'boolean' and value.strip().lower() in ('true', 'false'):
                return value.strip().lower() == 'true'
            if kind == 'null' and value.strip().lower() in ('', 'null', 'none'):
                return None
            if kind == 'string':
                return value
        return value
    rows = [{k: convert(v, props.get(k, {})) if isinstance(v, str) else v for k, v in r.items()} for r in rows]
    return rows[0] if schema.get('type') == 'object' and len(rows) == 1 else rows


def structured_candidates(text, fmt):
    """Where the data may be, most final first: fenced blocks (last first), then the whole response,
    then for JSON each value embedded in prose (last first). Explanation around the data is allowed;
    the grader uses the first candidate that parses, so the last answer given is the one that counts."""
    blocks = re.findall(r'```[^\n`]*\n(.*?)\n?```', text, re.S)
    seen = list(reversed(blocks)) + [text]
    if fmt == 'json':
        decoder, spans, i = json.JSONDecoder(), [], 0
        while i < len(text):
            if text[i] in '{[':
                try:
                    _, end = decoder.raw_decode(text, i)
                    spans.append(text[i:end])
                    i = end
                    continue
                except ValueError:
                    pass
            i += 1
        seen += reversed(spans)
    unique = []
    for candidate in seen:
        if candidate.strip() and candidate not in unique:
            unique.append(candidate)
    return unique


CODE_FENCE = re.compile(r"```[ \t]*([\w+.#-]*)[^\n]*\n(.*?)```", re.S)
PYTHON_FENCES = ("", "python", "py", "python3", "py3")


def candidate_code(text, entry_point=None):
    """(program, where it came from). The last Python code block is the answer; when a function is
    requested, the last block that defines it, so a usage example written after it is not run.
    An unfenced response runs as written; an unclosed final fence still counts as the block."""
    blocks = [code for lang, code in CODE_FENCE.findall(text) if lang.casefold() in PYTHON_FENCES]
    if not blocks:
        unclosed = re.match(r"\s*```[ \t]*(?:python3?|py3?)?[ \t]*\n(.*)", text, re.S | re.I)
        return (unclosed.group(1), "unclosed_block") if unclosed else (text, "whole_response")
    if entry_point:
        defines = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+" + re.escape(entry_point) + r"[ \t]*\(", re.M)
        for index in range(len(blocks) - 1, -1, -1):
            if defines.search(blocks[index]):
                return blocks[index], f"block {index + 1} of {len(blocks)}"
    return blocks[-1], f"block {len(blocks)} of {len(blocks)}"


def code_diagnostics(code, entry_point=None):
    """Why a program may have failed, for reports only; the sandbox verdict is the grade."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"syntax_error": f"line {e.lineno}: {e.msg}"}
    if not entry_point:
        return {}
    names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    names |= {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    return {"defines_entry_point": entry_point in names}


# How grade() can reach the judge for a row, ordered from never to every answer.
# none: never. on_miss: only after the deterministic check fails.
# to_pass: deterministic checks can only fail an answer; passing needs the judge.
# always: every non-empty, non-truncated answer. Keep in step with grade().
JUDGE_USES = ('none', 'on_miss', 'to_pass', 'always')


NCHALLENGE_MARKER = "Does the model's final response satisfy this criterion?"


def rubric_question(content):
    """Nemotron multichallenge checks embed the conversation and the expected verdict. The judge gets
    the conversation once, completed with the candidate, and pass_criteria is applied by the grader,
    so neither is repeated in the question (the expected verdict would anchor the judge)."""
    if NCHALLENGE_MARKER not in content:
        return content
    return re.sub(r'\s*Expected answer: (YES|NO)\s*$', '', NCHALLENGE_MARKER + content.split(NCHALLENGE_MARKER, 1)[1])


def judge_use(row):
    kind = row['verifier']
    if kind in ('math', 'choice', 'instruction', 'retrieval', 'calendar', 'humanevalplus', 'apps'):
        return 'none'
    if kind == 'exact':
        return 'on_miss' if row.get('domain') == 'logic' else 'none'
    if kind == 'alias':
        return 'on_miss'
    if kind == 'structure':
        return 'to_pass'
    if kind in ('grounded', 'equivalence', 'quality', 'rubric'):
        return 'always'
    raise GradingError('Unimplemented verifier: ' + kind)


class Grader:
    def __init__(self, judge=None, judge_max_tokens=8192, judge_context=32768,
                 code_image="suryavikram6/chimera-eval:0.1.1", code_timeout=15, code_concurrency=2,
                 judge_audit_dir=None, judge_attempts=3, judge_max_retry_tokens=None):
        self.judge = judge
        self.judge_max_tokens, self.judge_context = judge_max_tokens, judge_context
        self.code_image, self.code_timeout = code_image, code_timeout
        self.code_slots = threading.BoundedSemaphore(code_concurrency)
        self.judge_audit_dir = judge_audit_dir
        self.judge_attempts = judge_attempts
        self.judge_max_retry_tokens = judge_max_retry_tokens or judge_max_tokens
        if judge_attempts < 1 or self.judge_max_retry_tokens < judge_max_tokens:
            raise ValueError('Invalid judge attempt/token limits')

    def judge_json(self, payload, quality=False):
        if self.judge is None:
            raise GradingError("Judge endpoint required")
        messages = judge_messages(payload, quality)
        prompt_tokens = self.judge.token_count(messages)
        if prompt_tokens + self.judge_max_tokens > self.judge_context:
            raise GradingError("Judge context overflow; no truncation permitted")
        errors, audit_ids = [], []
        budget = self.judge_max_tokens
        from pathlib import Path
        for attempt in range(self.judge_attempts):
            ident = uuid.uuid4().hex
            trace = {'id':ident, 'protocol':protocol_id(), 'attempt':attempt,
                     'quality':quality, 'messages':messages, 'max_tokens':budget,
                     'model':getattr(self.judge,'model',None),
                     'sampling':getattr(self.judge,'sampling',None)}
            response = None
            try:
                response = self.judge.complete(messages, budget, response_format={"type":"json_object"})
                trace['response'] = response
                value, fenced = parse_judgment(response, quality)
                trace.update(status='valid', parsed=value.copy(), fenced=fenced)
                value['usage'] = response.get('usage', {})
                value['audit_ids'] = audit_ids + [ident]
                value['judge_protocol'] = protocol_id()
                return value
            except Exception as e:
                trace.update(status='error', error=f'{type(e).__name__}: {e}')
                errors.append(trace['error'])
                if response and response.get('finish_reason') == 'length':
                    next_budget = min(budget * 2, self.judge_max_retry_tokens)
                    if prompt_tokens + next_budget <= self.judge_context:
                        budget = next_budget
                elif response:
                    # Change the retry instruction, not the candidate or its reference.
                    # Repeating a greedy malformed response unchanged is ineffective.
                    schema = ('{"score": <integer 1..5>, "mandatory_pass": <boolean>, "reason": <nonempty string>}'
                              if quality else '{"verdict": <boolean>, "reason": <nonempty string>}')
                    messages = judge_messages(payload, quality) + [{'role':'user', 'content':
                        'Your previous attempt could not be parsed. Re-evaluate the same case and return ONLY '
                        'one JSON object with exactly this schema: ' + schema + '. No commentary or extra keys.'}]
                    prompt_tokens = self.judge.token_count(messages)
                    if prompt_tokens + budget > self.judge_context:
                        raise GradingError('Judge retry context overflow') from e
            finally:
                audit_ids.append(ident)
                if self.judge_audit_dir:
                    write_json(Path(self.judge_audit_dir)/(ident+'.json'), trace)
        raise GradingError("Invalid judge output; " + "; ".join(errors))

    def grade(self, row, response):
        answer = answer_text(response['text'])
        if answer is None and response.get('finish_reason') == 'stop':
            return result(0, False if row.get('binary', True) else None, failure='unfinished_reasoning',
                          incomplete=True, reasoning_tags=True)
        graded = self._grade(row, dict(response, text=response['text'] if answer is None else answer))
        if THINK_OPEN in response['text'] or THINK_CLOSE in response['text']:
            graded = dict(graded, components=dict(graded.get('components', {}), reasoning_tags=True))
        return graded

    def _grade(self, row, response):
        text = response["text"]
        meta = row.get("verification", {})
        kind = row["verifier"]
        if response.get("finish_reason") not in ('stop', 'length'):
            raise GradingError('Unsupported generation finish reason: ' + str(response.get('finish_reason')))
        if response.get('finish_reason') == 'length':
            return result(0, False if row.get('binary', True) else None,
                          failure='candidate_truncated', scoring_policy='fixed_budget_v1')
        if not text.strip():
            return result(0, False if row.get("binary", True) else None, failure="empty_response")
        final = extract_final(text, meta)
        if kind == "math":
            proc = subprocess.run([sys.executable, "-m", "eval_stack.native_math"],
                                  input=json.dumps({"answer": meta["answer"], "response": text,
                                                    "answer_policy":meta.get('answer_policy')}),
                                  text=True, capture_output=True, timeout=15)
            if proc.returncode:
                raise GradingError("Math verifier failed: " + proc.stderr[-500:])
            return result(**json.loads(proc.stdout))
        if kind == "choice":
            if row.get('task') == 'mcqa' and meta.get('output_regex') == r'\boxed\{\s*([A-Za-z0-9])\s*\}':
                raise GradingError('Broken v2 MCQA extraction metadata; use the quality-v3-mcqa-extraction dataset')
            if row.get('task') == 'mcqa' and (str(meta['answer']) not in meta['labels'] or
                    len(set(meta['labels'])) != len(meta['labels']) or len(meta['labels']) < 2):
                raise GradingError('Invalid MCQA gold/options; quarantine instead of scoring the candidate')
            # One explicit final label, never first letter anywhere in reasoning.
            def label(span):
                match = re.fullmatch(r"\s*[([]?([A-Z0-9]+)[)\].]?\s*", span)
                return match.group(1) if match else None
            chosen = label(final)
            if chosen not in meta['labels'] and meta.get('output_regex') in MCQA_FORMATS:
                other = next(f for f in MCQA_FORMATS if f != meta['output_regex'])
                alternative = label(extract_final(text, {'output_regex': other}))
                if alternative in meta['labels']:
                    correct = alternative == str(meta['answer'])
                    return result(OTHER_FORMAT_CREDIT if correct else 0, False, extracted=alternative,
                                  requested_format=False)
            passed = chosen in meta["labels"] and chosen == str(meta["answer"])
            return result(passed, passed, extracted=chosen)
        if kind in ("alias", "grounded"):
            em, f1 = answer_metrics(final, meta["aliases"])
            check = None
            # F1 and literal EM are diagnostics only. A correct explained answer gets full credit.
            # Grounding always checks the entire answer, including claims outside the final span.
            if kind == 'grounded' or not em or text.strip() != final.strip():
                question = ('Is the candidate answer correct and equivalent to one of the reference aliases for this question? '
                            'Allow additional relevant explanation and paraphrases; do not penalize length. '
                            'Reject incompatible alternative answers, contradictions and material factual errors. '
                            'A mention of the reference somewhere in a wrong answer is not sufficient.')
                if kind == 'grounded':
                    question += ' Also require all material answer claims to be supported by the supplied context.'
                check = self.judge_json({'question': question,
                                        'task': meta.get('question', row['messages']),
                                        'reference_aliases': meta['aliases'], 'response': text})
            passed = bool(check['verdict']) if check is not None else bool(em)
            return result(passed, passed, answer_em=em, answer_f1=f1, correctness_judge=check)
        if kind == "exact":
            expected = str(meta["answer"]).strip()
            passed = final.casefold().strip("(). ") == expected.casefold().strip("(). ")
            if not passed and row.get('domain') == 'logic':
                check = self.judge_json({'question': 'Is the final answer correct and equivalent to the reference? Ignore explanation length; reject conflicting final answers or contradictions.',
                                         'task':row['messages'], 'reference':expected, 'response':text})
                return result(check['verdict'], check['verdict'], extracted=final, correctness_judge=check)
            return result(passed, passed, extracted=final)
        if kind == "equivalence":
            if not final:
                return result(0, False, failure="answer_extraction")
            check = self.judge_json({"question": "Is the candidate answer equivalent to the reference for this problem? Reject missing required content and material contradictions.",
                                     "task": row["messages"], "reference": meta["answer"], "response": final})
            return result(check["verdict"], check["verdict"], judge=check)
        if kind == "quality":
            check = self.judge_json({"task": row["messages"], "response": text,
                                     "rubric": meta.get("rubric"), "reference": meta.get("reference", ""),
                                     "essential_checks": meta.get("essential_checks", [])}, quality=True)
            score = (check["score"] - 1) / 4 if check["mandatory_pass"] else 0
            # Quality is not implicitly converted to pass@k.
            return result(score, None, judge=check, acceptability=check["mandatory_pass"] and check["score"] >= 4)
        if kind == "rubric":
            checks = meta["checks"]
            if not checks:
                raise GradingError("Empty rubric")
            verdicts, raw = [], []
            # The judge reads the completed conversation once, the candidate as its final turn.
            conversation = row["messages"] + [{"role": "assistant", "content": text}]
            for check in checks:
                j = self.judge_json({"question": rubric_question(check["content"]), "conversation": conversation})
                passed = j["verdict"] == (check.get("pass_criteria", "YES") == "YES")
                if check.get("source") == "user" and check.get("is_misalignment_check"):
                    passed = not passed
                verdicts.append(passed)
                raw.append(j)
            return result(all(verdicts), all(verdicts), checks=verdicts, judge=raw,
                          fraction_passed=sum(verdicts) / len(verdicts))
        if kind == "instruction":
            checks = instruction_checks(text, meta["ids"], meta["kwargs"], row["messages"][-1]["content"])
            return result(all(checks), all(checks), checks=checks, fraction_passed=sum(checks) / len(checks))
        if kind == "structure":
            import yaml
            import csv
            from xml.etree.ElementTree import ParseError
            from defusedxml.common import DefusedXmlException
            import jsonschema
            fmt, schema = meta["format"].lower(), meta.get("schema")
            if fmt not in ("json", "yaml", "xml", "toml", "csv"):
                raise GradingError("Unsupported structure format: " + fmt)
            syntax_errors = (ValueError, TypeError, yaml.YAMLError, ParseError, DefusedXmlException, csv.Error)
            chosen, obj, error = None, None, 'no structured content'
            for candidate in structured_candidates(text, fmt):
                try:
                    obj = parse_structured(candidate, fmt, schema)
                except syntax_errors as e:
                    error = error if error != 'no structured content' else str(e)
                    continue
                chosen = candidate
                break
            if chosen is None:
                return result(0, False, syntax=False, reason=error)
            if schema:
                try:
                    jsonschema.validate(obj, schema)
                except jsonschema.ValidationError as e:
                    return result(0, False, schema=False, reason=e.message)
            # Explanation around the data is fine: the judge checks the extracted data itself.
            check = self.judge_json({"question": "Does the response satisfy ALL requested structural features and correctly preserve/extract required facts? Check values, not only keys. No unsupported invented content.",
                                     "task": row["messages"], "response": chosen,
                                     "requirements": meta.get("requirements", [])})
            return result(check["verdict"], check["verdict"], syntax=True, content=check,
                          extracted_from_prose=chosen.strip() != text.strip())
        if kind == "retrieval":
            targets = meta["targets"]
            # RULER native recall uses containment; strict all-target uses the same target matching.
            checks = [str(t).casefold() in final.casefold() for t in targets]
            if not checks:
                raise GradingError("Missing targets")
            passed = any(checks) if meta.get('any_alias') else all(checks)
            return result(passed, passed, recall=sum(checks) / len(checks))
        if kind == 'calendar':
            from .calendar import verify
            # The calendar may come with explanation; the last JSON value given is the final calendar.
            final_calendar = next((c for c in structured_candidates(text, 'json') if _parses_json(c)), text)
            passed = verify(final_calendar, meta['calendar'])
            return result(passed, passed, extracted_from_prose=final_calendar.strip() != text.strip())
        if kind in ("humanevalplus", "apps"):
            return self.execute_code(row, text)
        raise GradingError("Unimplemented verifier: " + kind)

    def self_check(self, rows):
        """Run each task's non-judge grading machinery once; {task: error} for tasks that cannot be graded.

        Surfaces missing packages, checker data or the code sandbox before training starts,
        instead of on the first sampled row. The judge is checked separately.
        """
        errors, passed = {}, set()
        for row in rows:
            task, kind, meta = row['task'], row['verifier'], row.get('verification', {})
            if task in errors:
                continue
            keys = [kind] + (['regex'] if meta.get('output_regex') else [])
            if kind == 'instruction':
                keys += [('instruction', ident) for ident in meta['ids']]
            for key in keys:
                if key in passed:
                    continue
                try:
                    self._check(key, row)
                except Exception as e:
                    detail = (str(e).strip().splitlines() or [''])[-1]  # subprocess errors carry a traceback
                    errors[task] = f"{type(e).__name__}: {detail} (row {row['id'][:12]})"
                    break
                passed.add(key)
        return errors

    def _check(self, key, row):
        meta = row['verification']
        if key == 'regex':
            import regex  # noqa: F401  pinned output_regex extraction
        elif key == 'math':
            self.grade(row, {'text': '\\boxed{' + str(meta['answer']) + '}', 'finish_reason': 'stop'})
        elif isinstance(key, tuple):
            index = meta['ids'].index(key[1])
            instruction_checks('x', [key[1]], [meta['kwargs'][index]], row['messages'][-1]['content'])
        elif key == 'structure':
            import yaml, jsonschema, defusedxml.ElementTree  # noqa: F401
        elif key in ('apps', 'humanevalplus'):
            if not shutil.which('docker'):
                raise GradingError('docker CLI not found; the code sandbox cannot run')
            probe = subprocess.run(['docker', 'image', 'inspect', self.code_image], capture_output=True, timeout=30)
            if probe.returncode:
                raise GradingError('Sandbox image not available locally: ' + self.code_image)

    def execute_code(self, row, text):
        code, source = candidate_code(text, row["verification"].get("entry_point"))
        name = 'chimera-code-' + uuid.uuid4().hex
        command = ["docker", "run", "--pull=never", "--name", name, "--rm", "-i", "--network", "none", "--read-only",
                   "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "64",
                   "--memory", "768m", "--cpus", "1", "--user", "65534:65534",
                   "--tmpfs", "/tmp:rw,nosuid,size=128m", "--entrypoint", "python",
                   self.code_image, "-m", "eval_stack.code_worker"]
        with self.code_slots:
            try:
                proc = subprocess.run(command, input=json.dumps({"kind": row["verifier"],
                                      "code": code, "problem": row["verification"],
                                      "timeout": self.code_timeout}), text=True,
                                      capture_output=True, timeout=self.code_timeout + 30)
            except subprocess.TimeoutExpired as e:
                # Docker startup/whole-worker timeout is infrastructure, not candidate failure.
                raise GradingError("Sandbox worker deadline exceeded") from e
            finally:
                subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=30)
        if proc.returncode:
            raise GradingError("Sandbox infrastructure: " + proc.stderr[-500:])
        try:
            data = json.loads(proc.stdout)
        except ValueError as e:
            raise GradingError("Malformed sandbox result") from e
        if data.get("status") != "valid":
            raise GradingError(str(data))
        data.setdefault("components", {})["code_source"] = source
        if not data.get("passed"):
            data["components"].update(code_diagnostics(code, row["verification"].get("entry_point")))
        return data
