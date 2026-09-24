"""One task-aware grader for reward and evaluation; native metrics stay explicit."""
from __future__ import annotations

import collections
import json
import os
import re
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


class Grader:
    def __init__(self, judge=None, judge_max_tokens=8192, judge_context=32768,
                 code_image="chimera-eval:0.1.0", code_timeout=15, code_concurrency=2,
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
            # One explicit final label, never first letter anywhere in reasoning.
            match = re.fullmatch(r"\s*[([]?([A-Z0-9]+)[)\].]?\s*", final)
            chosen = match.group(1) if match else None
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
            for check in checks:
                j = self.judge_json({"question": check["content"], "task": row["messages"], "response": text})
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
            fmt = meta["format"].lower()
            fenced = re.fullmatch(r'\s*```(?:json|yaml|yml|xml|csv|toml)?\s*\n(.*?)\n```\s*', text, re.S)
            if fenced:
                text = fenced.group(1)
            try:
                if fmt == "json":
                    obj = json.loads(text)
                elif fmt == "yaml":
                    import yaml
                    obj = yaml.safe_load(text)
                elif fmt == "xml":
                    from defusedxml.ElementTree import fromstring
                    obj = fromstring(text)
                elif fmt == "toml":
                    import tomllib
                    obj = tomllib.loads(text)
                elif fmt == "csv":
                    import csv, io
                    obj = list(csv.DictReader(io.StringIO(text), strict=True))
                    if not obj:
                        raise ValueError("empty CSV")
                else:
                    raise GradingError("Unsupported structure format: " + fmt)
                if meta.get("schema"):
                    import jsonschema
                    try:
                        jsonschema.validate(obj, meta["schema"])
                    except jsonschema.ValidationError as e:
                        return result(0, False, schema=False, reason=e.message)
            except (ValueError, TypeError, yaml.YAMLError, ParseError, DefusedXmlException, csv.Error) as e:
                return result(0, False, syntax=False, reason=str(e))
            check = self.judge_json({"question": "Does the response satisfy ALL requested structural features and correctly preserve/extract required facts? Check values, not only keys. No unsupported invented content.",
                                     "task": row["messages"], "response": text,
                                     "requirements": meta.get("requirements", [])})
            return result(check["verdict"], check["verdict"], syntax=True, content=check)
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
            passed = verify(text, meta['calendar'])
            return result(passed, passed)
        if kind in ("humanevalplus", "apps"):
            return self.execute_code(row, text)
        raise GradingError("Unimplemented verifier: " + kind)

    def execute_code(self, row, text):
        code_blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
        code = code_blocks[-1] if code_blocks else text
        name = 'chimera-code-' + uuid.uuid4().hex
        command = ["docker", "run", "--name", name, "--rm", "-i", "--network", "none", "--read-only",
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
        return data
