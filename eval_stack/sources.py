"""Published-source adapters. Gold/rubrics never go into candidate payloads."""
import csv
import json
import re
from pathlib import Path

from .common import digest, family, read_jsonl, write_json

BBH = ["boolean_expressions", "date_understanding", "disambiguation_qa", "formal_fallacies",
       "logical_deduction_five_objects", "navigate", "object_counting", "penguins_in_a_table",
       "reasoning_about_colored_objects", "temporal_sequences", "tracking_shuffled_objects_five_objects", "web_of_lies"]

# task, repository, config, split, adapter, domain, train quota, val quota, test quota
SPECS = [
 ("gsm8k_train", "openai/gsm8k", "main", "train", "gsm", "math", 5000, 13, 0),
 ("gsm8k", "openai/gsm8k", "main", "test", "gsm", "math", 0, 0, 400),
 ("math500", "HuggingFaceH4/MATH-500", None, "test", "math", "math", 0, 0, 200),
 ("nemotron_math", "nvidia/Nemotron-RL-Math-v2", None, "train", "nmath", "math", 5000, 13, 0),
 ("mcqa", "nvidia/Nemotron-RL-knowledge-mcqa", None, "train", "mcqa", "knowledge", 9500, 8, 0),
 ("openqa", "nvidia/Nemotron-RL-knowledge-openqa", None, "train", "openqa", "knowledge", 10000, 6, 0),
 ("science", "nvidia/Nemotron-RL-Science-v1", None, "so_openq", "openqa", "knowledge", 4000, 5, 0),
 ("arc", "allenai/ai2_arc", "ARC-Challenge", "test", "arc", "knowledge", 0, 0, 160),
 ("mmlu_pro", "TIGER-Lab/MMLU-Pro", None, "test", "mmlu", "knowledge", 0, 0, 280),
 ("triviaqa", "mandarjoshi/trivia_qa", "rc.nocontext", "validation", "trivia", "knowledge", 0, 0, 200),
 ("hotpot_train", "hotpotqa/hotpot_qa", "distractor", "train", "hotpot", "grounding", 20000, 26, 0),
 ("hotpot", "hotpotqa/hotpot_qa", "distractor", "validation", "hotpot", "grounding", 0, 0, 400),
 ("cascade_chat", "nvidia/Nemotron-Cascade-RL-RLHF", None, "train", "chat", "quality", 12000, 19, 0),
 ("cascade_lists", "nvidia/Nemotron-Cascade-RL-RLHF", None, "train", "lists", "quality", 4000, 3, 0),
 ("cascade_plans", "nvidia/Nemotron-Cascade-RL-RLHF", None, "train", "plans", "quality", 2000, 1, 0),
 ("biggen", "prometheus-eval/BiGGen-Bench", None, "test", "biggen", "quality", 0, 0, 400),
 ("nvidia_multichallenge", "nvidia/Nemotron-RL-Multichallenge-v1", "vanilla", "train", "nchallenge", "multiturn", 750, 7, 0),
 ("nvidia_multichallenge_advanced", "nvidia/Nemotron-RL-Multichallenge-v1", "advanced", "train", "nchallenge", "multiturn", 750, 6, 0),
 ("multichallenge", "ScaleAI/MultiChallenge", None, "test", "challenge", "multiturn", 0, 0, 200),
 ("multi_if", "facebook/Multi-IF", None, "train", "multiif", "multiturn", 0, 0, 200),
 ("nemotron_if", "nvidia/Nemotron-RL-instruction_following", None, "train", "instruction", "instruction", 10000, 10, 0),
 ("ifeval", "google/IFEval", None, "train", "instruction", "instruction", 0, 0, 200),
 ("ifbench", "allenai/IFBench_test", None, "train", "instruction", "instruction", 0, 0, 200),
 ("structured_train", "nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2", None, "train", "nstruct", "structure", 6000, 3, 0),
 ("structeval", "TIGER-Lab/StructEval", None, "test", "struct", "structure", 0, 0, 200),
 ("reasoning_gym", "nvidia/Nemotron-RL-ReasoningGym-v1", None, "train", "reasoning", "logic", 6000, 3, 0),
 ("calendar", "nvidia/Nemotron-RL-Instruction-Following-Calendar-v2", None, "train", "calendar", "logic", 3000, 2, 0),
 ("apps", "codeparrot/apps", None, "train", "apps", "python", 2000, 3, 0),
]
SPECS += [("bbh_" + name, "lukaemon/bbh", name, "test", "bbh", "logic", 0, 0, 30) for name in BBH]
SPECS += [("humanevalplus", "evalplus/humanevalplus", None, "test", "humanevalplus", "python", 0, 0, 163)]


class Rejected(ValueError):
    pass


def messages(row):
    value = row.get("responses_create_params", {}).get("input", row.get("prompt", []))
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list):
        raise Rejected("unsupported_prompt")
    result = []
    for m in value:
        content = m.get("content", "")
        if isinstance(content, list):
            if any(x.get("type") not in ("text", "input_text") for x in content):
                raise Rejected("nontext")
            content = "\n".join(x.get("text", "") for x in content)
        result.append({"role": m.get("role", "user"), "content": content})
    return result


def adapt(spec, row, index, revision):
    task, repo, config, split, kind, domain, *_ = spec
    verifier, verification, binary, stratum, extra = "exact", {}, True, "default", {}
    prompt = row.get("question", row.get("problem", row.get("input", "")))
    msgs = [{"role": "user", "content": prompt}]
    if kind == "gsm":
        verification = {"answer": row["answer"].split("####")[-1].strip()}
        verifier = "math"
    elif kind == "math":
        verifier, verification = "math", {"answer": row["answer"]}
        stratum = str(row.get("subject", "math")) + "/" + str(row.get("level", ""))
    elif kind in ("nmath", "openqa", "mcqa"):
        msgs = messages(row)
        prompt = row.get("question") or row.get("problem") or msgs[-1]["content"]
        if row.get("_hf_question_placeholder") or not row.get("expected_answer"):
            raise Rejected("placeholder_or_missing_gold")
        verification = {"answer": row["expected_answer"],
                        "output_regex": row.get("template_metadata", {}).get("output_regex")}
        if verification['output_regex']:
            # Publisher metadata contains JSON-escaped regex text one level too deep.
            verification['output_regex'] = verification['output_regex'].replace('\\\\', '\\')
        verifier = "math" if kind == "nmath" and "judge" not in row.get("verifier_type", "") else "equivalence"
        if kind == "mcqa":
            verifier = "choice"
            opts = row.get("options", [])
            verification["labels"] = [str(k) for o in opts for k, v in o.items() if v is not None]
            if not verification["labels"]:
                raise Rejected("missing_option_labels")
        stratum = str(row.get("metadata", {}).get("topic", "default"))
    elif kind in ("arc", "mmlu"):
        verifier = "choice"
        if kind == "arc":
            labels, options = row["choices"]["label"], row["choices"]["text"]
            gold = row["answerKey"]
        else:
            options = row["options"]
            labels, gold = list("ABCDEFGHIJ")[:len(options)], row["answer"]
            stratum = row["category"]
        verification = {"answer": gold, "labels": labels}
        msgs = [{"role": "user", "content": prompt + "\n" + "\n".join(f"{a}. {b}" for a, b in zip(labels, options)) +
                 "\nYou may reason. End with 'Final answer: X' where X is one option label."}]
    elif kind == "trivia":
        verifier, verification = "alias", {"aliases": row["answer"]["aliases"]}
        msgs[0]["content"] += "\nEnd with 'Final answer: <short answer>'."
    elif kind == "hotpot":
        context = row["context"]
        evidence = "\n\n".join(title + "\n" + " ".join(sentences) for title, sentences in zip(context["title"], context["sentences"]))
        msgs = [{"role": "user", "content": f"Use only the supplied passages.\n\n{evidence}\n\nQuestion: {prompt}\nEnd with 'Final answer: <short answer>'."}]
        verifier, verification = "grounded", {"aliases": [row["answer"]]}
        stratum = row.get("type", "default")
    elif kind in ("chat", "lists", "plans"):
        msgs = messages(row)
        prompt = msgs[-1]["content"]
        category = str(row.get("category", ""))
        if re.search("safety|identity|refus|abstain", category, re.I):
            raise Rejected("excluded_category")
        route = "lists" if re.search(r"\b(list|enumerate)\b", prompt, re.I) else "plans" if re.search(r"\b(plan|itinerary|schedule)\b", prompt, re.I) else "chat"
        if route != kind:
            raise Rejected("other_quality_route")
        verifier, binary = "quality", False
        verification = {"rubric": {"criteria": "Fulfill the actual request accurately, coherently and completely; obey essential constraints. No redundant repeated list items. Do not reward verbosity."}}
    elif kind == "biggen":
        if row["capability"] in ("safety", "tool_usage", "multilingual", "reasoning", "knowledge"):
            raise Rejected("other_capability")
        if row['task'] in ('revision_with_tools', 'executable_actions', 'alignment', 'false_presupposition'):
            raise Rejected('excluded_tool_or_abstention_task')
        msgs = [{"role": "system", "content": row["system_prompt"]}, {"role": "user", "content": row["input"]}]
        verifier, binary = "quality", False
        verification = {"reference": row["reference_answer"], "rubric": row["score_rubric"]}
        stratum = row["capability"] + "/" + row["task"]
    elif kind in ("nchallenge", "challenge"):
        verifier = "rubric"
        if kind == "nchallenge":
            msgs = messages(row)
            verification = {"checks": row["llm_judge"]}
        else:
            conv = row["conversation"]
            msgs = [{"role": a, "content": b} for a, b in zip(conv["role"], conv["content"])]
            verification = {"checks": [{"content": row["target_question"], "pass_criteria": row["pass_criteria"]}]}
            stratum = row["axis"]
        prompt = "\n".join(m["content"] for m in msgs if m["role"] == "user")
    elif kind == "instruction":
        msgs = messages(row)
        prompt = msgs[-1]["content"]
        verifier = "instruction"
        verification = {"ids": row["instruction_id_list"], "kwargs": row["kwargs"]}
        stratum = verification["ids"][0].split(":")[0]
    elif kind == "multiif":
        if row["language"] != "English":
            raise Rejected("non_english")
        turns = []
        for i in range(1, 4):
            msg = json.loads(row[f"turn_{i}_prompt"])
            kw = json.loads(row[f"turn_{i}_kwargs"])
            kw = [json.loads(k) if isinstance(k, str) else k for k in kw]
            turns.append({"message": msg, "ids": json.loads(row[f"turn_{i}_instruction_id_list"]), "kwargs": kw})
        msgs, verifier = [turns[0]["message"]], "instruction"
        verification = {"ids": turns[0]["ids"], "kwargs": turns[0]["kwargs"]}
        prompt, extra = msgs[0]["content"], {"turns": turns}
    elif kind in ("nstruct", "struct"):
        verifier = "structure"
        if kind == "nstruct":
            msgs = messages(row)
            fmt = row["schema_type"].lower()
            verification = {"schema": json.loads(row["schema_str"]), "format": fmt}
            if fmt == "xml":
                raise Rejected("xml_schema_adapter_pending")
        else:
            if row.get("rendering"):
                raise Rejected("rendered_format")
            fmt = row["output_type"].lower()
            prompt = row["query"]
            msgs = [{"role": "user", "content": prompt}]
            verification = {"format": fmt, "requirements": row.get("raw_output_metric", [])}
        if fmt not in ("json", "yaml", "xml", "csv", "toml"):
            raise Rejected("unsupported_format")
        stratum = fmt
    elif kind == "bbh":
        prompt, verification = row["input"], {"answer": row["target"]}
        msgs = [{"role": "user", "content": prompt + "\nEnd with 'Final answer: <answer>'."}]
    elif kind == "reasoning":
        # Only exact-output tasks whose strict success is actually exact; no false exact checks on alternate solutions.
        subtype = row["metadata"]["source_dataset"]
        if subtype not in ("aiw", "leg_counting", "number_filtering", "string_splitting", "string_synthesis", "acre", "self_reference"):
            raise Rejected("reasoning_task_not_allowlisted")
        msgs = messages(row)
        verification = {"answer": row["answer"], "native_metadata": row["metadata"]}
        stratum = subtype
    elif kind == "calendar":
        msgs = messages(row)
        verifier = "calendar"
        verification = {"calendar": row["exp_cal_state"]}
        if not verification["calendar"]:
            raise Rejected("empty_calendar")
    elif kind == "apps":
        tests = json.loads(row["input_output"])
        if not tests or not tests.get("inputs") or tests.get("fn_name") or row["difficulty"] == "competition":
            raise Rejected("code_scope_or_tests")
        verifier, verification = "apps", {"tests": tests, "reference_solutions": json.loads(row["solutions"])}
        msgs = [{"role": "user", "content": row["question"] + "\nReturn a complete Python program using stdin/stdout in one python code block."}]
        stratum = row["difficulty"]
    elif kind == "humanevalplus":
        verifier, verification = "humanevalplus", row
        prompt = row['prompt']
        msgs = [{'role': 'user', 'content': 'Complete this Python function. Return the complete function (including its signature and any imports) in one python code block.\n\n' + prompt}]
    else:
        raise Rejected("unsupported_adapter")
    if row.get("responses_create_params", {}).get("tools"):
        raise Rejected("tool_dependent")
    if not msgs or msgs[-1]["role"] != "user" or any(not m["content"] for m in msgs):
        raise Rejected("invalid_conversation")
    content = "\n".join(m["content"] for m in msgs)
    refusal_target = r"\b(refuse to answer|refusal|abstain from answering|decline to (?:answer|comply)|should not answer|must not answer|cannot (?:answer|provide|assist)|unable to (?:answer|provide|assist)|insufficient information to answer)\b"
    if re.search(refusal_target, content + json.dumps(verification), re.I):
        raise Rejected("refusal_target_candidate")
    if len(content) > 100000 or len(content) < 8:
        raise Rejected("length_or_empty")
    if kind in ("nstruct", "instruction", "chat", "lists", "plans", "calendar"):
        prompt = content
    rid = str(row.get("uuid", row.get("id", row.get("question_id", row.get("key", row.get("task_id", index))))))
    return {"id": digest([task, repo, revision, config, split, rid]), "family_id": family(prompt),
            "family_text": prompt, "task": task, "domain": domain, "stratum": stratum,
            "messages": msgs, "verifier": verifier, "verification": verification, "binary": binary,
            "max_new_tokens": 4096 if domain in ("math", "logic", "quality", "python") else 2048,
            "source": {"repo": repo, "revision": revision, "config": config, "split": split, "row_id": rid}, **extra}


def source_rows(spec, root):
    from huggingface_hub import HfApi, hf_hub_download
    from datasets import load_dataset
    task, repo, config, split, adapter, *_ = spec
    root = Path(root)
    if adapter == 'humanevalplus':
        from evalplus.data import get_human_eval_plus
        for row in get_human_eval_plus().values():
            yield row, 'HumanEvalPlus-v0.1.10/evalplus-0.3.1'
        return
    lock_path = root / "source-locks" / (repo.replace("/", "--") + ".json")
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
    else:
        pin_path = Path(__file__).resolve().parent.parent / 'source_revisions.json'
        pins = json.loads(pin_path.read_text())
        if repo not in pins:
            raise Rejected('Unpinned source; add an explicit revision before downloading: ' + repo)
        info = HfApi().dataset_info(repo, revision=pins[repo])
        lock = {"repo": repo, "revision": info.sha, "files": [f.rfilename for f in info.siblings],
                "license": (info.card_data or {}).get("license", "unverified")}
        write_json(lock_path, lock)
    rev = lock["revision"]
    files = lock["files"]
    if repo == 'codeparrot/apps':
        paths = [f for f in files if f.endswith('.parquet') and f.startswith('all/train/')]
        if not paths:
            raise Rejected('APPS converted parquet unavailable; no arbitrary dataset script execution')
        ds = load_dataset('parquet', data_files=[f'hf://datasets/{repo}@{rev}/{f}' for f in paths], split='train', streaming=True)
        for row in ds:
            yield row, rev
        return
    raw = [f for f in files if f.endswith(".jsonl") and (not config or config in f)]
    if adapter == 'nstruct':
        import pyarrow.parquet as pq
        for f in files:
            if not f.endswith('.parquet') or 'train-' not in f:
                continue
            p = hf_hub_download(repo, f, repo_type='dataset', revision=rev, cache_dir=str(root/'hf'))
            for batch in pq.ParquetFile(p).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    yield row, rev
        return
    if raw and repo.startswith("nvidia/"):
        for f in raw:
            if "validation" in f and split == "train":
                continue
            path = hf_hub_download(repo, f, repo_type="dataset", revision=rev, cache_dir=str(root / "hf"))
            for row in read_jsonl(path):
                yield row, rev
        return
    dataset = load_dataset(repo, config, split=split, revision=rev, streaming=True, cache_dir=str(root / "hf"))
    for row in dataset:
        yield row, rev
