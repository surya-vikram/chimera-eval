from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

DOMAINS = ("math", "knowledge", "logic", "grounding", "quality", "instruction",
           "multiturn", "structure", "python", "long_context")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def normalized(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", str(text)).casefold()))


def family(text):
    # Preserve operators and brackets: '<' and '(' can define different code/math tasks.
    return digest(' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split()))


def read_jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in records:
            f.write(canonical(row) + "\n")
    tmp.replace(path)


def env(name, default, cast=str):
    return cast(os.environ.get(name, str(default)))


def validate_record(row):
    for key in ("id", "family_id", "task", "domain", "messages", "verifier", "source"):
        if not row.get(key):
            raise ValueError(f"Missing {key}: {row.get('id')}")
    if row["domain"] not in DOMAINS:
        raise ValueError("Unknown domain")
    if row["messages"][-1]["role"] != "user":
        raise ValueError("Candidate input must end with user turn")
    if any(m["role"] not in ("system", "user", "assistant") or not isinstance(m["content"], str)
           for m in row["messages"]):
        raise ValueError("Malformed conversation")
    return row
