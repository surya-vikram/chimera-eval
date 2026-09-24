"""Math-Verify in a short-lived process: timeout applies even off the main thread."""
import json
import re
import sys
from math_verify import parse, verify

if __name__ == "__main__":
    data = json.load(sys.stdin)
    gold = parse("$" + str(data["answer"]) + "$")
    if not gold:
        raise ValueError("Invalid gold math answer")
    if data.get('answer_policy')=='explicit_final_box_v2' and len(re.findall(r'\\boxed\s*\{',data['response']))!=1:
        print(json.dumps({'score':0,'passed':False,'extracted':'missing or multiple explicit final boxes'}))
        raise SystemExit(0)
    candidate = parse(data["response"])
    passed = bool(verify(gold, candidate))
    print(json.dumps({"score": int(passed), "passed": passed, "extracted": str(candidate)}))
