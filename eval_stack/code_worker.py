"""Run ONLY in the resource-limited no-network container launched by Grader."""
import contextlib
import io
import json
import subprocess
import sys


def main(data):
    problem, code = data["problem"], data["code"]
    if data["kind"] == "humanevalplus":
        from evalplus.gen.util import trusted_exec
        from evalplus.eval import untrusted_check, PASS
        inputs = problem["base_input"] + problem["plus_input"]
        if not inputs:
            raise ValueError("No tests")
        with contextlib.redirect_stdout(io.StringIO()):
            expected, times = trusted_exec(problem["prompt"] + problem["canonical_solution"], inputs,
                                           problem["entry_point"], record_time=True)
            entry = problem['entry_point']
            if entry == 'find_zero':
                raise ValueError('HumanEval/32 quarantined: published reference and native oracle audit failures')
            status, details = untrusted_check("humaneval", code, inputs, entry,
                                              expected, problem.get("atol", 0), times,
                                              fast_check=True, min_time_limit=1, gt_time_limit_factor=4)
        return status == PASS, {"execution_status": status, "tests": len(inputs),
                               'failed_test_indices': [i for i, ok in enumerate(details) if not ok][:10]}
    if data["kind"] == "apps":
        tests = problem["tests"]
        if tests.get("fn_name"):
            raise ValueError("APPS function-call adapter not admitted yet")
        if not tests["inputs"] or len(tests["inputs"]) != len(tests["outputs"]):
            raise ValueError("Invalid APPS tests")
        for inp, expected in zip(tests["inputs"], tests["outputs"]):
            try:
                r = subprocess.run([sys.executable, "-I", "-c", code], input=inp if isinstance(inp, str) else "\n".join(inp),
                                   text=True, capture_output=True, timeout=data["timeout"])
            except subprocess.TimeoutExpired:
                return False, {"execution_status": "timeout"}
            expected = expected if isinstance(expected, str) else "\n".join(expected)
            if r.returncode or r.stdout.split() != expected.split():
                return False, {"execution_status": "wrong_or_runtime_error"}
        return True, {"execution_status": "pass", "tests": len(tests["inputs"])}
    raise ValueError("Unsupported code task")


if __name__ == "__main__":
    try:
        passed, components = main(json.load(sys.stdin))
        print(json.dumps({"status": "valid", "score": int(passed), "passed": passed, "components": components}))
    except Exception as e:
        print(json.dumps({"status": "infrastructure_error", "error": str(e)}))
