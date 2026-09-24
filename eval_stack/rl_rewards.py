"""Fail-closed reward boundary; integration into Slime's sampler is a separate stage."""
import math

class InvalidRewardGroup(ValueError):
    pass


def validated_group(records, expected_samples):
    """Caller must replace a rejected whole group; never substitute zeros or partial advantages."""
    if expected_samples < 2 or len(records) != expected_samples:
        raise InvalidRewardGroup('Incomplete rollout group')
    if len({r.get('id') for r in records}) != 1 or None in {r.get('id') for r in records}:
        raise InvalidRewardGroup('Mixed/missing prompt identities')
    if {r.get('sample') for r in records} != set(range(expected_samples)):
        raise InvalidRewardGroup('Missing/duplicate sample indices')
    values = []
    for record in sorted(records, key=lambda r:r['sample']):
        grade = record.get('grade', {})
        score = grade.get('score')
        if grade.get('status') != 'valid' or type(score) not in (int,float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise InvalidRewardGroup('Invalid judgment; retry or replace whole group')
        if grade.get('passed') is False and score != 0:
            raise InvalidRewardGroup('Quality cannot override a failed correctness check')
        values.append(float(score))
    return values
