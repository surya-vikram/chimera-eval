"""Strict validation of the published calendar's final state and supported constraints."""
import json
import re


def minutes(value):
    m = re.fullmatch(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', value.strip().lower())
    if not m:
        raise ValueError('Invalid time')
    h, minute, meridiem = int(m[1]), int(m[2] or 0), m[3]
    if meridiem:
        if not 1 <= h <= 12:
            raise ValueError('Invalid hour')
        h = h % 12 + (12 if meridiem == 'pm' else 0)
    if h > 23 or minute > 59:
        raise ValueError('Invalid time')
    return 60 * h + minute


def verify(text, expected):
    try:
        events = json.loads(text)
        if not isinstance(events, list) or not events or len(events) != len(expected):
            return False
        ids, spans = set(), []
        for event in events:
            ident = str(event['event_id'])
            if ident in ids or ident not in expected:
                return False
            ids.add(ident)
            gold = expected[ident]
            duration = event['duration']
            if type(duration) is not int or duration <= 0 or duration != gold['duration']:
                return False
            if not isinstance(event.get('event_name'), str) or not event['event_name'].strip():
                return False
            start, end = minutes(event['start_time']), minutes(event['start_time']) + duration
            if start < minutes(gold['min_time']) or end > minutes(gold['max_time']):
                return False
            c = gold.get('constraint')
            if c:
                if c.startswith('before ') and end > minutes(c[7:]): return False
                elif c.startswith('after ') and start < minutes(c[6:]): return False
                elif c.startswith('at ') and start != minutes(c[3:]): return False
                elif c.startswith('between '):
                    lower, upper = c[8:].split(' and ')
                    if start < minutes(lower) or end > minutes(upper): return False
                elif not c.startswith(('before ', 'after ', 'at ')):
                    raise RuntimeError('Unsupported admitted calendar constraint')
            spans.append((start, end))
        spans.sort()
        return all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    except (ValueError, TypeError, KeyError):
        return False
