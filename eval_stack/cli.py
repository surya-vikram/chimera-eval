import argparse
import sys
from .common import env


def check_report(report):
    if report.get('truncated_samples', 0):
        total = report.get('operations', {}).get('candidate_turns')
        print(f"TRUNCATION DETECTED: {report['truncated_samples']}" + (f' of {total}' if total else '') +
              ' samples hit their response cap and scored 0; compare models at identical caps.', file=sys.stderr)
    if report['infrastructure_errors'] or report['incomplete_prompts']:
        raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description='Prepare frozen splits or evaluate hosted models.')
    p.add_argument('command', choices=['prepare', 'evaluate', 'regrade', 'revise-data'], nargs='?', default='evaluate')
    args = p.parse_args()
    if args.command == 'prepare':
        from .prepare import prepare
        prepare(env('DATA_DIR', 'data'), env('TASKS', '').split(',') if env('TASKS', '') else None)
    elif args.command == 'revise-data':
        if not env('SOURCE_DATA_DIR', ''):
            p.error('revise-data requires SOURCE_DATA_DIR and a new DATA_DIR destination')
        from .data_revision import revise
        revise(env('SOURCE_DATA_DIR','data'),env('DATA_DIR','data-v2'),env('MODEL_URL','http://127.0.0.1:8000/v1'),env('MODEL_NAME','eval-smoke'),env('MODEL_CONTEXT',131072,int))
    elif args.command == 'regrade':
        from .regrade import regrade
        report = regrade()
        check_report(report)
    else:
        from .runner import evaluate
        report = evaluate()
        check_report(report)


if __name__ == '__main__':
    main()
