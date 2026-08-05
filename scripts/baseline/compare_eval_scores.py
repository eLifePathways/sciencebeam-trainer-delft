"""Compares two eval JSON files against the parity tolerance.

Both are produced by `--eval-output-format=json --eval-output-path=...`, which
is the same code on the TensorFlow and PyTorch branches, so the shapes match.

    python scripts/baseline/compare_eval_scores.py BASELINE.json CANDIDATE.json

Exits non-zero when the candidate is outside the tolerance: micro F1 down by
more than 0.5 points, or any field down by more than 1 point.
"""
import argparse
import json
import sys
from typing import Dict, List, Optional, Tuple


MICRO_F1_TOLERANCE = 0.005
FIELD_F1_TOLERANCE = 0.01


def load_scores(path: str) -> Tuple[float, Dict[str, dict]]:
    with open(path, encoding='utf-8') as fp:
        evaluation = json.load(fp)
    return evaluation['micro_averages']['f1'], evaluation['scores']


def iter_field_rows(
    baseline_scores: Dict[str, dict], candidate_scores: Dict[str, dict]
):
    for field in sorted(set(baseline_scores) | set(candidate_scores)):
        baseline = baseline_scores.get(field, {})
        candidate = candidate_scores.get(field, {})
        yield (
            field,
            baseline.get('f1'),
            candidate.get('f1'),
            baseline.get('support'),
            candidate.get('support')
        )


def format_value(value: Optional[float]) -> str:
    return '-' if value is None else '%.4f' % value


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', help='eval JSON from the reference run')
    parser.add_argument('candidate', help='eval JSON from the run being checked')
    args = parser.parse_args(argv)

    baseline_micro_f1, baseline_scores = load_scores(args.baseline)
    candidate_micro_f1, candidate_scores = load_scores(args.candidate)

    failures: List[str] = []

    print('%-24s %9s %9s %9s' % ('field', 'baseline', 'candidate', 'delta'))
    print('-' * 56)
    for field, baseline_f1, candidate_f1, baseline_support, candidate_support in iter_field_rows(
        baseline_scores, candidate_scores
    ):
        if not baseline_support and not candidate_support:
            # no entities of this type in either run, so nothing to compare
            continue
        if baseline_f1 is None or candidate_f1 is None:
            print('%-24s %9s %9s %9s  MISSING' % (
                field, format_value(baseline_f1), format_value(candidate_f1), '-'
            ))
            failures.append('%s is present in only one of the two runs' % field)
            continue
        delta = candidate_f1 - baseline_f1
        flag = ''
        # a field with no support in the baseline carries no signal
        if baseline_support and -delta > FIELD_F1_TOLERANCE:
            flag = '  BELOW TOLERANCE'
            failures.append(
                '%s down %.4f (support %s)' % (field, -delta, baseline_support)
            )
        print('%-24s %9.4f %9.4f %+9.4f%s' % (
            field, baseline_f1, candidate_f1, delta, flag
        ))

    micro_delta = candidate_micro_f1 - baseline_micro_f1
    print('-' * 56)
    print('%-24s %9.4f %9.4f %+9.4f' % (
        'micro average', baseline_micro_f1, candidate_micro_f1, micro_delta
    ))
    if -micro_delta > MICRO_F1_TOLERANCE:
        failures.append('micro F1 down %.4f' % -micro_delta)

    print()
    if failures:
        print('OUTSIDE TOLERANCE:')
        for failure in failures:
            print('  -', failure)
        return 1
    print('within tolerance (micro %.1f pt, field %.1f pt)' % (
        MICRO_F1_TOLERANCE * 100, FIELD_F1_TOLERANCE * 100
    ))
    return 0


if __name__ == '__main__':
    sys.exit(main())
