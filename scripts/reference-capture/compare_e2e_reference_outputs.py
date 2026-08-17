"""Compares two end-to-end reference captures, case by case.

`capture_e2e_reference_outputs.py` goes through this repo's own tag and eval
helpers, so it records whatever the checkout it runs in produces and needs no
TensorFlow. Capture once on a TensorFlow-era commit, once here, and compare:

    python scripts/reference-capture/capture_e2e_reference_outputs.py \\
        --output-path=data/reference/e2e-torch
    python scripts/reference-capture/compare_e2e_reference_outputs.py \\
        data/reference/e2e data/reference/e2e-torch

Identical weights through a faithful architecture give identical output, so
both the tagged XML and the scores are compared exactly. Exits non-zero on any
difference; a difference is something to explain, not to tolerate.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def get_case_ids(capture_path: Path) -> List[str]:
    return sorted(
        path.name for path in capture_path.iterdir()
        if path.is_dir() and (path / 'eval.json').exists()
    )


def get_micro_f1(case_path: Path) -> Optional[float]:
    eval_path = case_path / 'eval.json'
    if not eval_path.exists():
        return None
    return json.loads(eval_path.read_text())['micro_averages']['f1']


def get_tagged_output(case_path: Path) -> Optional[str]:
    tagged_path = case_path / 'tagged_output.xml'
    return tagged_path.read_text(encoding='utf-8') if tagged_path.exists() else None


def compare_case(
    reference_path: Path, candidate_path: Path
) -> Tuple[List[str], str]:
    """Returns the differences for one case, and a short status."""
    differences = []

    reference_f1 = get_micro_f1(reference_path)
    candidate_f1 = get_micro_f1(candidate_path)
    if reference_f1 is None or candidate_f1 is None:
        differences.append('eval.json missing from one of the two captures')
    elif reference_f1 != candidate_f1:
        differences.append(
            'micro f1 %.6f -> %.6f' % (reference_f1, candidate_f1)
        )

    reference_xml = get_tagged_output(reference_path)
    candidate_xml = get_tagged_output(candidate_path)
    if reference_xml is None or candidate_xml is None:
        differences.append('tagged_output.xml missing from one of the two captures')
    elif reference_xml != candidate_xml:
        differences.append('tagged output differs (%d vs %d characters)' % (
            len(reference_xml), len(candidate_xml)
        ))

    status = 'ok' if not differences else 'DIFFERS'
    return differences, '%s  f1 %s' % (
        status,
        '%.4f' % reference_f1 if reference_f1 is not None else '-'
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', help='the recorded capture')
    parser.add_argument('candidate', help='the capture being checked')
    args = parser.parse_args(argv)

    reference_path = Path(args.reference)
    candidate_path = Path(args.candidate)

    reference_cases = get_case_ids(reference_path)
    candidate_cases = get_case_ids(candidate_path)
    if not reference_cases:
        raise FileNotFoundError('no cases found in %s' % reference_path)

    failures = []
    missing = sorted(set(reference_cases) - set(candidate_cases))
    for case_id in missing:
        failures.append('%s is missing from the candidate capture' % case_id)

    for case_id in reference_cases:
        if case_id in missing:
            print('%-52s MISSING' % case_id)
            continue
        differences, status = compare_case(
            reference_path / case_id, candidate_path / case_id
        )
        print('%-52s %s' % (case_id, status))
        for difference in differences:
            print('    %s' % difference)
            failures.append('%s: %s' % (case_id, difference))

    print()
    print('%d case(s) compared' % len(reference_cases))
    if failures:
        print('%d difference(s):' % len(failures))
        for failure in failures:
            print('  -', failure)
        return 1
    print('every case reproduces the recorded output and score exactly')
    return 0


if __name__ == '__main__':
    sys.exit(main())
