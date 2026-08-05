"""Runs the header parity baseline and writes the record for it.

Trains the same configuration on each seed, on the TensorFlow checkout and on
this one, and reports what requirement 1 asks for: per-field scores, seconds
per epoch, the epoch the best score was reached at, and the epoch training
stopped at. Run-to-run spread comes first, since the parity tolerance is only
meaningful once it is known.

    python scripts/baseline/run_header_baseline.py \\
        --tf-venv ../../.venv --seeds 42 43

Omit --tf-venv to run only this checkout. Timings compare only when both sides
run on the same machine.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional


LOGGER = logging.getLogger(__name__)

HEADER_TRAIN_URL = (
    'https://github.com/elifesciences/sciencebeam-datasets/releases/download'
    '/grobid-0.8.2/delft-grobid-0.8.2-header.train.gz'
)

# the published header model's configuration, from its saved config
TRAIN_ARGS = [
    '--batch-size=20',
    '--no-embedding',
    '--max-sequence-length=100',
    '--architecture=CustomBidLSTM_CRF',
    '--use-features',
    '--features-indices=9-30',
    '--features-embedding-size=0',
    '--features-lstm-units=0',
    '--word-lstm-units=200',
    '--early-stopping-patience=10'
]

REPORTED_PACKAGES = [
    'delft', 'torch', 'tensorflow', 'tf-keras', 'numpy', 'scikit-learn'
]

MICRO_F1_TOLERANCE = 0.005
FIELD_F1_TOLERANCE = 0.01


def get_package_versions(python_path: str) -> Dict[str, str]:
    script = (
        'import importlib.metadata as md, json;'
        'names = %r;'
        'print(json.dumps({'
        ' n: (md.version(n) if _found(n) else None) for n in names'
        '}))' % REPORTED_PACKAGES
    )
    helper = (
        'import importlib.metadata as md\n'
        'def _found(name):\n'
        '    try:\n'
        '        md.version(name)\n'
        '        return True\n'
        '    except Exception:\n'
        '        return False\n'
    )
    result = subprocess.run(
        [python_path, '-c', helper + script],
        capture_output=True, text=True, check=False
    )
    if result.returncode:
        LOGGER.warning('could not read package versions: %s', result.stderr.strip())
        return {}
    return json.loads(result.stdout)


def get_checkpoint_epochs(checkpoint_directory: Path) -> List[dict]:
    """Returns each checkpoint's epoch, score and timestamp, oldest first.

    Both frameworks write this, so it is the one per-epoch record that does
    not depend on how the trainer logs.
    """
    checkpoints_path = checkpoint_directory / 'checkpoints.json'
    if not checkpoints_path.exists():
        return []
    checkpoints = json.loads(checkpoints_path.read_text())['checkpoints']
    epochs = []
    for checkpoint in sorted(checkpoints, key=lambda c: c['epoch']):
        meta_path = Path(checkpoint['path']) / 'meta.json'
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        epochs.append({
            'epoch': checkpoint['epoch'],
            'timestamp': checkpoint.get('timestamp'),
            # the Keras path records this epoch's f1; the torch path records
            # the running best, which is enough to find where it last improved
            'f1': meta.get('f1'),
            'best': (meta.get('early_stopping') or {}).get('best')
        })
    return epochs


def get_best_epoch(epochs: List[dict]) -> Optional[int]:
    best_epoch = None
    best_score = None
    for entry in epochs:
        score = entry['f1'] if entry['f1'] is not None else entry['best']
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = entry['epoch']
    return best_epoch


def get_seconds_per_epoch(epochs: List[dict], total_seconds: float) -> Optional[float]:
    timestamps = [
        datetime.fromisoformat(entry['timestamp'])
        for entry in epochs if entry.get('timestamp')
    ]
    if len(timestamps) >= 2:
        deltas = [
            (later - earlier).total_seconds()
            for earlier, later in zip(timestamps, timestamps[1:])
        ]
        return mean(deltas)
    if epochs:
        return total_seconds / len(epochs)
    return None


def run_training(
    python_path: str,
    label: str,
    seed: int,
    output_directory: Path,
    max_epoch: int,
    limit: Optional[int],
    use_legacy_keras: bool
) -> dict:
    run_directory = output_directory / ('%s-seed%d' % (label, seed))
    run_directory.mkdir(parents=True, exist_ok=True)
    eval_path = run_directory / 'eval.json'
    checkpoint_directory = run_directory / 'checkpoints'

    command = [
        python_path, '-m',
        'sciencebeam_trainer_delft.sequence_labelling.grobid_trainer',
        'header', 'train_eval',
        *TRAIN_ARGS,
        '--input=%s' % HEADER_TRAIN_URL,
        '--max-epoch=%d' % max_epoch,
        '--random-seed=%d' % seed,
        '--eval-output-format=json',
        '--eval-output-path=%s' % eval_path,
        '--checkpoint=%s' % checkpoint_directory,
        '--output=%s' % (run_directory / 'models')
    ]
    if limit:
        command.append('--limit=%d' % limit)

    environment = dict(os.environ)
    if use_legacy_keras:
        environment['TF_USE_LEGACY_KERAS'] = '1'

    LOGGER.info('running %s seed %d', label, seed)
    log_path = run_directory / 'train.log'
    started = time.monotonic()
    with open(log_path, 'w', encoding='utf-8') as log_fp:
        result = subprocess.run(
            command, stdout=log_fp, stderr=subprocess.STDOUT,
            env=environment, check=False
        )
    total_seconds = time.monotonic() - started
    if result.returncode:
        raise RuntimeError(
            '%s seed %d failed (exit %d), see %s'
            % (label, seed, result.returncode, log_path)
        )

    evaluation = json.loads(eval_path.read_text())
    epochs = get_checkpoint_epochs(checkpoint_directory)
    summary = {
        'label': label,
        'seed': seed,
        'command': ' '.join(command),
        'micro_f1': evaluation['micro_averages']['f1'],
        'scores': evaluation['scores'],
        'total_seconds': round(total_seconds, 1),
        'epochs_run': len(epochs),
        'stopped_epoch': epochs[-1]['epoch'] if epochs else None,
        'best_epoch': get_best_epoch(epochs),
        'seconds_per_epoch': get_seconds_per_epoch(epochs, total_seconds),
        'package_versions': get_package_versions(python_path)
    }
    (run_directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    LOGGER.info(
        '%s seed %d: micro f1=%.4f, best epoch=%s, %s epochs',
        label, seed, summary['micro_f1'], summary['best_epoch'],
        summary['epochs_run']
    )
    return summary


def format_run_table(summaries: List[dict]) -> List[str]:
    lines = [
        '| run | micro F1 | best epoch | stopped at | s/epoch |',
        '| --- | --- | --- | --- | --- |'
    ]
    for summary in summaries:
        seconds_per_epoch = summary['seconds_per_epoch']
        lines.append('| %s seed %d | %.4f | %s | %s | %s |' % (
            summary['label'], summary['seed'], summary['micro_f1'],
            summary['best_epoch'], summary['stopped_epoch'],
            '%.1f' % seconds_per_epoch if seconds_per_epoch else '-'
        ))
    return lines


def format_variance(summaries: List[dict], label: str) -> List[str]:
    of_label = [s for s in summaries if s['label'] == label]
    if len(of_label) < 2:
        return ['%s: one run only, so the spread is unknown' % label]
    scores = [s['micro_f1'] for s in of_label]
    return ['%s: micro F1 spread across seeds is %.4f (%s)' % (
        label, max(scores) - min(scores),
        ', '.join('%.4f' % score for score in scores)
    )]


def format_comparison(summaries: List[dict]) -> List[str]:
    baseline_runs = [s for s in summaries if s['label'] == 'tf']
    candidate_runs = [s for s in summaries if s['label'] == 'torch']
    if not baseline_runs or not candidate_runs:
        return [
            'No comparison: a baseline needs runs on both the TensorFlow and'
            ' the PyTorch checkout.'
        ]
    baseline_f1 = mean(s['micro_f1'] for s in baseline_runs)
    candidate_f1 = mean(s['micro_f1'] for s in candidate_runs)
    delta = candidate_f1 - baseline_f1
    lines = [
        'Mean micro F1: TensorFlow %.4f, PyTorch %.4f (%+.4f).' % (
            baseline_f1, candidate_f1, delta
        )
    ]

    fields_below = []
    baseline_scores = baseline_runs[0]['scores']
    candidate_scores = candidate_runs[0]['scores']
    for field, baseline in baseline_scores.items():
        candidate = candidate_scores.get(field)
        if not candidate or not baseline.get('support'):
            continue
        field_delta = candidate['f1'] - baseline['f1']
        if -field_delta > FIELD_F1_TOLERANCE:
            fields_below.append('%s (%+.4f)' % (field, field_delta))

    if -delta > MICRO_F1_TOLERANCE or fields_below:
        lines.append('')
        lines.append('**Outside the tolerance.** The spec blocks on this.')
        if -delta > MICRO_F1_TOLERANCE:
            lines.append('- micro F1 is down more than %.1f points'
                         % (MICRO_F1_TOLERANCE * 100))
        for field in fields_below:
            lines.append('- %s is down more than %.1f point'
                         % (field, FIELD_F1_TOLERANCE * 100))
    else:
        lines.append('')
        lines.append('Within the tolerance: micro F1 within %.1f points and no'
                     ' field down more than %.1f point.'
                     % (MICRO_F1_TOLERANCE * 100, FIELD_F1_TOLERANCE * 100))

    baseline_epochs = [s['best_epoch'] for s in baseline_runs if s['best_epoch']]
    candidate_epochs = [s['best_epoch'] for s in candidate_runs if s['best_epoch']]
    if baseline_epochs and candidate_epochs:
        lines.append('')
        lines.append(
            'Epochs to best: TensorFlow %.1f, PyTorch %.1f. A schedule that'
            ' converges more slowly shows up here rather than in the score.'
            % (mean(baseline_epochs), mean(candidate_epochs))
        )
    return lines


def write_note(note_path: Path, summaries: List[dict], max_epoch: int, limit):
    versions = {}
    for summary in summaries:
        versions.setdefault(summary['label'], summary['package_versions'])
    lines = [
        '# Header model parity baseline',
        '',
        'Produced by `scripts/baseline/run_header_baseline.py`.',
        '',
        '## Configuration',
        '',
        '- dataset: `%s`' % HEADER_TRAIN_URL,
        '- documents: %s' % ('all' if not limit else limit),
        '- max epoch: %d' % max_epoch,
        '- flags: `%s`' % ' '.join(TRAIN_ARGS),
        ''
    ]
    for label, package_versions in versions.items():
        installed = ', '.join(
            '%s %s' % (name, version)
            for name, version in package_versions.items() if version
        )
        lines.append('- %s: %s' % (label, installed))
    lines.extend(['', '## Runs', ''])
    lines.extend(format_run_table(summaries))
    lines.extend(['', '## Run-to-run spread', ''])
    for label in ['tf', 'torch']:
        if any(s['label'] == label for s in summaries):
            lines.extend('- %s' % line for line in format_variance(summaries, label))
    lines.extend(['', '## Comparison', ''])
    lines.extend(format_comparison(summaries))
    lines.append('')
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text('\n'.join(lines), encoding='utf-8')
    LOGGER.info('wrote %s', note_path)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--tf-venv',
        help='path to the venv of the TensorFlow checkout; omitted runs only this one'
    )
    parser.add_argument('--torch-venv', default='.venv', help='this checkout venv')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    parser.add_argument('--max-epoch', type=int, default=50)
    parser.add_argument(
        '--limit', type=int,
        help='documents to use; the whole dataset by default'
    )
    parser.add_argument('--output-dir', default='./data/baseline')
    parser.add_argument('--note-path', default='.project-notes/header-baseline.md')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level='INFO', format='%(levelname)s %(message)s')
    args = parse_args(argv)
    output_directory = Path(args.output_dir)

    runs = []
    if args.tf_venv:
        runs.append(('tf', str(Path(args.tf_venv) / 'bin' / 'python'), True))
    runs.append(('torch', str(Path(args.torch_venv) / 'bin' / 'python'), False))

    summaries = []
    for label, python_path, use_legacy_keras in runs:
        if not Path(python_path).exists():
            raise FileNotFoundError('no interpreter at %s' % python_path)
        for seed in args.seeds:
            summaries.append(run_training(
                python_path=python_path,
                label=label,
                seed=seed,
                output_directory=output_directory,
                max_epoch=args.max_epoch,
                limit=args.limit,
                use_legacy_keras=use_legacy_keras
            ))

    (output_directory / 'summaries.json').write_text(json.dumps(summaries, indent=2))
    write_note(Path(args.note_path), summaries, args.max_epoch, args.limit)
    print()
    print('\n'.join(format_run_table(summaries)))
    print()
    print('\n'.join(format_comparison(summaries)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
