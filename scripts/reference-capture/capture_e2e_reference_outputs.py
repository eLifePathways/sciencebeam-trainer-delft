"""Capture tagged output and eval scores for the end-to-end delft cases.

Cases are read from the regression suite's own YAML so the two stay in step.
Scores are written without the eval report's timestamp, so a re-run is
byte-identical.

See scripts/reference-capture/README.md for usage.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import sciencebeam_trainer_delft.utils.configure_keras  # noqa, pylint: disable=unused-import
# pylint: disable=wrong-import-order, ungrouped-imports

import yaml

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.embedding.manager import EmbeddingManager
from sciencebeam_trainer_delft.resources.default_config import DEFAULT_RESOURCE_REGISTRY_FILE

from sciencebeam_trainer_delft.sequence_labelling.saving import ModelLoader

from sciencebeam_trainer_delft.sequence_labelling.tools.grobid_trainer.utils import (
    load_data_and_labels,
    load_delft_model,
    tag_input
)


LOGGER = logging.getLogger(__name__)


DEFAULT_TEST_DATA_PATH = (
    'tests/e2e/regression/sequence_labelling/tag_using_existing_models_test.yaml'
)

TEST_NAME = 'test_tag_using_existing_model'


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Capture reference output for the end-to-end delft cases'
    )
    parser.add_argument('--test-data-path', default=DEFAULT_TEST_DATA_PATH)
    parser.add_argument('--output-path', required=True)
    parser.add_argument(
        '--tag-limit', type=int, default=1,
        help='documents to tag, matching the regression suite'
    )
    parser.add_argument(
        '--eval-limit', type=int, default=20,
        help='documents to score; fixed rather than large, so the score is reproducible'
    )
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument(
        '--eval-max-sequence-length', type=int,
        help="unset by default, meaning the model's own recorded sequence length is used"
    )
    parser.add_argument('--case-id', action='append', help='capture only these case ids')
    return parser.parse_args(argv)


def to_builtin(value):
    """Convert numpy scalars, which the support counts are, to JSON-writable types."""
    item = getattr(value, 'item', None)
    if item is not None:
        return item()
    raise TypeError(f'unsupported type: {type(value)}')


def get_delft_test_cases(test_data_path: str) -> List[dict]:
    with open(test_data_path, encoding='utf-8') as data_file:
        test_data = yaml.safe_load(data_file)
    return [
        test_case
        for test_case in test_data[TEST_NAME]
        if test_case.get('engine', 'delft') == 'delft'
    ]


def get_package_versions() -> Dict[str, str]:
    from importlib.metadata import (  # pylint: disable=import-outside-toplevel
        PackageNotFoundError,
        version
    )

    def get_version(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return 'not installed'

    return {
        name: get_version(name)
        for name in ['delft', 'tensorflow', 'tf-keras', 'numpy', 'scikit-learn']
    }


def capture_case(
    test_case: dict,
    case_output_path: Path,
    args: argparse.Namespace,
    download_manager: DownloadManager,
    embedding_manager: EmbeddingManager
) -> dict:
    model_path = test_case['model_path']
    input_path = test_case['input_path']
    case_output_path.mkdir(parents=True, exist_ok=True)

    # tagging mirrors the regression suite: same helper, same limit, same format
    tag_input(
        model_name='dummy-model-name',
        model_path=model_path,
        input_paths=[input_path],
        download_manager=download_manager,
        embedding_manager=embedding_manager,
        limit=args.tag_limit,
        batch_size=args.batch_size,
        tag_output_path=str(case_output_path / 'tagged_output.xml'),
        tag_output_format='xml'
    )

    model = load_delft_model(
        model_name='dummy-model-name',
        model_path=model_path,
        max_sequence_length=args.eval_max_sequence_length,
        batch_size=args.batch_size,
        embedding_manager=embedding_manager
    )

    # Scoring uses the sequence length and batch size the model itself records,
    # rather than this tool's defaults: those are what it was trained and
    # published with. Loading overrides the saved batch size, so re-read the
    # config from the downloaded directory. Segmentation documents reach 11k
    # tokens, so an unbounded sequence length here exhausts memory on a batch
    # that production would never build.
    saved_model_config = ModelLoader(
        download_manager=download_manager
    ).load_model_config_from_directory(model.model_path)
    if args.eval_max_sequence_length is None:
        model.max_sequence_length = saved_model_config.max_sequence_length
    model.model_config.batch_size = saved_model_config.batch_size
    LOGGER.info(
        'scoring with max_sequence_length=%s batch_size=%s',
        model.max_sequence_length, model.model_config.batch_size
    )

    x_eval, y_eval, features_eval = load_data_and_labels(
        input_paths=[input_path],
        limit=args.eval_limit,
        shuffle_input=False,
        download_manager=download_manager
    )
    classification_result = model.get_evaluation_result(
        x_eval, y_eval, features=features_eval
    )
    eval_json = {
        'micro_averages': classification_result.micro_averages,
        'scores': classification_result.scores
    }
    (case_output_path / 'eval.json').write_text(
        json.dumps(eval_json, indent=2, default=to_builtin) + '\n', encoding='utf-8'
    )
    LOGGER.info(
        'case %s: micro f1=%.4f over %d sequences',
        test_case['id'], classification_result.micro_averages['f1'], len(x_eval)
    )
    return {
        'id': test_case['id'],
        'model_path': model_path,
        'input_path': input_path,
        'architecture': model.model_config.architecture,
        'eval_sequence_count': len(x_eval),
        'eval_max_sequence_length': model.max_sequence_length,
        'eval_batch_size': model.model_config.batch_size,
        'micro_f1': classification_result.micro_averages['f1']
    }


def capture_e2e_reference_outputs(args: argparse.Namespace):
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    download_manager = DownloadManager()
    embedding_manager = EmbeddingManager(
        path=DEFAULT_RESOURCE_REGISTRY_FILE,
        download_manager=download_manager
    )

    test_cases = get_delft_test_cases(args.test_data_path)
    if args.case_id:
        test_cases = [
            test_case for test_case in test_cases if test_case['id'] in args.case_id
        ]
    LOGGER.info('capturing %d case(s)', len(test_cases))

    summaries = []
    for index, test_case in enumerate(test_cases, 1):
        LOGGER.info('[%d/%d] %s', index, len(test_cases), test_case['id'])
        summaries.append(capture_case(
            test_case,
            output_path / test_case['id'],
            args=args,
            download_manager=download_manager,
            embedding_manager=embedding_manager
        ))

    metadata = {
        'test_data_path': args.test_data_path,
        'tag_limit': args.tag_limit,
        'eval_limit': args.eval_limit,
        'batch_size': args.batch_size,
        'eval_max_sequence_length': args.eval_max_sequence_length,
        'package_versions': get_package_versions(),
        'cases': summaries
    }
    (output_path / 'metadata.json').write_text(
        json.dumps(metadata, indent=2, default=to_builtin) + '\n', encoding='utf-8'
    )
    LOGGER.info('written capture for %d case(s) to %s', len(summaries), output_path)


def main(argv: Optional[List[str]] = None):
    logging.basicConfig(level='INFO')
    capture_e2e_reference_outputs(parse_args(argv))


if __name__ == '__main__':
    main()
