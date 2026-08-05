"""Capture per-token reference output from a TensorFlow-trained model.

Records, for a small fixed sample: the tensors fed to the model after
preprocessing, the pre-CRF logits, and the predicted tags. The tensors are
recorded by observing what the tagger passes to the model, so they are what
inference uses rather than a reconstruction.

See scripts/reference-capture/README.md for usage.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.embedding.manager import EmbeddingManager
from sciencebeam_trainer_delft.resources.default_config import DEFAULT_RESOURCE_REGISTRY_FILE

from sciencebeam_trainer_delft.sequence_labelling.tools.grobid_trainer.utils import (
    load_data_and_labels,
    load_delft_model
)


LOGGER = logging.getLogger(__name__)


PRE_CRF_LAYER_NAME = 'dense_ntags'


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Capture per-token reference outputs from a TF-trained model'
    )
    parser.add_argument('--model-path', required=True, help='model directory or tar.gz URL')
    parser.add_argument('--input-path', required=True, help='dataset path or URL')
    parser.add_argument('--output-path', required=True, help='directory to write the capture to')
    parser.add_argument('--model-name', default='header')
    parser.add_argument(
        '--limit', type=int, default=3,
        help='number of documents to capture (kept small deliberately)'
    )
    parser.add_argument(
        '--max-sequence-length', type=int,
        help=(
            'truncate sequences to this length; unset by default, matching the'
            ' library default, so that whole documents are captured'
        )
    )
    parser.add_argument(
        '--input-window-stride', type=int,
        help='capture the sliding-window path, requires --max-sequence-length'
    )
    parser.add_argument('--batch-size', type=int, default=20)
    return parser.parse_args(argv)


def get_keras_model(delft_model):
    keras_model = delft_model.model
    if not hasattr(keras_model, 'get_layer'):
        raise ValueError(f'unsupported model type: {type(keras_model)}')
    return keras_model


def get_pre_crf_model(keras_model):
    from tf_keras.models import Model  # pylint: disable=import-outside-toplevel
    layer_names = [layer.name for layer in keras_model.layers]
    if PRE_CRF_LAYER_NAME not in layer_names:
        raise ValueError(
            f'no {PRE_CRF_LAYER_NAME!r} layer, cannot capture pre-CRF logits'
            f' (layers: {layer_names})'
        )
    return Model(
        inputs=keras_model.inputs,
        outputs=keras_model.get_layer(PRE_CRF_LAYER_NAME).output
    )


def get_input_names(keras_model) -> List[str]:
    return [
        # the tensor name carries a ":0" suffix and sometimes a scope prefix
        tensor.name.split(':')[0].split('/')[-1]
        for tensor in keras_model.inputs
    ]


def get_package_versions() -> Dict[str, str]:
    # delft exposes no __version__, so ask the installed distribution metadata
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


def capture_reference_outputs(args: argparse.Namespace):
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    download_manager = DownloadManager()
    embedding_manager = EmbeddingManager(
        path=DEFAULT_RESOURCE_REGISTRY_FILE,
        download_manager=download_manager
    )

    x_all, y_all, features_all = load_data_and_labels(
        input_paths=[args.input_path],
        limit=args.limit,
        shuffle_input=False,
        download_manager=download_manager
    )
    LOGGER.info('loaded %d documents', len(x_all))

    model = load_delft_model(
        model_name=args.model_name,
        model_path=args.model_path,
        max_sequence_length=args.max_sequence_length,
        input_window_stride=args.input_window_stride,
        batch_size=args.batch_size,
        embedding_manager=embedding_manager
    )
    delft_model = model.model
    keras_model = get_keras_model(delft_model)
    input_names = get_input_names(keras_model)
    LOGGER.info('model inputs: %s', input_names)

    pre_crf_model = get_pre_crf_model(keras_model)

    captured_batches: List[List[np.ndarray]] = []
    original_predict_on_batch = keras_model.predict_on_batch

    def recording_predict_on_batch(inputs):
        captured_batches.append([np.asarray(item) for item in inputs])
        return original_predict_on_batch(inputs)

    # shadows the attribute delft's BaseModel would otherwise delegate
    delft_model.predict_on_batch = recording_predict_on_batch

    tag_result = list(model.iter_tag(
        x_all, output_format=None, features=features_all
    ))
    LOGGER.info('tagged %d documents in %d batch(es)', len(tag_result), len(captured_batches))
    if not captured_batches:
        raise AssertionError('no batches were captured')

    input_arrays = {}
    logit_arrays = {}
    for batch_index, batch_inputs in enumerate(captured_batches):
        for name, array in zip(input_names, batch_inputs):
            input_arrays[f'batch{batch_index:02d}.{name}'] = array
        logits = np.asarray(pre_crf_model.predict_on_batch(batch_inputs))
        logit_arrays[f'batch{batch_index:02d}.logits'] = logits
        LOGGER.info(
            'batch %d: inputs=%s logits=%s',
            batch_index,
            {name: array.shape for name, array in zip(input_names, batch_inputs)},
            logits.shape
        )

    np.savez_compressed(output_path / 'inputs.npz', **input_arrays)
    np.savez_compressed(output_path / 'pre_crf_logits.npz', **logit_arrays)

    tags_json = [
        {
            'tokens': [token for token, _ in document],
            'predicted_tags': [tag for _, tag in document],
            'expected_tags': list(expected_tags)
        }
        for document, expected_tags in zip(tag_result, y_all)
    ]
    (output_path / 'tags.json').write_text(
        json.dumps(tags_json, indent=2) + '\n', encoding='utf-8'
    )

    model_config = vars(model.model_config)
    metadata = {
        'model_path': args.model_path,
        'input_path': args.input_path,
        'limit': args.limit,
        'max_sequence_length': args.max_sequence_length,
        'input_window_stride': args.input_window_stride,
        'batch_size': args.batch_size,
        'package_versions': get_package_versions(),
        'model_config': {
            key: value for key, value in sorted(model_config.items())
            if not key.startswith('_')
        },
        'preprocessor': {
            'vocab_tag': list(model.p.vocab_tag),
            'vocab_char_size': len(model.p.vocab_char),
            'return_features': model.p.return_features,
            'return_casing': model.p.return_casing
        },
        'input_names': input_names,
        'batch_shapes': [
            {name: list(array.shape) for name, array in zip(input_names, batch_inputs)}
            for batch_inputs in captured_batches
        ],
        'document_token_counts': [len(document) for document in tag_result]
    }
    (output_path / 'metadata.json').write_text(
        json.dumps(metadata, indent=2, default=str) + '\n', encoding='utf-8'
    )
    LOGGER.info('written capture to %s', output_path)


def main(argv: Optional[List[str]] = None):
    logging.basicConfig(level='INFO')
    capture_reference_outputs(parse_args(argv))


if __name__ == '__main__':
    main()
