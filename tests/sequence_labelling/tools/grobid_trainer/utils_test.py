import logging
import os
from pathlib import Path
from unittest.mock import call, patch, MagicMock
from typing import Iterator, List, Optional, cast

from typing_extensions import TypedDict

import pytest

import numpy as np

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.saving import ModelLoader

from sciencebeam_trainer_delft.sequence_labelling.wrapper import (
    get_model_directory
)
from sciencebeam_trainer_delft.sequence_labelling.transfer_learning import (
    TransferLearningConfig
)

import sciencebeam_trainer_delft.sequence_labelling.tools.grobid_trainer.utils as utils_module
from sciencebeam_trainer_delft.sequence_labelling.tools.grobid_trainer.utils import (
    set_random_seeds,
    load_data_and_labels,
    train,
    train_eval,
    tag_input,
    wapiti_train,
    wapiti_train_eval,
    wapiti_eval_model,
    wapiti_tag_input,
)

from sciencebeam_trainer_delft.sequence_labelling.reader import load_data_and_labels_crf_file
from sciencebeam_trainer_delft.sequence_labelling.tag_formatter import TagOutputFormats

from sciencebeam_trainer_delft.sequence_labelling.data_generator import (
    NBSP
)

from ....embedding.test_data import TEST_DATA_PATH
from ....test_utils import log_on_exception
from ...tagger_test import (
    TOKEN_1, TOKEN_2,
    UNROLLED_TOKEN_1, UNROLLED_TOKEN_2, UNROLLED_TOKEN_3, UNROLLED_TOKEN_4,
    TAG_1, TAG_2
)


LOGGER = logging.getLogger(__name__)


EMBEDDING_NAME_1 = 'embedding1'

EMBEDDING_1 = {
    "name": EMBEDDING_NAME_1,
    "path": os.path.join(TEST_DATA_PATH, 'embedding1.txt'),
    "type": "glove",
    "format": "vec",
    "lang": "en",
    "item": "word"
}


RESOURCE_REGISTRY_1: dict = {
    'embedding-lmdb-path': None,
    'embeddings': [EMBEDDING_1],
    'embeddings-contextualized': []
}


MODEL_NAME_1 = 'model1'

INPUT_PATH_1 = '/path/to/dataset1'
INPUT_PATH_2 = '/path/to/dataset2'

FEATURE_INDICES_1 = [9, 10, 11]

FEATURES_EMBEDDING_SIZE_1 = 13


@pytest.fixture(name='embedding_registry_path')
def _embedding_registry_path():
    return 'delft/resources-registry.json'


@pytest.fixture(name='embedding_manager_load_mock', autouse=True)
def _embedding_manager_load_mock() -> Iterator[MagicMock]:
    with patch(
        'sciencebeam_trainer_delft.embedding.EmbeddingManager._load'
    ) as mock:
        mock.return_value = RESOURCE_REGISTRY_1
        yield mock


@pytest.fixture(name='load_resource_registry_mock', autouse=True)
def _load_resource_registry_mock() -> Iterator[MagicMock]:
    with patch('delft.utilities.Embeddings.load_resource_registry') as mock:
        mock.return_value = RESOURCE_REGISTRY_1
        yield mock


@pytest.fixture(name='seq_wrapper_load_resource_registry_mock', autouse=True)
def _seq_wrapper_load_resource_registry_mock(
    load_resource_registry_mock: MagicMock
) -> Iterator[MagicMock]:
    with patch(
        'delft.sequenceLabelling.wrapper.load_resource_registry',
        load_resource_registry_mock
    ) as mock:
        yield mock


@pytest.fixture(name='text_models_load_resource_registry_mock', autouse=True)
def _text_models_load_resource_registry_mock(
    load_resource_registry_mock: MagicMock
) -> Iterator[MagicMock]:
    with patch(
        'delft.textClassification.models.load_resource_registry',
        load_resource_registry_mock
    ) as mock:
        yield mock


@pytest.fixture(name='text_wrapper_load_resource_registry_mock', autouse=True)
def _text_wrapper_load_resource_registry_mock(
    load_resource_registry_mock: MagicMock
) -> Iterator[MagicMock]:
    with patch(
        'delft.textClassification.wrapper.load_resource_registry',
        load_resource_registry_mock
    ) as mock:
        yield mock


@pytest.fixture(name='wrapper_load_resource_registry_mock', autouse=True)
def _wrapper_load_resource_registry_mock(
    load_resource_registry_mock: MagicMock
) -> Iterator[MagicMock]:
    with patch(
        'delft.textClassification.wrapper.load_resource_registry',
        load_resource_registry_mock
    ) as mock:
        yield mock


# @pytest.fixture(autouse=True)
# def _embedding_class(embedding_registry_path: Path):
#     embedding_class_with_defaults = partial(Embeddings, path=str(embedding_registry_path))
#     target = 'delft.sequenceLabelling.wrapper.Embeddings'
#     with patch(target, new=embedding_class_with_defaults) as mock:
#         yield mock


@pytest.fixture(name='trainer_class_mock')
def _trainer_class_mock():
    with patch('sciencebeam_trainer_delft.sequence_labelling.wrapper.Trainer') as mock:
        yield mock


@pytest.fixture(name='trainer_mock')
def _trainer_mock(trainer_class_mock: MagicMock):
    return trainer_class_mock.return_value


@pytest.fixture(name='get_default_training_data_mock')
def _get_default_training_data_mock():
    with patch.object(utils_module, 'get_default_training_data') as mock:
        yield mock


@pytest.fixture(name='load_data_and_labels_crf_file_mock')
def _load_data_and_labels_crf_file_mock():
    with patch.object(utils_module, 'load_data_and_labels_crf_file') as mock:
        mock.return_value = (MagicMock(), MagicMock(), MagicMock())
        yield mock


def get_mock_shuffled_array(a: np.ndarray) -> np.ndarray:
    # reverse array
    return a[::-1]


def _mock_shuffle_array(a: np.ndarray):
    a[:] = get_mock_shuffled_array(a)


def _mock_shuffle_arrays(arrays: List[np.ndarray], **_) -> None:
    for a in arrays:
        _mock_shuffle_array(a)


def get_mock_shuffle_arrays(arrays: List[np.ndarray], **_) -> List[np.ndarray]:
    return [
        get_mock_shuffled_array(a)
        for a in arrays
    ]


@pytest.fixture(name='shuffle_arrays_mock')
def _shuffle_arrays_mock():
    with patch.object(utils_module, 'shuffle_arrays') as mock:
        mock.side_effect = _mock_shuffle_arrays
        yield mock


@pytest.fixture(name='download_manager_mock')
def _download_manager_mock():
    mock = MagicMock(name='download_manager_mock')
    mock.download_if_url.side_effect = str
    return mock


@pytest.fixture(name='model_base_path')
def _model_base_path(temp_dir: Path):
    p = temp_dir.joinpath('models')
    p.mkdir(parents=True, exist_ok=True)
    return p


class DefaultArgsDict(TypedDict):
    model_name: str
    embeddings_name: Optional[str]
    input_paths: List[str]
    download_manager: Optional[DownloadManager]
    output_path: Optional[str]
    architecture: str
    word_lstm_units: int
    features_indices: Optional[List[int]]
    embedding_registry_path: Optional[str]


class TrainArgsDict(DefaultArgsDict):
    pass


@pytest.fixture(name='default_args')
def _default_args(
    sample_train_file: str,
    model_base_path: Path,
    embedding_registry_path: Path,
    download_manager_mock: MagicMock
) -> DefaultArgsDict:
    return dict(
        model_name='header',
        embeddings_name=EMBEDDING_NAME_1,
        input_paths=[sample_train_file],
        download_manager=download_manager_mock,
        output_path=str(model_base_path),
        architecture='CustomBidLSTM_CRF',
        word_lstm_units=11,
        features_indices=list(range(7, 1 + 10)),
        embedding_registry_path=str(embedding_registry_path)
    )


@pytest.fixture(name='default_model_directory')
def _default_model_directory(default_args: dict):
    return get_model_directory(
        model_name=default_args['model_name'],
        dir_path=default_args['output_path']
    )


def load_model_config(model_path: str) -> ModelConfig:
    LOGGER.debug('model_path: %s', model_path)
    return ModelLoader().load_model_config_from_directory(model_path)


class TestGrobidTrainerUtils:
    class TestSetRandomSeeds:
        def test_should_not_fail(self):
            set_random_seeds(123)

    @pytest.mark.usefixtures(
        'get_default_training_data_mock', 'load_data_and_labels_crf_file_mock'
    )
    class TestLoadDataAndLabels:
        def test_should_load_single_input_without_limit(
                self,
                download_manager_mock: MagicMock):
            load_data_and_labels(
                input_paths=[INPUT_PATH_1],
                download_manager=download_manager_mock
            )
            download_manager_mock.download_if_url.assert_called_with(
                INPUT_PATH_1
            )

        def test_should_load_single_input_with_limit(
                self,
                load_data_and_labels_crf_file_mock: MagicMock,
                download_manager_mock: MagicMock):
            load_data_and_labels(
                input_paths=[INPUT_PATH_1],
                limit=123,
                download_manager=download_manager_mock
            )
            load_data_and_labels_crf_file_mock.assert_called_with(
                INPUT_PATH_1,
                limit=123
            )

        def test_should_load_multiple_inputs_with_limit(
                self,
                load_data_and_labels_crf_file_mock: MagicMock,
                download_manager_mock: MagicMock):
            download_manager_mock.download_if_url.side_effect = lambda input_path: input_path
            load_data_and_labels_crf_file_mock.side_effect = [
                (np.array([['x1']]), np.array([['y1']]), np.array([[['f1']]])),
                (np.array([['x2']]), np.array([['y2']]), np.array([[['f2']]]))
            ]
            x, y, f = load_data_and_labels(
                input_paths=[INPUT_PATH_1, INPUT_PATH_2],
                limit=123,
                download_manager=download_manager_mock
            )
            assert (x == np.array([['x1'], ['x2']])).all()
            assert (y == np.array([['y1'], ['y2']])).all()
            assert (f == np.array([[['f1']], [['f2']]])).all()
            load_data_and_labels_crf_file_mock.assert_has_calls(
                [call(INPUT_PATH_1, limit=123), call(INPUT_PATH_2, limit=123)]
            )

        def test_should_shuffle_single_source_input_data(
                self,
                load_data_and_labels_crf_file_mock: MagicMock,
                shuffle_arrays_mock: MagicMock,
                download_manager_mock: MagicMock):
            download_manager_mock.download_if_url.side_effect = lambda input_path: input_path
            x_unshuffled = np.array([['x1'], ['x2']])
            y_unshuffled = np.array([['y1'], ['y2']])
            f_unshuffled = np.array([[['f1']], [['f2']]])
            load_data_and_labels_crf_file_mock.side_effect = [
                (x_unshuffled, y_unshuffled, f_unshuffled)
            ]
            x, y, f = load_data_and_labels(
                input_paths=[INPUT_PATH_1],
                limit=123,
                shuffle_input=True,
                random_seed=424,
                download_manager=download_manager_mock
            )
            shuffle_arrays_mock.assert_called()
            assert shuffle_arrays_mock.call_args[1] == {'random_seed': 424}
            input_arrays = shuffle_arrays_mock.call_args[0][0]
            assert (input_arrays[0] == x_unshuffled).all()
            assert (input_arrays[1] == y_unshuffled).all()
            assert (input_arrays[2] == f_unshuffled).all()
            assert (x == get_mock_shuffle_arrays(x.tolist())).all()
            assert (y == get_mock_shuffle_arrays(y.tolist())).all()
            assert (f == get_mock_shuffle_arrays(f.tolist())).all()

    @pytest.mark.slow
    @pytest.mark.very_slow
    class TestEndToEnd:
        @log_on_exception
        def test_should_be_able_to_train_without_features(
                self, default_args: dict, default_model_directory: str):
            train(
                use_features=False,
                **default_args
            )
            model_config = load_model_config(default_model_directory)
            assert not model_config.use_features
            tag_input(
                model_name=default_args['model_name'],
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path']
            )

        @log_on_exception
        def test_should_be_able_to_train_with_features_without_feature_embeddings(
                self, default_args: dict, default_model_directory: str):
            train(
                use_features=True,
                **cast(TrainArgsDict, {
                    **default_args,
                    'features_indices': FEATURE_INDICES_1,
                    'features_embedding_size': None
                })
            )
            model_config = load_model_config(default_model_directory)
            assert model_config.use_features
            assert model_config.features_indices == FEATURE_INDICES_1
            assert model_config.features_embedding_size is None
            tag_input(
                model_name=default_args['model_name'],
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path']
            )

        @log_on_exception
        def test_should_be_able_to_train_with_features_and_features_embeddings(
                self, default_args: dict, default_model_directory: str):
            train(
                use_features=True,
                **cast(TrainArgsDict, {
                    **default_args,
                    'features_indices': FEATURE_INDICES_1,
                    'features_embedding_size': FEATURES_EMBEDDING_SIZE_1
                })
            )
            model_config = load_model_config(default_model_directory)
            assert model_config.use_features
            assert model_config.features_indices == FEATURE_INDICES_1
            assert model_config.features_embedding_size == FEATURES_EMBEDDING_SIZE_1
            tag_input(
                model_name=default_args['model_name'],
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path']
            )

        @log_on_exception
        def test_should_be_able_to_train_without_word_embeddings(
                self, default_args: dict, default_model_directory: str):
            train(
                **cast(TrainArgsDict, {
                    **default_args,
                    'embeddings_name': None,
                    'config_props': {
                        **default_args.get('config_props', {}),
                        'use_word_embeddings': False
                    }
                })
            )
            model_config = load_model_config(default_model_directory)
            assert model_config.embeddings_name is None
            tag_input(
                model_name=default_args['model_name'],
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path']
            )

        @log_on_exception
        def test_should_be_able_to_train_with_additional_token_feature_indices(
                self, default_args: dict, default_model_directory: str):
            train(
                **cast(TrainArgsDict, {
                    **default_args,
                    'config_props': {
                        **default_args.get('config_props', {}),
                        'max_char_length': 60,
                        'additional_token_feature_indices': [0]
                    }
                })
            )
            model_config = load_model_config(default_model_directory)
            assert model_config.max_char_length == 60
            assert model_config.additional_token_feature_indices == [0]
            tag_input(
                model_name=default_args['model_name'],
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path']
            )

        @log_on_exception
        def test_should_be_able_to_train_with_text_feature_indices(
                self, default_args: dict, default_model_directory: str):
            train(
                **cast(TrainArgsDict, {
                    **default_args,
                    'config_props': {
                        **default_args.get('config_props', {}),
                        'max_char_length': 60,
                        'text_feature_indices': [0],
                        'concatenated_embeddings_token_count': 2
                    }
                })
            )
            model_config = load_model_config(default_model_directory)
            assert model_config.max_char_length == 60
            assert model_config.text_feature_indices == [0]
            assert model_config.concatenated_embeddings_token_count == 2
            tag_input(
                model_name=default_args['model_name'],
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path']
            )

        @log_on_exception
        def test_should_be_able_to_train_with_unrolled_text_feature_index(
            self, default_args: dict, default_model_directory: str,
            tmp_path: Path
        ):
            input_path = tmp_path / 'input.data'
            tag_output_path = tmp_path / 'output.data'
            input_path.write_text('\n'.join([
                f'{TOKEN_1}\t{UNROLLED_TOKEN_1}{NBSP}{UNROLLED_TOKEN_2}\tB-{TAG_1}',
                f'{TOKEN_2}\t{UNROLLED_TOKEN_3}{NBSP}{UNROLLED_TOKEN_4}\tB-{TAG_2}'
            ]))
            tag_input_paths = [str(input_path)]
            # duplicate the input paths to get past the train test split
            input_paths = [str(input_path)] * 10
            train_eval(
                **cast(TrainArgsDict, {
                    **default_args,
                    'input_paths': input_paths,
                    'config_props': {
                        **default_args.get('config_props', {}),
                        'max_char_length': 60,
                        'unroll_text_feature_index': 0
                    }
                })
            )
            model_config = load_model_config(default_model_directory)
            assert model_config.max_char_length == 60
            assert model_config.unroll_text_feature_index == 0
            tag_input(
                model_name=default_args['model_name'],
                model_path=default_model_directory,
                input_paths=tag_input_paths,
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path'],
                tag_output_format=TagOutputFormats.DATA,
                tag_output_path=str(tag_output_path),
                tag_transformed=False
            )
            output_texts, output_labels, _ = load_data_and_labels_crf_file(
                str(tag_output_path)
            )
            LOGGER.debug('output_texts=%s', output_texts)
            LOGGER.debug('output_labels=%s', output_labels)
            assert output_texts.tolist() == [[TOKEN_1, TOKEN_2]]
            tag_input(
                model_name=default_args['model_name'],
                model_path=default_model_directory,
                input_paths=tag_input_paths,
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path'],
                tag_output_format=TagOutputFormats.DATA,
                tag_output_path=str(tag_output_path),
                tag_transformed=True
            )
            transformed_texts, transformed_labels, _ = load_data_and_labels_crf_file(
                str(tag_output_path)
            )
            LOGGER.debug('transformed_texts=%s', transformed_texts)
            LOGGER.debug('transformed_labels=%s', transformed_labels)
            assert transformed_texts.tolist() == [
                [UNROLLED_TOKEN_1, UNROLLED_TOKEN_2, UNROLLED_TOKEN_3, UNROLLED_TOKEN_4]
            ]

        @pytest.mark.usefixtures('trainer_mock')
        @pytest.mark.parametrize("copy_preprocessor", [False, True])
        @log_on_exception
        def test_should_be_able_to_copy_weights_from_previous_model(
            self,
            tmp_path: Path,
            default_args: dict,
            copy_preprocessor: bool
        ):
            source_model_output_path = tmp_path / 'source_model'
            train(
                **cast(TrainArgsDict, {
                    **default_args,
                    'output_path': str(source_model_output_path),
                    'embeddings_name': None,
                    'config_props': {
                        **default_args.get('config_props', {}),
                        'use_word_embeddings': False
                    }
                })
            )
            train(
                **cast(DefaultArgsDict, {
                    **default_args,
                    'transfer_learning_config': TransferLearningConfig(
                        source_model_path=(
                            str(source_model_output_path / default_args['model_name'])
                        ),
                        copy_layers={
                            'char_embeddings': 'char_embeddings',
                            'char_lstm': 'char_lstm'
                        },
                        copy_preprocessor=copy_preprocessor,
                        copy_preprocessor_fields=['vocab_char'],
                        freeze_layers=['char_embeddings']
                    ),
                    'embeddings_name': None,
                    'config_props': {
                        **default_args.get('config_props', {}),
                        'use_word_embeddings': False
                    }
                })
            )

        @log_on_exception
        def test_should_be_able_to_train_BidLSTM_CRF_FEATURES(
                self, default_args: dict, default_model_directory: str):
            train_args = cast(TrainArgsDict, {
                **default_args,
                'architecture': 'BidLSTM_CRF_FEATURES',
                'features_embedding_size': 4,
                'config_props': {
                    'features_lstm_units': 4
                }
            })
            train(
                **train_args
            )
            model_config = load_model_config(default_model_directory)
            assert model_config.architecture == 'BidLSTM_CRF_FEATURES'
            assert model_config.features_embedding_size == 4
            assert model_config.features_lstm_units == 4
            tag_input(
                model_name=default_args['model_name'],
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path']
            )

        @log_on_exception
        def test_should_be_able_to_train_CustomBidLSTM_CRF_FEATURES(
                self, default_args: DefaultArgsDict, default_model_directory: str):
            train_args = cast(TrainArgsDict, {
                **default_args,
                'architecture': 'CustomBidLSTM_CRF_FEATURES',
                'features_embedding_size': 4,
                'config_props': {
                    'features_lstm_units': 4
                }
            })
            train(
                **train_args
            )
            model_config = load_model_config(default_model_directory)
            assert model_config.architecture == 'CustomBidLSTM_CRF_FEATURES'
            assert model_config.features_embedding_size == 4
            assert model_config.features_lstm_units == 4
            tag_input(
                model_name=default_args['model_name'],
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager'],
                embedding_registry_path=default_args['embedding_registry_path']
            )

        @log_on_exception
        def test_should_be_able_to_train_wapiti(
                self, default_args: dict, default_model_directory: str,
                temp_dir: Path):
            template_path = temp_dir.joinpath('template')
            template_path.write_text('U00:%x[-4,0]')
            wapiti_train(
                model_name=default_args['model_name'],
                template_path=str(template_path),
                input_paths=default_args['input_paths'],
                output_path=default_args['output_path'],
                download_manager=default_args['download_manager'],
                gzip_enabled=False
            )
            wapiti_eval_model(
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager']
            )
            wapiti_tag_input(
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager']
            )

        @log_on_exception
        def test_should_be_able_to_train_eval_wapiti(
                self, default_args: dict, default_model_directory: str,
                temp_dir: Path):
            template_path = temp_dir.joinpath('template')
            template_path.write_text('U00:%x[-4,0]')
            wapiti_train_eval(
                model_name=default_args['model_name'],
                template_path=str(template_path),
                input_paths=default_args['input_paths'],
                output_path=default_args['output_path'],
                download_manager=default_args['download_manager']
            )
            wapiti_tag_input(
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager']
            )

        @log_on_exception
        def test_should_be_able_to_train_and_gzip_wapiti(
                self, default_args: dict, default_model_directory: str,
                temp_dir: Path):
            template_path = temp_dir.joinpath('template')
            template_path.write_text('U00:%x[-4,0]')
            wapiti_train(
                model_name=default_args['model_name'],
                template_path=str(template_path),
                input_paths=default_args['input_paths'],
                output_path=default_args['output_path'],
                download_manager=default_args['download_manager'],
                gzip_enabled=True
            )
            wapiti_eval_model(
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager']
            )
            wapiti_tag_input(
                model_path=default_model_directory,
                input_paths=default_args['input_paths'],
                download_manager=default_args['download_manager']
            )

        def test_should_be_able_to_train_eval(self, default_args: dict):
            train_eval(
                **default_args
            )

        def test_should_be_able_to_train_eval_nfold(self, default_args: dict):
            train_eval(
                fold_count=2,
                **default_args
            )

        def test_should_be_able_to_train_eval_with_eval_input_window_stride(
                self, default_args: dict):
            train_eval(
                eval_max_sequence_length=5,
                eval_input_window_stride=3,
                **default_args
            )
