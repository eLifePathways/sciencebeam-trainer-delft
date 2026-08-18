import logging
from pathlib import Path

import pytest

import torch

from delft.sequenceLabelling.preprocess import (
    Preprocessor as DelftWordPreprocessor,
    FeaturesPreprocessor as DelftFeaturesPreprocessor
)

from sciencebeam_trainer_delft.resources.default_config import DEFAULT_RESOURCE_REGISTRY_FILE
from sciencebeam_trainer_delft.utils.device import log_device_info_once
from sciencebeam_trainer_delft.sequence_labelling.preprocess import (
    Preprocessor as ScienceBeamPreprocessor,
    FeaturesPreprocessor as ScienceBeamFeaturesPreprocessor
)

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.wrapper import (
    get_preprocessor,
    Sequence
)

from sciencebeam_trainer_delft.sequence_labelling.transfer_learning import (
    TransferLearningConfig,
    TransferModelWrapper
)

from ..test_utils import log_on_exception


LOGGER = logging.getLogger(__name__)


MODEL_NAME_1 = 'DummyModel1'

INVALID_DEVICE = 'gpu'

TOKEN_1 = 'token1'
TOKEN_2 = 'token2'

LABEL_1 = 'label1'
LABEL_2 = 'label2'

FEATURE_VALUE_1 = 'feature1'
FEATURE_VALUE_2 = 'feature2'

TOKEN_FEATURES_1 = [FEATURE_VALUE_1, FEATURE_VALUE_2]


class TestGetPreprocessor:
    def test_should_use_default_preprocessor_if_not_using_features(self):
        model_config = ModelConfig(use_features=False)
        preprocessor = get_preprocessor(model_config, features=[[TOKEN_FEATURES_1]])
        assert isinstance(preprocessor, DelftWordPreprocessor)
        assert not isinstance(preprocessor, ScienceBeamPreprocessor)
        assert preprocessor.feature_preprocessor is None

    def test_should_use_default_preprocessor_if_using_features_indices_input(self):
        model_config = ModelConfig(use_features=True, use_features_indices_input=True)
        preprocessor = get_preprocessor(model_config, features=[[TOKEN_FEATURES_1]])
        assert isinstance(preprocessor, DelftWordPreprocessor)
        assert not isinstance(preprocessor, ScienceBeamPreprocessor)
        assert preprocessor.feature_preprocessor is not None
        assert isinstance(preprocessor.feature_preprocessor, DelftFeaturesPreprocessor)

    def test_should_create_preprocessor_with_feature_preprocessor(self):
        model_config = ModelConfig(use_features=True, use_features_indices_input=False)
        preprocessor = get_preprocessor(model_config, features=[[TOKEN_FEATURES_1]])
        assert isinstance(preprocessor, DelftWordPreprocessor)
        assert not isinstance(preprocessor, ScienceBeamPreprocessor)
        assert preprocessor.feature_preprocessor is not None
        assert isinstance(preprocessor.feature_preprocessor, ScienceBeamFeaturesPreprocessor)


class TestSequence:
    def test_should_create_embedding_manager_with_default_regisry_path(self):
        model = Sequence(MODEL_NAME_1)
        assert model.embedding_registry_path == DEFAULT_RESOURCE_REGISTRY_FILE
        assert model.embedding_manager.path == DEFAULT_RESOURCE_REGISTRY_FILE

    def test_should_log_the_device_it_will_use(self, caplog):
        log_device_info_once.cache_clear()
        with caplog.at_level('INFO'):
            model = Sequence(MODEL_NAME_1)
        assert f'using device: {model.device}' in caplog.text

    def test_should_log_the_device_once_per_process(self, caplog):
        log_device_info_once.cache_clear()
        with caplog.at_level('INFO'):
            Sequence(MODEL_NAME_1)
            Sequence(MODEL_NAME_1)
        assert caplog.text.count('using device') == 1

    def test_should_reject_an_invalid_device(self):
        with pytest.raises(ValueError, match=repr(INVALID_DEVICE)):
            Sequence(MODEL_NAME_1, device=INVALID_DEVICE)


def get_layer_weights(model, layer_name: str):
    wrapped_model = TransferModelWrapper(model)
    LOGGER.debug('layer_names: %s', wrapped_model.layer_names)
    return wrapped_model.get_layer_weights(layer_name)


@pytest.mark.slow
@pytest.mark.very_slow
class TestSequenceEndToEnd:
    @log_on_exception
    def test_should_copy_weights_from_source_model(
        self, tmp_path: Path
    ):
        x_train = [[TOKEN_1, TOKEN_2]]
        y_train = [[LABEL_1, LABEL_2]]
        model_kwargs = dict(
            architecture='CustomBidLSTM_CRF',
            char_emb_size=2,
            max_char_length=3,
            char_lstm_units=4,
            word_lstm_units=5,
            max_sequence_length=6,
            multiprocessing=False,
            max_epoch=1
        )
        train_kwargs = dict(
            x_train=x_train, y_train=y_train,
            x_valid=x_train, y_valid=y_train
        )
        model_wrapper = Sequence(
            MODEL_NAME_1,
            **model_kwargs  # type: ignore
        )
        model_wrapper.train(**train_kwargs)  # type: ignore
        layer_name = 'word_lstm'
        expected_weights = get_layer_weights(model_wrapper.model, layer_name)
        model_wrapper.save(str(tmp_path))
        model_wrapper_2 = Sequence(
            MODEL_NAME_1,
            transfer_learning_config=TransferLearningConfig(
                source_model_path=str(tmp_path / MODEL_NAME_1),
                copy_layers={layer_name: layer_name},
                freeze_layers=[layer_name]
            ),
            **model_kwargs  # type: ignore
        )
        model_wrapper_2.train(**train_kwargs)  # type: ignore
        actual_weights = get_layer_weights(model_wrapper_2.model, layer_name)
        assert set(actual_weights) == set(expected_weights)
        for name, expected_weight in expected_weights.items():
            LOGGER.debug('expected_weights(%s):\n%s', name, expected_weight)
            LOGGER.debug('actual_weights(%s):\n%s', name, actual_weights[name])
            # the layer was frozen, so training cannot have changed it
            assert torch.equal(actual_weights[name], expected_weight)
