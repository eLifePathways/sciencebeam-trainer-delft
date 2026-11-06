import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import torch

from delft.sequenceLabelling.models import BaseSequenceLabeler
from delft.sequenceLabelling.tagger import Tagger as DelftTagger
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
from sciencebeam_trainer_delft.sequence_labelling.models import CustomBidLSTM_CRF, get_model
from sciencebeam_trainer_delft.sequence_labelling.saving import ModelSaver
from sciencebeam_trainer_delft.sequence_labelling.wrapper import (
    get_preprocessor,
    get_vocab_size,
    prepare_preprocessor,
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


class TestPreparePreprocessorFeaturesIndices:
    """The columns taken as indices: exactly those asked for, else those within the limit."""

    # column 0 has a value per token, 13 in all, column 1 a single one
    X = [[f'token{index}' for index in range(13)]]
    Y = [[LABEL_1] * 13]
    FEATURES = [[[f'value{index}', FEATURE_VALUE_1] for index in range(13)]]

    def _prepare(self, **config_kwargs):
        model_config = ModelConfig(
            use_features=True, use_features_indices_input=True, **config_kwargs
        )
        prepare_preprocessor(
            self.X, self.Y, model_config=model_config,
            features=self.FEATURES  # type: ignore[arg-type]
        )
        return model_config

    def test_should_refuse_a_column_asked_for_over_the_vocabulary_size(self):
        with pytest.raises(ValueError, match='features_vocabulary_size'):
            self._prepare(features_indices=[0, 1])

    def test_should_use_a_column_asked_for_within_a_raised_vocabulary_size(self):
        model_config = self._prepare(features_indices=[0, 1], features_vocabulary_size=13)
        assert model_config.features_indices == [0, 1]

    def test_should_revise_the_indices_to_the_columns_within_the_limit_when_none_asked(self):
        assert self._prepare().features_indices == [1]


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


def _sequence_with_model(model, **config_kwargs) -> Sequence:
    sequence = Sequence(MODEL_NAME_1)
    sequence.model = model
    sequence.model_config = ModelConfig(**config_kwargs)
    return sequence


class TestSequenceTaggerChoice:
    def test_should_label_with_delft_tagger_for_an_upstream_architecture(self):
        sequence = _sequence_with_model(MagicMock(spec=BaseSequenceLabeler))
        assert sequence.is_tagged_by_delft()

    def test_should_label_with_the_data_generator_for_an_architecture_of_this_repo(self):
        sequence = _sequence_with_model(MagicMock(spec=CustomBidLSTM_CRF))
        assert not sequence.is_tagged_by_delft()

    @pytest.mark.parametrize('config_kwargs', [
        {'additional_token_feature_indices': [1]},
        {'text_feature_indices': [1]},
        {'concatenated_embeddings_token_count': 2},
        {'unroll_text_feature_index': 0}
    ])
    def test_should_label_with_the_data_generator_for_inputs_only_it_builds(
        self, config_kwargs: dict
    ):
        sequence = _sequence_with_model(MagicMock(spec=BaseSequenceLabeler), **config_kwargs)
        assert not sequence.is_tagged_by_delft()


class TestSequenceDelftTagModelConfig:
    def _tag_max_sequence_length(self, asked, trained, transformer_name=None):
        sequence = _sequence_with_model(
            MagicMock(spec=BaseSequenceLabeler),
            max_sequence_length=trained,
            transformer_name=transformer_name
        )
        sequence.max_sequence_length = asked
        # pylint: disable-next=protected-access
        tag_model_config = sequence._get_delft_tag_model_config()
        # the model's own configuration is left as it is
        assert sequence.model_config.max_sequence_length == trained
        return tag_model_config.max_sequence_length

    def test_should_take_the_windows_asked_for_here(self):
        assert self._tag_max_sequence_length(asked=2000, trained=3000) == 2000

    def test_should_not_limit_what_is_not_limited_here(self):
        # as the data generator does
        assert self._tag_max_sequence_length(asked=None, trained=3000) is None

    @pytest.mark.parametrize('asked', [2000, None])
    def test_should_give_a_transformer_no_more_than_it_was_trained_with(self, asked):
        assert self._tag_max_sequence_length(
            asked=asked, trained=512, transformer_name='bert-base-cased'
        ) == 512

    def test_should_give_a_transformer_less_when_asked(self):
        assert self._tag_max_sequence_length(
            asked=256, trained=512, transformer_name='bert-base-cased'
        ) == 256


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

    @log_on_exception
    def test_should_imply_use_features_from_the_architecture_when_loading(
        self, tmp_path: Path
    ):
        x_train = [[TOKEN_1, TOKEN_2]]
        y_train = [[LABEL_1, LABEL_2]]
        features_train = [[TOKEN_FEATURES_1, TOKEN_FEATURES_1]]
        model_config = ModelConfig(
            model_name=MODEL_NAME_1,
            architecture='BidLSTM_CRF_FEATURES',
            embeddings_name=None,
            use_features=True,
            use_features_indices_input=True,
            features_indices=[0, 1],
            char_embedding_size=2,
            max_char_length=3,
            num_char_lstm_units=4,
            num_word_lstm_units=5,
            word_embedding_size=0
        )
        preprocessor = prepare_preprocessor(
            x_train, y_train, model_config=model_config,
            features=features_train  # type: ignore[arg-type]
        )
        # sized from the fitted preprocessor, as training does
        model_config.char_vocab_size = get_vocab_size(preprocessor.vocab_char)
        model_config.case_vocab_size = get_vocab_size(preprocessor.vocab_case)
        model = get_model(model_config, preprocessor, ntags=len(preprocessor.vocab_tag))
        ModelSaver(preprocessor=preprocessor, model_config=model_config).save_to(
            str(tmp_path), model=model
        )
        # a config delft saved does not say use_features
        config_path = tmp_path / 'config.json'
        config = json.loads(config_path.read_text())
        del config['use_features']
        config_path.write_text(json.dumps(config))

        loaded_model_wrapper = Sequence(MODEL_NAME_1, multiprocessing=False)
        loaded_model_wrapper.load_from(str(tmp_path))
        assert loaded_model_wrapper.model_config.use_features
        tagged = loaded_model_wrapper.tag(x_train, features=features_train, output_format=None)
        assert [token for token, _ in tagged[0]] == x_train[0]

    @log_on_exception
    def test_should_label_an_upstream_architecture_whole_in_windows_with_delft_tagger(
        self, tmp_path: Path
    ):
        x_train = [[TOKEN_1, TOKEN_2]]
        y_train = [[LABEL_1, LABEL_2]]
        model_config = ModelConfig(
            model_name=MODEL_NAME_1,
            architecture='BidLSTM_CRF',
            embeddings_name=None,
            char_embedding_size=2,
            max_char_length=3,
            num_char_lstm_units=4,
            num_word_lstm_units=5,
            word_embedding_size=0
        )
        preprocessor = prepare_preprocessor(x_train, y_train, model_config=model_config)
        model_config.char_vocab_size = get_vocab_size(preprocessor.vocab_char)
        model_config.case_vocab_size = get_vocab_size(preprocessor.vocab_case)
        model = get_model(model_config, preprocessor, ntags=len(preprocessor.vocab_tag))
        ModelSaver(preprocessor=preprocessor, model_config=model_config).save_to(
            str(tmp_path), model=model
        )

        # windows of 3 tokens, one every 2
        loaded_model_wrapper = Sequence(
            MODEL_NAME_1, multiprocessing=False, max_sequence_length=3, input_window_stride=2
        )
        loaded_model_wrapper.load_from(str(tmp_path))
        assert loaded_model_wrapper.is_tagged_by_delft()
        tokens = [TOKEN_1, TOKEN_2] * 3 + [TOKEN_1]
        with patch.object(DelftTagger, 'tag', autospec=True, side_effect=DelftTagger.tag) as tag:
            tagged = loaded_model_wrapper.tag([tokens], output_format=None)
        assert tag.call_args.kwargs['window_stride'] == 2
        # every token labelled, the model being untrained
        assert [token for token, _ in tagged[0]] == tokens
        assert all(label in preprocessor.vocab_tag for _, label in tagged[0])
