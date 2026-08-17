import pytest
import torch

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.models import CustomBidLSTM_CRF
from sciencebeam_trainer_delft.sequence_labelling.transfer_learning import (
    TransferLearningConfig,
    TransferLearningSource,
    TransferModelWrapper,
    freeze_model_layers
)


NTAGS = 5
CHAR_VOCAB_SIZE = 12
MAX_CHAR_LENGTH = 5
MAX_FEATURE_SIZE = 7
WORD_EMBEDDING_SIZE = 3

WORD_LSTM_LAYER = 'word_lstm'
CHAR_EMBEDDINGS_LAYER = 'char_encoder.char_embeddings'


def _model_config() -> ModelConfig:
    return ModelConfig(
        architecture='CustomBidLSTM_CRF',
        char_vocab_size=CHAR_VOCAB_SIZE,
        char_embedding_size=5,
        num_char_lstm_units=4,
        max_char_length=MAX_CHAR_LENGTH,
        num_word_lstm_units=6,
        word_embedding_size=WORD_EMBEDDING_SIZE,
        dropout=0.0,
        use_features=True,
        max_feature_size=MAX_FEATURE_SIZE,
        features_embedding_size=0
    )


def _model() -> CustomBidLSTM_CRF:
    return CustomBidLSTM_CRF(_model_config(), NTAGS)


class TestTransferModelWrapper:
    def test_should_name_layers_by_module_path(self):
        wrapped_model = TransferModelWrapper(_model())
        assert WORD_LSTM_LAYER in wrapped_model.layer_names
        assert CHAR_EMBEDDINGS_LAYER in wrapped_model.layer_names

    def test_should_not_expose_the_root_module_as_a_layer(self):
        assert '' not in TransferModelWrapper(_model()).layer_names

    def test_should_round_trip_layer_weights(self):
        source_model = _model()
        target_model = _model()
        weights = TransferModelWrapper(source_model).get_layer_weights(WORD_LSTM_LAYER)
        TransferModelWrapper(target_model).set_layer_weights(WORD_LSTM_LAYER, weights)
        assert torch.equal(
            target_model.word_lstm.weight_ih_l0, source_model.word_lstm.weight_ih_l0
        )

    def test_should_freeze_every_parameter_of_a_layer(self):
        model = _model()
        TransferModelWrapper(model).freeze_layer(WORD_LSTM_LAYER)
        assert not any(
            parameter.requires_grad for parameter in model.word_lstm.parameters()
        )
        # other layers stay trainable
        assert all(
            parameter.requires_grad for parameter in model.dense_ntags.parameters()
        )


class TestApplyWeights:
    def _source(self, copy_layers) -> TransferLearningSource:
        return TransferLearningSource(
            transfer_learning_config=TransferLearningConfig(copy_layers=copy_layers),
            source_model=_model(),
            source_preprocessor=None  # type: ignore[arg-type]
        )

    def test_should_copy_the_requested_layer_only(self):
        target_model = _model()
        original_dense_weight = target_model.dense_ntags.weight.clone()
        transfer_learning_source = self._source({WORD_LSTM_LAYER: WORD_LSTM_LAYER})
        transfer_learning_source.apply_weights(target_model)
        assert torch.equal(
            target_model.word_lstm.weight_ih_l0,
            transfer_learning_source.source_model.word_lstm.weight_ih_l0  # type: ignore
        )
        assert torch.equal(target_model.dense_ntags.weight, original_dense_weight)

    def test_should_reject_an_unknown_source_layer(self):
        with pytest.raises(ValueError, match='missing source layers'):
            self._source({WORD_LSTM_LAYER: 'not_a_layer'}).apply_weights(_model())

    def test_should_reject_an_unknown_target_layer(self):
        with pytest.raises(ValueError, match='missing target layers'):
            self._source({'not_a_layer': WORD_LSTM_LAYER}).apply_weights(_model())

    def test_should_report_the_layer_a_copy_failed_for(self):
        with pytest.raises(RuntimeError, match=WORD_LSTM_LAYER):
            # the shapes of these two layers do not match
            self._source({WORD_LSTM_LAYER: CHAR_EMBEDDINGS_LAYER}).apply_weights(_model())

    def test_should_copy_nothing_without_requested_layers(self):
        target_model = _model()
        original_weight = target_model.word_lstm.weight_ih_l0.clone()
        self._source(None).apply_weights(target_model)
        assert torch.equal(target_model.word_lstm.weight_ih_l0, original_weight)


class TestFreezeModelLayers:
    def test_should_freeze_the_requested_layers(self):
        model = _model()
        freeze_model_layers(model, [WORD_LSTM_LAYER, CHAR_EMBEDDINGS_LAYER])
        assert not any(
            parameter.requires_grad for parameter in model.word_lstm.parameters()
        )
        assert not model.char_encoder.char_embeddings.weight.requires_grad

    def test_should_freeze_nothing_without_requested_layers(self):
        model = _model()
        freeze_model_layers(model, None)
        assert all(parameter.requires_grad for parameter in model.parameters())
