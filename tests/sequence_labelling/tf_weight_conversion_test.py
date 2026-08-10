from pathlib import Path
from typing import Any, Dict, cast

import h5py
import numpy as np
import pytest
import torch

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.models import CustomBidLSTM_CRF
from sciencebeam_trainer_delft.sequence_labelling.tf_weight_conversion import (
    TfWeightConversionError,
    load_keras_weights_into_model,
    read_keras_layer_weights
)


NTAGS = 5
CHAR_VOCAB_SIZE = 12
CHAR_EMBEDDING_SIZE = 5
NUM_CHAR_LSTM_UNITS = 4
NUM_WORD_LSTM_UNITS = 6
WORD_EMBEDDING_SIZE = 3
MAX_FEATURE_SIZE = 7
MAX_CHAR_LENGTH = 5

WORD_LSTM_INPUT_SIZE = WORD_EMBEDDING_SIZE + 2 * NUM_CHAR_LSTM_UNITS + MAX_FEATURE_SIZE


@pytest.fixture(name='model_config')
def _model_config() -> ModelConfig:
    return ModelConfig(
        model_name='test-model',
        architecture='CustomBidLSTM_CRF',
        char_vocab_size=CHAR_VOCAB_SIZE,
        char_embedding_size=CHAR_EMBEDDING_SIZE,
        num_char_lstm_units=NUM_CHAR_LSTM_UNITS,
        max_char_length=MAX_CHAR_LENGTH,
        num_word_lstm_units=NUM_WORD_LSTM_UNITS,
        word_embedding_size=WORD_EMBEDDING_SIZE,
        dropout=0.0,
        use_features=True,
        max_feature_size=MAX_FEATURE_SIZE,
        features_embedding_size=0
    )


@pytest.fixture(name='model')
def _model(model_config: ModelConfig) -> CustomBidLSTM_CRF:
    return CustomBidLSTM_CRF(model_config, NTAGS)


def _tensor(value: Any) -> torch.Tensor:
    """The LSTM and CRF parameters are registered dynamically, so not typed."""
    return cast(torch.Tensor, value)


def _values(*shape: int) -> np.ndarray:
    """Distinct, ordered values, so a transposed result is visibly wrong."""
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100


def _keras_arrays() -> Dict[str, np.ndarray]:
    """The weights a Keras `CustomBidLSTM_CRF` of this config would hold."""
    arrays = {
        'char_embeddings/char_embeddings/embeddings':
            _values(CHAR_VOCAB_SIZE, CHAR_EMBEDDING_SIZE),
        'dense_1/dense_1/kernel': _values(2 * NUM_WORD_LSTM_UNITS, NUM_WORD_LSTM_UNITS),
        'dense_1/dense_1/bias': _values(NUM_WORD_LSTM_UNITS),
        'dense_ntags/dense_ntags/kernel': _values(NUM_WORD_LSTM_UNITS, NTAGS),
        'dense_ntags/dense_ntags/bias': _values(NTAGS),
        'chain_crf_1/chain_crf_1/U': _values(NTAGS, NTAGS),
        'chain_crf_1/chain_crf_1/b_start': _values(NTAGS),
        'chain_crf_1/chain_crf_1/b_end': _values(NTAGS),
    }
    for direction in ('forward', 'backward'):
        prefix = f'char_lstm/char_lstm/{direction}_lstm_1'
        arrays[f'{prefix}/kernel'] = _values(CHAR_EMBEDDING_SIZE, 4 * NUM_CHAR_LSTM_UNITS)
        arrays[f'{prefix}/recurrent_kernel'] = _values(
            NUM_CHAR_LSTM_UNITS, 4 * NUM_CHAR_LSTM_UNITS
        )
        arrays[f'{prefix}/bias'] = _values(4 * NUM_CHAR_LSTM_UNITS)
        prefix = f'bidirectional_2/bidirectional_2/{direction}_lstm_2'
        arrays[f'{prefix}/kernel'] = _values(WORD_LSTM_INPUT_SIZE, 4 * NUM_WORD_LSTM_UNITS)
        arrays[f'{prefix}/recurrent_kernel'] = _values(
            NUM_WORD_LSTM_UNITS, 4 * NUM_WORD_LSTM_UNITS
        )
        arrays[f'{prefix}/bias'] = _values(4 * NUM_WORD_LSTM_UNITS)
    return arrays


def _write_keras_weights(
    filepath: Path,
    arrays: Dict[str, np.ndarray],
    prefix: str = 'model_weights/'
) -> str:
    with h5py.File(str(filepath), 'w') as h5_file:
        for name, value in arrays.items():
            h5_file.create_dataset(f'{prefix}{name}:0', data=value)
    return str(filepath)


@pytest.fixture(name='weights_file')
def _weights_file(temp_dir: Path) -> str:
    return _write_keras_weights(temp_dir / 'model_weights.hdf5', _keras_arrays())


class TestReadKerasLayerWeights:
    def test_should_group_datasets_by_layer_and_strip_the_variable_index(
        self, weights_file: str
    ):
        layers = {layer.path: layer for layer in read_keras_layer_weights(weights_file)}
        assert 'char_embeddings/char_embeddings' in layers
        assert set(layers['chain_crf_1/chain_crf_1'].weights) == {'U', 'b_start', 'b_end'}

    def test_should_ignore_optimizer_state(self, temp_dir: Path):
        arrays = _keras_arrays()
        filepath = _write_keras_weights(temp_dir / 'model_weights.hdf5', arrays)
        with h5py.File(filepath, 'a') as h5_file:
            h5_file.create_dataset('optimizer_weights/Adam/Variable_1:0', data=_values(3, 4))
        paths = {layer.path for layer in read_keras_layer_weights(filepath)}
        assert not any(path.startswith('optimizer') for path in paths)

    def test_should_read_weights_saved_without_the_model_weights_prefix(
        self, temp_dir: Path
    ):
        filepath = _write_keras_weights(
            temp_dir / 'model_weights.hdf5', _keras_arrays(), prefix=''
        )
        paths = {layer.path for layer in read_keras_layer_weights(filepath)}
        assert 'char_embeddings/char_embeddings' in paths


class TestLoadKerasWeightsIntoModel:
    def test_should_copy_the_embedding_unchanged(
        self, model: CustomBidLSTM_CRF, weights_file: str
    ):
        load_keras_weights_into_model(weights_file, model)
        np.testing.assert_allclose(
            model.char_encoder.char_embeddings.weight.detach().numpy(),
            _values(CHAR_VOCAB_SIZE, CHAR_EMBEDDING_SIZE)
        )

    def test_should_transpose_dense_kernels_and_copy_their_bias(
        self, model: CustomBidLSTM_CRF, weights_file: str
    ):
        load_keras_weights_into_model(weights_file, model)
        np.testing.assert_allclose(
            model.dense_ntags.weight.detach().numpy(),
            _values(NUM_WORD_LSTM_UNITS, NTAGS).T
        )
        np.testing.assert_allclose(model.dense_ntags.bias.detach().numpy(), _values(NTAGS))

    def test_should_put_the_whole_keras_bias_on_the_input_side(
        self, model: CustomBidLSTM_CRF, weights_file: str
    ):
        load_keras_weights_into_model(weights_file, model)
        np.testing.assert_allclose(
            _tensor(model.word_lstm.bias_ih_l0).detach().numpy(),
            _values(4 * NUM_WORD_LSTM_UNITS)
        )
        # torch adds the two biases, so the hidden side has to be zero or the
        # bias would be applied twice
        np.testing.assert_allclose(
            _tensor(model.word_lstm.bias_hh_l0).detach().numpy(),
            np.zeros(4 * NUM_WORD_LSTM_UNITS)
        )

    def test_should_map_the_backward_direction_to_the_reverse_parameters(
        self, model: CustomBidLSTM_CRF, temp_dir: Path
    ):
        arrays = _keras_arrays()
        # make the two directions distinguishable
        arrays['bidirectional_2/bidirectional_2/backward_lstm_2/bias'] = np.full(
            4 * NUM_WORD_LSTM_UNITS, 7, dtype=np.float32
        )
        filepath = _write_keras_weights(temp_dir / 'model_weights.hdf5', arrays)
        load_keras_weights_into_model(filepath, model)
        np.testing.assert_allclose(
            _tensor(model.word_lstm.bias_ih_l0_reverse).detach().numpy(),
            np.full(4 * NUM_WORD_LSTM_UNITS, 7)
        )
        assert not np.allclose(_tensor(model.word_lstm.bias_ih_l0).detach().numpy(), 7)

    def test_should_copy_the_chain_crf_parameters(
        self, model: CustomBidLSTM_CRF, weights_file: str
    ):
        load_keras_weights_into_model(weights_file, model)
        np.testing.assert_allclose(
            _tensor(model.crf.U).detach().numpy(), _values(NTAGS, NTAGS)
        )

    def test_should_produce_a_usable_model(
        self, model: CustomBidLSTM_CRF, weights_file: str
    ):
        load_keras_weights_into_model(weights_file, model)
        model.eval()
        inputs = {
            'word_input': torch.zeros(2, 3, WORD_EMBEDDING_SIZE),
            'char_input': torch.zeros(2, 3, MAX_CHAR_LENGTH, dtype=torch.long),
            'features_input': torch.zeros(2, 3, MAX_FEATURE_SIZE)
        }
        with torch.no_grad():
            logits = model.get_logits(inputs)
        assert logits.shape == (2, 3, NTAGS)


class TestLoadKerasWeightsIntoModelRefusal:
    def test_should_refuse_an_unrecognised_layer(
        self, model: CustomBidLSTM_CRF, temp_dir: Path
    ):
        arrays = _keras_arrays()
        arrays['attention_1/attention_1/something'] = _values(3, 4)
        filepath = _write_keras_weights(temp_dir / 'model_weights.hdf5', arrays)
        with pytest.raises(TfWeightConversionError, match='unrecognised'):
            load_keras_weights_into_model(filepath, model)

    def test_should_refuse_a_layer_no_destination_has_the_shape_for(
        self, model: CustomBidLSTM_CRF, temp_dir: Path
    ):
        arrays = _keras_arrays()
        arrays['dense_ntags/dense_ntags/kernel'] = _values(NUM_WORD_LSTM_UNITS, NTAGS + 1)
        arrays['dense_ntags/dense_ntags/bias'] = _values(NTAGS + 1)
        filepath = _write_keras_weights(temp_dir / 'model_weights.hdf5', arrays)
        with pytest.raises(TfWeightConversionError, match='no dense layer matches'):
            load_keras_weights_into_model(filepath, model)

    def test_should_refuse_when_a_destination_gets_no_weights(
        self, model: CustomBidLSTM_CRF, temp_dir: Path
    ):
        arrays = {
            name: value for name, value in _keras_arrays().items()
            if not name.startswith('char_embeddings/')
        }
        filepath = _write_keras_weights(temp_dir / 'model_weights.hdf5', arrays)
        with pytest.raises(TfWeightConversionError, match='no converted weights for'):
            load_keras_weights_into_model(filepath, model)

    def test_should_refuse_an_lstm_direction_it_cannot_identify(
        self, model: CustomBidLSTM_CRF, temp_dir: Path
    ):
        arrays = {
            name.replace('forward_lstm_1', 'lstm_1'): value
            for name, value in _keras_arrays().items()
        }
        filepath = _write_keras_weights(temp_dir / 'model_weights.hdf5', arrays)
        with pytest.raises(TfWeightConversionError, match='direction'):
            load_keras_weights_into_model(filepath, model)

    def test_should_refuse_a_file_holding_no_weights(
        self, model: CustomBidLSTM_CRF, temp_dir: Path
    ):
        filepath = _write_keras_weights(temp_dir / 'model_weights.hdf5', {})
        with pytest.raises(TfWeightConversionError, match='no weights'):
            load_keras_weights_into_model(filepath, model)

    def test_should_refuse_a_corrupted_file(
        self, model: CustomBidLSTM_CRF, temp_dir: Path
    ):
        filepath = temp_dir / 'model_weights.hdf5'
        filepath.write_bytes(b'not an hdf5 file at all')
        with pytest.raises(OSError):
            load_keras_weights_into_model(str(filepath), model)
