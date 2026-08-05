import json
from pathlib import Path
from typing import Any, Dict, Iterator
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.optim import Adam

from delft.utilities.crf_pytorch import ChainCRF

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.models import (
    BidLSTM_CRF_FEATURES,
    CustomBidLSTM_CRF,
    CustomBidLSTM_CRF_FEATURES,
    get_model,
    get_model_names,
    is_model_stateful,
    to_tag_indices_array,
    updated_implicit_model_config_props
)
from sciencebeam_trainer_delft.sequence_labelling.upstream_patches import (
    patch_chain_crf_eager_build
)


# the published header model this repo's reference capture uses
NTAGS = 34
REFERENCE_CONFIG = {
    'char_vocab_size': 288,
    'char_embedding_size': 25,
    'num_char_lstm_units': 25,
    'max_char_length': 30,
    'num_word_lstm_units': 200,
    'word_embedding_size': 0,
    'dropout': 0.5,
    'use_features': True,
    'max_feature_size': 53,
    'features_embedding_size': 0
}

REFERENCE_CAPTURE_PATH = (
    Path(__file__).parents[2] / 'data' / 'reference' / 'header-2020-10-04'
)


def _model_config(**kwargs) -> ModelConfig:
    # ModelConfig sets keyword arguments as attributes without passing them to
    # the upstream constructor, so these are the resolved attribute names
    # (num_word_lstm_units, not word_lstm_units) rather than its parameter names
    values: Dict[str, Any] = {**REFERENCE_CONFIG, **kwargs}
    return ModelConfig(architecture='CustomBidLSTM_CRF', **values)


@pytest.fixture(name='patched_chain_crf', autouse=True)
def _patched_chain_crf() -> Iterator[None]:
    original_init = ChainCRF.__init__
    patch_chain_crf_eager_build()
    yield
    ChainCRF.__init__ = original_init  # type: ignore[method-assign]


def _batch(model_config: ModelConfig, batch_size: int = 2, sequence_length: int = 4):
    torch.manual_seed(42)
    inputs = {
        'word_input': torch.randn(
            batch_size, sequence_length, model_config.word_embedding_size
        ),
        'char_input': torch.randint(
            0, model_config.char_vocab_size,
            (batch_size, sequence_length, model_config.max_char_length)
        ),
        'features_input': torch.randn(
            batch_size, sequence_length, model_config.max_feature_size
        )
    }
    labels = torch.randint(1, NTAGS, (batch_size, sequence_length))
    return inputs, labels


class TestCustomBidLSTMCRF:
    def test_should_pass_features_through_without_a_projection(self):
        model = CustomBidLSTM_CRF(_model_config(), NTAGS)
        assert model.features_embeddings_dense is None
        # word embeddings 0 + char encoder 2 x 25 + features 53
        assert model.word_lstm.input_size == 103

    def test_should_project_features_when_an_embedding_size_is_set(self):
        model = CustomBidLSTM_CRF(_model_config(features_embedding_size=8), NTAGS)
        assert model.features_embeddings_dense is not None
        assert model.word_lstm.input_size == 58

    def test_should_have_the_expected_parameter_shapes(self):
        model = CustomBidLSTM_CRF(_model_config(), NTAGS)
        shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
        assert shapes['char_encoder.char_embeddings.weight'] == (288, 25)
        assert shapes['char_encoder.char_lstm.weight_ih_l0'] == (100, 25)
        assert shapes['char_encoder.char_lstm.weight_ih_l0_reverse'] == (100, 25)
        assert shapes['word_lstm.weight_ih_l0'] == (800, 103)
        assert shapes['word_lstm.weight_hh_l0'] == (800, 200)
        assert shapes['word_lstm_dense.weight'] == (200, 400)
        assert shapes['dense_ntags.weight'] == (NTAGS, 200)
        assert shapes['crf.U'] == (NTAGS, NTAGS)

    def test_should_produce_logits_of_the_expected_shape(self):
        model_config = _model_config()
        model = CustomBidLSTM_CRF(model_config, NTAGS)
        model.eval()
        inputs, _ = _batch(model_config)
        logits = model(inputs)['logits']
        assert logits.shape == (2, 4, NTAGS)

    def test_should_decode_a_tag_for_every_token(self):
        model_config = _model_config()
        model = CustomBidLSTM_CRF(model_config, NTAGS)
        model.eval()
        inputs, _ = _batch(model_config)
        decoded = torch.as_tensor(model.decode(inputs))
        assert decoded.shape == (2, 4)

    def test_should_reduce_the_loss_and_train_the_crf_transitions(self):
        model_config = _model_config(dropout=0.0)
        model = CustomBidLSTM_CRF(model_config, NTAGS)
        optimizer = Adam(model.parameters(), lr=0.05)
        inputs, labels = _batch(model_config)
        model.train()
        transitions_before = model.crf.U.detach().clone()  # pylint: disable=not-callable
        first_loss = None
        for _ in range(10):
            optimizer.zero_grad()
            loss = model(inputs, labels)['loss']
            if first_loss is None:
                first_loss = loss.item()
            loss.backward()
            optimizer.step()
        assert loss.item() < first_loss
        assert not torch.equal(
            transitions_before,
            model.crf.U.detach()  # pylint: disable=not-callable
        )

    def test_should_mask_pad_positions_when_masked_crf_loss_is_enabled(self):
        model_config = _model_config(dropout=0.0, masked_crf_loss=True)
        model = CustomBidLSTM_CRF(model_config, NTAGS)
        model.eval()
        inputs, labels = _batch(model_config)
        labels[:, -2:] = 0
        masked_loss = model(inputs, labels)['loss'].item()
        model.config.masked_crf_loss = False
        unmasked_loss = model(inputs, labels)['loss'].item()
        assert masked_loss != unmasked_loss


@pytest.mark.skipif(
    not REFERENCE_CAPTURE_PATH.exists(),
    reason=f'no reference capture in {REFERENCE_CAPTURE_PATH}'
)
class TestWithReferenceCapture:
    """Checks the port against the tensors the TensorFlow model was actually fed.

    This validates the input contract and the output shape. It cannot check the
    values, because that needs the Keras weights mapped into this model, which
    is a separate piece of work.
    """

    def test_should_accept_the_captured_inputs_and_match_the_logits_shape(self):
        metadata = json.loads((REFERENCE_CAPTURE_PATH / 'metadata.json').read_text())
        captured_config = metadata['model_config']
        model_config = _model_config(
            max_feature_size=captured_config['max_feature_size'],
            num_word_lstm_units=captured_config['num_word_lstm_units'],
            char_vocab_size=captured_config['char_vocab_size']
        )
        ntags = len(metadata['preprocessor']['vocab_tag'])

        inputs_npz = np.load(REFERENCE_CAPTURE_PATH / 'inputs.npz')
        logits_npz = np.load(REFERENCE_CAPTURE_PATH / 'pre_crf_logits.npz')
        inputs = {
            name.split('.', 1)[1]: torch.as_tensor(inputs_npz[name])
            for name in inputs_npz.files
            if name.startswith('batch00.')
        }
        inputs['char_input'] = inputs['char_input'].long()

        model = CustomBidLSTM_CRF(model_config, ntags)
        model.eval()
        logits = model(inputs)['logits']
        assert logits.shape == logits_npz['batch00.logits'].shape


class TestUpdatedImplicitModelConfigProps:
    def test_should_enable_features_for_the_stock_features_architecture(self):
        model_config = _model_config(use_features=False)
        model_config.architecture = BidLSTM_CRF_FEATURES.name
        model_config.use_features_indices_input = False
        updated_implicit_model_config_props(model_config)
        assert model_config.use_features
        assert model_config.use_features_indices_input

    def test_should_leave_other_architectures_alone(self):
        model_config = _model_config(use_features=False)
        updated_implicit_model_config_props(model_config)
        assert not model_config.use_features


class TestGetModel:
    def test_should_build_a_registered_architecture(self):
        preprocessor = MagicMock(name='preprocessor')
        model = get_model(_model_config(), preprocessor, ntags=NTAGS)
        assert isinstance(model, CustomBidLSTM_CRF)

    def test_should_set_the_crf_flags_from_the_model(self):
        model_config = _model_config()
        get_model(model_config, MagicMock(name='preprocessor'), ntags=NTAGS)
        assert model_config.use_crf
        assert model_config.use_chain_crf

    def test_should_tell_the_preprocessor_to_return_features(self):
        preprocessor = MagicMock(name='preprocessor')
        get_model(_model_config(), preprocessor, ntags=NTAGS)
        assert preprocessor.return_features is True
        assert preprocessor.return_casing is False

    def test_should_list_the_custom_architecture_among_the_names(self):
        assert 'CustomBidLSTM_CRF' in get_model_names()
        assert 'BidLSTM_CRF' in get_model_names()


class TestIsModelStateful:
    def test_should_report_the_architectures_as_stateless(self):
        model = CustomBidLSTM_CRF(_model_config(), NTAGS)
        assert is_model_stateful(model) is False


FEATURES_INDICES = [9, 10, 11]
FEATURES_VOCABULARY_SIZE = 12
FEATURES_EMBEDDING_SIZE = 4
FEATURES_LSTM_UNITS = 5


def _features_model_config(
    architecture: str = CustomBidLSTM_CRF_FEATURES.name, **kwargs
) -> ModelConfig:
    return ModelConfig(
        architecture=architecture,
        char_vocab_size=int(REFERENCE_CONFIG['char_vocab_size']),
        char_embedding_size=5,
        num_char_lstm_units=4,
        max_char_length=6,
        num_word_lstm_units=6,
        word_embedding_size=0,
        dropout=0.0,
        use_features=True,
        use_features_indices_input=True,
        features_indices=FEATURES_INDICES,
        features_vocabulary_size=FEATURES_VOCABULARY_SIZE,
        features_embedding_size=FEATURES_EMBEDDING_SIZE,
        features_lstm_units=FEATURES_LSTM_UNITS,
        **kwargs
    )


def _features_inputs(batch_size: int = 2, sequence_length: int = 4) -> Dict[str, Any]:
    torch.manual_seed(11)
    return {
        'char_input': torch.randint(
            0, int(REFERENCE_CONFIG['char_vocab_size']), (batch_size, sequence_length, 6)
        ),
        'features_input': torch.randint(
            0, FEATURES_VOCABULARY_SIZE, (batch_size, sequence_length, len(FEATURES_INDICES))
        )
    }


class TestCustomBidLSTMCRFFeatures:
    def test_should_embed_features_rather_than_concatenate_them(self):
        model = CustomBidLSTM_CRF_FEATURES(_features_model_config(), NTAGS)
        # one shared embedding, sized for every value of every feature plus padding
        assert model.features_embedding.num_embeddings == (
            FEATURES_VOCABULARY_SIZE * len(FEATURES_INDICES) + 1
        )
        assert model.features_embedding.padding_idx == 0

    def test_should_size_the_word_lstm_from_the_features_lstm(self):
        model = CustomBidLSTM_CRF_FEATURES(_features_model_config(), NTAGS)
        char_output_size = 4 * 2
        assert model.word_lstm.input_size == char_output_size + FEATURES_LSTM_UNITS * 2

    def test_should_use_the_plain_crf_not_the_chain_crf(self):
        # the Keras model used the default CRF wrapper, unlike CustomBidLSTM_CRF
        assert CustomBidLSTM_CRF_FEATURES.use_crf is True
        assert CustomBidLSTM_CRF_FEATURES.use_chain_crf is False

    def test_should_produce_logits_of_the_expected_shape(self):
        model = CustomBidLSTM_CRF_FEATURES(_features_model_config(), NTAGS)
        model.eval()
        outputs = model(_features_inputs())
        assert outputs['logits'].shape == (2, 4, NTAGS)

    def test_should_decode_a_tag_for_every_token_as_a_tensor(self):
        model = CustomBidLSTM_CRF_FEATURES(_features_model_config(), NTAGS)
        model.eval()
        decoded = model.decode(_features_inputs())
        # the tagger and the scorer expect a tensor, as ChainCRF returns
        assert isinstance(decoded, torch.Tensor)
        assert decoded.shape == (2, 4)

    def test_should_reduce_the_loss_over_steps(self):
        model = CustomBidLSTM_CRF_FEATURES(_features_model_config(), NTAGS)
        inputs = _features_inputs()
        labels = torch.randint(1, NTAGS, (2, 4))
        optimizer = Adam(model.parameters(), lr=0.05)
        first_loss = model(inputs, labels)['loss'].item()
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(inputs, labels)['loss']
            loss.backward()
            optimizer.step()
        assert loss.item() < first_loss

    def test_should_be_built_by_get_model(self):
        preprocessor = MagicMock(name='preprocessor')
        model_config = _features_model_config()
        model = get_model(model_config, preprocessor, ntags=NTAGS)
        assert isinstance(model, CustomBidLSTM_CRF_FEATURES)
        assert model_config.use_chain_crf is False
        assert preprocessor.return_features is True


class TestToTagIndicesArray:
    def test_should_pass_through_a_tensor(self):
        result = to_tag_indices_array(torch.tensor([[1, 2], [3, 4]]))
        assert result.tolist() == [[1, 2], [3, 4]]

    def test_should_convert_the_list_the_plain_crf_returns(self):
        assert to_tag_indices_array([[1, 2], [3, 4]]).tolist() == [[1, 2], [3, 4]]


class TestGetModelForUpstreamArchitecture:
    def test_should_configure_the_preprocessor_for_an_upstream_architecture(self):
        # the delegating path used to skip this, leaving the data generator
        # without the features the architecture requires
        preprocessor = MagicMock(name='preprocessor')
        model_config = _features_model_config(
            architecture=BidLSTM_CRF_FEATURES.name
        )
        get_model(model_config, preprocessor, ntags=NTAGS)
        assert preprocessor.return_features is True
        assert model_config.use_crf is True
        assert model_config.use_chain_crf is False
