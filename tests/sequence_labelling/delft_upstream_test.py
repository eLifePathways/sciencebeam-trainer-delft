"""Guards upstream delft behaviour this repo's published models depend on.

Every model published from this repo uses the ChainCRF path, so the defects
covered here are not avoidable by configuration: the CRF transitions are left
out of the optimizer, and a saved model cannot be loaded back.

The xfail markers are strict: when a fixed delft is released, these tests pass
unexpectedly and the run fails, which is the signal to drop the marker and the
version fallback along with it.
"""
from typing import Any

import pytest
import torch
from torch import nn
from torch.optim import Adam

from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.models import BidLSTM_ChainCRF
from delft.utilities.crf_pytorch import ChainCRF

from sciencebeam_trainer_delft.sequence_labelling import upstream_patches


NTAGS = 5
CHAR_VOCAB_SIZE = 12
MAX_CHAR_LENGTH = 5

CHAIN_CRF_PARAMETER_NAMES = ['U', 'b_start', 'b_end']

LAZY_CHAIN_CRF_REASON = (
    'delft 1.0.1 ChainCRF creates U/b_start/b_end on the first forward pass'
)


@pytest.fixture(name='unpatched_chain_crf', autouse=True)
def _unpatched_chain_crf():
    """Restores upstream's own ChainCRF for the duration of each test.

    `get_model` patches it for the rest of the process, so without this these
    tests would report on the patch rather than on the installed delft.
    """
    patched_init = ChainCRF.__init__
    ChainCRF.__init__ = (  # type: ignore[method-assign]
        upstream_patches.ORIGINAL_CHAIN_CRF_INIT
    )
    yield
    ChainCRF.__init__ = patched_init  # type: ignore[method-assign]


def _snapshot(tensor: Any) -> torch.Tensor:
    # upstream annotates the CRF parameters as None until they are built
    return tensor.detach().clone()  # pylint: disable=not-callable


@pytest.fixture(name='model_config')
def _model_config() -> ModelConfig:
    model_config = ModelConfig(
        architecture='BidLSTM_ChainCRF',
        word_embedding_size=8,
        char_emb_size=4,
        char_lstm_units=3,
        word_lstm_units=6,
        dropout=0.0,
        recurrent_dropout=0.3,
        use_crf=True,
        use_chain_crf=True
    )
    model_config.char_vocab_size = CHAR_VOCAB_SIZE
    return model_config


@pytest.fixture(name='batch')
def _batch(model_config: ModelConfig):
    torch.manual_seed(42)
    inputs = {
        'word_input': torch.randn(2, 4, model_config.word_embedding_size),
        'char_input': torch.randint(0, CHAR_VOCAB_SIZE, (2, 4, MAX_CHAR_LENGTH))
    }
    labels = torch.randint(0, NTAGS, (2, 4))
    return inputs, labels


class TestChainCRF:
    @pytest.mark.xfail(strict=True, reason=LAZY_CHAIN_CRF_REASON)
    def test_should_register_parameters_before_first_forward_pass(self):
        chain_crf = ChainCRF(NTAGS)
        assert sorted(chain_crf.state_dict().keys()) == sorted(CHAIN_CRF_PARAMETER_NAMES)

    @pytest.mark.xfail(strict=True, reason=LAZY_CHAIN_CRF_REASON)
    def test_should_train_transitions_via_optimizer_created_before_first_forward_pass(
        self, model_config: ModelConfig, batch
    ):
        # the trainer compiles the optimizer before the first batch, so any
        # parameter created later is silently left out of it
        model = BidLSTM_ChainCRF(model_config, NTAGS)
        optimizer = Adam(model.parameters(), lr=0.1)
        inputs, labels = batch
        model.train()
        model(inputs, labels)
        transitions_before = _snapshot(model.crf.U)
        dense_before = _snapshot(model.dense2.weight)
        for _ in range(5):
            optimizer.zero_grad()
            model(inputs, labels)['loss'].backward()
            optimizer.step()
        # the dense layer is the control: training itself is working
        assert not torch.equal(dense_before, _snapshot(model.dense2.weight))
        assert not torch.equal(transitions_before, _snapshot(model.crf.U))

    @pytest.mark.xfail(strict=True, reason=LAZY_CHAIN_CRF_REASON)
    def test_should_load_saved_model_into_fresh_instance(
        self, model_config: ModelConfig, batch
    ):
        model = BidLSTM_ChainCRF(model_config, NTAGS)
        inputs, labels = batch
        model(inputs, labels)
        fresh_model = BidLSTM_ChainCRF(model_config, NTAGS)
        fresh_model.load_state_dict(model.state_dict())


class TestRecurrentDropout:
    @pytest.mark.xfail(
        strict=True,
        reason='delft 1.0.1 passes recurrent_dropout to a single-layer nn.LSTM'
    )
    def test_should_not_pass_dropout_to_a_single_layer_lstm(
        self, model_config: ModelConfig
    ):
        # nn.LSTM applies dropout between layers, so a single-layer LSTM
        # discards it: whatever recurrent_dropout was configured has no effect
        model = BidLSTM_ChainCRF(model_config, NTAGS)
        assert model_config.recurrent_dropout
        assert not [
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.LSTM) and module.num_layers == 1 and module.dropout
        ]
