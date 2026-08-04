from typing import Any, Iterator

import pytest
import torch
from torch.optim import Adam

from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.models import BidLSTM_ChainCRF
from delft.utilities.crf_pytorch import ChainCRF

from sciencebeam_trainer_delft.sequence_labelling.upstream_patches import (
    is_chain_crf_eager_build_required,
    patch_chain_crf_eager_build
)


NTAGS = 5
CHAR_VOCAB_SIZE = 12
MAX_CHAR_LENGTH = 5

CHAIN_CRF_PARAMETER_NAMES = ['U', 'b_start', 'b_end']


def _snapshot(tensor: Any) -> torch.Tensor:
    # upstream annotates the CRF parameters as None until they are built
    return tensor.detach().clone()  # pylint: disable=not-callable


@pytest.fixture(name='restore_chain_crf', autouse=True)
def _restore_chain_crf() -> Iterator[None]:
    original_init = ChainCRF.__init__
    yield
    ChainCRF.__init__ = original_init  # type: ignore[method-assign]


@pytest.fixture(name='model_config')
def _model_config() -> ModelConfig:
    model_config = ModelConfig(
        architecture='BidLSTM_ChainCRF',
        word_embedding_size=8,
        char_emb_size=4,
        char_lstm_units=3,
        word_lstm_units=6,
        dropout=0.0,
        recurrent_dropout=0.0,
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


class TestPatchChainCrfEagerBuild:
    def test_should_register_parameters_before_first_forward_pass(self):
        patch_chain_crf_eager_build()
        chain_crf = ChainCRF(NTAGS)
        assert sorted(chain_crf.state_dict().keys()) == sorted(CHAIN_CRF_PARAMETER_NAMES)

    def test_should_train_transitions_via_optimizer_created_before_first_forward_pass(
        self, model_config: ModelConfig, batch
    ):
        patch_chain_crf_eager_build()
        model = BidLSTM_ChainCRF(model_config, NTAGS)
        optimizer = Adam(model.parameters(), lr=0.1)
        inputs, labels = batch
        model.train()
        transitions_before = _snapshot(model.crf.U)
        dense_before = _snapshot(model.dense2.weight)
        for _ in range(5):
            optimizer.zero_grad()
            model(inputs, labels)['loss'].backward()
            optimizer.step()
        assert not torch.equal(dense_before, _snapshot(model.dense2.weight))
        assert not torch.equal(transitions_before, _snapshot(model.crf.U))

    def test_should_load_saved_model_into_fresh_instance(
        self, model_config: ModelConfig, batch
    ):
        patch_chain_crf_eager_build()
        model = BidLSTM_ChainCRF(model_config, NTAGS)
        inputs, labels = batch
        model(inputs, labels)
        fresh_model = BidLSTM_ChainCRF(model_config, NTAGS)
        fresh_model.load_state_dict(model.state_dict())
        assert torch.equal(_snapshot(fresh_model.crf.U), _snapshot(model.crf.U))

    def test_should_keep_lazy_build_without_num_tags(self):
        patch_chain_crf_eager_build()
        assert not ChainCRF().state_dict()

    def test_should_not_apply_twice(self):
        patch_chain_crf_eager_build()
        patched_init = ChainCRF.__init__
        patch_chain_crf_eager_build()
        assert ChainCRF.__init__ is patched_init

    def test_should_report_the_defect_as_no_longer_present_once_patched(self):
        assert is_chain_crf_eager_build_required()
        patch_chain_crf_eager_build()
        assert not is_chain_crf_eager_build_required()
