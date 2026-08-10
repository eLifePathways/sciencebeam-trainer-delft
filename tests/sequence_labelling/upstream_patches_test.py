from typing import Any, Iterator

import pytest
import torch
from torch.optim import Adam

from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.models import BidLSTM_ChainCRF, BidLSTM_CRF
from delft.utilities.crf_pytorch import ChainCRF

from sciencebeam_trainer_delft.sequence_labelling.upstream_patches import (
    ORIGINAL_BID_LSTM_CRF_DECODE,
    ORIGINAL_BID_LSTM_CRF_FORWARD,
    ORIGINAL_BID_LSTM_CRF_INIT,
    ORIGINAL_CHAIN_CRF_INIT,
    is_bid_lstm_crf_token_masking_required,
    is_char_encoder_masking_required,
    is_chain_crf_eager_build_required,
    patch_bid_lstm_crf_char_masking,
    patch_bid_lstm_crf_token_masking,
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
    """Starts each test from upstream's own ChainCRF.

    Importing the models module patches it for the rest of the process, and
    these tests are about what the patch changes, so they have to begin from
    the unpatched behaviour.
    """
    patched_init = ChainCRF.__init__
    ChainCRF.__init__ = ORIGINAL_CHAIN_CRF_INIT  # type: ignore[method-assign]
    yield
    ChainCRF.__init__ = patched_init  # type: ignore[method-assign]


@pytest.fixture(name='restore_bid_lstm_crf', autouse=True)
def _restore_bid_lstm_crf() -> Iterator[None]:
    """Starts each test from upstream's own BidLSTM_CRF, as above."""
    patched_init = BidLSTM_CRF.__init__
    patched_forward = BidLSTM_CRF.forward
    patched_decode = BidLSTM_CRF.decode
    BidLSTM_CRF.__init__ = ORIGINAL_BID_LSTM_CRF_INIT  # type: ignore[method-assign]
    BidLSTM_CRF.forward = ORIGINAL_BID_LSTM_CRF_FORWARD  # type: ignore[method-assign]
    BidLSTM_CRF.decode = ORIGINAL_BID_LSTM_CRF_DECODE  # type: ignore[method-assign]
    yield
    BidLSTM_CRF.__init__ = patched_init  # type: ignore[method-assign]
    BidLSTM_CRF.forward = patched_forward  # type: ignore[method-assign]
    BidLSTM_CRF.decode = patched_decode  # type: ignore[method-assign]


@pytest.fixture(name='bid_lstm_crf_config')
def _bid_lstm_crf_config(model_config: ModelConfig) -> ModelConfig:
    model_config.architecture = 'BidLSTM_CRF'
    model_config.use_chain_crf = False
    return model_config


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


class TestPatchBidLstmCrfCharMasking:
    def test_should_make_the_encoding_independent_of_trailing_padding(
        self, bid_lstm_crf_config: ModelConfig
    ):
        patch_bid_lstm_crf_char_masking()
        encoder = BidLSTM_CRF(bid_lstm_crf_config, NTAGS).char_encoder
        encoder.eval()
        with torch.no_grad():
            narrow = encoder(torch.tensor([[[1, 2, 0]]]))
            wide = encoder(torch.tensor([[[1, 2, 0, 0, 0]]]))
        assert torch.allclose(narrow, wide, atol=1e-6)

    def test_should_leave_bid_lstm_chain_crf_unmasked(
        self, model_config: ModelConfig
    ):
        # its Keras counterpart set mask_zero=False, and it converts exactly as
        # upstream stands, so masking it would be the regression
        patch_bid_lstm_crf_char_masking()
        encoder = BidLSTM_ChainCRF(model_config, NTAGS).char_encoder
        encoder.eval()
        with torch.no_grad():
            narrow = encoder(torch.tensor([[[1, 2, 0]]]))
            wide = encoder(torch.tensor([[[1, 2, 0, 0, 0]]]))
        assert not torch.allclose(narrow, wide, atol=1e-6)

    def test_should_keep_the_state_dict_unchanged(
        self, bid_lstm_crf_config: ModelConfig
    ):
        unpatched_keys = sorted(BidLSTM_CRF(bid_lstm_crf_config, NTAGS).state_dict())
        patch_bid_lstm_crf_char_masking()
        assert sorted(BidLSTM_CRF(bid_lstm_crf_config, NTAGS).state_dict()) == unpatched_keys

    def test_should_encode_a_fully_padded_token_as_zeros(
        self, bid_lstm_crf_config: ModelConfig
    ):
        patch_bid_lstm_crf_char_masking()
        encoder = BidLSTM_CRF(bid_lstm_crf_config, NTAGS).char_encoder
        encoder.eval()
        with torch.no_grad():
            encoded = encoder(torch.tensor([[[0, 0, 0]]]))
        assert torch.allclose(encoded, torch.zeros_like(encoded))

    def test_should_not_apply_twice(self):
        patch_bid_lstm_crf_char_masking()
        patched_init = BidLSTM_CRF.__init__
        patch_bid_lstm_crf_char_masking()
        assert BidLSTM_CRF.__init__ is patched_init

    def test_should_keep_reporting_the_upstream_defect_after_patching(self):
        # unlike the ChainCRF patch, this one replaces the encoder on one
        # architecture rather than fixing the shared class, so the detector
        # goes on reporting the defect -- which is what has to change when
        # upstream fixes it, and is the signal to drop this patch
        assert is_char_encoder_masking_required()
        patch_bid_lstm_crf_char_masking()
        assert is_char_encoder_masking_required()


class TestPatchBidLstmCrfTokenMasking:
    def test_should_make_a_documents_logits_independent_of_batch_padding(
        self, bid_lstm_crf_config: ModelConfig
    ):
        patch_bid_lstm_crf_token_masking()
        model = BidLSTM_CRF(bid_lstm_crf_config, NTAGS)
        model.eval()
        char_input = torch.tensor([[[1, 2], [3, 1], [0, 0]]])
        padded = {'word_input': torch.zeros(1, 3, 8), 'char_input': char_input}
        unpadded = {'word_input': torch.zeros(1, 2, 8), 'char_input': char_input[:, :2]}
        with torch.no_grad():
            assert torch.allclose(
                model(padded)['logits'][:, :2], model(unpadded)['logits'], atol=1e-6
            )

    def test_should_decode_one_tag_per_position_including_padding(
        self, bid_lstm_crf_config: ModelConfig
    ):
        # a masked decode returns only the real positions, where the caller
        # expects a rectangular result it can trim itself
        patch_bid_lstm_crf_token_masking()
        model = BidLSTM_CRF(bid_lstm_crf_config, NTAGS)
        model.eval()
        inputs = {
            'word_input': torch.zeros(2, 3, 8),
            'char_input': torch.tensor([[[1, 2], [3, 1], [0, 0]], [[1, 1], [2, 3], [1, 2]]])
        }
        decoded = model.decode(inputs)
        assert torch.as_tensor(decoded).shape == (2, 3)

    def test_should_leave_bid_lstm_chain_crf_alone(self, model_config: ModelConfig):
        patch_bid_lstm_crf_token_masking()
        model = BidLSTM_ChainCRF(model_config, NTAGS)
        model.eval()
        char_input = torch.tensor([[[1, 2], [3, 1], [0, 0]]])
        padded = {'word_input': torch.zeros(1, 3, 8), 'char_input': char_input}
        unpadded = {'word_input': torch.zeros(1, 2, 8), 'char_input': char_input[:, :2]}
        with torch.no_grad():
            assert not torch.allclose(
                model(padded)['logits'][:, :2], model(unpadded)['logits'], atol=1e-6
            )

    def test_should_not_apply_twice(self):
        patch_bid_lstm_crf_token_masking()
        patched_forward = BidLSTM_CRF.forward
        patch_bid_lstm_crf_token_masking()
        assert BidLSTM_CRF.forward is patched_forward

    def test_should_report_the_defect_as_no_longer_present_once_patched(self):
        assert is_bid_lstm_crf_token_masking_required()
        patch_bid_lstm_crf_token_masking()
        assert not is_bid_lstm_crf_token_masking_required()
