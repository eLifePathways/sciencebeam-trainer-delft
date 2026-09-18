"""Guards upstream delft behaviour the architectures here depend on.

The xfail markers are strict: when a fixed delft is released, these tests pass
unexpectedly and the run fails, which is the signal to drop the marker and the
patch it guards.

What is covered here is what this repo relies on and upstream does not yet
promise. The character masking is the other way round -- upstream promises it
now, so `TestCharacterEncoderMasking` asserts it rather than expecting it to
fail.
"""
import pytest
import torch
from torch import nn

from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.models import (
    BidLSTM_ChainCRF,
    BidLSTM_CRF,
    CharacterEncoder
)


NTAGS = 5
CHAR_VOCAB_SIZE = 12
MAX_CHAR_LENGTH = 5


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


class TestRecurrentDropout:
    @pytest.mark.xfail(
        strict=True,
        reason='delft 1.1.0 passes recurrent_dropout to a single-layer nn.LSTM'
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


@pytest.fixture(name='bid_lstm_crf_config')
def _bid_lstm_crf_config(model_config: ModelConfig) -> ModelConfig:
    model_config.architecture = 'BidLSTM_CRF'
    model_config.use_chain_crf = False
    return model_config


class TestCharacterEncoderMasking:
    """`BidLSTM_CRF` has to keep the masking its Keras counterpart had.

    Upstream's shared `CharacterEncoder` is unmasked by default and masks on
    request, which is what the Keras implementations did per architecture. These
    assert the choice each architecture makes, not the default: changing the
    default would be upstream's business, changing `BidLSTM_CRF` would break a
    published model.
    """

    def test_should_not_let_trailing_padding_change_a_masked_encoding(self):
        encoder = CharacterEncoder(CHAR_VOCAB_SIZE, 4, 3, mask_zero=True)
        encoder.eval()
        with torch.no_grad():
            narrow = encoder(torch.tensor([[[1, 2, 0]]]))
            wide = encoder(torch.tensor([[[1, 2, 0, 0, 0]]]))
        assert torch.allclose(narrow, wide, atol=1e-6)

    def test_should_mask_the_character_encoder_of_bid_lstm_crf(
        self, bid_lstm_crf_config: ModelConfig
    ):
        model = BidLSTM_CRF(bid_lstm_crf_config, NTAGS)
        assert model.char_encoder.mask_zero is True

    def test_should_not_let_batch_padding_change_a_documents_logits(
        self, bid_lstm_crf_config: ModelConfig
    ):
        # the Keras mask reached the word LSTM, so a document scored the same
        # whatever it was batched with; the backward direction otherwise starts
        # in the padding and carries it into every real position
        model = BidLSTM_CRF(bid_lstm_crf_config, NTAGS)
        model.eval()
        char_input = torch.tensor([[[1, 2], [3, 1], [0, 0]]])
        padded = {'word_input': torch.zeros(1, 3, 8), 'char_input': char_input}
        unpadded = {'word_input': torch.zeros(1, 2, 8), 'char_input': char_input[:, :2]}
        with torch.no_grad():
            assert torch.allclose(
                model(padded)['logits'][:, :2], model(unpadded)['logits'], atol=1e-6
            )
