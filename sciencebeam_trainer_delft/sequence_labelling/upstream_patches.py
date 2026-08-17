"""Local fixes for defects in the installed delft, applied at runtime.

Each patch is conditional on the defect still being present, so a delft
release that fixes it makes the patch a no-op without any change here. The
corresponding tests in ``tests/sequence_labelling/delft_upstream_test.py`` are
strict xfails and will go red on such a release, which is the prompt to delete
the patch rather than to leave it running.
"""
import logging
from typing import Dict, Optional

import torch
from torch import nn

from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.models import BidLSTM_CRF, CharacterEncoder
from delft.utilities.crf_pytorch import ChainCRF


LOGGER = logging.getLogger(__name__)


# kept so that the tests guarding the upstream defect can restore it
ORIGINAL_CHAIN_CRF_INIT = ChainCRF.__init__


def _chain_crf_init_with_eager_build(self, num_tags: Optional[int] = None):
    # upstream annotates num_tags as int while defaulting it to None
    ORIGINAL_CHAIN_CRF_INIT(self, num_tags)  # type: ignore[arg-type]
    if num_tags:
        # upstream defers this to the first forward pass, by which point the
        # optimizer has already been constructed without these parameters
        self.build(num_tags)


def is_chain_crf_eager_build_required() -> bool:
    return not ChainCRF(1).state_dict()


def patch_chain_crf_eager_build():
    """Register the CRF transition parameters at construction time.

    Without this, ``U``, ``b_start`` and ``b_end`` appear only after the first
    forward pass: they are missing from any optimizer built before it, so the
    transitions never train, and missing from a freshly constructed model's
    ``state_dict``, so loading a saved model raises on unexpected keys.
    """
    if not is_chain_crf_eager_build_required():
        LOGGER.debug('ChainCRF already builds its parameters eagerly')
        return
    ChainCRF.__init__ = _chain_crf_init_with_eager_build  # type: ignore[method-assign]
    LOGGER.info('patched ChainCRF to build its parameters eagerly')


# kept so that the tests guarding the upstream defect can restore it
ORIGINAL_BID_LSTM_CRF_INIT = BidLSTM_CRF.__init__


class MaskedCharacterEncoder(CharacterEncoder):
    """A `CharacterEncoder` that skips padded character positions.

    The Keras implementations set `mask_zero` per architecture -- `True` for
    `BidLSTM_CRF`, `False` for `BidLSTM_ChainCRF`, and from the config for this
    repo's `CustomBidLSTM_CRF`. delft 1.0.x has one shared encoder that
    implements the unmasked behaviour, so the architectures that masked lost it.

    With masking, a bidirectional LSTM returning only its final state returns
    the state at the last *real* character. Without it, the LSTM runs on through
    the padding -- for a three-character token in a thirty-character window,
    that is twenty-seven further steps, and the state that reaches the rest of
    the model is mostly a function of the padding embedding.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, max_char_length = x.shape
        flattened = x.reshape(batch_size * sequence_length, max_char_length)
        lengths = (flattened != 0).sum(dim=1)
        embedded = self.char_embeddings(flattened)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            # a length of zero is not packable; those rows are zeroed below
            lengths.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, (hidden, _) = self.bilstm(packed)
        encoded = torch.cat([hidden[0], hidden[1]], dim=-1)
        # a token that is entirely padding is masked out completely in Keras,
        # leaving the initial state rather than whatever one step produced
        encoded = encoded * (lengths > 0).unsqueeze(-1).to(encoded.dtype)
        return encoded.view(batch_size, sequence_length, self.output_size)


def _bid_lstm_crf_init_with_char_masking(self, config, ntags: Optional[int] = None):
    # upstream annotates ntags as int while defaulting it to None
    ORIGINAL_BID_LSTM_CRF_INIT(self, config, ntags)  # type: ignore[arg-type]
    # replacing the class keeps the module and its parameters, so the state
    # dict is unchanged and only the forward pass differs
    self.char_encoder.__class__ = MaskedCharacterEncoder


def is_char_encoder_masking_required() -> bool:
    """Reports whether the installed `CharacterEncoder` ignores padding.

    Trailing padding must not change the encoding, so the same token in two
    differently sized windows has to encode identically.
    """
    encoder = CharacterEncoder(4, 3, 2)
    encoder.eval()
    with torch.no_grad():
        narrow = encoder(torch.tensor([[[1, 2, 0]]]))
        wide = encoder(torch.tensor([[[1, 2, 0, 0, 0]]]))
    return not torch.allclose(narrow, wide, atol=1e-6)


def patch_bid_lstm_crf_char_masking():
    """Restore the character masking `BidLSTM_CRF` had under Keras.

    Only `BidLSTM_CRF` is patched. `BidLSTM_ChainCRF` had `mask_zero=False` and
    converts exactly as it stands, so patching the shared encoder itself would
    break it.
    """
    if not is_char_encoder_masking_required():
        LOGGER.debug('CharacterEncoder already masks padded characters')
        return
    BidLSTM_CRF.__init__ = (  # type: ignore[method-assign]
        _bid_lstm_crf_init_with_char_masking
    )
    LOGGER.info('patched BidLSTM_CRF to mask padded characters')


# kept so that the tests guarding the upstream defect can restore them
ORIGINAL_BID_LSTM_CRF_FORWARD = BidLSTM_CRF.forward
ORIGINAL_BID_LSTM_CRF_DECODE = BidLSTM_CRF.decode


def _get_token_mask(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Marks the token positions that are not padding.

    Keras derived this from the character embedding's own mask, reduced over
    the character axis, so a token counts as padding when every one of its
    characters does. Taking it from `char_input` rather than from the `length`
    input reproduces that, and works whether or not a length was supplied.
    """
    mask = (inputs['char_input'] != 0).any(dim=-1)
    # the CRF requires the first position of every sequence to be unmasked; a
    # sequence that is padding throughout would otherwise be rejected outright
    mask[:, 0] = True
    return mask


def _run_masked_lstm(
    lstm: nn.LSTM, x: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    lengths = mask.sum(dim=1)
    packed = nn.utils.rnn.pack_padded_sequence(
        x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
    )
    packed_output, _ = lstm(packed)
    output, _ = nn.utils.rnn.pad_packed_sequence(
        packed_output, batch_first=True, total_length=x.shape[1]
    )
    return output


def _bid_lstm_crf_forward_with_token_masking(
    self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    mask = _get_token_mask(inputs)
    char_encoded = self.char_encoder(inputs['char_input'])
    x = torch.cat([inputs['word_input'], char_encoded], dim=-1)
    x = self.dropout(x)
    lstm_output = self.dropout(_run_masked_lstm(self.bilstm, x, mask))
    emissions = self.linear(torch.tanh(self.dense(lstm_output)))
    outputs = {'logits': emissions}
    if labels is not None:
        outputs['loss'] = self.crf(emissions, labels, mask=mask)
    return outputs


def _bid_lstm_crf_decode_with_token_masking(
    self, inputs: Dict[str, torch.Tensor]
) -> torch.Tensor:
    with torch.no_grad():
        mask = _get_token_mask(inputs)
        emissions = self.forward(inputs)['logits']
        decoded = self.crf.decode(emissions, mask=mask)
    # a masked decode returns one list per sequence, of that sequence's length;
    # the caller expects one tag per position, so the padding is filled back in
    sequence_length = emissions.shape[1]
    return torch.tensor([
        tags + [0] * (sequence_length - len(tags))
        for tags in decoded
    ])


def is_bid_lstm_crf_token_masking_required() -> bool:
    """Reports whether padding a batch changes the real positions' logits.

    Under Keras the mask reached the word LSTM, so a document scored the same
    whatever it was batched with. Without it the backward direction starts in
    the padding and runs back through it before reaching any real token, so the
    padding reaches every position rather than only its own.
    """
    config = ModelConfig(
        architecture='BidLSTM_CRF',
        word_embedding_size=2,
        char_emb_size=3,
        char_lstm_units=2,
        word_lstm_units=2,
        dropout=0.0,
        recurrent_dropout=0.0,
        use_crf=True,
        use_chain_crf=False
    )
    config.char_vocab_size = 4
    model = BidLSTM_CRF(config, 3)
    model.eval()
    char_input = torch.tensor([[[1, 2], [3, 1], [0, 0]]])
    inputs = {
        'word_input': torch.zeros(1, 3, 2),
        'char_input': char_input
    }
    unpadded = {
        'word_input': torch.zeros(1, 2, 2),
        'char_input': char_input[:, :2]
    }
    with torch.no_grad():
        padded_logits = model(inputs)['logits'][:, :2]
        unpadded_logits = model(unpadded)['logits']
    return not torch.allclose(padded_logits, unpadded_logits, atol=1e-6)


def patch_bid_lstm_crf_token_masking():
    """Restore the token masking `BidLSTM_CRF` had under Keras.

    Without it the word LSTM runs through the padded token positions, so a
    document's predictions depend on what it happens to be batched with. The
    character masking alone is not enough: it fixes what each token encodes to,
    not whether padded tokens take part in the sequence over them.
    """
    if not is_bid_lstm_crf_token_masking_required():
        LOGGER.debug('BidLSTM_CRF already masks padded tokens')
        return
    BidLSTM_CRF.forward = (  # type: ignore[method-assign]
        _bid_lstm_crf_forward_with_token_masking
    )
    BidLSTM_CRF.decode = (  # type: ignore[method-assign]
        _bid_lstm_crf_decode_with_token_masking
    )
    LOGGER.info('patched BidLSTM_CRF to mask padded tokens')
