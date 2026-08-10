"""Local fixes for defects in the installed delft, applied at runtime.

Each patch is conditional on the defect still being present, so a delft
release that fixes it makes the patch a no-op without any change here. The
corresponding tests in ``tests/sequence_labelling/delft_upstream_test.py`` are
strict xfails and will go red on such a release, which is the prompt to delete
the patch rather than to leave it running.
"""
import logging
from typing import Optional

import torch
from torch import nn

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
