"""Local fixes for defects in the installed delft, applied at runtime.

Each patch is conditional on the defect still being present, so a delft
release that fixes it makes the patch a no-op without any change here. The
corresponding tests in ``tests/sequence_labelling/delft_upstream_test.py`` are
strict xfails and will go red on such a release, which is the prompt to delete
the patch rather than to leave it running.
"""
import logging
from typing import Dict, Optional, cast

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


# kept so that the tests guarding the upstream defects can restore them
ORIGINAL_CHAIN_CRF_DECODE = ChainCRF.decode
ORIGINAL_CHAIN_CRF_FREE_ENERGY = ChainCRF._free_energy  # pylint: disable=protected-access


# A two-tag CRF, chosen so that both defects change the answer rather than
# happening to agree. The transition matrix is asymmetric, so the best previous
# tag depends on which tag follows it -- the case the decode defect turns on --
# and the padding's emissions pull hard towards the second tag.
_PROBE_TRANSITIONS = torch.tensor([[0.0, 10.0], [0.0, -1.0]])
_PROBE_EMISSIONS = torch.zeros(1, 2, 2)
_PROBE_PADDING_EMISSIONS = torch.tensor([[[0.0, 100.0]]])


def _get_probe_chain_crf() -> ChainCRF:
    """A ChainCRF with fixed parameters, so the probes below are deterministic."""
    ntags = _PROBE_TRANSITIONS.shape[0]
    crf = ChainCRF(ntags)
    if getattr(crf, 'U', None) is None:
        crf.build(ntags)
    with torch.no_grad():
        # upstream annotates the CRF parameters as None until they are built
        cast(torch.Tensor, crf.U).copy_(_PROBE_TRANSITIONS)
        cast(torch.Tensor, crf.b_start).zero_()
        cast(torch.Tensor, crf.b_end).zero_()
    crf.eval()
    return crf


def _get_probe_inputs():
    """Returns `(emissions, mask, padded_emissions, padded_mask)` to compare."""
    real_length = _PROBE_EMISSIONS.shape[1]
    padding_length = _PROBE_PADDING_EMISSIONS.shape[1]
    mask = torch.ones(1, real_length, dtype=torch.bool)
    padded_emissions = torch.cat(
        [_PROBE_EMISSIONS, _PROBE_PADDING_EMISSIONS], dim=1
    )
    padded_mask = torch.cat(
        [mask, torch.zeros(1, padding_length, dtype=torch.bool)], dim=1
    )
    return _PROBE_EMISSIONS, mask, padded_emissions, padded_mask


def _chain_crf_free_energy_with_masked_positions(
    self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    _, sequence_length, _ = x.shape
    alpha = x[:, 0, :]
    for index in range(1, sequence_length):
        next_alpha = (
            alpha.unsqueeze(2) + self.U.unsqueeze(0) + x[:, index, :].unsqueeze(1)
        )
        updated = torch.logsumexp(next_alpha, dim=1)
        if mask is None:
            alpha = updated
        else:
            # a masked step carries the previous alpha forward rather than
            # accumulating the padding's emissions into the partition function
            alpha = torch.where(mask[:, index:index + 1].bool(), updated, alpha)
    return torch.logsumexp(alpha, dim=1)


def is_chain_crf_masked_free_energy_required() -> bool:
    """Reports whether padding changes the CRF loss despite a mask.

    Upstream's masked branch reads ``torch.where(mask, alpha, alpha)``, which
    returns the same value either way, so the partition function keeps
    accumulating over the padded positions.
    """
    crf = _get_probe_chain_crf()
    emissions, mask, padded_emissions, padded_mask = _get_probe_inputs()
    labels = torch.ones(1, emissions.shape[1], dtype=torch.long)
    padded_labels = torch.zeros(1, padded_emissions.shape[1], dtype=torch.long)
    padded_labels[:, :labels.shape[1]] = labels
    with torch.no_grad():
        unpadded_loss = crf(emissions, labels, mask=mask)
        padded_loss = crf(padded_emissions, padded_labels, mask=padded_mask)
    return not torch.allclose(unpadded_loss, padded_loss, atol=1e-5)


def patch_chain_crf_masked_free_energy():
    """Make the CRF partition function ignore masked positions.

    Without this a mask reduces the path energy only, so the loss still depends
    on how much padding the batch happened to carry -- which defeats
    ``--masked-crf-loss`` and any token masking in front of it.
    """
    if not is_chain_crf_masked_free_energy_required():
        LOGGER.debug('ChainCRF already ignores masked positions in the free energy')
        return
    ChainCRF._free_energy = (  # type: ignore[method-assign] # pylint: disable=protected-access
        _chain_crf_free_energy_with_masked_positions
    )
    LOGGER.info('patched ChainCRF to ignore masked positions in the free energy')


def _chain_crf_decode_from_last_real_position(
    self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    batch_size, sequence_length, _ = emissions.shape
    device = emissions.device
    x = self._add_boundary_energy(emissions, mask)  # pylint: disable=protected-access
    alpha = x[:, 0, :]
    backpointers = []
    for index in range(1, sequence_length):
        next_score = (
            alpha.unsqueeze(2) + self.U.unsqueeze(0) + x[:, index, :].unsqueeze(1)
        )
        updated, backpointer = next_score.max(dim=1)
        backpointers.append(backpointer)
        if mask is None:
            alpha = updated
        else:
            # carry the previous alpha forward, so it still holds the scores at
            # the last real position once the loop reaches the end
            alpha = torch.where(mask[:, index:index + 1].bool(), updated, alpha)

    if mask is None:
        lengths = torch.full(
            (batch_size,), sequence_length, dtype=torch.long, device=device
        )
    else:
        lengths = mask.long().sum(dim=1)

    best_paths = torch.zeros(
        batch_size, sequence_length, dtype=torch.long, device=device
    )
    # start the backtrack at each sequence's own last real position rather than
    # at the end of the padded tensor
    best_paths.scatter_(
        1,
        (lengths - 1).clamp(min=0).unsqueeze(1),
        alpha.argmax(dim=1, keepdim=True)
    )
    for index in range(sequence_length - 2, -1, -1):
        follows = backpointers[index].gather(
            1, best_paths[:, index + 1:index + 2]
        ).squeeze(1)
        # only follow the backpointer where the next position is a real one; at
        # the last real position the tag is the one scattered above
        best_paths[:, index] = torch.where(
            (index + 1) < lengths, follows, best_paths[:, index]
        )

    if mask is not None:
        best_paths = best_paths * mask.long()
    return best_paths


def is_chain_crf_masked_decode_required() -> bool:
    """Reports whether padding changes the decoded tags despite a mask.

    Upstream resets alpha to the emissions at a masked step and then starts the
    backtrack from the end of the padded tensor, so the walk back through the
    padding decides the last real token's tag and, through it, every tag before.
    """
    crf = _get_probe_chain_crf()
    emissions, mask, padded_emissions, padded_mask = _get_probe_inputs()
    real_length = emissions.shape[1]
    with torch.no_grad():
        unpadded = crf.decode(emissions, mask=mask)
        padded = crf.decode(padded_emissions, mask=padded_mask)
    return not torch.equal(
        torch.as_tensor(unpadded), torch.as_tensor(padded)[:, :real_length]
    )


def patch_chain_crf_masked_decode():
    """Decode from each sequence's last real position.

    Without this, masking the loss and the LSTMs still leaves tagging dependent
    on batch composition, because the Viterbi backtrack enters the real
    positions carrying a tag chosen by the padding.
    """
    if not is_chain_crf_masked_decode_required():
        LOGGER.debug('ChainCRF already decodes from the last real position')
        return
    ChainCRF.decode = (  # type: ignore[method-assign]
        _chain_crf_decode_from_last_real_position
    )
    LOGGER.info('patched ChainCRF to decode from the last real position')
