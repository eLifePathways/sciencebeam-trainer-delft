"""Padding masks for the sequence models.

Keras took the mask from the character embedding's `mask_zero=True` and
propagated it through the concatenate into the word LSTM and the CRF, so a
document encoded, scored and decoded the same whatever batch it sat in. torch
propagates nothing, so each architecture has to derive the mask and apply it at
every layer that runs over the sequence.
"""
from typing import Optional

import torch
from torch import nn


def get_token_mask(char_input: torch.Tensor) -> torch.Tensor:
    """Marks the token positions that are not padding.

    Keras reduced the character embedding's own mask over the character axis,
    so a token counted as padding when every one of its characters did. Taking
    it from `char_input` rather than from the `length` input reproduces that,
    and works whether or not a length was supplied.
    """
    mask = (char_input != 0).any(dim=-1)
    # the CRF requires the first position of every sequence to be unmasked; a
    # sequence that is padding throughout would otherwise be rejected outright
    mask[:, 0] = True
    return mask


def get_lengths_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.sum(dim=1)


def run_masked_lstm(
    lstm: nn.RNNBase, x: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Runs a sequence LSTM over the real positions only, returning every step."""
    lengths = get_lengths_from_mask(mask)
    packed = nn.utils.rnn.pack_padded_sequence(
        x,
        # a length of zero is not packable; `get_token_mask` keeps position 0
        lengths.clamp(min=1).cpu(),
        batch_first=True,
        enforce_sorted=False
    )
    packed_output, _ = lstm(packed)
    output, _ = nn.utils.rnn.pad_packed_sequence(
        packed_output, batch_first=True, total_length=x.shape[1]
    )
    return output


def run_masked_final_state_lstm(
    lstm: nn.RNNBase, x: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Returns each direction's final state, taken at the last real step.

    Without this the LSTM runs on through the padding: for a three-character
    token in a thirty-character window that is twenty-seven further steps, and
    the state that reaches the rest of the model is mostly a function of the
    padding embedding.
    """
    packed = nn.utils.rnn.pack_padded_sequence(
        x,
        # a length of zero is not packable; those rows are zeroed below
        lengths.clamp(min=1).cpu(),
        batch_first=True,
        enforce_sorted=False
    )
    _, (hidden, _) = lstm(packed)
    encoded = torch.cat([hidden[0], hidden[1]], dim=-1)
    # a row that is entirely padding is masked out completely in Keras, leaving
    # the initial state rather than whatever a single step produced
    return encoded * (lengths > 0).unsqueeze(-1).to(encoded.dtype)


def get_mask_for_char_input(
    char_input: torch.Tensor, enabled: bool
) -> Optional[torch.Tensor]:
    """The token mask when masking is enabled, otherwise nothing.

    A `None` mask is what the layers here take to mean "run over the padding
    too", which is the behaviour a model from before this option was trained
    with.
    """
    if not enabled:
        return None
    return get_token_mask(char_input)
