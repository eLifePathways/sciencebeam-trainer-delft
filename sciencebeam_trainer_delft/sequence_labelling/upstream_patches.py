"""Local fixes for defects in the installed delft, applied at runtime.

Each patch is conditional on the defect still being present, so a delft
release that fixes it makes the patch a no-op without any change here. The
corresponding tests in ``tests/sequence_labelling/upstream_patches_test.py``
assert that the defect is there to fix, so they go red on such a release,
which is the prompt to delete the patch rather than to leave it running.

Both defects here are reported upstream, with patches prepared against v1.1.0;
see ``.project-notes/upstream/``.
"""
import logging
from typing import List, Optional, cast

import torch

from delft.utilities.crf_pytorch import ChainCRF


LOGGER = logging.getLogger(__name__)


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
    # delft >= 1.1.0 builds the parameters in the constructor, so they are here
    crf = ChainCRF(ntags)
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
) -> List[List[int]]:
    """Returns what upstream's decode does: a list of tags for each sequence of
    the batch, of the width of the batch, holding 0 where the mask leaves a
    position out. DeLFT's tagger reads the tags as integers."""
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
    return best_paths.tolist()


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
