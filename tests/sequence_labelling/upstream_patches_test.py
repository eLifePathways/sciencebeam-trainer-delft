from typing import Any, Iterator

import pytest
import torch

from delft.utilities.crf_pytorch import ChainCRF

from sciencebeam_trainer_delft.sequence_labelling.upstream_patches import (
    ORIGINAL_CHAIN_CRF_DECODE,
    ORIGINAL_CHAIN_CRF_FREE_ENERGY,
    is_chain_crf_masked_decode_required,
    is_chain_crf_masked_free_energy_required,
    patch_chain_crf_masked_decode,
    patch_chain_crf_masked_free_energy
)


@pytest.fixture(name='restore_chain_crf', autouse=True)
def _restore_chain_crf() -> Iterator[None]:
    """Starts each test from upstream's own ChainCRF.

    Importing the models module patches it for the rest of the process, and
    these tests are about what the patch changes, so they have to begin from
    the unpatched behaviour.
    """
    patched_decode = ChainCRF.decode
    patched_free_energy = ChainCRF._free_energy  # pylint: disable=protected-access
    ChainCRF.decode = ORIGINAL_CHAIN_CRF_DECODE  # type: ignore[method-assign]
    # pylint: disable=protected-access
    ChainCRF._free_energy = ORIGINAL_CHAIN_CRF_FREE_ENERGY  # type: ignore[method-assign]
    yield
    ChainCRF.decode = patched_decode  # type: ignore[method-assign]
    ChainCRF._free_energy = patched_free_energy  # type: ignore[method-assign]


def _as_parameter(value: Any) -> torch.Tensor:
    # upstream annotates the CRF parameters as None until they are built
    return value


def _built_chain_crf(ntags: int = 2) -> ChainCRF:
    crf = ChainCRF(ntags)
    with torch.no_grad():
        # asymmetric, so the best previous tag depends on the tag after it
        _as_parameter(crf.U).copy_(torch.tensor([[0.0, 10.0], [0.0, -1.0]]))
        _as_parameter(crf.b_start).zero_()
        _as_parameter(crf.b_end).zero_()
    crf.eval()
    return crf


def _padded(tensor: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    return torch.cat([tensor, padding], dim=1)


REAL_EMISSIONS = torch.zeros(1, 2, 2)
PADDING_EMISSIONS = torch.tensor([[[0.0, 100.0]]])
REAL_MASK = torch.ones(1, 2, dtype=torch.bool)
PADDING_MASK = torch.zeros(1, 1, dtype=torch.bool)


class TestPatchChainCrfMaskedFreeEnergy:
    def test_should_leave_the_loss_padding_dependent_without_the_patch(self):
        crf = _built_chain_crf()
        labels = torch.ones(1, 2, dtype=torch.long)
        with torch.no_grad():
            unpadded = crf(REAL_EMISSIONS, labels, mask=REAL_MASK)
            padded = crf(
                _padded(REAL_EMISSIONS, PADDING_EMISSIONS),
                _padded(labels, torch.zeros(1, 1, dtype=torch.long)),
                mask=_padded(REAL_MASK, PADDING_MASK)
            )
        assert not torch.allclose(unpadded, padded, atol=1e-5)

    def test_should_make_the_masked_loss_independent_of_the_padding(self):
        patch_chain_crf_masked_free_energy()
        crf = _built_chain_crf()
        labels = torch.ones(1, 2, dtype=torch.long)
        with torch.no_grad():
            unpadded = crf(REAL_EMISSIONS, labels, mask=REAL_MASK)
            padded = crf(
                _padded(REAL_EMISSIONS, PADDING_EMISSIONS),
                _padded(labels, torch.zeros(1, 1, dtype=torch.long)),
                mask=_padded(REAL_MASK, PADDING_MASK)
            )
        assert torch.allclose(unpadded, padded, atol=1e-5)

    def test_should_leave_the_unmasked_loss_unchanged(self):
        labels = torch.ones(1, 2, dtype=torch.long)
        crf = _built_chain_crf()
        with torch.no_grad():
            before = crf(REAL_EMISSIONS, labels)
        patch_chain_crf_masked_free_energy()
        with torch.no_grad():
            after = crf(REAL_EMISSIONS, labels)
        assert torch.allclose(before, after, atol=1e-6)

    def test_should_not_apply_twice(self):
        patch_chain_crf_masked_free_energy()
        patched = ChainCRF._free_energy  # pylint: disable=protected-access
        patch_chain_crf_masked_free_energy()
        assert ChainCRF._free_energy is patched  # pylint: disable=protected-access

    def test_should_report_the_defect_as_no_longer_present_once_patched(self):
        assert is_chain_crf_masked_free_energy_required()
        patch_chain_crf_masked_free_energy()
        assert not is_chain_crf_masked_free_energy_required()


class TestPatchChainCrfMaskedDecode:
    def test_should_leave_the_tags_padding_dependent_without_the_patch(self):
        crf = _built_chain_crf()
        with torch.no_grad():
            unpadded = crf.decode(REAL_EMISSIONS, mask=REAL_MASK)
            padded = crf.decode(
                _padded(REAL_EMISSIONS, PADDING_EMISSIONS),
                mask=_padded(REAL_MASK, PADDING_MASK)
            )
        assert not torch.equal(
            torch.as_tensor(unpadded), torch.as_tensor(padded)[:, :2]
        )

    def test_should_decode_the_real_positions_the_same_whatever_the_padding(self):
        patch_chain_crf_masked_decode()
        crf = _built_chain_crf()
        with torch.no_grad():
            unpadded = crf.decode(REAL_EMISSIONS, mask=REAL_MASK)
            for padding_length in (1, 2, 5):
                padding = PADDING_EMISSIONS.repeat(1, padding_length, 1)
                padded = crf.decode(
                    _padded(REAL_EMISSIONS, padding),
                    mask=_padded(
                        REAL_MASK, torch.zeros(1, padding_length, dtype=torch.bool)
                    )
                )
                assert torch.equal(
                    torch.as_tensor(unpadded), torch.as_tensor(padded)[:, :2]
                )

    def test_should_zero_the_padded_positions(self):
        patch_chain_crf_masked_decode()
        crf = _built_chain_crf()
        with torch.no_grad():
            padded = crf.decode(
                _padded(REAL_EMISSIONS, PADDING_EMISSIONS),
                mask=_padded(REAL_MASK, PADDING_MASK)
            )
        assert torch.as_tensor(padded)[:, 2:].tolist() == [[0]]

    def test_should_leave_the_unmasked_decode_unchanged(self):
        crf = _built_chain_crf()
        emissions = _padded(REAL_EMISSIONS, PADDING_EMISSIONS)
        with torch.no_grad():
            before = torch.as_tensor(crf.decode(emissions))
        patch_chain_crf_masked_decode()
        with torch.no_grad():
            after = torch.as_tensor(crf.decode(emissions))
        assert torch.equal(before, after)

    def test_should_not_apply_twice(self):
        patch_chain_crf_masked_decode()
        patched = ChainCRF.decode
        patch_chain_crf_masked_decode()
        assert ChainCRF.decode is patched

    def test_should_report_the_defect_as_no_longer_present_once_patched(self):
        assert is_chain_crf_masked_decode_required()
        patch_chain_crf_masked_decode()
        assert not is_chain_crf_masked_decode_required()
