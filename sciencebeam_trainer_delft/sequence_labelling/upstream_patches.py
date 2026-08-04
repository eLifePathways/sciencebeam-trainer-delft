"""Local fixes for defects in the installed delft, applied at runtime.

Each patch is conditional on the defect still being present, so a delft
release that fixes it makes the patch a no-op without any change here. The
corresponding tests in ``tests/sequence_labelling/delft_upstream_test.py`` are
strict xfails and will go red on such a release, which is the prompt to delete
the patch rather than to leave it running.
"""
import logging
from typing import Optional

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
