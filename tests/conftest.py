import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Iterator

import pytest


LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope='session', autouse=True)
def setup_logging():
    logging.root.handlers = []
    logging.basicConfig(level='INFO')
    logging.getLogger('tests').setLevel('DEBUG')
    logging.getLogger('sciencebeam_trainer_delft').setLevel('DEBUG')


def _backport_assert_called(mock: MagicMock):
    assert mock.called


@pytest.fixture(scope='session', autouse=True)
def patch_magicmock():
    try:
        MagicMock.assert_called
    except AttributeError:
        MagicMock.assert_called = _backport_assert_called


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    # no longer need to convert to standard Path (remove this fixture)
    return tmp_path


@pytest.fixture
def env_mock() -> Iterator[dict]:
    env: dict = {}
    with patch('os.environ', env):
        yield env
