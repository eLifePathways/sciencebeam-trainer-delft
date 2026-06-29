from unittest.mock import patch, MagicMock

import pytest

import sciencebeam_trainer_delft.utils.io as io_module
from sciencebeam_trainer_delft.utils.io import (
    is_external_location,
    auto_uploading_output_file
)


@pytest.fixture(name='copy_file_mock', autouse=True)
def _copy_file_mock():
    with patch.object(io_module, 'copy_file') as mock:
        yield mock


class TestIsExternalLocation:
    def test_should_return_false_for_name(self):
        assert not is_external_location('name')

    def test_should_return_true_for_url(self):
        assert is_external_location('http://name')


class TestIsAutoUploadingOutputFile:
    def test_should_return_file_pointer_for_local_path_for_binary_file(self, tmp_path):
        file_path = tmp_path / 'file.bin'
        with auto_uploading_output_file(str(file_path), mode='wb') as fp:
            fp.write(b'def')
        assert file_path.read_bytes() == b'def'

    def test_should_return_temp_file_pointer_for_external_path_for_binary_file(
        self,
        copy_file_mock: MagicMock
    ):
        file_path = 'http://name/file.bin'
        with auto_uploading_output_file(file_path, mode='wb') as fp:
            fp.write(b'def')
        copy_file_mock.assert_called_once_with(fp.name, file_path)
