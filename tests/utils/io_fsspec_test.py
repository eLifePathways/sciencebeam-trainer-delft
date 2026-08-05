"""Covers the remote path of the file utilities.

fsspec's in-memory filesystem is an external location as far as these
utilities are concerned, so it exercises the same branch a `gs://` URL takes
without needing credentials or a network.
"""
from pathlib import Path
from typing import Iterator

import fsspec
import pytest

from sciencebeam_trainer_delft.utils.io import (
    copy_file,
    file_exists,
    is_external_location,
    list_files,
    open_file
)


MEMORY_DIRECTORY = 'memory://test-directory'


@pytest.fixture(name='memory_filesystem', autouse=True)
def _memory_filesystem() -> Iterator[None]:
    filesystem = fsspec.filesystem('memory')
    filesystem.store.clear()
    filesystem.pseudo_dirs.clear()
    yield
    filesystem.store.clear()
    filesystem.pseudo_dirs.clear()


class TestIsExternalLocation:
    def test_should_treat_a_memory_url_as_external(self):
        assert is_external_location(MEMORY_DIRECTORY)


class TestOpenFile:
    def test_should_round_trip_bytes(self):
        file_url = f'{MEMORY_DIRECTORY}/file.bin'
        with open_file(file_url, mode='wb') as fp:
            fp.write(b'binary content')
        with open_file(file_url, mode='rb') as fp:
            assert fp.read() == b'binary content'

    def test_should_round_trip_text(self):
        file_url = f'{MEMORY_DIRECTORY}/file.txt'
        with open_file(file_url, mode='w') as fp:
            fp.write('text content')
        with open_file(file_url, mode='r') as fp:
            assert fp.read() == 'text content'

    def test_should_raise_file_not_found_for_a_missing_file(self):
        with pytest.raises(FileNotFoundError):
            with open_file(f'{MEMORY_DIRECTORY}/missing.txt', mode='r') as fp:
                fp.read()

    def test_should_round_trip_a_gzipped_file(self):
        file_url = f'{MEMORY_DIRECTORY}/file.txt.gz'
        with open_file(file_url, mode='wb') as fp:
            fp.write(b'gzipped content')
        with open_file(file_url, mode='rb') as fp:
            assert fp.read() == b'gzipped content'


class TestFileExists:
    def test_should_report_a_written_file_as_existing(self):
        file_url = f'{MEMORY_DIRECTORY}/file.txt'
        assert not file_exists(file_url)
        with open_file(file_url, mode='w') as fp:
            fp.write('content')
        assert file_exists(file_url)


class TestListFiles:
    def test_should_return_names_without_the_directory(self):
        for name in ['a.txt', 'b.txt']:
            with open_file(f'{MEMORY_DIRECTORY}/{name}', mode='w') as fp:
                fp.write('content')
        assert sorted(list_files(MEMORY_DIRECTORY)) == ['a.txt', 'b.txt']


class TestCopyFile:
    def test_should_copy_from_local_to_remote(self, tmp_path: Path):
        source_path = tmp_path / 'source.txt'
        source_path.write_text('content')
        target_url = f'{MEMORY_DIRECTORY}/target.txt'
        copy_file(str(source_path), target_url)
        with open_file(target_url, mode='r') as fp:
            assert fp.read() == 'content'

    def test_should_copy_from_remote_to_local(self, tmp_path: Path):
        source_url = f'{MEMORY_DIRECTORY}/source.txt'
        with open_file(source_url, mode='w') as fp:
            fp.write('content')
        target_path = tmp_path / 'target.txt'
        copy_file(source_url, str(target_path))
        assert target_path.read_text() == 'content'

    def test_should_skip_an_existing_target_when_not_overwriting(self):
        source_url = f'{MEMORY_DIRECTORY}/source.txt'
        target_url = f'{MEMORY_DIRECTORY}/target.txt'
        for url, content in [(source_url, 'source'), (target_url, 'original')]:
            with open_file(url, mode='w') as fp:
                fp.write(content)
        copy_file(source_url, target_url, overwrite=False)
        with open_file(target_url, mode='r') as fp:
            assert fp.read() == 'original'
