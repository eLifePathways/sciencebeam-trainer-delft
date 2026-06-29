# pylint: disable=contextmanager-generator-missing-cleanup
import logging
import os
import tarfile
import tempfile
import zipfile
from abc import ABC, abstractmethod
from shutil import copyfileobj
from contextlib import contextmanager
from gzip import GzipFile
from lzma import LZMAFile
from urllib.error import HTTPError
from urllib.request import urlretrieve
from typing import List, IO, Iterator, Optional

try:
    from tensorflow.python.lib.io import file_io as tf_file_io  # type: ignore
    from tensorflow.python.framework.errors_impl import (  # type: ignore
        NotFoundError as tf_NotFoundError
    )
except ImportError:
    tf_file_io = None
    tf_NotFoundError = None


LOGGER = logging.getLogger(__name__)


def is_external_location(filepath: str):
    return isinstance(filepath, str) and '://' in filepath


def path_join(parent, child):
    return os.path.join(str(parent), str(child))


def is_gzip_filename(filepath: str):
    return filepath.endswith('.gz')


def is_xz_filename(filepath: str):
    return filepath.endswith('.xz')


def strip_gzip_filename_ext(filepath: str):
    if not is_gzip_filename(filepath):
        raise ValueError('not a gzip filename: %s' % filepath)
    return os.path.splitext(filepath)[0]


def strip_xz_filename_ext(filepath: str):
    if not is_xz_filename(filepath):
        raise ValueError('not a xz filename: %s' % filepath)
    return os.path.splitext(filepath)[0]


class CompressionWrapper(ABC):
    @abstractmethod
    def strip_compression_filename_ext(self, filepath: str):
        pass

    @abstractmethod
    def wrap_fileobj(self, filename: str, fileobj: IO, mode: Optional[str] = None):
        pass

    @contextmanager
    def open(self, filename: str, mode: str) -> Iterator[IO]:
        LOGGER.debug('opening file: %r, mode=%r', filename, mode)
        with _open_raw(filename, mode=mode) as fp:
            yield self.wrap_fileobj(
                filename=filename,
                fileobj=fp,
                mode=mode
            )


class ClosingGzipFile(GzipFile):
    # GzipFile doesn't close the underlying fileobj, we will do that here
    def close(self):
        fileobj = self.fileobj
        LOGGER.debug('ClosingGzipFile.close, fileobj: %s', fileobj)
        try:
            super().close()
        finally:
            if fileobj is not None:
                LOGGER.debug('closing: %s', fileobj)
                fileobj.close()


class GzipCompressionWrapper(CompressionWrapper):
    def strip_compression_filename_ext(self, filepath: str):
        return strip_gzip_filename_ext(filepath)

    def wrap_fileobj(self, filename: str, fileobj: IO, mode: Optional[str] = None):
        return ClosingGzipFile(filename=filename, fileobj=fileobj, mode=mode)

    @contextmanager
    def open(self, filename: str, mode: str):
        if is_external_location(filename):
            # there seem to be an issue with GzipFile and fileobj
            with tempfile.TemporaryDirectory(suffix='-gzip') as gzip_dir:
                local_gzip_file = os.path.join(gzip_dir, os.path.basename(filename))
                with ClosingGzipFile(filename=local_gzip_file, mode=mode) as local_fp:
                    yield local_fp
                tf_file_io.copy(local_gzip_file, filename, overwrite=True)
        else:
            with ClosingGzipFile(filename=filename, mode=mode) as local_fp:
                yield local_fp


class XzCompressionWrapper(CompressionWrapper):
    def strip_compression_filename_ext(self, filepath: str):
        return strip_xz_filename_ext(filepath)

    def wrap_fileobj(self, filename: str, fileobj: IO, mode: Optional[str] = None):
        return LZMAFile(filename=fileobj, mode=mode or 'r')


class DummyCompressionWrapper(CompressionWrapper):
    def strip_compression_filename_ext(self, filepath: str):
        return filepath

    def wrap_fileobj(self, filename: str, fileobj: IO, mode: Optional[str] = None):
        return fileobj


GZIP_COMPRESSION_WRAPPER = GzipCompressionWrapper()
XZ_COMPRESSION_WRAPPER = XzCompressionWrapper()
DUMMY_COMPRESSION_WRAPPER = DummyCompressionWrapper()


def get_compression_wrapper(filepath: str):
    if is_gzip_filename(filepath):
        return GZIP_COMPRESSION_WRAPPER
    if is_xz_filename(filepath):
        return XZ_COMPRESSION_WRAPPER
    return DUMMY_COMPRESSION_WRAPPER


def strip_compression_filename_ext(filepath: str) -> str:
    return get_compression_wrapper(filepath).strip_compression_filename_ext(filepath)


@contextmanager
def _open_raw(filepath: str, mode: str) -> Iterator[IO]:
    if filepath.startswith('https://'):
        try:
            with tempfile.TemporaryDirectory(suffix='download') as temp_dir:
                temp_file = os.path.join(temp_dir, os.path.basename(filepath))
                urlretrieve(filepath, temp_file)
                encoding: Optional[str] = 'utf-8' if 'b' not in mode else None
                with open(temp_file, mode=mode, encoding=encoding) as fp:
                    yield fp
        except HTTPError as error:
            if error.code == 404:
                raise FileNotFoundError('file not found: %s' % filepath) from error
            raise
    else:
        try:
            with tf_file_io.FileIO(filepath, mode=mode) as fp:
                yield fp
        except tf_NotFoundError as e:
            raise FileNotFoundError('file not found: %s' % filepath) from e


@contextmanager
def open_file(filepath: str, mode: str, compression_wrapper: Optional[CompressionWrapper] = None):
    if compression_wrapper is None:
        compression_wrapper = get_compression_wrapper(filepath)
    LOGGER.debug(
        'open_file, filepath=%s, mode=%s, compression_wrapper=%s',
        filepath, mode, compression_wrapper
    )
    if mode in {'rb', 'r'}:
        with _open_raw(filepath, mode=mode) as source_fp:
            yield compression_wrapper.wrap_fileobj(
                filename=filepath,
                fileobj=source_fp,
                mode=mode
            )
    elif mode in {'wb', 'w'}:
        tf_file_io.recursive_create_dir(os.path.dirname(filepath))
        with compression_wrapper.open(filepath, mode=mode) as target_fp:
            yield target_fp
    else:
        raise ValueError('unsupported mode: %s' % mode)


def _require_tf_file_io():
    if tf_file_io is None:
        raise ImportError('Cloud storage file transfer requires TensorFlow.')


def copy_file(source_filepath: str, target_filepath: str, overwrite: bool = True):
    _require_tf_file_io()
    if not overwrite and tf_file_io.file_exists(target_filepath):
        LOGGER.info('skipping already existing file: %s', target_filepath)
        return
    with open_file(str(source_filepath), mode='rb') as source_fp:
        with open_file(str(target_filepath), mode='wb') as target_fp:
            copyfileobj(source_fp, target_fp)


def list_files(directory_path: str) -> List[str]:
    _require_tf_file_io()
    return tf_file_io.list_directory(directory_path)


def is_binary_mode(mode: str) -> bool:
    return 'b' in mode


@contextmanager
def auto_uploading_output_file(filepath: str, mode: str = 'w', encoding='utf-8', **kwargs):
    if not is_external_location(filepath):
        file_dirname = os.path.dirname(filepath)
        if file_dirname:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if is_binary_mode(mode):
            encoding = None
        with open(filepath, mode=mode, encoding=encoding, **kwargs) as fp:
            yield fp
            return
    with tempfile.TemporaryDirectory(suffix='-output') as temp_dir:
        temp_file = os.path.join(
            temp_dir,
            get_compression_wrapper(filepath).strip_compression_filename_ext(
                os.path.basename(filepath)
            )
        )
        if is_binary_mode(mode):
            encoding = None
        try:
            with open(temp_file, mode=mode, encoding=encoding, **kwargs) as fp:
                yield fp
        finally:
            if os.path.exists(temp_file):
                copy_file(temp_file, filepath)


@contextmanager
def auto_download_input_file(filepath: str, auto_decompress: bool = False) -> Iterator[str]:
    if not is_external_location(filepath):
        yield filepath
        return
    with tempfile.TemporaryDirectory(suffix='-input') as temp_dir:
        file_basename = os.path.basename(filepath)
        if auto_decompress:
            file_basename = get_compression_wrapper(filepath).strip_compression_filename_ext(
                file_basename
            )
        temp_file = os.path.join(temp_dir, file_basename)
        copy_file(filepath, temp_file, overwrite=True)
        yield temp_file


def write_text(filepath: str, text: str, **kwargs):
    with auto_uploading_output_file(filepath, mode='w', **kwargs) as fp:
        fp.write(text)


def read_text(filepath: str, **kwargs) -> str:
    with open_file(filepath, mode='r', **kwargs) as fp:
        return fp.read()


def read_binary(filepath: str, **kwargs) -> bytes:
    with open_file(filepath, mode='rb', **kwargs) as fp:
        return fp.read()


class FileRef(ABC):
    def __init__(self, file_url: str):
        self.file_url = file_url

    @property
    def basename(self):
        return os.path.basename(self.file_url)

    def __str__(self):
        return self.file_url

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.file_url)

    @abstractmethod
    def copy_to(self, target_url: str):
        pass


class FileContainer(ABC):
    def __init__(self, directory_url: str):
        self.directory_url = directory_url

    @abstractmethod
    def list_files(self) -> List[FileRef]:
        pass

    def __str__(self):
        return self.directory_url

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.directory_url)


class FileUrlRef(FileRef):
    def copy_to(self, target_url: str):
        copy_file(self.file_url, target_url)


class DirectoryFileContainer(FileContainer):
    def list_files(self) -> List[FileRef]:
        return [
            FileUrlRef(path_join(self.directory_url, file_url))
            for file_url in list_files(self.directory_url)
        ]


class TarFileRef(FileRef):
    def __init__(
            self,
            file_url: str,
            tar_file: tarfile.TarFile,
            tar_info: tarfile.TarInfo):
        super().__init__(file_url)
        self.tar_file = tar_file
        self.tar_info = tar_info

    def open_tar_file(self) -> IO:
        fp = self.tar_file.extractfile(self.tar_info)
        assert fp
        return fp

    def copy_to(self, target_url: str):
        with self.open_tar_file() as source_fp:
            with open_file(
                    target_url,
                    mode='wb',
                    compression_wrapper=DUMMY_COMPRESSION_WRAPPER) as target_fp:
                copyfileobj(source_fp, target_fp)


class TarFileContainer(FileContainer):
    def __init__(self, directory_url, tar_file: tarfile.TarFile):
        super().__init__(directory_url)
        self.tar_file = tar_file

    def list_files(self) -> List[FileRef]:
        return [
            TarFileRef(
                path_join(self.directory_url, tar_info.name),
                tar_file=self.tar_file,
                tar_info=tar_info
            )
            for tar_info in self.tar_file.getmembers()
        ]


class ZipFileRef(FileRef):
    def __init__(
            self,
            file_url: str,
            zip_file: zipfile.ZipFile,
            zip_info: zipfile.ZipInfo):
        super().__init__(file_url)
        self.zip_file = zip_file
        self.zip_info = zip_info

    def copy_to(self, target_url: str):
        with self.zip_file.open(self.zip_info.filename) as source_fp:
            with open_file(
                    target_url,
                    mode='wb',
                    compression_wrapper=DUMMY_COMPRESSION_WRAPPER) as target_fp:
                copyfileobj(source_fp, target_fp)


class ZipFileContainer(FileContainer):
    def __init__(self, directory_url, zip_file: zipfile.ZipFile):
        super().__init__(directory_url)
        self.zip_file = zip_file

    def list_files(self) -> List[FileRef]:
        return [
            ZipFileRef(
                path_join(self.directory_url, zip_info.filename),
                zip_file=self.zip_file,
                zip_info=zip_info
            )
            for zip_info in self.zip_file.infolist()
        ]


@contextmanager
def open_file_container(directory_url: str) -> Iterator[FileContainer]:
    if str(directory_url).endswith('.tar.gz'):
        with auto_download_input_file(directory_url) as local_tar_file:
            with tarfile.open(local_tar_file) as tar_file:
                yield TarFileContainer(directory_url, tar_file=tar_file)
                return
    if str(directory_url).endswith('.zip'):
        with auto_download_input_file(directory_url) as local_zip_file:
            with zipfile.ZipFile(local_zip_file, mode='r') as zip_file:
                yield ZipFileContainer(directory_url, zip_file=zip_file)
                return
    yield DirectoryFileContainer(directory_url)
