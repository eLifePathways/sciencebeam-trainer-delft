"""Loading the training/evaluation input, and checking it before it is used.

The feature length checks live here rather than in the caller because they have to run per
input and before the documents are combined: that is what lets a failure name a file, and
what keeps the outcome independent of the shuffle.
"""
import logging
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from typing import List, Optional, Set, Tuple

import numpy as np

from sciencebeam_trainer_delft.sequence_labelling.typing import (
    T_Batch_Features_Array,
    T_Batch_Label_Array,
    T_Batch_Token_Array
)
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.utils.numpy import shuffle_arrays

from sciencebeam_trainer_delft.sequence_labelling.reader import load_data_and_labels_crf_file

from sciencebeam_trainer_delft.sequence_labelling.feature_lengths import (
    DEFAULT_FEATURE_LENGTH_MODE,
    FeatureLengthCheckResult,
    FeatureLengthModes,
    FeatureLengthReport,
    check_feature_lengths,
    get_feature_length_report
)

from sciencebeam_trainer_delft.sequence_labelling.input_info import (
    iter_flat_batch_tokens,
    iter_flat_features,
    get_quantiles,
    get_quantiles_feature_value_length_by_index,
    get_feature_counts,
    get_suggested_feature_indices,
    format_dict,
    format_indices
)


LOGGER = logging.getLogger(__name__)


DEFAULT_RANDOM_SEED = 42


def log_data_info(x: np.ndarray, y: np.ndarray, features: np.ndarray):
    LOGGER.info('x sample: %s (y: %s)', x[:1][:10], y[:1][:1])
    LOGGER.info(
        'feature dimensions of first sample, word: %s',
        [{index: value for index, value in enumerate(features[0][0])}]  # noqa pylint: disable=unnecessary-comprehension
    )


@dataclass
class LoadedInputData:
    """What one input contributed, kept separate so a check can name the file it came from."""
    input_path: str
    x: T_Batch_Token_Array
    y: T_Batch_Label_Array
    features: T_Batch_Features_Array

    def apply_document_mask(self, mask: List[bool]) -> 'LoadedInputData':
        if all(mask):
            return self
        return LoadedInputData(
            input_path=self.input_path,
            x=self.x[mask],
            y=self.y[mask],
            features=self.features[mask]
        )


def _load_data_and_labels_crf_files(
    input_paths: List[str],
    downloaded_input_paths: List[str],
    limit: Optional[int] = None
) -> List[LoadedInputData]:
    loaded_input_data_list = []
    for input_path, downloaded_input_path in zip(input_paths, downloaded_input_paths):
        LOGGER.debug('calling load_data_and_labels_crf_file: %s', downloaded_input_path)
        x, y, f = load_data_and_labels_crf_file(
            downloaded_input_path,
            limit=limit
        )
        loaded_input_data_list.append(LoadedInputData(
            input_path=input_path, x=x, y=y, features=f
        ))
    return loaded_input_data_list


def to_document_object_array(array: np.ndarray) -> np.ndarray:
    """One object entry per document, whatever shape the documents have.

    The reader returns a multi-dimensional array where every document happens to have the same
    number of rows, and a one-dimensional object array otherwise. Two such arrays only
    concatenate when their trailing dimensions agree, which they do not when the inputs carry
    different numbers of features.
    """
    if array.ndim <= 1:
        return array
    document_array = np.empty(len(array), dtype='object')
    for index, document in enumerate(array):
        document_array[index] = document
    return document_array


def _concatenate_arrays(arrays: List[np.ndarray]) -> np.ndarray:
    try:
        return np.concatenate(arrays)
    except ValueError:
        # the inputs disagree on a trailing dimension, e.g. on how many features they carry;
        # a mode that allows that has to be able to combine them regardless
        LOGGER.debug('falling back to concatenating documents as objects')
        return np.concatenate([to_document_object_array(array) for array in arrays])


def _concatenate_loaded_input_data(
    loaded_input_data_list: List[LoadedInputData]
) -> Tuple[T_Batch_Token_Array, T_Batch_Label_Array, T_Batch_Features_Array]:
    if len(loaded_input_data_list) == 1:
        # avoid the copy np.concatenate would make of a single, potentially large input
        loaded_input_data = loaded_input_data_list[0]
        return loaded_input_data.x, loaded_input_data.y, loaded_input_data.features
    return (
        _concatenate_arrays([item.x for item in loaded_input_data_list]),
        _concatenate_arrays([item.y for item in loaded_input_data_list]),
        _concatenate_arrays([item.features for item in loaded_input_data_list])
    )


def _load_and_check_data_and_labels(
    input_paths: List[str],
    limit: Optional[int] = None,
    feature_length_mode: str = DEFAULT_FEATURE_LENGTH_MODE,
    read_feature_indices: Optional[Set[int]] = None,
    expected_feature_count: Optional[int] = None,
    download_manager: Optional[DownloadManager] = None
) -> Tuple[List[LoadedInputData], FeatureLengthCheckResult]:
    assert download_manager
    assert input_paths
    LOGGER.info('loading data from: %s', input_paths)
    downloaded_input_paths = [
        download_manager.download_if_url(input_path)
        for input_path in input_paths
    ]
    loaded_input_data_list = _load_data_and_labels_crf_files(
        input_paths,
        downloaded_input_paths,
        limit=limit
    )
    # checked per input and before the shuffle, so the diagnosis names a file and
    # does not depend on the seed
    check_result = check_feature_lengths(
        get_feature_length_report(
            input_paths=[item.input_path for item in loaded_input_data_list],
            features_list=[item.features for item in loaded_input_data_list]
        ),
        read_feature_indices=read_feature_indices,
        mode=feature_length_mode,
        expected_feature_count=expected_feature_count
    )
    if check_result.masks is not None:
        loaded_input_data_list = [
            item.apply_document_mask(mask)
            for item, mask in zip(loaded_input_data_list, check_result.masks)
        ]
    return loaded_input_data_list, check_result


def load_data_and_labels_with_check_result(
    input_paths: Optional[List[str]] = None,
    limit: Optional[int] = None,
    shuffle_input: bool = False,
    feature_length_mode: str = DEFAULT_FEATURE_LENGTH_MODE,
    read_feature_indices: Optional[Set[int]] = None,
    expected_feature_count: Optional[int] = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    download_manager: Optional[DownloadManager] = None
) -> Tuple[
    T_Batch_Token_Array, T_Batch_Label_Array, T_Batch_Features_Array, FeatureLengthCheckResult
]:
    assert input_paths
    loaded_input_data_list, check_result = _load_and_check_data_and_labels(
        input_paths=input_paths,
        limit=limit,
        feature_length_mode=feature_length_mode,
        read_feature_indices=read_feature_indices,
        expected_feature_count=expected_feature_count,
        download_manager=download_manager
    )
    x_all, y_all, f_all = _concatenate_loaded_input_data(loaded_input_data_list)
    if shuffle_input:
        shuffle_arrays([x_all, y_all, f_all], random_seed=random_seed)
    # logged after the check, so the feature dimensions it reports are representative
    log_data_info(x_all, y_all, f_all)
    return x_all, y_all, f_all, check_result


def load_data_and_labels(
    input_paths: Optional[List[str]] = None,
    limit: Optional[int] = None,
    shuffle_input: bool = False,
    feature_length_mode: str = DEFAULT_FEATURE_LENGTH_MODE,
    read_feature_indices: Optional[Set[int]] = None,
    expected_feature_count: Optional[int] = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    download_manager: Optional[DownloadManager] = None
) -> Tuple[T_Batch_Token_Array, T_Batch_Label_Array, T_Batch_Features_Array]:
    x_all, y_all, f_all, _ = load_data_and_labels_with_check_result(
        input_paths=input_paths,
        limit=limit,
        shuffle_input=shuffle_input,
        feature_length_mode=feature_length_mode,
        read_feature_indices=read_feature_indices,
        expected_feature_count=expected_feature_count,
        random_seed=random_seed,
        download_manager=download_manager
    )
    return x_all, y_all, f_all


def print_feature_length_info(report: FeatureLengthReport):
    """Reports feature length per input, so a mismatch can be attributed to a file."""
    for input_feature_lengths in report.input_lengths:
        print('input: %s' % input_feature_lengths.input_path)
        print('  number of sequences: %d' % input_feature_lengths.document_count)
        print('  feature lengths: %s' % input_feature_lengths.format_widths())
    if report.is_uniform:
        print('feature length across inputs: consistent')
    else:
        print('feature length across inputs: INCONSISTENT (%s)' % report.format_summary())


def print_input_info(
    input_paths: List[str],
    limit: Optional[int] = None,
    download_manager: Optional[DownloadManager] = None
):
    loaded_input_data_list, check_result = _load_and_check_data_and_labels(
        input_paths=input_paths, limit=limit,
        # the diagnostic: it loads everything, reports, and changes nothing
        feature_length_mode=FeatureLengthModes.IGNORE,
        download_manager=download_manager
    )
    print_feature_length_info(check_result.report)
    x_all, y_all, features_all = _concatenate_loaded_input_data(loaded_input_data_list)

    seq_lengths = np.array([len(seq) for seq in x_all])
    y_counts = Counter(
        y_row
        for y_doc in y_all
        for y_row in y_doc
    )
    flat_features = list(iter_flat_features(features_all))
    feature_lengths = Counter(map(len, flat_features))

    print('number of input sequences: %d' % len(x_all))
    print('sequence lengths: %s' % format_dict(get_quantiles(seq_lengths)))
    print('token lengths: %s' % format_dict(get_quantiles(
        map(len, iter_flat_batch_tokens(x_all))
    )))
    print('number of features: %d' % len(features_all[0][0]))
    if len(feature_lengths) > 1:
        print('inconsistent feature length counts: %s' % format_dict(feature_lengths))
        for feature_length in feature_lengths:
            print('examples with feature length=%d:\n%s' % (
                feature_length,
                '\n'.join(islice((
                    ' '.join(features_vector)
                    for features_vector in flat_features
                    if len(features_vector) == feature_length
                ), 3))
            ))
    quantiles_feature_value_lengths = get_quantiles_feature_value_length_by_index(features_all)
    feature_counts = get_feature_counts(features_all)
    print('feature value lengths: %s' % format_dict(quantiles_feature_value_lengths))
    print('feature counts: %s' % format_dict(feature_counts))
    print('suggested feature indices: %s' % format_indices(
        get_suggested_feature_indices(feature_counts)
    ))
    print('label counts: %s' % format_dict(y_counts))
