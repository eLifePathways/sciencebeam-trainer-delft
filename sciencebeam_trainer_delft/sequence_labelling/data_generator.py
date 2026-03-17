import logging
from typing import Iterable, List, Optional, Tuple, Union

from typing_extensions import Protocol

import numpy as np
import tf_keras as keras

from delft.utilities.Embeddings import Embeddings
from delft.sequenceLabelling.preprocess import (
    to_vector_single, to_casing_single,
    # to_vector_simple_with_bert,
    Preprocessor,
    PAD
)
from delft.utilities.Tokenizer import tokenizeAndFilterSimple

from sciencebeam_trainer_delft.utils.typing import T
from sciencebeam_trainer_delft.utils.numpy import shuffle_arrays


LOGGER = logging.getLogger(__name__)

NBSP = '\u00A0'


def left_pad_batch_values(batch_values: np.ndarray, max_sequence_length: int, dtype=None):
    if dtype is None:
        dtype = batch_values.dtype
    batch_size = len(batch_values)
    value_dimension = 0
    for batch_value in batch_values:
        try:
            value_dimension = len(batch_value[0])
            continue
        except IndexError:
            pass
    result = np.zeros((batch_size, max_sequence_length, value_dimension), dtype=dtype)
    for batch_index in range(batch_size):
        values = batch_values[batch_index]
        if len(values) > max_sequence_length:
            values = values[:max_sequence_length]
        if len(values) > 0:
            result[batch_index, :len(values)] = values
    return result


def truncate_batch_values(batch_values: list, max_sequence_length: int) -> list:
    return [
        row[:max_sequence_length]
        for row in batch_values
    ]


def iter_stateless_window_indices_and_offset(
        sequence_lengths: List[int],
        window_stride: int) -> Iterable[Tuple[int, int]]:
    for sequence_index, sequence_length in enumerate(sequence_lengths):
        sequence_offset = 0
        while sequence_offset < sequence_length:
            yield sequence_index, sequence_offset
            if not window_stride:
                break
            sequence_offset += window_stride


def get_stateless_window_indices_and_offset(
        sequence_lengths: List[int],
        window_stride: int) -> List[Tuple[int, int]]:
    return list(iter_stateless_window_indices_and_offset(
        sequence_lengths, window_stride
    ))


def iter_batch_window_indices_and_offset(
        sequence_lengths: List[int],
        window_stride: int,
        batch_size: int) -> Iterable[List[Tuple[int, int]]]:
    batch_size = min(batch_size, len(sequence_lengths))
    next_sequence_indices = list(range(len(sequence_lengths)))
    batch_sequence_indices = next_sequence_indices[:batch_size]
    next_sequence_indices = next_sequence_indices[batch_size:]
    batch_offsets = [0] * batch_size
    batch_complete = [False] * batch_size
    yield list(zip(batch_sequence_indices, batch_offsets))
    while True:
        for batch_item_index in range(batch_size):
            current_sequence_length = sequence_lengths[batch_sequence_indices[batch_item_index]]
            batch_offsets[batch_item_index] += window_stride
            if batch_offsets[batch_item_index] >= current_sequence_length:
                # we already reached the end of the sequence
                if not next_sequence_indices:
                    batch_complete[batch_item_index] = True
                    continue
                batch_offsets[batch_item_index] = 0
                batch_sequence_indices[batch_item_index] = next_sequence_indices.pop(0)
        if all(batch_complete):
            return
        yield list(zip(batch_sequence_indices, batch_offsets))


def get_batch_window_indices_and_offset(
        sequence_lengths: List[int],
        window_stride: int,
        batch_size: int) -> List[List[Tuple[int, int]]]:
    return list(iter_batch_window_indices_and_offset(
        sequence_lengths=sequence_lengths,
        window_stride=window_stride,
        batch_size=batch_size
    ))


def get_chunk_at_offset(sequence: list, offset: int, max_sequence_length: Optional[int] = None):
    if not max_sequence_length:
        return sequence[offset:]
    return sequence[offset:min(len(sequence), offset + max_sequence_length)]


def take_with_offset(
    sequences: list,
    indices_and_offset: List[Tuple[int, int]],
    max_sequence_length: Optional[int] = None
) -> list:
    return [
        get_chunk_at_offset(sequences[index], offset, max_sequence_length)
        for index, offset in indices_and_offset
    ]


def to_batch_embedding_vector(
        batch_tokens: List[List[str]],
        embeddings: Embeddings,
        max_length: int = 300,
        **kwargs):
    batch_x = np.zeros((len(batch_tokens), max_length, embeddings.embed_size), dtype='float32')
    for i, tokens in enumerate(batch_tokens):
        batch_x[i] = to_vector_single(tokens, embeddings, max_length, **kwargs)
    return batch_x


def to_dummy_batch_embedding_vector(
        batch_tokens: List[List[str]],
        max_length: int = 300):
    return np.zeros((len(batch_tokens), max_length, 0), dtype='float32')


def get_tokens_from_text_features(
        token_features: List[str],
        text_feature_indices: List[int]) -> List[str]:
    return tokenizeAndFilterSimple(' '.join([
        (
            token_features[text_feature_index]
            if text_feature_index < len(token_features)
            else ''
        )
        for text_feature_index in text_feature_indices
    ]).replace(NBSP, ' '))


def iter_batch_text_from_tokens_with_additional_token_features(
    batch_tokens: Iterable[List[str]],
    batch_features: Iterable[List[List[str]]],
    additional_token_feature_indices: List[int]
) -> Iterable[List[str]]:
    if not additional_token_feature_indices:
        return batch_tokens
    return (
        [
            ' '.join([token] + [
                token_features[additional_token_feature_index]
                for additional_token_feature_index in additional_token_feature_indices
            ])
            for token, token_features in zip(doc_tokens, doc_features)
        ]
        for doc_tokens, doc_features in zip(batch_tokens, batch_features)
    )


def iter_batch_text_from_text_features(
        batch_features: Iterable[List[List[str]]],
        text_feature_indices: List[int]) -> Iterable[List[str]]:
    LOGGER.debug('text_feature_indices: %s', text_feature_indices)
    LOGGER.debug('batch_features: %s', batch_features)
    return (
        [
            ' '.join(get_tokens_from_text_features(
                token_features,
                text_feature_indices
            ))
            for token_features in doc_features
        ]
        for doc_features in batch_features
    )


def iter_batch_text_list(
    batch_tokens: List[List[str]],
    batch_features: Optional[Union[np.ndarray, List[List[List[str]]]]],
    additional_token_feature_indices: Optional[List[int]],
    text_feature_indices: Optional[List[int]]
) -> Iterable[List[str]]:
    if additional_token_feature_indices and text_feature_indices:
        raise ValueError('both, additional token and text features, not supported')
    if additional_token_feature_indices:
        assert batch_features is not None
        return iter_batch_text_from_tokens_with_additional_token_features(
            batch_tokens=batch_tokens,
            batch_features=batch_features,
            additional_token_feature_indices=additional_token_feature_indices
        )
    if text_feature_indices:
        assert batch_features is not None
        return iter_batch_text_from_text_features(
            batch_features=batch_features,
            text_feature_indices=text_feature_indices
        )
    return batch_tokens


def safe_list_get_at(some_list: List[T], index: int, default_value: T) -> T:
    try:
        return some_list[index]
    except IndexError:
        return default_value


def iter_batch_tokens_by_token_index(
        batch_text_list: List[List[str]],
        concatenated_embeddings_token_count: int,
        text_is_token: bool = False) -> Iterable[List[List[str]]]:
    if not concatenated_embeddings_token_count:
        yield batch_text_list
        return
    if text_is_token and concatenated_embeddings_token_count == 1:
        yield batch_text_list
        return
    batch_tokens_list = [
        [
            text.split(' ')
            for text in batch_text
        ]
        for batch_text in batch_text_list
    ]
    for token_index in range(concatenated_embeddings_token_count):
        yield [
            [
                safe_list_get_at(tokens, token_index, PAD)
                for tokens in batch_tokens
            ]
            for batch_tokens in batch_tokens_list
        ]


class ToBatchVectorCallableProtocol(Protocol):
    def __call__(
        self,
        batch_tokens: List[List[str]],
        max_length: int
    ) -> np.ndarray:
        pass


def to_concatenated_batch_vector_from_batch_text_list(
        to_batch_vector_fn: ToBatchVectorCallableProtocol,
        batch_text_list: List[List[str]],
        *args,
        concatenated_embeddings_token_count: int,
        text_is_token: bool,
        **kwargs):
    batch_tokens_iterable = iter_batch_tokens_by_token_index(
        batch_text_list=batch_text_list,
        concatenated_embeddings_token_count=concatenated_embeddings_token_count,
        text_is_token=text_is_token
    )
    batch_vector_list = [
        to_batch_vector_fn(
            batch_tokens,
            *args,
            **kwargs
        )
        for batch_tokens in batch_tokens_iterable
    ]
    concatenated_batch_vector = np.concatenate(
        batch_vector_list,
        axis=-1
    )
    return concatenated_batch_vector


def get_token_padded_batch_text_list(batch_text_list: List[List[str]]) -> List[List[str]]:
    batch_tokens_list = [
        [
            text.split(' ')
            for text in batch_text
        ]
        for batch_text in batch_text_list
    ]
    max_token_count = max(
        len(tokens)
        for batch_tokens in batch_tokens_list
        for tokens in batch_tokens
    )
    return [
        [
            ' '.join([
                safe_list_get_at(tokens, token_index, PAD)
                for token_index in range(max_token_count)
            ])
            for tokens in batch_tokens
        ]
        for batch_tokens in batch_tokens_list
    ]


def to_batch_casing(
        batch_tokens: List[List[str]],
        max_length: int = 300):
    batch_a = np.zeros((len(batch_tokens), max_length), dtype='float32')
    for i, tokens in enumerate(batch_tokens):
        batch_a[i] = to_casing_single(tokens, max_length)
    return batch_a


def get_concatenated_embeddings_token_count(
    concatenated_embeddings_token_count: Optional[int] = None,
    additional_token_feature_indices: Optional[List[int]] = None,
    use_word_embeddings: bool = True
) -> int:
    if not use_word_embeddings:
        return 0
    return (
        concatenated_embeddings_token_count
        or (1 + len(additional_token_feature_indices or []))
    )


# generate batch of data to feed sequence labelling model, both for training and prediction
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
        self,
        x: List[Union[str, List[str]]],
        y: Optional[List[List[str]]],
        preprocessor: Preprocessor,
        batch_size: int = 24,
        input_window_stride: Optional[int] = None,
        stateful: bool = True,
        char_embed_size: int = 25,
        use_word_embeddings: Optional[bool] = None,
        embeddings: Optional[Embeddings] = None,
        max_sequence_length: Optional[int] = None,
        tokenize: bool = False,
        shuffle: bool = True,
        features: Optional[List[List[List[str]]]] = None,
        additional_token_feature_indices: Optional[List[int]] = None,
        text_feature_indices: Optional[List[int]] = None,
        concatenated_embeddings_token_count: Optional[int] = None,
        is_deprecated_padded_batch_text_list_enabled: bool = False,
        name: Optional[str] = None,
        use_chain_crf: bool = False
    ):
        'Initialization'
        if use_word_embeddings is None:
            use_word_embeddings = embeddings is not None
        self.x = x
        self.y = y
        # features here are optional additional features provided
        # in the case of GROBID input for instance
        self.input_window_stride = input_window_stride
        self.features = features
        self.additional_token_feature_indices = additional_token_feature_indices
        self.text_feature_indices = text_feature_indices
        self.concatenated_embeddings_token_count = get_concatenated_embeddings_token_count(
            concatenated_embeddings_token_count=concatenated_embeddings_token_count,
            additional_token_feature_indices=additional_token_feature_indices,
            use_word_embeddings=use_word_embeddings
        )
        LOGGER.debug(
            'concatenated_embeddings_token_count: %s', self.concatenated_embeddings_token_count
        )
        self.preprocessor = preprocessor
        if preprocessor:
            self.labels = preprocessor.vocab_tag
        self.batch_size = batch_size
        self.use_word_embeddings = use_word_embeddings
        self.embeddings = embeddings
        self.char_embed_size = char_embed_size
        self.shuffle = shuffle
        self.tokenize = tokenize
        self.max_sequence_length = max_sequence_length
        if preprocessor.return_features and self.features is None:
            raise ValueError('features required')
        if additional_token_feature_indices and features is None:
            raise ValueError('features required for additional token values')
        self.batch_window_indices_and_offset = None
        self.window_indices_and_offset = None
        self.name = name
        self.use_chain_crf = use_chain_crf
        if self.shuffle:
            # do we need to shuffle here?, the input was already shuffled
            self._shuffle_dataset()
        if not stateful and self.input_window_stride:
            self.window_indices_and_offset = self.generate_stateless_window_indices_and_offset()
        elif stateful and self.input_window_stride:
            self.batch_window_indices_and_offset = self.generate_batch_window_indices_and_offset()
        else:
            self.log_window_config()
        if self.shuffle and self.input_window_stride and stateful:
            LOGGER.info(
                'not shuffling between epochs as number of batch windows could change (name=%s)',
                self.name
            )
            self.shuffle = False
        self.is_deprecated_padded_batch_text_list_enabled = (
            is_deprecated_padded_batch_text_list_enabled
        )
        if is_deprecated_padded_batch_text_list_enabled:
            LOGGER.warning('using deprecated padded batch_text_list (name=%s)', self.name)

    def get_sample_count(self) -> int:
        if self.window_indices_and_offset:
            return len(self.window_indices_and_offset)
        if self.batch_window_indices_and_offset:
            return len(self.batch_window_indices_and_offset) * self.batch_size
        return len(self.x)

    def get_batch_count(self) -> int:
        return int((self.get_sample_count() + self.batch_size - 1) / self.batch_size)

    def __len__(self):
        'Denotes the number of batches per epoch'
        # The number of batches is set so that each training sample is seen at most once per epoch
        return self.get_batch_count()

    def __getitem__(self, index):
        'Generate one batch of data'
        # generate data for the current batch index
        return self.__data_generation(index)

    def _shuffle_dataset(self):
        if self.window_indices_and_offset:
            np.random.shuffle(self.window_indices_and_offset)
            return
        arrays_to_shuffle = [self.x]
        if self.y is not None:
            arrays_to_shuffle.append(self.y)
        if self.features is not None:
            arrays_to_shuffle.append(self.features)
        shuffle_arrays(arrays_to_shuffle)

    def get_sequence_lengths(self) -> List[int]:
        return [len(item) for item in self.x]

    def log_window_config(self):
        LOGGER.info(
            ' '.join([
                'max sequence length: %s, input window stride: %s'
                ' (%d samples -> %d batches) (name=%s)'
            ]),
            self.max_sequence_length,
            self.input_window_stride,
            len(self.x),
            self.get_batch_count(),
            self.name
        )

    def generate_stateless_window_indices_and_offset(self):
        window_indices_and_offset = get_stateless_window_indices_and_offset(
            sequence_lengths=self.get_sequence_lengths(),
            window_stride=self.input_window_stride
        )
        LOGGER.info(
            'max sequence length: %s, input window stride: %s (%d samples -> %s windows) (name=%s)',
            self.max_sequence_length,
            self.input_window_stride,
            len(self.x),
            len(window_indices_and_offset),
            self.name
        )
        return window_indices_and_offset

    def generate_batch_window_indices_and_offset(self):
        batch_window_indices_and_offset = get_batch_window_indices_and_offset(
            sequence_lengths=self.get_sequence_lengths(),
            window_stride=self.input_window_stride,
            batch_size=self.batch_size
        )
        LOGGER.info(
            ' '.join([
                'max sequence length: %s, input window stride: %s'
                ' (%d samples -> %d batches) (name=%s)'
            ]),
            self.max_sequence_length,
            self.input_window_stride,
            len(self.x),
            len(batch_window_indices_and_offset),
            self.name
        )
        return batch_window_indices_and_offset

    def on_epoch_end(self):
        # shuffle dataset at each epoch
        if self.shuffle:
            self._shuffle_dataset()

    def get_batch_window_indices_and_offsets(self, batch_index: int) -> List[Tuple[int, int]]:
        if self.batch_window_indices_and_offset:
            return self.batch_window_indices_and_offset[batch_index]
        elif self.window_indices_and_offset:
            return self.window_indices_and_offset[
                batch_index * self.batch_size:(batch_index + 1) * self.batch_size
            ]
        else:
            indices = list(range(
                batch_index * self.batch_size,
                min(len(self.x), (batch_index + 1) * self.batch_size)
            ))
            return [(index, 0) for index in indices]

    def __data_generation(self, index):
        return self.get_window_batch_data(
            self.get_batch_window_indices_and_offsets(index)
        )

    def to_batch_embedding_vector(
        self,
        batch_tokens: List[List[str]],
        max_length: int
    ) -> np.ndarray:
        if not self.use_word_embeddings:
            return to_dummy_batch_embedding_vector(batch_tokens, max_length)
        assert self.embeddings is not None
        return to_batch_embedding_vector(batch_tokens, self.embeddings, max_length)

    def to_concatenated_batch_vector_from_batch_text_list(
        self,
        batch_text_list: List[List[str]],
        max_length: int,
        text_is_token: bool
    ) -> np.ndarray:
        return to_concatenated_batch_vector_from_batch_text_list(
            self.to_batch_embedding_vector,
            batch_text_list,
            max_length,
            concatenated_embeddings_token_count=self.concatenated_embeddings_token_count,
            text_is_token=text_is_token
        )

    def to_padded_batch_text_list(
            self,
            batch_text_list: List[List[str]]) -> List[List[str]]:
        if not self.is_deprecated_padded_batch_text_list_enabled:
            return batch_text_list
        return get_token_padded_batch_text_list(batch_text_list)

    def get_window_batch_data(  # pylint: disable=too-many-statements
            self,
            window_indices_and_offsets: List[Tuple[int, int]]):
        'Generates data containing batch_size samples'

        # restrict data to index window
        # Note: can't apply max_sequence_length here because we may tokenize
        sub_x = take_with_offset(self.x, window_indices_and_offsets)

        # tokenize texts in self.x if not already done
        if self.tokenize:
            x_tokenized = [
                tokenizeAndFilterSimple(text)
                for text in sub_x
            ]
        else:
            x_tokenized = sub_x

        max_length_x = max((len(tokens) for tokens in x_tokenized))

        if self.max_sequence_length and max_length_x > self.max_sequence_length:
            max_length_x = self.max_sequence_length
            # truncation of sequence at max_sequence_length
            x_tokenized = truncate_batch_values(x_tokenized, self.max_sequence_length)

        # prevent sequence of length 1 alone in a batch (this causes an error in tf)
        extend = False
        if max_length_x == 1:
            max_length_x += 1
            extend = True

        batch_y = None

        sub_f = None
        if (
                self.preprocessor.return_features
                or self.additional_token_feature_indices
                or self.text_feature_indices
        ):
            assert self.features is not None
            sub_f = take_with_offset(
                self.features,
                window_indices_and_offsets,
                max_sequence_length=max_length_x
            )

        batch_text_list = list(iter_batch_text_list(
            x_tokenized,
            batch_features=sub_f,
            additional_token_feature_indices=self.additional_token_feature_indices,
            text_feature_indices=self.text_feature_indices
        ))
        LOGGER.debug('batch_text_list: %s', batch_text_list)

        padded_batch_text_list = self.to_padded_batch_text_list(
            batch_text_list
        )
        LOGGER.debug('padded_batch_text_list: %s', padded_batch_text_list)

        batch_x = self.to_concatenated_batch_vector_from_batch_text_list(
            batch_text_list,
            max_length_x,
            text_is_token=(
                not self.additional_token_feature_indices
                and not self.text_feature_indices
            )
        )

        if self.preprocessor.return_casing:
            batch_a = to_batch_casing(x_tokenized, max_length_x)

        batch_y = None
        # store tag embeddings
        if self.y is not None:
            batch_y = take_with_offset(self.y, window_indices_and_offsets)
            max_length_y = max((len(y_row) for y_row in batch_y))
            if self.max_sequence_length and max_length_y > self.max_sequence_length:
                max_length_y = self.max_sequence_length
                # truncation of sequence at max_sequence_length
                batch_y = truncate_batch_values(batch_y, self.max_sequence_length)

            batches, batch_y = self.preprocessor.transform(
                padded_batch_text_list,
                batch_y,
                extend=extend,
                label_indices=not self.use_chain_crf
            )
            truncated_batch_y = truncate_batch_values(batch_y, max_length_x)
            try:
                batch_y = np.asarray(truncated_batch_y, dtype=np.int32)
            except ValueError:
                max_length_y = max(len(y_row) for y_row in truncated_batch_y)
                batch_y = np.zeros((len(truncated_batch_y), max_length_y), dtype=np.int32)
                for i, y_row in enumerate(truncated_batch_y):
                    batch_y[i, :len(y_row)] = y_row
        else:
            batches = self.preprocessor.transform(
                padded_batch_text_list, extend=extend
            )

        batch_c = left_pad_batch_values(
            np.asarray(batches[0], dtype=object),
            max_length_x,
            dtype=np.int32
        )

        batch_l = batches[1]

        inputs = []
        inputs.append(batch_x)
        inputs.append(batch_c)
        if self.preprocessor.return_casing:
            inputs.append(batch_a)
        if self.preprocessor.return_features:
            LOGGER.debug('extend: %s', extend)
            try:
                batch_features = self.preprocessor.transform_features(sub_f, extend=extend)
                batch_features = left_pad_batch_values(
                    batch_features,
                    max_length_x,
                    dtype=np.float32
                )
            except TypeError:
                batch_features = left_pad_batch_values(
                    self.preprocessor.transform_features(sub_f),
                    max_length_x,
                    dtype=np.float32
                )
            LOGGER.debug(
                'batch_features: shape=%s (dtype=%s)',
                batch_features.shape,
                batch_features.dtype
            )
            if batch_features.dtype == np.object0:
                LOGGER.warning(
                    'invalid object type of batch_features sample data[0]: %s',
                    batch_features[0][0:min(2, len(batch_features[0]))]
                )
            inputs.append(batch_features)
        inputs.append(batch_l)

        if LOGGER.isEnabledFor(logging.DEBUG):
            input_nparrays = [
                np.asarray(x)
                for x in inputs
            ]
            LOGGER.debug('inputs shapes: %s', [x.shape for x in input_nparrays])
            LOGGER.debug('inputs dtype: %s', [x.dtype for x in input_nparrays])
            for index, input_array in enumerate(input_nparrays):
                if input_array.dtype == np.object0:
                    LOGGER.warning(
                        'invalid object type inputs[%d] sample data[0]: %s',
                        index,
                        input_array[0][0:min(2, len(input_array[0]))]
                    )

        return inputs, batch_y
