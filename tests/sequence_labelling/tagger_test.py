import logging
from unittest.mock import MagicMock
from typing import Dict, List, Optional

import pytest

import torch

from delft.sequenceLabelling.preprocess import PAD

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.data_loader_torch import CHAR_INPUT
from sciencebeam_trainer_delft.sequence_labelling.preprocess import (
    WordPreprocessor
)
from sciencebeam_trainer_delft.sequence_labelling.dataset_transform.unroll_transform import (
    UnrollingTextFeatureDatasetTransformer
)

from sciencebeam_trainer_delft.sequence_labelling.tagger import (
    Tagger
)


LOGGER = logging.getLogger(__name__)


TOKEN_1 = 'token1'
TOKEN_2 = 'token2'
TOKEN_3 = 'token3'

UNROLLED_TOKEN_1 = 'unroll1'
UNROLLED_TOKEN_2 = 'unroll2'
UNROLLED_TOKEN_3 = 'unroll3'
UNROLLED_TOKEN_4 = 'unroll4'

UNROLLED_TOKENS = [UNROLLED_TOKEN_1, UNROLLED_TOKEN_2, UNROLLED_TOKEN_3, UNROLLED_TOKEN_4]

ALL_TOKENS = [TOKEN_1, TOKEN_2, TOKEN_3] + UNROLLED_TOKENS

TAG_BY_UNROLLED_TOKEN = {
    token: f'tag-{token}'
    for token in UNROLLED_TOKENS
}

TAG_1 = 'tag1'
TAG_2 = 'tag2'
TAG_3 = 'tag3'
ALL_TAGS = [TAG_1, TAG_2, TAG_3] + list(TAG_BY_UNROLLED_TOKEN.values())

DEFAULT_TAG_BY_TOKEN_MAP = dict(zip([''] + ALL_TOKENS, [PAD] + ALL_TAGS))


@pytest.fixture(name='model_mock')
def _model_mock():
    return MagicMock(name='model')


@pytest.fixture(name='model_config')
def _model_config():
    return ModelConfig(model_name='test_model', batch_size=1)


def get_preprocessor(
        tokens: List[str],
        tags: List[str],
        **kwargs) -> WordPreprocessor:
    preprocessor = WordPreprocessor(**kwargs)
    preprocessor.fit([tokens], [tags])
    return preprocessor


@pytest.fixture(name='preprocessor')
def _preprocessor() -> WordPreprocessor:
    return get_preprocessor(tokens=ALL_TOKENS, tags=ALL_TAGS)


@pytest.fixture(name='preprocessor_mock')
def _preprocessor_mock():
    preprocessor_mock = MagicMock(name='preprocessor')
    preprocessor_mock.return_features = False
    return preprocessor_mock


def inverse_char_transform(
        char_indices: List[List[List[int]]],
        char_by_index_map: Dict[int, str]):
    LOGGER.debug('char_by_index_map: %s', char_by_index_map)
    return [
        [
            ''.join([
                char_by_index_map[char_index]
                for char_index in seq
                if char_index
            ])
            for seq in doc
        ]
        for doc in char_indices
    ]


def get_decode_by_token_fn(
    tag_by_token_map: Dict[str, str],
    preprocessor: WordPreprocessor,
    batch_size: Optional[int] = None
):
    """Fakes the model, decoding each token to the tag the map gives it.

    The model returns one tag index per token, which is what the CRF's decode
    produces, so the tokens are recovered from the char input to look up.
    """
    vocab_tag = preprocessor.vocab_tag
    LOGGER.debug('vocab_tag=%s', vocab_tag)
    vocab_char: Dict[str, int] = preprocessor.vocab_char or {}
    char_by_index_map = {i: c for c, i in vocab_char.items()}

    def decode(inputs):
        char_inputs = inputs[CHAR_INPUT].numpy()
        inverse_char_inputs = inverse_char_transform(char_inputs, char_by_index_map)
        if batch_size:
            assert len(char_inputs) == batch_size, (
                "expected batch size: %d, but was %d" % (batch_size, len(char_inputs))
            )
        LOGGER.debug('inverse_char_inputs: %s', inverse_char_inputs)
        return torch.tensor([
            [
                vocab_tag[tag_by_token_map[token]]
                for token in doc
            ]
            for doc in inverse_char_inputs
        ])
    return decode


class TestTagger:
    def test_should_return_empty_result_for_empty_input(
            self,
            model_mock: MagicMock,
            model_config: ModelConfig,
            preprocessor: WordPreprocessor):
        tagger = Tagger(
            model=model_mock,
            model_config=model_config,
            preprocessor=preprocessor,
            max_sequence_length=2,
            input_window_stride=None
        )
        model_mock.decode.side_effect = get_decode_by_token_fn(
            DEFAULT_TAG_BY_TOKEN_MAP,
            preprocessor=preprocessor,
            batch_size=model_config.batch_size
        )
        tag_result = tagger.tag([], output_format=None)
        LOGGER.debug('tag_result: %s', tag_result)
        assert tag_result == []

    def test_should_truncate_without_input_window_stride(
            self,
            model_mock: MagicMock,
            model_config: ModelConfig,
            preprocessor: WordPreprocessor):
        tagger = Tagger(
            model=model_mock,
            model_config=model_config,
            preprocessor=preprocessor,
            max_sequence_length=2,
            input_window_stride=None
        )
        model_mock.decode.side_effect = get_decode_by_token_fn(
            DEFAULT_TAG_BY_TOKEN_MAP,
            preprocessor=preprocessor,
            batch_size=model_config.batch_size
        )
        tag_result = tagger.tag(
            [
                [TOKEN_1, TOKEN_2, TOKEN_3]
            ],
            output_format=None
        )
        LOGGER.debug('tag_result: %s', tag_result)
        assert tag_result == [
            [(TOKEN_1, TAG_1), (TOKEN_2, TAG_2)]
        ]

    def test_should_not_truncate_with_input_window_stride(
            self,
            model_mock: MagicMock,
            model_config: ModelConfig,
            preprocessor: WordPreprocessor):
        tagger = Tagger(
            model=model_mock,
            model_config=model_config,
            preprocessor=preprocessor,
            max_sequence_length=2,
            input_window_stride=2
        )
        model_mock.decode.side_effect = get_decode_by_token_fn(
            DEFAULT_TAG_BY_TOKEN_MAP,
            preprocessor=preprocessor,
            batch_size=model_config.batch_size
        )
        tag_result = tagger.tag(
            [
                [TOKEN_1, TOKEN_2, TOKEN_3]
            ],
            output_format=None
        )
        LOGGER.debug('tag_result: %s', tag_result)
        assert tag_result == [
            [(TOKEN_1, TAG_1), (TOKEN_2, TAG_2), (TOKEN_3, TAG_3)]
        ]

    def test_should_allow_shorter_input_window_stride(
            self,
            model_mock: MagicMock,
            model_config: ModelConfig,
            preprocessor: WordPreprocessor):
        tagger = Tagger(
            model=model_mock,
            model_config=model_config,
            preprocessor=preprocessor,
            max_sequence_length=2,
            input_window_stride=1
        )
        model_mock.decode.side_effect = get_decode_by_token_fn(
            DEFAULT_TAG_BY_TOKEN_MAP,
            preprocessor=preprocessor,
            batch_size=model_config.batch_size
        )
        tag_result = tagger.tag(
            [
                [TOKEN_1, TOKEN_2, TOKEN_3]
            ],
            output_format=None
        )
        LOGGER.debug('tag_result: %s', tag_result)
        assert tag_result == [
            [(TOKEN_1, TAG_1), (TOKEN_2, TAG_2), (TOKEN_3, TAG_3)]
        ]

    def test_should_not_truncate_without_max_sequence_length(
            self,
            model_mock: MagicMock,
            model_config: ModelConfig,
            preprocessor: WordPreprocessor):
        tagger = Tagger(
            model=model_mock,
            model_config=model_config,
            preprocessor=preprocessor,
            max_sequence_length=None,
            input_window_stride=None
        )
        model_mock.decode.side_effect = get_decode_by_token_fn(
            DEFAULT_TAG_BY_TOKEN_MAP,
            preprocessor=preprocessor,
            batch_size=model_config.batch_size
        )
        tag_result = tagger.tag(
            [
                [TOKEN_1, TOKEN_2, TOKEN_3]
            ],
            output_format=None
        )
        LOGGER.debug('tag_result: %s', tag_result)
        assert tag_result == [
            [(TOKEN_1, TAG_1), (TOKEN_2, TAG_2), (TOKEN_3, TAG_3)]
        ]

    def test_should_tag_tokenized_texts_with_varying_lengths_and_sliding_windows(
            self,
            model_mock: MagicMock,
            model_config: ModelConfig,
            preprocessor: WordPreprocessor):
        tagger = Tagger(
            model=model_mock,
            model_config=model_config,
            preprocessor=preprocessor,
            max_sequence_length=2,
            input_window_stride=2
        )
        model_mock.decode.side_effect = get_decode_by_token_fn(
            DEFAULT_TAG_BY_TOKEN_MAP,
            preprocessor=preprocessor,
            batch_size=model_config.batch_size
        )
        tag_result = tagger.tag(
            [
                [TOKEN_1, TOKEN_2, TOKEN_3],
                [TOKEN_2, TOKEN_3]
            ],
            output_format=None
        )
        LOGGER.debug('tag_result: %s', tag_result)
        assert tag_result == [
            [(TOKEN_1, TAG_1), (TOKEN_2, TAG_2), (TOKEN_3, TAG_3)],
            [(TOKEN_2, TAG_2), (TOKEN_3, TAG_3)]
        ]

    def test_should_tag_tokenized_texts_with_exact_batch_size_if_stateful(
            self,
            model_mock: MagicMock,
            model_config: ModelConfig,
            preprocessor: WordPreprocessor):
        model_config.stateful = True
        model_config.batch_size = 2
        tagger = Tagger(
            model=model_mock,
            model_config=model_config,
            preprocessor=preprocessor,
            max_sequence_length=2,
            input_window_stride=2
        )
        model_mock.decode.side_effect = get_decode_by_token_fn(
            DEFAULT_TAG_BY_TOKEN_MAP,
            preprocessor=preprocessor,
            batch_size=model_config.batch_size
        )
        tag_result = tagger.tag(
            [
                [TOKEN_1, TOKEN_2, TOKEN_3],
                [TOKEN_2, TOKEN_3]
            ],
            output_format=None
        )
        LOGGER.debug('tag_result: %s', tag_result)
        assert tag_result == [
            [(TOKEN_1, TAG_1), (TOKEN_2, TAG_2), (TOKEN_3, TAG_3)],
            [(TOKEN_2, TAG_2), (TOKEN_3, TAG_3)]
        ]

    def test_should_tag_with_dataset_transformer(
            self,
            model_mock: MagicMock,
            model_config: ModelConfig,
            preprocessor: WordPreprocessor):
        tagger = Tagger(
            model=model_mock,
            model_config=model_config,
            preprocessor=preprocessor,
            dataset_transformer_factory=lambda: (
                UnrollingTextFeatureDatasetTransformer(0)
            )
        )
        model_mock.decode.side_effect = get_decode_by_token_fn(
            DEFAULT_TAG_BY_TOKEN_MAP,
            preprocessor=preprocessor,
            batch_size=model_config.batch_size
        )
        texts = [
            [TOKEN_1, TOKEN_2],
        ]
        features = [
            [
                [f'{UNROLLED_TOKEN_1} {UNROLLED_TOKEN_2}'],
                [f'{UNROLLED_TOKEN_3} {UNROLLED_TOKEN_4}']
            ]
        ]
        tag_result = tagger.tag(
            texts, features=features, output_format=None
        )
        LOGGER.debug('tag_result: %s', tag_result)
        assert tag_result == [
            [
                (TOKEN_1, TAG_BY_UNROLLED_TOKEN[UNROLLED_TOKEN_1]),
                (TOKEN_2, TAG_BY_UNROLLED_TOKEN[UNROLLED_TOKEN_3])
            ]
        ]
        transformed_tag_result = tagger.tag(
            texts, features=features, output_format=None,
            tag_transformed=True
        )
        LOGGER.debug('transformed_tag_result: %s', transformed_tag_result)
        assert transformed_tag_result == [
            [
                (UNROLLED_TOKEN_1, TAG_BY_UNROLLED_TOKEN[UNROLLED_TOKEN_1]),
                (UNROLLED_TOKEN_2, TAG_BY_UNROLLED_TOKEN[UNROLLED_TOKEN_2]),
                (UNROLLED_TOKEN_3, TAG_BY_UNROLLED_TOKEN[UNROLLED_TOKEN_3]),
                (UNROLLED_TOKEN_4, TAG_BY_UNROLLED_TOKEN[UNROLLED_TOKEN_4])
            ]
        ]
