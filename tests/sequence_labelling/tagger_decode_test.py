"""Checks that tagging reports the CRF's best path.

Reference output captured from the TensorFlow models agrees with the decoded
path and not with the best tag at each position independently, so a regression
to per-token argmax has to be caught here rather than at review time.
"""
import numpy as np
import pytest
import torch

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.data_loader_torch import (
    get_input_names,
    to_model_inputs
)
from sciencebeam_trainer_delft.sequence_labelling.models import CustomBidLSTM_CRF
from sciencebeam_trainer_delft.sequence_labelling.wrapper import get_vocab_size
from sciencebeam_trainer_delft.sequence_labelling.preprocess import WordPreprocessor
from sciencebeam_trainer_delft.sequence_labelling.tagger import Tagger


TOKENS = ['One', 'Two', 'Three', 'Four']
TAGS = ['B-<title>', 'I-<title>', 'B-<author>', 'I-<author>']

MAX_CHAR_LENGTH = 5


@pytest.fixture(name='preprocessor')
def _preprocessor() -> WordPreprocessor:
    preprocessor = WordPreprocessor(max_char_length=MAX_CHAR_LENGTH)
    preprocessor.fit([TOKENS], [TAGS])
    return preprocessor


@pytest.fixture(name='model_config')
def _model_config(preprocessor: WordPreprocessor) -> ModelConfig:
    return ModelConfig(
        model_name='test-model',
        architecture='CustomBidLSTM_CRF',
        embeddings_name=None,
        char_vocab_size=get_vocab_size(preprocessor.vocab_char),
        char_embedding_size=5,
        num_char_lstm_units=4,
        max_char_length=MAX_CHAR_LENGTH,
        num_word_lstm_units=6,
        word_embedding_size=0,
        max_sequence_length=10,
        batch_size=1,
        dropout=0.0,
        use_features=False,
        use_chain_crf=True
    )


@pytest.fixture(name='model')
def _model(model_config: ModelConfig, preprocessor: WordPreprocessor) -> CustomBidLSTM_CRF:
    torch.manual_seed(7)
    model = CustomBidLSTM_CRF(model_config, get_vocab_size(preprocessor.vocab_tag))
    model.eval()
    return model


def _tagger(model, model_config, preprocessor) -> Tagger:
    return Tagger(
        model=model,
        model_config=model_config,
        preprocessor=preprocessor,
        max_sequence_length=model_config.max_sequence_length
    )


def _get_tags(model, model_config, preprocessor):
    tag_result = _tagger(model, model_config, preprocessor).tag(
        [list(TOKENS)], output_format=None
    )
    return [tag for _, tag in tag_result[0]]


def _get_batch_inputs(model_config: ModelConfig, preprocessor: WordPreprocessor):
    from sciencebeam_trainer_delft.sequence_labelling.data_generator import (  # noqa: E501 pylint: disable=import-outside-toplevel
        DataGenerator
    )
    data_generator = DataGenerator(
        x=[list(TOKENS)], y=None,
        batch_size=1,
        preprocessor=preprocessor,
        char_embed_size=model_config.char_embedding_size,
        max_sequence_length=model_config.max_sequence_length,
        shuffle=False,
        use_chain_crf=model_config.use_chain_crf
    )
    arrays, _ = data_generator[0]
    return to_model_inputs(get_input_names(data_generator), arrays)


class TestTaggerDecodesTheCrfPath:
    def test_should_report_the_decoded_tags(
        self, model, model_config: ModelConfig, preprocessor: WordPreprocessor
    ):
        inputs = _get_batch_inputs(model_config, preprocessor)
        expected_tags = preprocessor.inverse_transform(
            model.decode(inputs)[0][:len(TOKENS)].tolist()
        )
        assert _get_tags(model, model_config, preprocessor) == list(expected_tags)

    def test_should_not_report_per_token_argmax(
        self, model, model_config: ModelConfig, preprocessor: WordPreprocessor
    ):
        # a start transition strong enough that the best path cannot start at
        # the tag with the highest score of its own
        with torch.no_grad():
            model.crf.b_start.fill_(0.0)
            model.crf.b_start[1] = 100.0

        inputs = _get_batch_inputs(model_config, preprocessor)
        logits = model.get_logits(inputs).detach().numpy()
        argmax_tags = list(preprocessor.inverse_transform(
            np.argmax(logits[0][:len(TOKENS)], axis=-1).tolist()
        ))
        decoded_tags = list(preprocessor.inverse_transform(
            model.decode(inputs)[0][:len(TOKENS)].tolist()
        ))
        # the transition has to actually change the first tag, or this proves
        # nothing
        assert decoded_tags[0] != argmax_tags[0]
        assert _get_tags(model, model_config, preprocessor) == decoded_tags
