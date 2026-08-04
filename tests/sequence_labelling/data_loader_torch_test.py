from typing import List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.data_loader_torch import (
    DataLoader,
    get_input_names
)
from sciencebeam_trainer_delft.sequence_labelling.models_torch import CustomBidLSTM_CRF


BATCH_SIZE = 2
SEQUENCE_LENGTH = 4
NTAGS = 5

CHAR_VOCAB_SIZE = 12
MAX_CHAR_LENGTH = 6
WORD_EMBEDDING_SIZE = 3
MAX_FEATURE_SIZE = 7


def _fake_data_generator(
    batches: List[tuple],
    return_casing: bool = False,
    return_features: bool = True
) -> MagicMock:
    data_generator = MagicMock(name='data_generator')
    data_generator.preprocessor.return_casing = return_casing
    data_generator.preprocessor.return_features = return_features
    data_generator.__len__.return_value = len(batches)
    data_generator.__getitem__.side_effect = lambda index: batches[index]
    return data_generator


def _arrays(
    word_embedding_size: int = WORD_EMBEDDING_SIZE,
    with_features: bool = True
) -> List[np.ndarray]:
    arrays = [
        np.zeros((BATCH_SIZE, SEQUENCE_LENGTH, word_embedding_size), dtype=np.float32),
        np.ones((BATCH_SIZE, SEQUENCE_LENGTH, MAX_CHAR_LENGTH), dtype=np.int32)
    ]
    if with_features:
        arrays.append(
            np.zeros((BATCH_SIZE, SEQUENCE_LENGTH, MAX_FEATURE_SIZE), dtype=np.float32)
        )
    arrays.append(np.full((BATCH_SIZE, 1), SEQUENCE_LENGTH, dtype=np.int32))
    return arrays


def _one_hot_labels() -> np.ndarray:
    labels = np.zeros((BATCH_SIZE, SEQUENCE_LENGTH, NTAGS), dtype=np.int32)
    labels[0, :, 2] = 1
    labels[1, :, 3] = 1
    return labels


def _index_labels() -> np.ndarray:
    return np.full((BATCH_SIZE, SEQUENCE_LENGTH), 2, dtype=np.int32)


def _model_config() -> ModelConfig:
    return ModelConfig(
        architecture='CustomBidLSTM_CRF',
        char_vocab_size=CHAR_VOCAB_SIZE,
        char_embedding_size=5,
        num_char_lstm_units=4,
        max_char_length=MAX_CHAR_LENGTH,
        num_word_lstm_units=6,
        word_embedding_size=WORD_EMBEDDING_SIZE,
        dropout=0.0,
        use_features=True,
        max_feature_size=MAX_FEATURE_SIZE,
        features_embedding_size=0
    )


class TestGetInputNames:
    def test_should_name_the_inputs_the_generator_produces(self):
        data_generator = _fake_data_generator([])
        assert get_input_names(data_generator) == [
            'word_input', 'char_input', 'features_input', 'length_input'
        ]

    def test_should_omit_features_when_the_preprocessor_does_not_return_them(self):
        data_generator = _fake_data_generator([], return_features=False)
        assert get_input_names(data_generator) == [
            'word_input', 'char_input', 'length_input'
        ]

    def test_should_include_casing_before_features(self):
        data_generator = _fake_data_generator([], return_casing=True)
        assert get_input_names(data_generator) == [
            'word_input', 'char_input', 'casing_input', 'features_input', 'length_input'
        ]


class TestDataLoader:
    def test_should_yield_named_tensors_of_the_expected_dtype(self):
        data_loader = DataLoader(_fake_data_generator([(_arrays(), _index_labels())]))
        inputs, labels = data_loader.get_batch(0)
        assert set(inputs) == {
            'word_input', 'char_input', 'features_input', 'length_input'
        }
        assert inputs['word_input'].dtype == torch.float32
        assert inputs['features_input'].dtype == torch.float32
        assert inputs['char_input'].dtype == torch.long
        assert inputs['length_input'].dtype == torch.long
        assert labels is not None and labels.dtype == torch.long

    def test_should_convert_one_hot_labels_to_indices(self):
        data_loader = DataLoader(_fake_data_generator([(_arrays(), _one_hot_labels())]))
        _, labels = data_loader.get_batch(0)
        assert labels is not None
        assert labels.shape == (BATCH_SIZE, SEQUENCE_LENGTH)
        assert labels[0].tolist() == [2] * SEQUENCE_LENGTH
        assert labels[1].tolist() == [3] * SEQUENCE_LENGTH

    def test_should_keep_index_labels_unchanged(self):
        data_loader = DataLoader(_fake_data_generator([(_arrays(), _index_labels())]))
        _, labels = data_loader.get_batch(0)
        assert labels is not None
        assert labels.shape == (BATCH_SIZE, SEQUENCE_LENGTH)
        assert labels[0].tolist() == [2] * SEQUENCE_LENGTH

    def test_should_pass_through_missing_labels(self):
        labels: Optional[np.ndarray] = None
        data_loader = DataLoader(_fake_data_generator([(_arrays(), labels)]))
        _, label_tensor = data_loader.get_batch(0)
        assert label_tensor is None

    def test_should_reject_a_batch_with_unexpected_inputs(self):
        arrays = _arrays(with_features=False)
        data_loader = DataLoader(_fake_data_generator([(arrays, _index_labels())]))
        with pytest.raises(AssertionError):
            data_loader.get_batch(0)

    def test_should_iterate_every_batch_and_end_the_epoch(self):
        batches = [(_arrays(), _index_labels()) for _ in range(3)]
        data_generator = _fake_data_generator(batches)
        data_loader = DataLoader(data_generator)
        assert len(list(data_loader)) == 3
        data_generator.on_epoch_end.assert_called_once()

    def test_should_produce_batches_the_model_accepts(self):
        model_config = _model_config()
        model = CustomBidLSTM_CRF(model_config, NTAGS)
        model.eval()
        data_loader = DataLoader(_fake_data_generator([(_arrays(), _one_hot_labels())]))
        inputs, labels = data_loader.get_batch(0)
        outputs = model(inputs, labels)
        assert outputs['logits'].shape == (BATCH_SIZE, SEQUENCE_LENGTH, NTAGS)
        assert outputs['loss'].dim() == 0
