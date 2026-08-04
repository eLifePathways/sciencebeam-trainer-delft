from unittest.mock import MagicMock

import torch

from sciencebeam_trainer_delft.sequence_labelling.evaluation import (
    get_classification_result_for_model,
    get_f1_scorer
)


TAGS = {0: '<PAD>', 1: 'B-<title>', 2: 'I-<title>', 3: 'B-<author>'}


def _preprocessor() -> MagicMock:
    preprocessor = MagicMock(name='preprocessor')
    preprocessor.inverse_transform.side_effect = lambda indices: [
        TAGS[index] for index in indices
    ]
    return preprocessor


def _model(predictions) -> MagicMock:
    model = MagicMock(name='model')
    model.decode.return_value = predictions
    return model


def _data_loader(labels):
    inputs = {'char_input': torch.zeros(len(labels), len(labels[0]))}
    return [(inputs, torch.tensor(labels))]


class TestGetClassificationResultForModel:
    def test_should_score_a_perfect_prediction(self):
        labels = [[1, 2, 2]]
        result = get_classification_result_for_model(
            _model([[1, 2, 2]]), _data_loader(labels), _preprocessor()
        )
        assert result.micro_averages['f1'] == 1.0

    def test_should_score_a_wrong_prediction(self):
        labels = [[1, 2, 2]]
        result = get_classification_result_for_model(
            _model([[3, 3, 3]]), _data_loader(labels), _preprocessor()
        )
        assert result.micro_averages['f1'] == 0.0

    def test_should_truncate_predictions_to_the_expected_length(self):
        labels = [[1, 2]]
        # the model predicts for a padded batch, longer than the expected tags
        result = get_classification_result_for_model(
            _model([[1, 2, 3]]), _data_loader(labels), _preprocessor()
        )
        assert result.micro_averages['f1'] == 1.0

    def test_should_score_every_sequence_in_the_batch(self):
        labels = [[1, 2], [3, 0]]
        result = get_classification_result_for_model(
            _model([[1, 2], [3, 0]]), _data_loader(labels), _preprocessor()
        )
        assert result.micro_averages['support'] == 2


class TestGetF1Scorer:
    def test_should_return_the_micro_f1_for_a_model(self):
        scorer = get_f1_scorer(_data_loader([[1, 2, 2]]), _preprocessor())
        assert scorer(_model([[1, 2, 2]])) == 1.0
