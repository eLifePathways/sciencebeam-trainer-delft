import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig, TrainingConfig
from sciencebeam_trainer_delft.sequence_labelling.models import CustomBidLSTM_CRF
from sciencebeam_trainer_delft.sequence_labelling.preprocess import Preprocessor
from sciencebeam_trainer_delft.sequence_labelling.saving import ModelSaver
from sciencebeam_trainer_delft.sequence_labelling.trainer_torch import ModelTrainer


TOKENS_1 = ['One', 'Two', 'Three']
TOKENS_2 = ['Four', 'Five', 'Six']

TAGS_1 = ['B-<title>', 'I-<title>', 'B-<author>']
TAGS_2 = ['B-<author>', 'I-<author>', 'B-<title>']


@pytest.fixture(name='x')
def _x() -> np.ndarray:
    return np.asarray([TOKENS_1, TOKENS_2], dtype='object')


@pytest.fixture(name='y')
def _y() -> np.ndarray:
    return np.asarray([TAGS_1, TAGS_2], dtype='object')


@pytest.fixture(name='preprocessor')
def _preprocessor(x: np.ndarray, y: np.ndarray) -> Preprocessor:
    preprocessor = Preprocessor(max_char_length=5)
    preprocessor.fit(x, y)
    return preprocessor


@pytest.fixture(name='model_config')
def _model_config(preprocessor: Preprocessor) -> ModelConfig:
    return ModelConfig(
        model_name='test-model',
        architecture='CustomBidLSTM_CRF',
        embeddings_name=None,
        char_vocab_size=len(preprocessor.vocab_char),
        char_embedding_size=5,
        num_char_lstm_units=4,
        max_char_length=5,
        num_word_lstm_units=6,
        word_embedding_size=0,
        max_sequence_length=10,
        dropout=0.0,
        use_features=False,
        use_chain_crf=True
    )


def _training_config(**kwargs) -> TrainingConfig:
    values = {
        'learning_rate': 0.05,
        'batch_size': 2,
        'max_epoch': 2,
        'early_stop': False,
        'patience': 2,
        'lr_decay': 0.9,
        'clip_gradients': 5.0,
        **kwargs
    }
    return TrainingConfig(**values)  # type: ignore[arg-type]


def _model_trainer(
    model_config: ModelConfig,
    preprocessor: Preprocessor,
    training_config: TrainingConfig,
    checkpoint_path: Optional[str] = None,
    model_saver: Optional[ModelSaver] = None
) -> ModelTrainer:
    model = CustomBidLSTM_CRF(model_config, len(preprocessor.vocab_tag))
    return ModelTrainer(
        model,
        model_config=model_config,
        training_config=training_config,
        preprocessor=preprocessor,
        model_saver=model_saver,
        checkpoint_path=checkpoint_path
    )


class TestModelTrainerTrain:
    def test_should_train_for_the_configured_epochs(
        self, model_config: ModelConfig, preprocessor: Preprocessor,
        x: np.ndarray, y: np.ndarray
    ):
        model_trainer = _model_trainer(
            model_config, preprocessor, _training_config(max_epoch=2)
        )
        history = model_trainer.train(x, y)
        assert sorted(history) == ['epoch_0_loss', 'epoch_1_loss']

    def test_should_score_the_validation_set_when_stopping_early(
        self, model_config: ModelConfig, preprocessor: Preprocessor,
        x: np.ndarray, y: np.ndarray
    ):
        model_trainer = _model_trainer(
            model_config, preprocessor,
            _training_config(max_epoch=1, early_stop=True)
        )
        assert model_trainer.train(x, y, x, y)

    def test_should_require_validation_data_when_stopping_early(
        self, model_config: ModelConfig, preprocessor: Preprocessor,
        x: np.ndarray, y: np.ndarray
    ):
        model_trainer = _model_trainer(
            model_config, preprocessor,
            _training_config(max_epoch=1, early_stop=True)
        )
        with pytest.raises(ValueError):
            model_trainer.train(x, y)

    def test_should_train_on_the_validation_data_without_early_stopping(
        self, model_config: ModelConfig, preprocessor: Preprocessor,
        x: np.ndarray, y: np.ndarray
    ):
        model_trainer = _model_trainer(
            model_config, preprocessor,
            _training_config(max_epoch=1, early_stop=False, batch_size=4)
        )
        model_trainer.train(x, y, x, y)
        training_data_loader = model_trainer.create_data_loader(
            np.concatenate((x, x), axis=0),
            np.concatenate((y, y), axis=0),
            shuffle=False,
            features=None,
            name_suffix='check'
        )
        assert len(training_data_loader) == 1

    def test_should_require_features_when_the_preprocessor_returns_them(
        self, model_config: ModelConfig, preprocessor: Preprocessor,
        x: np.ndarray, y: np.ndarray
    ):
        preprocessor.return_features = True
        model_trainer = _model_trainer(
            model_config, preprocessor, _training_config(max_epoch=1)
        )
        with pytest.raises(ValueError):
            model_trainer.train(x, y)


class TestModelTrainerCheckpoints:
    def test_should_write_no_checkpoints_without_a_checkpoint_path(
        self, model_config: ModelConfig, preprocessor: Preprocessor,
        x: np.ndarray, y: np.ndarray
    ):
        model_trainer = _model_trainer(
            model_config, preprocessor, _training_config(max_epoch=1)
        )
        assert model_trainer.get_save_checkpoint() is None
        assert model_trainer.train(x, y)

    def test_should_write_a_one_based_epoch_directory_per_epoch(
        self, model_config: ModelConfig, preprocessor: Preprocessor,
        x: np.ndarray, y: np.ndarray, temp_dir: Path
    ):
        model_saver = ModelSaver(preprocessor=preprocessor, model_config=model_config)
        model_trainer = _model_trainer(
            model_config, preprocessor, _training_config(max_epoch=2),
            checkpoint_path=str(temp_dir), model_saver=model_saver
        )
        model_trainer.train(x, y)
        assert sorted(
            path.name for path in temp_dir.iterdir() if path.is_dir()
        ) == ['epoch-00001', 'epoch-00002']

    def test_should_record_the_resumable_epoch_in_the_checkpoints_json(
        self, model_config: ModelConfig, preprocessor: Preprocessor,
        x: np.ndarray, y: np.ndarray, temp_dir: Path
    ):
        model_saver = ModelSaver(preprocessor=preprocessor, model_config=model_config)
        model_trainer = _model_trainer(
            model_config, preprocessor, _training_config(max_epoch=2),
            checkpoint_path=str(temp_dir), model_saver=model_saver
        )
        model_trainer.train(x, y)
        checkpoints_json = json.loads((temp_dir / 'checkpoints.json').read_text())
        assert [
            checkpoint['epoch'] for checkpoint in checkpoints_json['checkpoints']
        ] == [1, 2]

    def test_should_write_meta_a_resume_can_read(
        self, model_config: ModelConfig, preprocessor: Preprocessor,
        x: np.ndarray, y: np.ndarray, temp_dir: Path
    ):
        model_saver = ModelSaver(preprocessor=preprocessor, model_config=model_config)
        model_trainer = _model_trainer(
            model_config, preprocessor,
            _training_config(max_epoch=1, early_stop=True),
            checkpoint_path=str(temp_dir), model_saver=model_saver
        )
        model_trainer.train(x, y, x, y)
        meta = json.loads((temp_dir / 'epoch-00001' / 'meta.json').read_text())
        assert meta['epoch'] == 1
        assert meta['early_stopping']['best'] is not None
        assert meta['optimizer']['type'] == 'torch.optim.adam.Adam'
        assert 'training_config' in meta

    def test_should_omit_the_initial_meta_from_the_saved_training_config(
        self, model_config: ModelConfig, preprocessor: Preprocessor
    ):
        model_trainer = _model_trainer(
            model_config, preprocessor,
            _training_config(initial_meta={'early_stopping': {'best': 0.5}})
        )
        assert 'initial_meta' not in model_trainer.get_meta()['training_config']

    def test_should_honour_the_checkpoint_epoch_interval(
        self, model_config: ModelConfig, preprocessor: Preprocessor,
        x: np.ndarray, y: np.ndarray, temp_dir: Path
    ):
        model_saver = ModelSaver(preprocessor=preprocessor, model_config=model_config)
        model_trainer = _model_trainer(
            model_config, preprocessor,
            _training_config(max_epoch=4, checkpoint_epoch_interval=2),
            checkpoint_path=str(temp_dir), model_saver=model_saver
        )
        model_trainer.train(x, y)
        assert sorted(
            path.name for path in temp_dir.iterdir() if path.is_dir()
        ) == ['epoch-00002', 'epoch-00004']
