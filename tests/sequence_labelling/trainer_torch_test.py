from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from delft.utilities.crf_pytorch import ChainCRF

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig, TrainingConfig
from sciencebeam_trainer_delft.sequence_labelling.models import CustomBidLSTM_CRF
from sciencebeam_trainer_delft.sequence_labelling.trainer_torch import (
    EarlyStopping,
    MetaKeys,
    Trainer,
    set_random_seed
)
from sciencebeam_trainer_delft.sequence_labelling.upstream_patches import (
    patch_chain_crf_eager_build
)


NTAGS = 5
BATCH_SIZE = 2
SEQUENCE_LENGTH = 4
CHAR_VOCAB_SIZE = 12
MAX_CHAR_LENGTH = 5
MAX_FEATURE_SIZE = 7


@pytest.fixture(name='patched_chain_crf', autouse=True)
def _patched_chain_crf():
    original_init = ChainCRF.__init__
    patch_chain_crf_eager_build()
    yield
    ChainCRF.__init__ = original_init  # type: ignore[method-assign]


def _model_config() -> ModelConfig:
    return ModelConfig(
        architecture='CustomBidLSTM_CRF',
        char_vocab_size=CHAR_VOCAB_SIZE,
        char_embedding_size=5,
        num_char_lstm_units=4,
        max_char_length=MAX_CHAR_LENGTH,
        num_word_lstm_units=6,
        word_embedding_size=3,
        dropout=0.0,
        use_features=True,
        max_feature_size=MAX_FEATURE_SIZE,
        features_embedding_size=0
    )


def _training_config(**kwargs) -> TrainingConfig:
    values = {
        'learning_rate': 0.05,
        'batch_size': BATCH_SIZE,
        'max_epoch': 3,
        'early_stop': False,
        'patience': 2,
        'lr_decay': 0.9,
        'clip_gradients': 5.0,
        **kwargs
    }
    return TrainingConfig(**values)  # type: ignore[arg-type]


def _batches(count: int = 2) -> List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
    torch.manual_seed(7)
    batches = []
    for _ in range(count):
        inputs = {
            'word_input': torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, 3),
            'char_input': torch.randint(
                0, CHAR_VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH, MAX_CHAR_LENGTH)
            ),
            'features_input': torch.randn(
                BATCH_SIZE, SEQUENCE_LENGTH, MAX_FEATURE_SIZE
            ),
            'length_input': torch.full((BATCH_SIZE, 1), SEQUENCE_LENGTH)
        }
        labels = torch.randint(1, NTAGS, (BATCH_SIZE, SEQUENCE_LENGTH))
        batches.append((inputs, labels))
    return batches


def _model() -> CustomBidLSTM_CRF:
    return CustomBidLSTM_CRF(_model_config(), NTAGS)


class TestEarlyStopping:
    def test_should_not_stop_while_the_score_improves(self):
        early_stopping = EarlyStopping(patience=2)
        assert not early_stopping(0.1, epoch=0)
        assert not early_stopping(0.2, epoch=1)
        assert early_stopping.wait == 0
        assert early_stopping.best == 0.2

    def test_should_stop_after_patience_epochs_without_improvement(self):
        early_stopping = EarlyStopping(patience=2)
        assert not early_stopping(0.5, epoch=0)
        assert not early_stopping(0.4, epoch=1)
        assert early_stopping(0.3, epoch=2)
        assert early_stopping.stopped_epoch == 2
        assert early_stopping.best == 0.5

    def test_should_expose_its_state_as_meta(self):
        early_stopping = EarlyStopping(patience=2)
        early_stopping(0.5, epoch=0)
        early_stopping(0.4, epoch=1)
        meta = early_stopping.get_meta()[MetaKeys.EARLY_STOPPING]
        assert meta == {
            MetaKeys.WAIT: 1, MetaKeys.STOPPED_EPOCH: 0, MetaKeys.BEST: 0.5
        }

    def test_should_restore_its_state_from_meta(self):
        original = EarlyStopping(patience=3)
        original(0.5, epoch=0)
        original(0.4, epoch=1)
        restored = EarlyStopping(patience=3, initial_meta=original.get_meta())
        assert restored.wait == 1
        assert restored.best == 0.5

    def test_should_carry_the_patience_count_across_a_resume(self):
        original = EarlyStopping(patience=2)
        original(0.5, epoch=0)
        original(0.4, epoch=1)
        restored = EarlyStopping(patience=2, initial_meta=original.get_meta())
        # one more epoch without improvement is enough, since the wait was resumed
        assert restored(0.3, epoch=2)


class TestTrainer:
    def test_should_reduce_the_loss_over_epochs(self):
        trainer = Trainer(_model(), _training_config(max_epoch=5))
        history = trainer.train(_batches())
        losses = [history[f'epoch_{epoch}_loss'] for epoch in range(5)]
        assert losses[-1] < losses[0]

    def test_should_decay_the_learning_rate_each_epoch(self):
        trainer = Trainer(_model(), _training_config(max_epoch=2, lr_decay=0.5))
        trainer.train(_batches(count=1))
        assert trainer.optimizer.param_groups[0]['lr'] == pytest.approx(0.05 * 0.25)

    def test_should_save_a_checkpoint_every_epoch_by_default(self):
        save_checkpoint = MagicMock(name='save_checkpoint')
        trainer = Trainer(
            _model(), _training_config(max_epoch=3), save_checkpoint=save_checkpoint
        )
        trainer.train(_batches(count=1))
        assert save_checkpoint.call_count == 3
        assert [
            call.kwargs['epoch'] for call in save_checkpoint.call_args_list
        ] == [0, 1, 2]

    def test_should_honour_the_checkpoint_epoch_interval(self):
        save_checkpoint = MagicMock(name='save_checkpoint')
        trainer = Trainer(
            _model(),
            _training_config(max_epoch=4, checkpoint_epoch_interval=2),
            save_checkpoint=save_checkpoint
        )
        trainer.train(_batches(count=1))
        assert [
            call.kwargs['epoch'] for call in save_checkpoint.call_args_list
        ] == [1, 3]

    def test_should_include_the_early_stopping_state_in_checkpoint_meta(self):
        save_checkpoint = MagicMock(name='save_checkpoint')
        trainer = Trainer(
            _model(),
            _training_config(max_epoch=1, early_stop=True),
            save_checkpoint=save_checkpoint,
            scorer=lambda model: 0.5
        )
        trainer.train(_batches(count=1))
        meta = save_checkpoint.call_args.kwargs['meta']
        assert meta['epoch'] == 0
        assert meta[MetaKeys.EARLY_STOPPING][MetaKeys.BEST] == 0.5

    def test_should_stop_early_when_the_score_stops_improving(self):
        scores = iter([0.5, 0.4, 0.3, 0.2, 0.1])
        trainer = Trainer(
            _model(),
            _training_config(max_epoch=5, early_stop=True, patience=2),
            scorer=lambda model: next(scores)
        )
        history = trainer.train(_batches(count=1))
        assert len(history) == 3

    def test_should_not_stop_early_when_early_stop_is_disabled(self):
        scores = iter([0.5, 0.4, 0.3, 0.2])
        trainer = Trainer(
            _model(),
            _training_config(max_epoch=4, early_stop=False, patience=1),
            scorer=lambda model: next(scores)
        )
        assert len(trainer.train(_batches(count=1))) == 4

    def test_should_resume_from_the_initial_epoch(self):
        trainer = Trainer(
            _model(), _training_config(max_epoch=4, initial_epoch=2)
        )
        history = trainer.train(_batches(count=1))
        assert sorted(history) == ['epoch_2_loss', 'epoch_3_loss']

    def test_should_resume_the_early_stopping_state(self):
        exhausted = EarlyStopping(patience=2)
        exhausted(0.5, epoch=0)
        exhausted(0.4, epoch=1)
        trainer = Trainer(
            _model(),
            _training_config(
                max_epoch=4, initial_epoch=2, early_stop=True, patience=2,
                initial_meta=exhausted.get_meta()
            ),
            scorer=lambda model: 0.3
        )
        # the resumed wait count means the first epoch without improvement stops it
        assert len(trainer.train(_batches(count=1))) == 1


class TestSetRandomSeed:
    def test_should_make_a_training_step_reproducible(self):
        losses = []
        for _ in range(2):
            set_random_seed(42)
            trainer = Trainer(_model(), _training_config(max_epoch=1))
            losses.append(trainer.train(_batches(count=1))['epoch_0_loss'])
        assert losses[0] == losses[1]

    def test_should_produce_different_results_for_different_seeds(self):
        losses = []
        for seed in [1, 2]:
            set_random_seed(seed)
            trainer = Trainer(_model(), _training_config(max_epoch=1))
            losses.append(trainer.train(_batches(count=1))['epoch_0_loss'])
        assert losses[0] != losses[1]


class TestOptionalDependencies:
    def test_should_train_without_a_scorer_or_checkpoints(self):
        trainer = Trainer(_model(), _training_config(max_epoch=1))
        assert trainer.train(_batches(count=1))

    def test_should_not_require_labels_to_be_one_hot(self):
        model = _model()
        inputs, labels = _batches(count=1)[0]
        assert labels.dtype == torch.long
        assert np.isfinite(model(inputs, labels)['loss'].item())


def test_should_expose_no_optional_typing_leaks():
    optional_meta: Optional[dict] = None
    assert EarlyStopping(patience=1, initial_meta=optional_meta).best is None
