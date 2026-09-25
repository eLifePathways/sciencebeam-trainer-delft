import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig, TrainingConfig
from sciencebeam_trainer_delft.sequence_labelling.models import CustomBidLSTM_CRF
from sciencebeam_trainer_delft.sequence_labelling.trainer_torch import (
    EarlyStopping,
    MetaKeys,
    Trainer,
    set_random_seed
)


TRAINER_LOGGER = 'sciencebeam_trainer_delft.sequence_labelling.trainer_torch'


NTAGS = 5
BATCH_SIZE = 2
SEQUENCE_LENGTH = 4
CHAR_VOCAB_SIZE = 12
MAX_CHAR_LENGTH = 5
MAX_FEATURE_SIZE = 7


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

    def test_should_decay_the_learning_rate_to_a_tenth_over_the_run(self):
        # what delft 0.4.3 did, and independent of max_epoch
        trainer = Trainer(_model(), _training_config(max_epoch=4))
        trainer.train(_batches(count=3))
        assert trainer.optimizer.param_groups[0]['lr'] == pytest.approx(0.05 * 0.1)

    def test_should_decay_the_learning_rate_per_step_not_per_epoch(self):
        trainer = Trainer(_model(), _training_config(max_epoch=2))
        trainer.train(_batches(count=2))
        # four steps in total, so halfway through is one epoch plus one step
        assert trainer.scheduler is not None
        assert trainer.scheduler.get_last_lr()[0] == pytest.approx(0.05 * 0.1)

    def test_should_not_use_lr_decay_for_the_schedule(self):
        # 0.4.3 hardcoded the decay rate; lr_decay never reached the schedule
        learning_rates = []
        for lr_decay in [0.5, 0.9]:
            trainer = Trainer(
                _model(), _training_config(max_epoch=2, lr_decay=lr_decay)
            )
            trainer.train(_batches(count=2))
            learning_rates.append(trainer.optimizer.param_groups[0]['lr'])
        assert learning_rates[0] == pytest.approx(learning_rates[1])

    def test_should_keep_the_learning_rate_usable_over_a_long_run(self):
        # decaying by lr_decay per epoch would reach 1e-14 here
        trainer = Trainer(_model(), _training_config(max_epoch=300))
        scheduler = trainer.create_scheduler(steps_per_epoch=10)
        for _ in range(300 * 10):
            scheduler.step()
        assert trainer.optimizer.param_groups[0]['lr'] == pytest.approx(0.05 * 0.1)

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


class _SlowLoader(list):
    """A loader whose batches take real time to arrive, as a real one's do."""

    def __iter__(self):
        for batch in list.__iter__(self):
            time.sleep(0.03)
            yield batch


def _timing_lines(caplog) -> List[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if 'timing:' in record.getMessage()
    ]


def _phase_seconds(line: str, phase: str) -> float:
    match = re.search(r'%s=([\d.]+)s' % phase, line)
    assert match, 'no %s in %r' % (phase, line)
    return float(match.group(1))


def _phase_cores(line: str, phase: str) -> Optional[float]:
    """Returns the phase's core count, or None where it is too short to report."""
    match = re.search(r'%s=[\d.]+s \(([\d.]+) cores\)' % phase, line)
    return float(match.group(1)) if match else None


class TestTrainerPhaseTiming:
    def test_should_charge_the_time_to_reach_a_batch_to_the_batch_phase(self):
        delay = 0.02
        batches = _batches()

        def _slow_batches():
            for batch in batches:
                time.sleep(delay)
                yield batch

        trainer = Trainer(_model(), _training_config())
        result = trainer.train_epoch(_slow_batches())
        assert result.batch.seconds >= len(batches) * delay
        assert result.step.seconds > 0

    def test_should_still_return_the_mean_loss(self):
        trainer = Trainer(_model(), _training_config())
        result = trainer.train_epoch(_batches())
        assert result.loss > 0

    def test_should_log_every_phase_of_every_epoch(self, caplog):
        trainer = Trainer(_model(), _training_config(max_epoch=2))
        with caplog.at_level(logging.INFO, logger=TRAINER_LOGGER):
            trainer.train(_batches())
        lines = _timing_lines(caplog)
        assert len(lines) == 2
        for line in lines:
            for phase in ('total', 'batch', 'step', 'evaluate', 'checkpoint'):
                _phase_seconds(line, phase)

    def test_should_charge_writing_a_checkpoint_to_the_checkpoint_phase(self, caplog):
        delay = 0.05
        save_checkpoint = MagicMock(name='save_checkpoint')
        save_checkpoint.side_effect = lambda **_: time.sleep(delay)
        trainer = Trainer(
            _model(), _training_config(max_epoch=1), save_checkpoint=save_checkpoint
        )
        with caplog.at_level(logging.INFO, logger=TRAINER_LOGGER):
            trainer.train(_batches())
        assert _phase_seconds(_timing_lines(caplog)[0], 'checkpoint') >= delay

    def test_should_report_no_checkpoint_time_when_none_is_written(self, caplog):
        trainer = Trainer(_model(), _training_config(max_epoch=1))
        with caplog.at_level(logging.INFO, logger=TRAINER_LOGGER):
            trainer.train(_batches())
        assert _phase_seconds(_timing_lines(caplog)[0], 'checkpoint') == 0

    def test_should_report_the_cores_of_each_phase_of_a_real_epoch(self, caplog):
        # the sleeps make the phases long enough to measure without depending
        # on how fast this machine trains a toy model
        trainer = Trainer(
            _model(), _training_config(max_epoch=1),
            scorer=lambda _: time.sleep(0.05) or 0.5
        )
        with caplog.at_level(logging.INFO, logger=TRAINER_LOGGER):
            trainer.train(_SlowLoader(_batches()))
        line = _timing_lines(caplog)[0]
        for phase in ('total', 'batch', 'evaluate'):
            cores = _phase_cores(line, phase)
            assert cores is not None, 'no cores for %s in %r' % (phase, line)
            assert 0 <= cores <= os.cpu_count()
        # nothing is asserted about one phase against another: the CPU side is
        # the whole process, so a phase that only waits is still credited with
        # whatever else is running, and torch's pools spin between operations

    def test_should_log_the_peak_memory_once_the_run_ends(self, caplog):
        trainer = Trainer(_model(), _training_config(max_epoch=2))
        with caplog.at_level(logging.INFO):
            trainer.train(_batches())
        peak_lines = [
            record.getMessage() for record in caplog.records
            if 'peak memory:' in record.getMessage()
        ]
        assert len(peak_lines) == 1

    def test_should_charge_scoring_to_the_evaluate_phase(self, caplog):
        delay = 0.05
        scorer = MagicMock(name='scorer')
        scorer.side_effect = lambda _: time.sleep(delay) or 0.5
        trainer = Trainer(_model(), _training_config(max_epoch=1), scorer=scorer)
        with caplog.at_level(logging.INFO, logger=TRAINER_LOGGER):
            trainer.train(_batches())
        assert _phase_seconds(_timing_lines(caplog)[0], 'evaluate') >= delay
