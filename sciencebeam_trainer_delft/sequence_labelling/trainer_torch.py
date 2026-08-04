"""Training loop for the PyTorch models.

Upstream's trainer keeps only the best model by F1 and has no per-epoch
checkpoint interval, no resume and no meta, so the lifecycle this repo exposes
is implemented here rather than adapted.

Early stopping keeps its state under the same `early_stopping` meta key, with
the same `wait`, `stopped_epoch` and `best` fields, so checkpoint metadata
written before and after the migration stays readable by `--auto-resume`.
"""
import logging
from typing import Callable, Dict, Optional, Protocol

import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR

from sciencebeam_trainer_delft.sequence_labelling.config import TrainingConfig


LOGGER = logging.getLogger(__name__)


class MetaKeys:
    EARLY_STOPPING = 'early_stopping'
    WAIT = 'wait'
    STOPPED_EPOCH = 'stopped_epoch'
    BEST = 'best'


class SaveCheckpointCallable(Protocol):
    def __call__(self, epoch: int, meta: dict) -> None:
        pass


class ScorerCallable(Protocol):
    def __call__(self, model: nn.Module) -> float:
        pass


class EarlyStopping:
    """Stops training when the monitored score has not improved for `patience` epochs.

    A higher score is better. The state is restored from checkpoint meta so that
    a resumed run does not start its patience count again.
    """

    def __init__(
        self,
        patience: int = 5,
        initial_meta: Optional[dict] = None
    ):
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.best: Optional[float] = None
        self.restore_state(initial_meta)

    def restore_state(self, initial_meta: Optional[dict]):
        early_stopping_meta = (initial_meta or {}).get(MetaKeys.EARLY_STOPPING)
        if not early_stopping_meta:
            return
        self.wait = early_stopping_meta.get(MetaKeys.WAIT, 0)
        self.stopped_epoch = early_stopping_meta.get(MetaKeys.STOPPED_EPOCH, 0)
        self.best = early_stopping_meta.get(MetaKeys.BEST)
        LOGGER.info(
            'restored early stopping state: wait=%s, stopped_epoch=%s, best=%s',
            self.wait, self.stopped_epoch, self.best
        )

    def get_meta(self) -> dict:
        return {
            MetaKeys.EARLY_STOPPING: {
                MetaKeys.WAIT: self.wait,
                MetaKeys.STOPPED_EPOCH: self.stopped_epoch,
                MetaKeys.BEST: self.best
            }
        }

    def is_improvement(self, score: float) -> bool:
        return self.best is None or score > self.best

    def __call__(self, score: float, epoch: int) -> bool:
        """Records the score for an epoch and returns whether to stop."""
        if self.is_improvement(score):
            self.best = score
            self.wait = 0
            return False
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            LOGGER.info('early stopping at epoch %d (best=%s)', epoch, self.best)
            return True
        return False


class Trainer:
    """Trains a model for the configured number of epochs.

    `save_checkpoint` is called every `checkpoint_epoch_interval` epochs and
    receives the meta to store alongside the weights, which is what makes a
    resume possible.
    """

    def __init__(
        self,
        model: nn.Module,
        training_config: TrainingConfig,
        save_checkpoint: Optional[SaveCheckpointCallable] = None,
        scorer: Optional[ScorerCallable] = None,
        device: Optional[str] = None
    ):
        self.model = model
        self.training_config = training_config
        self.save_checkpoint = save_checkpoint
        self.scorer = scorer
        self.device = device
        self.optimizer: Optimizer = Adam(
            model.parameters(), lr=training_config.learning_rate
        )
        # matches the exponential decay the Keras path applied per epoch
        self.scheduler = ExponentialLR(self.optimizer, gamma=training_config.lr_decay)
        self.early_stopping = EarlyStopping(
            patience=training_config.patience,
            initial_meta=training_config.initial_meta
        )

    def get_meta(self, epoch: int) -> dict:
        return {
            'epoch': epoch,
            **self.early_stopping.get_meta()
        }

    def should_save_checkpoint(self, epoch: int) -> bool:
        if self.save_checkpoint is None:
            return False
        interval = self.training_config.checkpoint_epoch_interval or 1
        return (epoch + 1) % interval == 0

    def train_epoch(self, data_loader) -> float:
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        for inputs, labels in data_loader:
            self.optimizer.zero_grad()
            loss = self.model(inputs, labels)['loss']
            loss.backward()
            if self.training_config.clip_gradients:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.training_config.clip_gradients
                )
            self.optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        return total_loss / max(batch_count, 1)

    def train(self, data_loader) -> Dict[str, float]:
        initial_epoch = self.training_config.initial_epoch or 0
        max_epoch = self.training_config.max_epoch
        history: Dict[str, float] = {}
        for epoch in range(initial_epoch, max_epoch):
            loss = self.train_epoch(data_loader)
            self.scheduler.step()
            history[f'epoch_{epoch}_loss'] = loss
            score = self.scorer(self.model) if self.scorer is not None else None
            LOGGER.info('epoch %d: loss=%.4f score=%s', epoch, loss, score)
            # record the score before checkpointing, so that the meta a resume
            # reads includes this epoch rather than the state before it
            should_stop = bool(
                score is not None
                and self.training_config.early_stop
                and self.early_stopping(score, epoch)
            )
            if self.should_save_checkpoint(epoch):
                assert self.save_checkpoint is not None
                self.save_checkpoint(epoch=epoch, meta=self.get_meta(epoch))
            if should_stop:
                break
        return history


def set_random_seed(random_seed: int):
    """Seeds torch so that a run with the same seed on the same device matches."""
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def get_scorer_from_evaluation(
    evaluate: Callable[[nn.Module], float]
) -> ScorerCallable:
    def scorer(model: nn.Module) -> float:
        model.eval()
        with torch.no_grad():
            return evaluate(model)
    return scorer
