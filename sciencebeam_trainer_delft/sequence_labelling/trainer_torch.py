"""Training loop for the PyTorch models.

Upstream's trainer keeps only the best model by F1 and has no per-epoch
checkpoint interval, no resume and no meta, so the lifecycle this repo exposes
is implemented here rather than adapted.

Early stopping keeps its state under the same `early_stopping` meta key, with
the same `wait`, `stopped_epoch` and `best` fields, so checkpoint metadata
written before and after the migration stays readable by `--auto-resume`.
"""
import logging
import os
from typing import Callable, Dict, Optional, Protocol

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR

from delft.sequenceLabelling.preprocess import Preprocessor

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig, TrainingConfig
from sciencebeam_trainer_delft.sequence_labelling.data_generator import DataGenerator
from sciencebeam_trainer_delft.sequence_labelling.data_loader_torch import DataLoader
from sciencebeam_trainer_delft.sequence_labelling.evaluation import get_f1_scorer
from sciencebeam_trainer_delft.sequence_labelling.saving import ModelSaver
from sciencebeam_trainer_delft.sequence_labelling.typing import (
    T_Batch_Features_Array,
    T_Batch_Label_Array,
    T_Batch_Token_Array
)
from sciencebeam_trainer_delft.utils.numpy import concatenate_or_none


LOGGER = logging.getLogger(__name__)


CHECKPOINT_DIRECTORY_NAME_FORMAT = 'epoch-{epoch:05d}'


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

    def get_optimizer_meta(self) -> dict:
        optimizer_type = type(self.optimizer)
        return {
            'type': '%s.%s' % (optimizer_type.__module__, optimizer_type.__name__),
            'lr': float(self.optimizer.param_groups[0]['lr'])
        }

    def get_meta(self, epoch: int) -> dict:
        return {
            'epoch': epoch,
            'optimizer': self.get_optimizer_meta(),
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


class ModelTrainer:
    """Trains a model from the untransformed training data.

    Builds the data generators the model config asks for, scores each epoch
    against the validation set, and writes the checkpoints `--auto-resume`
    later reads.
    """

    def __init__(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        preprocessor: Preprocessor,
        embeddings=None,
        model_saver: Optional[ModelSaver] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.model = model
        self.model_config = model_config
        self.training_config = training_config
        self.preprocessor = preprocessor
        self.embeddings = embeddings
        self.model_saver = model_saver
        self.checkpoint_path = checkpoint_path
        self.device = device

    def get_meta(self) -> dict:
        training_config_meta = vars(self.training_config).copy()
        training_config_meta.pop('initial_meta', None)
        return {'training_config': training_config_meta}

    def create_data_generator(
        self, *args, name_suffix: str, **kwargs
    ) -> DataGenerator:
        return DataGenerator(  # type: ignore
            *args,
            batch_size=self.training_config.batch_size,
            input_window_stride=self.training_config.input_window_stride,
            stateful=self.model_config.stateful,
            preprocessor=self.preprocessor,
            additional_token_feature_indices=self.model_config.additional_token_feature_indices,
            text_feature_indices=self.model_config.text_feature_indices,
            concatenated_embeddings_token_count=(
                self.model_config.concatenated_embeddings_token_count
            ),
            char_embed_size=self.model_config.char_embedding_size,
            use_chain_crf=self.model_config.use_chain_crf,
            is_deprecated_padded_batch_text_list_enabled=(
                self.model_config.is_deprecated_padded_batch_text_list_enabled
            ),
            max_sequence_length=self.model_config.max_sequence_length,
            embeddings=self.embeddings,
            name='%s.%s' % (self.model_config.model_name, name_suffix),
            **kwargs
        )

    def create_data_loader(self, *args, name_suffix: str, **kwargs) -> DataLoader:
        return DataLoader(
            self.create_data_generator(*args, name_suffix=name_suffix, **kwargs),
            device=self.device
        )

    def get_save_checkpoint(self) -> Optional[SaveCheckpointCallable]:
        if not self.checkpoint_path or self.model_saver is None:
            return None
        base_meta = self.get_meta()

        def save_checkpoint(epoch: int, meta: dict):
            assert self.model_saver is not None
            # the epoch is one-based on disk, which is the epoch a resume
            # continues from
            directory = os.path.join(
                str(self.checkpoint_path),
                CHECKPOINT_DIRECTORY_NAME_FORMAT.format(epoch=1 + epoch)
            )
            self.model_saver.save_to(
                directory,
                model=self.model,
                meta={**base_meta, **meta, 'epoch': 1 + epoch}
            )
            self.model_saver.add_checkpoint_meta(directory, epoch=epoch)

        return save_checkpoint

    def get_scorer(self, validation_data_loader: DataLoader) -> ScorerCallable:
        return get_scorer_from_evaluation(
            get_f1_scorer(validation_data_loader, self.preprocessor)
        )

    def train(
        self,
        x_train: T_Batch_Token_Array,
        y_train: T_Batch_Label_Array,
        x_valid: Optional[T_Batch_Token_Array] = None,
        y_valid: Optional[T_Batch_Label_Array] = None,
        features_train: Optional[T_Batch_Features_Array] = None,
        features_valid: Optional[T_Batch_Features_Array] = None
    ) -> Dict[str, float]:
        if self.preprocessor.return_features and features_train is None:
            raise ValueError('features required')
        scorer: Optional[ScorerCallable] = None
        if self.training_config.early_stop:
            if x_valid is None or y_valid is None:
                raise ValueError('validation data required for early stopping')
            training_data_loader = self.create_data_loader(
                x_train, y_train,
                shuffle=True,
                features=features_train,
                name_suffix='training_generator'
            )
            scorer = self.get_scorer(self.create_data_loader(
                x_valid, y_valid,
                shuffle=False,
                features=features_valid,
                name_suffix='validation_generator'
            ))
        else:
            # without a score to stop on, the validation data is trained on too
            if x_valid is not None and y_valid is not None:
                x_train = np.concatenate((x_train, x_valid), axis=0)
                y_train = np.concatenate((y_train, y_valid), axis=0)
                if features_valid is not None:
                    features_train = concatenate_or_none(
                        (features_train, features_valid), axis=0
                    )
            training_data_loader = self.create_data_loader(
                x_train, y_train,
                shuffle=True,
                features=features_train,
                name_suffix='training_generator'
            )
        trainer = Trainer(
            self.model,
            self.training_config,
            save_checkpoint=self.get_save_checkpoint(),
            scorer=scorer,
            device=self.device
        )
        return trainer.train(training_data_loader)
