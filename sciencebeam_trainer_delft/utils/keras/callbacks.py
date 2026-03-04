import logging
import warnings
from typing import Any, Dict, Optional, Callable

from typing_extensions import Protocol

import numpy as np

from tf_keras.callbacks import Callback, EarlyStopping


LOGGER = logging.getLogger(__name__)


class ResumableEarlyStopping(EarlyStopping):
    class MetaKeys:
        EARLY_STOPPING = 'early_stopping'
        WAIT = 'wait'
        STOPPED_EPOCH = 'stopped_epoch'
        BEST = 'best'

    def __init__(
        self,
        initial_meta: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.best: Optional[float] = None
        self.initial_wait = 0
        self.initial_stopped_epoch = 0
        self.initial_best: Optional[float] = None
        self.restore_state(initial_meta)

    def restore_state(self, initial_meta: Optional[dict]):
        if not initial_meta:
            return
        early_stopping_meta = initial_meta.get(ResumableEarlyStopping.MetaKeys.EARLY_STOPPING)
        if not early_stopping_meta:
            return
        self.initial_wait = early_stopping_meta.get(
            ResumableEarlyStopping.MetaKeys.WAIT,
            0
        )
        self.initial_stopped_epoch = early_stopping_meta.get(
            ResumableEarlyStopping.MetaKeys.STOPPED_EPOCH,
            0
        )
        self.initial_best = early_stopping_meta.get(
            ResumableEarlyStopping.MetaKeys.BEST,
            None
        )
        LOGGER.info(
            (
                'restored early stopping state: initial_wait=%s, initial_stopped_epoch=%s'
                ', initial_best=%s'
            ),
            self.initial_wait, self.initial_stopped_epoch, self.initial_best
        )

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        self.wait = self.initial_wait
        self.stopped_epoch = self.stopped_epoch
        if self.initial_best is not None:
            self.best = self.initial_best

    def _get_early_stopping_meta(self):
        return {
            ResumableEarlyStopping.MetaKeys.WAIT: self.wait,
            ResumableEarlyStopping.MetaKeys.STOPPED_EPOCH: self.stopped_epoch,
            ResumableEarlyStopping.MetaKeys.BEST: self.best
        }

    def _add_early_stopping_meta_to_logs(self, logs: dict):
        logs[ResumableEarlyStopping.MetaKeys.EARLY_STOPPING] = (
            self._get_early_stopping_meta()
        )

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)
        self._add_early_stopping_meta_to_logs(logs)
        LOGGER.info('on_epoch_end: logs=%s', logs)


class SaveFunctionProtocol(Protocol):
    def __call__(self, epoch: int, logs: Dict[str, Any], **kwargs):
        pass


class ModelSaverCallback(Callback):
    """Similar to ModelCheckpoint but leaves the actual saving to the save_fn.
    """
    def __init__(
        self,
        save_fn: Optional[SaveFunctionProtocol],
        monitor: str = 'val_loss',
        mode: str = 'auto',
        period: int = 1,
        save_best_only: bool = False,
        save_kwargs: Optional[dict] = None
    ):
        super().__init__()
        self.monitor = monitor
        self.save_fn = save_fn
        self.period = period
        self.save_best_only = save_best_only
        self.save_kwargs = save_kwargs or {}
        self.epochs_since_last_save = 0

        # explicitly type the monitor comparison callable to avoid mypy
        # inferring a specific numpy ufunc literal
        self.monitor_op: Callable[[Any, Any], bool]

        if mode not in ['auto', 'min', 'max']:
            warnings.warn(
                'ModelCheckpoint mode %s is unknown, '
                'fallback to auto mode.' % (mode),
                RuntimeWarning
            )
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def _save(self, epoch, logs=None):
        self.save_fn(epoch=epoch, logs=logs, **self.save_kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        'Can save best model only with %s available, '
                        'skipping.' % (self.monitor), RuntimeWarning
                    )
                else:
                    if self.monitor_op(current, self.best):
                        LOGGER.info(
                            'Epoch %05d: %s improved from %0.5f to %0.5f',
                            epoch + 1, self.monitor, self.best, current
                        )
                        self.best = current
                        self._save(epoch=epoch, logs=logs)
                    else:
                        LOGGER.info(
                            'Epoch %05d: %s did not improve from %0.5f',
                            epoch + 1, self.monitor, self.best
                        )
            else:
                self._save(epoch=epoch, logs=logs)
