import logging
from typing import Optional

import tf_keras.backend as K  # type: ignore[import-untyped]

from sciencebeam_trainer_delft.utils.keras.callbacks import ModelSaverCallback
from sciencebeam_trainer_delft.sequence_labelling.saving import ModelSaver


LOGGER = logging.getLogger(__name__)


class ModelWithMetadataCheckpoint(ModelSaverCallback):
    """Similar to ModelCheckpoint but saves model metadata such as the config.
    """
    def __init__(
        self, base_path: str,
        model_saver: ModelSaver,
        add_checkpoint_meta: bool = True,
        meta: Optional[dict] = None,
        **kwargs
    ):
        self.base_path = base_path
        self.model_saver = model_saver
        self.add_checkpoint_meta = add_checkpoint_meta
        self.meta = meta
        super().__init__(save_fn=self._save_model, **kwargs)

    def _get_meta(self, epoch: int, logs: dict) -> dict:
        optimizer = self.model.optimizer
        optimizer_type = type(optimizer)
        optimizer_fullname = '%s.%s' % (optimizer_type.__module__, optimizer_type.__name__)
        return {
            **logs,
            **(self.meta or {}),
            'epoch': 1 + epoch,
            'optimizer': {
                'type': optimizer_fullname,
                'lr': float(K.get_value(optimizer.lr))
            }
        }

    def _save_model(self, epoch: int, logs: dict, **_):
        meta = self._get_meta(epoch=epoch, logs=logs)
        LOGGER.info('meta: %s', meta)
        base_path = self.base_path.format(epoch=epoch + 1, **logs)
        self.model_saver.save_to(base_path, model=self.model, meta=meta)
        if self.add_checkpoint_meta:
            self.model_saver.add_checkpoint_meta(
                base_path, epoch=epoch
            )
