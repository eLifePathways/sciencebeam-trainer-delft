import logging
import math
import os
from typing import List, Optional

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from keras.models import Model
from keras.callbacks import Callback

from sciencebeam_trainer_delft.text_classification.saving import (
    ModelSaver
)

from sciencebeam_trainer_delft.text_classification.callbacks import (
    ModelWithMetadataCheckpoint
)


LOGGER = logging.getLogger(__name__)


def get_callbacks(
    model_saver: ModelSaver,
    log_dir: Optional[str] = None,
    meta: Optional[dict] = None
) -> List[Callback]:
    callbacks = []

    if log_dir:
        epoch_dirname = 'epoch-{epoch:05d}'
        assert model_saver
        save_callback = ModelWithMetadataCheckpoint(
            os.path.join(log_dir, epoch_dirname),
            model_saver=model_saver,
            monitor='f1',
            meta=meta
        )
        callbacks.append(save_callback)

    return callbacks


# mostly copied from:
# https://github.com/kermitt2/delft/blob/v0.2.3/delft/textClassification/models.py


def train_model(  # pylint: disable=too-many-statements
    model: Model,
    list_classes: List[str],
    batch_size: int,  # pylint: disable=unused-argument
    max_epoch: int,
    use_roc_auc: bool,
    class_weights,
    training_generator,
    validation_generator,
    val_y,
    multiprocessing: bool = True,
    nb_workers: int = 6,
    callbacks: Optional[List[Callback]] = None
):
    best_loss = -1.0
    best_roc_auc = -1.0
    best_weights = None
    best_epoch = 0
    current_epoch = 1

    while current_epoch <= max_epoch:
        model.fit_generator(
            generator=training_generator,
            use_multiprocessing=multiprocessing,
            workers=nb_workers,
            class_weight=class_weights,
            epochs=current_epoch,
            initial_epoch=(current_epoch - 1),
            callbacks=callbacks)

        y_pred = model.predict_generator(
            generator=validation_generator,
            use_multiprocessing=multiprocessing,
            workers=nb_workers)

        total_loss = 0.0
        total_roc_auc = 0.0

        # we distinguish 1-class and multiclass problems
        if len(list_classes) == 1:
            total_loss = log_loss(val_y, y_pred)
            total_roc_auc = roc_auc_score(val_y, y_pred)
        else:
            for j in range(0, len(list_classes)):
                labels = [0, 1]
                loss = log_loss(val_y[:, j], y_pred[:, j], labels=labels)
                total_loss += loss
                try:
                    roc_auc = roc_auc_score(val_y[:, j], y_pred[:, j])
                except ValueError as e:
                    LOGGER.debug('could not calculate roc (index=%d): %s', j, e)
                    roc_auc = np.nan
                total_roc_auc += roc_auc

        total_loss /= len(list_classes)
        total_roc_auc /= len(list_classes)
        if np.isnan(total_roc_auc):
            use_roc_auc = False
        if use_roc_auc:
            LOGGER.info(
                "Epoch %s loss %s best_loss %s (for info)",
                current_epoch, total_loss, best_loss
            )
            LOGGER.info(
                "Epoch %s roc_auc %s best_roc_auc %s (for early stop)",
                current_epoch, total_roc_auc, best_roc_auc
            )
        else:
            LOGGER.info(
                "Epoch %s loss %s best_loss %s (for early stop)",
                current_epoch, total_loss, best_loss
            )
            LOGGER.info(
                "Epoch %s roc_auc %s best_roc_auc %s (for info)",
                current_epoch, total_roc_auc, best_roc_auc
            )

        current_epoch += 1
        if total_loss < best_loss or best_loss == -1 or math.isnan(best_loss) is True:
            best_loss = total_loss
            if use_roc_auc is False:
                best_weights = model.get_weights()
                best_epoch = current_epoch
        elif use_roc_auc is False:
            if current_epoch - best_epoch == 5:
                break

        if total_roc_auc > best_roc_auc or best_roc_auc == -1:
            best_roc_auc = total_roc_auc
            if use_roc_auc:
                best_weights = model.get_weights()
                best_epoch = current_epoch
        elif use_roc_auc:
            if current_epoch - best_epoch == 5:
                break

    model.set_weights(best_weights)

    if use_roc_auc:
        return model, best_roc_auc
    else:
        return model, best_loss
