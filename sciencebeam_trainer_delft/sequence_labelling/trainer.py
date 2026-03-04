import logging
import os
from typing import List, NamedTuple, Optional

import numpy as np

import tensorflow as tf
from keras.callbacks import ProgbarLogger

from delft.sequenceLabelling.preprocess import (
    Preprocessor as DelftWordPreprocessor
)
from delft.sequenceLabelling.evaluation import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)

from delft.sequenceLabelling.trainer import Trainer as _Trainer
from delft.sequenceLabelling.trainer import Scorer as _Scorer
from delft.sequenceLabelling.models import BaseModel

from sciencebeam_trainer_delft.sequence_labelling.typing import (
    T_Batch_Label_List,
    T_Batch_Token_Array,
    T_Batch_Features_Array,
    T_Batch_Label_Array,
    T_Document_Label_List
)
from sciencebeam_trainer_delft.utils.keras.callbacks import ResumableEarlyStopping

from sciencebeam_trainer_delft.sequence_labelling.evaluation import classification_report
from sciencebeam_trainer_delft.sequence_labelling.config import TrainingConfig
from sciencebeam_trainer_delft.sequence_labelling.data_generator import DataGenerator
from sciencebeam_trainer_delft.sequence_labelling.callbacks import ModelWithMetadataCheckpoint
from sciencebeam_trainer_delft.sequence_labelling.saving import ModelSaver


LOGGER = logging.getLogger(__name__)


class SafeProgbarLogger(ProgbarLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ensure early_stopping is treated as stateful
        self.stateful_metrics = set(getattr(self, 'stateful_metrics', []) or [])
        self.stateful_metrics.add('early_stopping')

    def on_train_batch_end(self, batch, logs=None):
        logs = dict(logs or {})
        if 'early_stopping' in logs and isinstance(logs['early_stopping'], dict):
            # prevent dict from going into internal averaging
            logs.pop('early_stopping', None)
        super().on_train_batch_end(batch, logs=logs)


def get_callbacks(
    model_saver: ModelSaver,
    use_crf: bool,  # only required for Scorer (valid passed in)
    use_chain_crf: bool,  # only required for Scorer (valid passed in)
    log_dir: Optional[str] = None,
    log_period: int = 1,
    valid: tuple = (),
    early_stopping: bool = True,
    early_stopping_patience: int = 5,
    initial_meta: Optional[dict] = None,
    meta: Optional[dict] = None
):
    """
    Get callbacks.

    Args:
        log_dir (str): the destination to save logs
        valid (tuple): data for validation.
        early_stopping (bool): whether to use early stopping.

    Returns:
        list: list of callbacks
    """
    callbacks = []

    if valid:
        callbacks.append(Scorer(  # pylint: disable=no-value-for-parameter
            *valid,
            use_crf=use_crf,
            use_chain_crf=use_chain_crf
        ))  # pylint: disable=no-value-for-parameter

    if early_stopping:
        # Note: ensure we are not restoring weights
        #   as that would affect saving the model.
        #   The saving checkpoint need to be last,
        #   in order to save the state meta data of this checkpoint.
        callbacks.append(ResumableEarlyStopping(
            initial_meta=initial_meta,
            monitor='f1',
            patience=early_stopping_patience,
            mode='max',
            restore_best_weights=False
        ))

    if log_dir:
        epoch_dirname = 'epoch-{epoch:05d}'
        assert model_saver
        save_callback = ModelWithMetadataCheckpoint(
            os.path.join(log_dir, epoch_dirname),
            period=log_period,
            model_saver=model_saver,
            monitor='f1',
            meta=meta
        )
        callbacks.append(save_callback)

    callbacks.append(SafeProgbarLogger())

    return callbacks


class PredictedResults(NamedTuple):
    y_pred: T_Batch_Label_List
    y_true: T_Batch_Label_List


def get_model_results(
    model,
    valid_batches: list,
    preprocessor: DelftWordPreprocessor,
    *,
    use_crf: bool = False,
    use_chain_crf: bool = False
) -> PredictedResults:
    """
    Compute y_pred / y_true in label form (strings) for evaluation.

    - Non‑CRF or ChainCRF (use_crf=False or use_chain_crf=True):
        * labels and preds are dense/one‑hot → argmax over last axis.
    - TFA CRF wrapper (use_crf=True and use_chain_crf=False):
        * labels and preds are sparse integer indices → no argmax.
    """
    y_pred: List[T_Document_Label_List] = []
    y_true: List[T_Document_Label_List] = []
    valid_steps = len(valid_batches)

    for i, (data, label) in enumerate(valid_batches):
        if i == valid_steps:
            break

        sequence_lengths = np.reshape(data[-1], (-1,))  # shape (batch_size,)

        # ----- labels -----
        y_true_batch = np.asarray(label)

        if not use_crf or use_chain_crf:
            # non‑CRF or ChainCRF: one‑hot / logits → argmax to get indices
            if y_true_batch.ndim >= 2:
                y_true_batch = np.argmax(y_true_batch, axis=-1)
        # else: TFA CRF: y_true_batch already integer indices

        # normalise to list of 1D sequences
        if y_true_batch.ndim == 2:
            y_true_seqs = list(y_true_batch)
        elif y_true_batch.ndim == 1:
            y_true_seqs = [np.atleast_1d(y) for y in y_true_batch]
        else:
            y_true_seqs = [np.atleast_1d(y_true_batch[0])]

        # ----- predictions -----
        y_pred_batch = np.asarray(model.predict_on_batch(data))

        if not use_crf or use_chain_crf:
            # non‑CRF or ChainCRF: model outputs logits / one‑hot
            if y_pred_batch.ndim >= 2:
                y_pred_batch = np.argmax(y_pred_batch, axis=-1)
        # else: TFA CRF: y_pred_batch already integer indices (decoded sequence)

        if y_pred_batch.ndim == 2:
            y_pred_seqs = list(y_pred_batch)
        elif y_pred_batch.ndim == 1:
            y_pred_seqs = [np.atleast_1d(y) for y in y_pred_batch]
        else:
            y_pred_seqs = [np.atleast_1d(y_pred_batch[0])]

        # ----- map indices -> tag strings with sequence length -----
        y_pred_batch_list: List[T_Document_Label_List] = [
            preprocessor.inverse_transform(y[:l])
            for y, l in zip(y_pred_seqs, sequence_lengths)
        ]
        y_true_batch_list: List[T_Document_Label_List] = [
            preprocessor.inverse_transform(y[:l])
            for y, l in zip(y_true_seqs, sequence_lengths)
        ]

        if i == 0:
            y_pred = y_pred_batch_list
            y_true = y_true_batch_list
        else:
            y_pred += y_pred_batch_list
            y_true += y_true_batch_list

    return PredictedResults(y_pred=y_pred, y_true=y_true)


class Scorer(_Scorer):
    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        prediction_results = get_model_results(
            model=self.model,
            valid_batches=self.valid_batches,
            preprocessor=self.p,
            use_crf=self.use_crf,
            use_chain_crf=self.use_chain_crf
        )
        y_pred = prediction_results.y_pred
        y_true = prediction_results.y_true

        f1 = f1_score(y_true, y_pred)
        print("\tf1 (micro): {:04.2f}".format(f1 * 100))

        if self.evaluation:
            self.accuracy = accuracy_score(y_true, y_pred)
            self.precision = precision_score(y_true, y_pred)
            self.recall = recall_score(y_true, y_pred)
            self.report = classification_report(y_true, y_pred, digits=4)
            print(self.report)

        # save eval
        if logs:
            logs['f1'] = f1
        self.f1 = f1


class Trainer(_Trainer):
    def __init__(
            self,
            *args,
            model_saver: ModelSaver,
            training_config: TrainingConfig,
            multiprocessing: bool = True,
            **kwargs):
        self.model_saver = model_saver
        self.multiprocessing = multiprocessing
        self.model: Optional[BaseModel] = None
        super().__init__(*args, training_config=training_config, **kwargs)

    def train(  # pylint: disable=arguments-differ
        self,
        x_train,
        y_train,
        x_valid,
        y_valid,
        features_train: Optional[np.ndarray] = None,
        features_valid: Optional[np.ndarray] = None
    ):
        assert self.model is not None
        self.model.summary()

        LOGGER.debug('Training model with config: %s', vars(self.model_config))
        if self.model_config.use_crf and not self.model_config.use_chain_crf:
            LOGGER.info('Enabling eager execution for CRF training')
            # Note: this avoids "indicates an invalid graph that escaped type checking"
            tf.config.run_functions_eagerly(True)
        self.model = self.compile_model(self.model, len(x_train))

        self.model = self.train_model(
            self.model, x_train, y_train, x_valid, y_valid,
            self.training_config.max_epoch,
            features_train=features_train, features_valid=features_valid
        )

    def get_meta(self):
        training_config_meta = vars(self.training_config).copy()
        try:
            training_config_meta.pop('initial_meta')
        except KeyError:
            pass
        return {
            'training_config': training_config_meta
        }

    def create_data_generator(self, *args, name_suffix: str, **kwargs) -> DataGenerator:
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

    def train_model(  # pylint: disable=arguments-differ
        self,
        local_model,
        x_train: T_Batch_Token_Array,
        y_train: T_Batch_Label_Array,
        x_valid: Optional[T_Batch_Token_Array] = None,
        y_valid: Optional[T_Batch_Label_Array] = None,
        max_epoch: int = 50,
        features_train: Optional[T_Batch_Features_Array] = None,
        features_valid: Optional[T_Batch_Features_Array] = None
    ):
        """ parameter model local_model must be compiled before calling this method
            this model will be returned with trained weights """
        # todo: if valid set if None, create it as random segment of the shuffled train set

        if self.preprocessor.return_features and features_train is None:
            raise ValueError('features required')

        if self.training_config.early_stop:
            training_generator = self.create_data_generator(
                x_train, y_train,
                shuffle=True,
                features=features_train,
                name_suffix='training_generator'
            )

            validation_generator = self.create_data_generator(
                x_valid, y_valid,
                shuffle=False,
                features=features_valid,
                name_suffix='validation_generator'
            )

            callbacks = get_callbacks(
                model_saver=self.model_saver,
                use_crf=self.model_config.use_crf,
                use_chain_crf=self.model_config.use_chain_crf,
                log_dir=self.checkpoint_path,
                log_period=self.training_config.checkpoint_epoch_interval,
                early_stopping=True,
                early_stopping_patience=self.training_config.patience,
                initial_meta=self.training_config.initial_meta,
                valid=(validation_generator, self.preprocessor),
                meta=self.get_meta()
            )
        else:
            if x_valid is not None and y_valid is not None:
                x_train = np.concatenate((x_train, x_valid), axis=0)
                y_train = np.concatenate((y_train, y_valid), axis=0)
            features_all = None
            if features_train is not None:
                if features_valid is not None:
                    features_all = np.concatenate((features_train, features_valid), axis=0)
            training_generator = self.create_data_generator(
                x_train, y_train,
                shuffle=True,
                features=features_all,
                name_suffix='training_generator'
            )

            callbacks = get_callbacks(
                model_saver=self.model_saver,
                use_crf=self.model_config.use_crf,
                use_chain_crf=self.model_config.use_chain_crf,
                log_dir=self.checkpoint_path,
                early_stopping=False,
                meta=self.get_meta()
            )
        nb_workers = 6
        multiprocessing = self.multiprocessing

        local_model.fit_generator(
            generator=training_generator,
            initial_epoch=self.training_config.initial_epoch or 0,
            epochs=max_epoch,
            use_multiprocessing=multiprocessing,
            workers=nb_workers,
            callbacks=callbacks,
            verbose=0
        )

        return local_model

    def train_nfold(  # pylint: disable=arguments-differ
        self,
        x_train: T_Batch_Token_Array,
        y_train: T_Batch_Label_Array,
        x_valid: Optional[T_Batch_Token_Array] = None,
        y_valid: Optional[T_Batch_Label_Array] = None,
        features_train: Optional[T_Batch_Features_Array] = None,
        features_valid: Optional[T_Batch_Features_Array] = None
    ):
        """ n-fold training for the instance model
            the n models are stored in self.models, and self.model left unset at this stage """
        fold_count = len(self.models)
        fold_size = len(x_train) // fold_count

        train_x: T_Batch_Token_Array
        train_y: T_Batch_Label_Array
        train_features: Optional[T_Batch_Features_Array]
        val_y: Optional[T_Batch_Label_Array]
        val_features: Optional[T_Batch_Features_Array]

        for fold_id in range(0, fold_count):
            print(
                '\n------------------------ fold %s--------------------------------------'
                % fold_id
            )

            if x_valid is None:
                # segment train and valid
                fold_start = fold_size * fold_id
                fold_end = fold_start + fold_size

                if fold_id == fold_size - 1:
                    fold_end = len(x_train)

                train_x = np.concatenate([x_train[:fold_start], x_train[fold_end:]])
                train_y = np.concatenate([y_train[:fold_start], y_train[fold_end:]])

                val_x = x_train[fold_start:fold_end]
                val_y = y_train[fold_start:fold_end]

                if features_train is not None:
                    train_features = np.concatenate(
                        [features_train[:fold_start], features_train[fold_end:]]
                    )
                    val_features = features_train[fold_start:fold_end]
                else:
                    train_features = None
                    val_features = None
            else:
                # reuse given segmentation
                train_x = x_train
                train_y = y_train
                train_features = features_train

                val_x = x_valid
                val_y = y_valid
                val_features = features_valid

            foldModel = self.models[fold_id]
            foldModel.summary()
            if self.model_config.use_crf:
                foldModel.compile(
                    loss=foldModel.crf.loss,
                    optimizer='adam'
                )
            else:
                foldModel.compile(
                    loss='categorical_crossentropy',
                    optimizer='adam'
                )

            foldModel = self.train_model(
                foldModel,
                train_x,
                train_y,
                val_x,
                val_y,
                features_train=train_features,
                features_valid=val_features,
                max_epoch=self.training_config.max_epoch
            )
            self.models[fold_id] = foldModel
