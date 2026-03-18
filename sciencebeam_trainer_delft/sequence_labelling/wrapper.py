import logging
import os
import time
from functools import partial
from typing import Callable, Iterable, List, Optional, Tuple, Union, cast
from typing import Sequence as TypingSequence

import numpy as np

from delft.sequenceLabelling.models import BaseModel
from delft.sequenceLabelling.preprocess import Preprocessor, FeaturesPreprocessor
from delft.sequenceLabelling.wrapper import Sequence as _Sequence
from delft.sequenceLabelling.config import TrainingConfig as DelftTrainingConfig

from sciencebeam_trainer_delft.resources.default_config import DEFAULT_RESOURCE_REGISTRY_FILE
from sciencebeam_trainer_delft.sequence_labelling.typing import (
    T_Batch_Features_Array,
    T_Batch_Label_Array,
    T_Batch_Token_Array
)
from sciencebeam_trainer_delft.utils.typing import T
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.utils.numpy import concatenate_or_none
from sciencebeam_trainer_delft.utils.misc import str_to_bool

from sciencebeam_trainer_delft.sequence_labelling.tools.install_models import (
    copy_directory_with_source_meta
)

from sciencebeam_trainer_delft.embedding import Embeddings, EmbeddingManager

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig, TrainingConfig
from sciencebeam_trainer_delft.sequence_labelling.data_generator import (
    DataGenerator,
    iter_batch_text_list,
    get_concatenated_embeddings_token_count
)
from sciencebeam_trainer_delft.sequence_labelling.trainer import (
    Scorer,
    Trainer
)
from sciencebeam_trainer_delft.sequence_labelling.models import (
    is_model_stateful,
    get_model,
    updated_implicit_model_config_props
)
from sciencebeam_trainer_delft.sequence_labelling.preprocess import (
    T_FeaturesPreprocessor,
    FeaturesPreprocessor as ScienceBeamFeaturesPreprocessor,
    faster_preprocessor_fit
)
from sciencebeam_trainer_delft.sequence_labelling.saving import ModelSaver, ModelLoader
from sciencebeam_trainer_delft.sequence_labelling.tagger import Tagger
from sciencebeam_trainer_delft.sequence_labelling.evaluation import ClassificationResult

from sciencebeam_trainer_delft.sequence_labelling.debug import get_tag_debug_reporter_if_enabled

from sciencebeam_trainer_delft.sequence_labelling.tools.checkpoints import (
    get_checkpoints_json,
    get_last_checkpoint_url
)
from sciencebeam_trainer_delft.sequence_labelling.transfer_learning import (
    TransferLearningConfig,
    TransferLearningSource,
    freeze_model_layers
)

from sciencebeam_trainer_delft.sequence_labelling.dataset_transform import (
    DummyDatasetTransformer
)

from sciencebeam_trainer_delft.sequence_labelling.dataset_transform.unroll_transform import (
    UnrollingTextFeatureDatasetTransformer
)


LOGGER = logging.getLogger(__name__)


DEFAUT_MODEL_PATH = 'data/models/sequenceLabelling/'


DEFAUT_BATCH_SIZE = 10


class EnvironmentVariables:
    # environment variables are mainly intended for GROBID, as we can't pass in arguments
    MAX_SEQUENCE_LENGTH = 'SCIENCEBEAM_DELFT_MAX_SEQUENCE_LENGTH'
    INPUT_WINDOW_STRIDE = 'SCIENCEBEAM_DELFT_INPUT_WINDOW_STRIDE'
    BATCH_SIZE = 'SCIENCEBEAM_DELFT_BATCH_SIZE'
    STATEFUL = 'SCIENCEBEAM_DELFT_STATEFUL'


def get_typed_env(
    key: str,
    type_fn: Callable[[str], T],
    default_value: Optional[T] = None
) -> Optional[T]:
    max_sequence_length_str = os.getenv(key)
    if not max_sequence_length_str:
        return default_value
    return type_fn(max_sequence_length_str)


def get_default_max_sequence_length() -> Optional[int]:
    return get_typed_env(EnvironmentVariables.MAX_SEQUENCE_LENGTH, int, default_value=None)


def get_default_input_window_stride() -> Optional[int]:
    return get_typed_env(EnvironmentVariables.INPUT_WINDOW_STRIDE, int, default_value=None)


def get_default_batch_size() -> Optional[int]:
    return get_typed_env(EnvironmentVariables.BATCH_SIZE, int, default_value=DEFAUT_BATCH_SIZE)


def get_default_stateful() -> Optional[bool]:
    return get_typed_env(
        EnvironmentVariables.STATEFUL,
        str_to_bool,
        default_value=None
    )


def get_features_preprocessor(
    model_config: ModelConfig,
    features: Optional[T_Batch_Features_Array] = None
) -> Optional[T_FeaturesPreprocessor]:
    if not model_config.use_features:
        LOGGER.info('features not enabled')
        return None
    if features is None:
        LOGGER.info('no features available')
        return None
    if model_config.use_features_indices_input:
        LOGGER.info(
            'using feature indices as input, features_indices=%s, features_vocab_size=%s',
            model_config.features_indices, model_config.features_vocabulary_size
        )
        return FeaturesPreprocessor(
            features_indices=model_config.features_indices,
            features_vocabulary_size=model_config.features_vocabulary_size
        )
    LOGGER.info(
        'using feature indices=%s', model_config.features_indices
    )
    return ScienceBeamFeaturesPreprocessor(
        features_indices=model_config.features_indices,
        continuous_features_indices=model_config.continuous_features_indices
    )


def get_preprocessor(
    model_config: ModelConfig,
    features: Optional[T_Batch_Features_Array] = None
) -> Preprocessor:
    feature_preprocessor = get_features_preprocessor(
        model_config,
        features=features
    )
    return Preprocessor(
        max_char_length=model_config.max_char_length,
        feature_preprocessor=feature_preprocessor
    )


def prepare_preprocessor(
    X, y,
    model_config: ModelConfig,
    features: Optional[T_Batch_Features_Array] = None
):
    preprocessor = get_preprocessor(model_config, features=features)
    batch_text_list_iterable = iter_batch_text_list(
        X, features,
        additional_token_feature_indices=model_config.additional_token_feature_indices,
        text_feature_indices=model_config.text_feature_indices
    )
    if isinstance(preprocessor, Preprocessor):
        LOGGER.info('fitting preprocessor (faster)')
        faster_preprocessor_fit(preprocessor, batch_text_list_iterable, y)
    else:
        LOGGER.info('fitting preprocessor (default)')
        preprocessor.fit(batch_text_list_iterable, y)
    if model_config.use_features and features is not None:
        LOGGER.info('fitting features preprocessor')
        preprocessor.fit_features(features)
        if model_config.features_indices != preprocessor.feature_preprocessor.features_indices:
            LOGGER.info('revised features_indices: %s', model_config.features_indices)
            model_config.features_indices = preprocessor.feature_preprocessor.features_indices
        model_config.features_map_to_index = preprocessor.feature_preprocessor.features_map_to_index
    LOGGER.info('done fitting preprocessor')
    return preprocessor


def get_model_directory(model_name: str, dir_path: Optional[str] = None):
    return os.path.join(dir_path or DEFAUT_MODEL_PATH, model_name)


class Sequence(_Sequence):
    def __init__(
        self,
        *args,
        use_features: bool = False,
        features_indices: Optional[List[int]] = None,
        features_embedding_size: Optional[int] = None,
        multiprocessing: bool = False,
        embedding_registry_path: Optional[str] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        config_props: Optional[dict] = None,
        training_props: Optional[dict] = None,
        max_sequence_length: Optional[int] = None,
        input_window_stride: Optional[int] = None,
        eval_max_sequence_length: Optional[int] = None,
        eval_input_window_stride: Optional[int] = None,
        batch_size: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        stateful: Optional[bool] = None,
        transfer_learning_config: Optional[TransferLearningConfig] = None,
        tag_transformed: bool = False,
        **kwargs
    ):
        # initialise logging if not already initialised
        logging.basicConfig(level='INFO')
        LOGGER.debug('Sequence, args=%s, kwargs=%s', args, kwargs)
        self.embedding_registry_path = (
            embedding_registry_path
            or (embedding_manager.path if embedding_manager else None)
            or DEFAULT_RESOURCE_REGISTRY_FILE
        )
        if embedding_manager is None:
            embedding_manager = EmbeddingManager(
                path=self.embedding_registry_path,
                download_manager=DownloadManager()
            )
        self.download_manager = embedding_manager.download_manager
        self.embedding_manager = embedding_manager
        self.embeddings: Optional[Embeddings] = None
        if not batch_size:
            batch_size = get_default_batch_size()
        if not max_sequence_length:
            max_sequence_length = get_default_max_sequence_length()
        self.max_sequence_length = max_sequence_length
        if not input_window_stride:
            input_window_stride = get_default_input_window_stride()
        self.input_window_stride = input_window_stride
        self.eval_max_sequence_length = eval_max_sequence_length
        self.eval_input_window_stride = eval_input_window_stride
        self.eval_batch_size = eval_batch_size
        self.model_path: Optional[str] = None
        if stateful is None:
            # use a stateful model, if supported
            stateful = get_default_stateful()
        self.stateful = stateful
        self.transfer_learning_config = transfer_learning_config
        self.dataset_transformer_factory = DummyDatasetTransformer
        self.tag_transformed = tag_transformed
        super().__init__(
            *args,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            resource_registry_path=self.embedding_registry_path,
            **kwargs
        )
        LOGGER.debug('use_features=%s', use_features)
        self.model_config: ModelConfig = ModelConfig(
            **{  # type: ignore
                **vars(self.model_config),
                **(config_props or {}),
                'features_indices': features_indices,
                'features_embedding_size': features_embedding_size
            },
            use_features=use_features
        )
        self.update_model_config_word_embedding_size()
        updated_implicit_model_config_props(self.model_config)
        self.update_dataset_transformer_factor()
        self.training_config: TrainingConfig = TrainingConfig(
            **vars(cast(DelftTrainingConfig, self.training_config)),
            **(training_props or {})
        )
        LOGGER.info('training_config: %s', vars(self.training_config))
        self.multiprocessing = multiprocessing
        self.tag_debug_reporter = get_tag_debug_reporter_if_enabled()
        self._load_exception: Optional[Exception] = None
        self.p: Optional[Preprocessor] = None
        self.model: Optional[BaseModel] = None
        self.models: List[BaseModel] = []

    def update_model_config_word_embedding_size(self):
        if self.embeddings:
            token_count = get_concatenated_embeddings_token_count(
                concatenated_embeddings_token_count=(
                    self.model_config.concatenated_embeddings_token_count
                ),
                additional_token_feature_indices=(
                    self.model_config.additional_token_feature_indices
                )
            )
            self.model_config.word_embedding_size = (
                self.embeddings.embed_size * token_count
            )

    def update_dataset_transformer_factor(self):
        self.dataset_transformer_factory = DummyDatasetTransformer
        if self.model_config.unroll_text_feature_index is not None:
            LOGGER.info(
                'using unrolling text feature dataset transformer, index=%s',
                self.model_config.unroll_text_feature_index
            )
            self.dataset_transformer_factory = partial(
                UnrollingTextFeatureDatasetTransformer,
                self.model_config.unroll_text_feature_index,
                used_features_indices=self.model_config.features_indices
            )

    def train(  # pylint: disable=arguments-differ
        self,
        x_train,
        y_train,
        x_valid=None,
        y_valid=None,
        features_train: Optional[np.ndarray] = None,
        features_valid: Optional[np.ndarray] = None
    ):
        # TBD if valid is None, segment train to get one
        dataset_fransformer = self.dataset_transformer_factory()
        x_train, y_train, features_train = dataset_fransformer.fit_transform(
            x_train, y_train, features_train
        )
        if x_valid is not None:
            x_valid, y_valid, features_valid = dataset_fransformer.fit_transform(
                x_valid, y_valid, features_valid
            )
        x_all = np.concatenate((x_train, x_valid), axis=0)
        y_all = np.concatenate((y_train, y_valid), axis=0)
        features_all = concatenate_or_none((features_train, features_valid), axis=0)
        transfer_learning_source: Optional[TransferLearningSource] = None
        if self.p is None or self.model is None:
            transfer_learning_source = TransferLearningSource.from_config(
                self.transfer_learning_config,
                download_manager=self.download_manager
            )
        if self.p is None:
            if transfer_learning_source:
                self.p = transfer_learning_source.copy_preprocessor_if_enabled()
            if self.p is None:
                self.p = prepare_preprocessor(
                    x_all, y_all,
                    features=features_all,
                    model_config=self.model_config
                )
            if transfer_learning_source:
                transfer_learning_source.apply_preprocessor(target_preprocessor=self.p)
            self.model_config.char_vocab_size = len(self.p.vocab_char)
            self.model_config.case_vocab_size = len(self.p.vocab_case)

            if self.model_config.use_features and features_train is not None:
                LOGGER.info('x_train.shape: %s', x_train.shape)
                LOGGER.info('features_train.shape: %s', features_train.shape)
                sample_transformed_features = self.p.transform_features(features_train[:1])
                try:
                    if isinstance(sample_transformed_features, tuple):
                        sample_transformed_features = sample_transformed_features[0]
                    LOGGER.info(
                        'sample_transformed_features.shape: %s',
                        sample_transformed_features.shape
                    )
                    self.model_config.max_feature_size = sample_transformed_features.shape[-1]
                    LOGGER.info('max_feature_size: %s', self.model_config.max_feature_size)
                except Exception:  # pylint: disable=broad-except
                    LOGGER.info('features do not implement shape, set max_feature_size manually')

        if self.model is None:
            self.model = get_model(self.model_config, self.p, len(self.p.vocab_tag))
            if transfer_learning_source:
                transfer_learning_source.apply_weights(target_model=self.model)
        if self.transfer_learning_config:
            freeze_model_layers(self.model, self.transfer_learning_config.freeze_layers)
        trainer = Trainer(
            self.model,
            self.models,
            self.embeddings,
            self.model_config,
            training_config=self.training_config,
            model_saver=self.get_model_saver(),
            multiprocessing=self.multiprocessing,
            checkpoint_path=self.log_dir,
            preprocessor=self.p
        )
        trainer.train(
            x_train, y_train, x_valid, y_valid,
            features_train=features_train, features_valid=features_valid
        )

    def get_model_saver(self):
        return ModelSaver(
            preprocessor=self.p,
            model_config=self.model_config
        )

    def train_nfold(  # pylint: disable=arguments-differ
        self,
        x_train,
        y_train,
        x_valid=None,
        y_valid=None,
        fold_number=10,
        features_train: Optional[np.ndarray] = None,
        features_valid: Optional[np.ndarray] = None
    ):
        if x_valid is not None and y_valid is not None:
            x_all = np.concatenate((x_train, x_valid), axis=0)
            y_all = np.concatenate((y_train, y_valid), axis=0)
            features_all = concatenate_or_none((features_train, features_valid), axis=0)
            self.p = prepare_preprocessor(
                x_all, y_all,
                features=features_all,
                model_config=self.model_config
            )
        else:
            self.p = prepare_preprocessor(
                x_train, y_train,
                features=features_train,
                model_config=self.model_config
            )
        self.model_config.char_vocab_size = len(self.p.vocab_char)
        self.model_config.case_vocab_size = len(self.p.vocab_case)
        self.p.return_lengths = True

        self.models = []

        for _ in range(0, fold_number):
            model = get_model(self.model_config, self.p, len(self.p.vocab_tag))
            self.models.append(model)

        trainer = Trainer(
            self.model,
            self.models,
            self.embeddings,
            self.model_config,
            training_config=self.training_config,
            model_saver=self.get_model_saver(),
            checkpoint_path=self.log_dir,
            preprocessor=self.p
        )
        trainer.train_nfold(
            x_train, y_train,
            x_valid, y_valid,
            features_train=features_train,
            features_valid=features_valid
        )

    def eval(  # pylint: disable=arguments-differ
        self,
        x_test,
        y_test,
        features: Optional[np.ndarray] = None
    ):
        should_eval_nfold = (
            self.model_config.fold_number > 1
            and self.models
            and len(self.models) == self.model_config.fold_number
        )
        if should_eval_nfold:
            self.eval_nfold(x_test, y_test, features=features)
        else:
            self.eval_single(x_test, y_test, features=features)

    def create_eval_data_generator(self, *args, **kwargs) -> DataGenerator:
        return DataGenerator(  # type: ignore
            *args,
            batch_size=(
                self.eval_batch_size
                or self.training_config.batch_size
            ),
            preprocessor=self.p,
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
            max_sequence_length=self.eval_max_sequence_length,
            embeddings=self.embeddings,
            **kwargs
        )

    def get_evaluation_result(
        self,
        x_test: T_Batch_Token_Array,
        y_test: T_Batch_Label_Array,
        features: Optional[Union[np.ndarray, List[List[List[str]]]]] = None
    ) -> ClassificationResult:
        self._require_model()
        if self.model_config.use_features and features is None:
            raise ValueError('features required')
        assert self.p is not None
        tagger = Tagger(
            model=self.model,
            model_config=self.model_config,
            preprocessor=self.p,
            embeddings=self.embeddings,
            dataset_transformer_factory=self.dataset_transformer_factory,
            max_sequence_length=self.eval_max_sequence_length,
            input_window_stride=self.eval_input_window_stride,
        )
        tag_result = tagger.tag(
            list(x_test),
            output_format=None,
            features=features
        )
        y_pred = [
            [token_tag for _, token_tag in doc_pred]
            for doc_pred in tag_result
        ]
        # convert to list, get_entities is type checking for list but not ndarray
        y_true = [list(true_doc) for true_doc in y_test]
        return ClassificationResult(y_pred=y_pred, y_true=y_true)

    def eval_single(  # pylint: disable=arguments-differ
        self,
        x_test: T_Batch_Token_Array,
        y_test: T_Batch_Label_Array,
        features: Optional[T_Batch_Features_Array] = None
    ):
        classification_result = self.get_evaluation_result(
            x_test=x_test,
            y_test=y_test,
            features=features
        )
        print(classification_result.get_formatted_report(digits=4))

    def eval_nfold(  # pylint: disable=arguments-differ
        self,
        x_test,
        y_test,
        features: Optional[np.ndarray] = None
    ):
        if self.models is not None:
            total_f1 = 0
            best_f1 = 0
            best_index = 0
            worst_f1 = 1
            worst_index = 0
            reports = []
            total_precision = 0
            total_recall = 0
            for i in range(0, self.model_config.fold_number):
                print(
                    '\n------------------------ fold %s --------------------------------------'
                    % i
                )

                # Prepare test data(steps, generator)
                test_generator = self.create_eval_data_generator(
                    x_test, y_test,
                    features=features,
                    shuffle=False
                )

                # Build the evaluator and evaluate the model
                scorer = Scorer(
                    test_generator,
                    self.p,
                    evaluation=True,
                    use_crf=self.model_config.use_crf,
                    use_chain_crf=self.model_config.use_chain_crf
                )
                scorer.model = self.models[i]
                scorer.on_epoch_end(epoch=-1)
                f1 = scorer.f1
                precision = scorer.precision
                recall = scorer.recall
                reports.append(scorer.report)

                if best_f1 < f1:
                    best_f1 = f1
                    best_index = i
                if worst_f1 > f1:
                    worst_f1 = f1
                    worst_index = i
                total_f1 += f1
                total_precision += precision
                total_recall += recall

            macro_f1 = total_f1 / self.model_config.fold_number
            macro_precision = total_precision / self.model_config.fold_number
            macro_recall = total_recall / self.model_config.fold_number

            print("\naverage over", self.model_config.fold_number, "folds")
            print("\tmacro f1 =", macro_f1)
            print("\tmacro precision =", macro_precision)
            print("\tmacro recall =", macro_recall, "\n")

            print("\n** Worst ** model scores - \n")
            print(reports[worst_index])

            self.model = self.models[best_index]
            print("\n** Best ** model scores - \n")
            print(reports[best_index])

    def iter_tag(
        self, texts, output_format: Optional[str], features=None
    ) -> Union[dict, Iterable[TypingSequence[Tuple[str, str]]]]:
        # annotate a list of sentences, return the list of annotations in the
        # specified output_format
        self._require_model()
        if self.model_config.use_features and features is None:
            raise ValueError('features required')
        assert self.p is not None
        tagger = Tagger(
            model=self.model,
            model_config=self.model_config,
            preprocessor=self.p,
            embeddings=self.embeddings,
            dataset_transformer_factory=self.dataset_transformer_factory,
            max_sequence_length=self.max_sequence_length,
            input_window_stride=self.input_window_stride
        )
        LOGGER.debug('tag_transformed: %s', self.tag_transformed)
        annotations: Union[dict, Iterable[TypingSequence[Tuple[str, str]]]]
        if output_format == 'json':
            start_time = time.time()
            annotations = tagger.tag(
                list(texts), output_format,
                features=features,
                tag_transformed=self.tag_transformed
            )
            runtime = round(time.time() - start_time, 3)
            assert isinstance(annotations, dict)
            annotations["runtime"] = runtime
        else:
            annotations = tagger.iter_tag(  # type: ignore
                list(texts), output_format,
                features=features,
                tag_transformed=self.tag_transformed
            )
        if self.tag_debug_reporter:
            if not isinstance(annotations, dict):
                # the tag debug reporter only supports lists
                # additionally should not consume the iterable
                annotations = list(annotations)
            self.tag_debug_reporter.report_tag_results(
                texts=texts,
                features=features,
                annotations=annotations,
                model_name=self._get_model_name()
            )
        return annotations

    def tag(
        self,
        *args,
        **kwargs
    ) -> Union[dict, TypingSequence[TypingSequence[Tuple[str, str]]]]:
        iterable_or_dict = self.iter_tag(*args, **kwargs)
        if isinstance(iterable_or_dict, dict):
            return iterable_or_dict
        return list(iterable_or_dict)

    def _require_model(self):
        if not self.model:
            try:
                raise OSError('Model not loaded: %s (previous load exception: %r)' % (
                    self._get_model_name(), self._load_exception
                )) from self._load_exception
            except Exception as exc:
                LOGGER.exception('Model required but not loaded: %r', exc, exc_info=exc)
                raise

    def _get_model_name(self):
        return self.model_config.model_name

    @property
    def last_checkpoint_path(self) -> Optional[str]:
        if not self.log_dir:
            return None
        return get_last_checkpoint_url(get_checkpoints_json(self.log_dir))

    @property
    def model_summary_props(self) -> dict:
        return {
            'model_type': 'delft',
            'architecture': self.model_config.architecture,
            'model_config': vars(self.model_config)
        }

    def get_model_output_path(self, dir_path: Optional[str] = None) -> str:
        return get_model_directory(model_name=self.model_config.model_name, dir_path=dir_path)

    def _get_model_directory(self, dir_path: Optional[str] = None) -> str:
        return self.get_model_output_path(dir_path=dir_path)

    def get_embedding_for_model_config(self, model_config: ModelConfig):
        embedding_name = model_config.embeddings_name
        if not model_config.use_word_embeddings or not embedding_name:
            return None
        embedding_name = self.embedding_manager.ensure_available(embedding_name)
        LOGGER.info('embedding_name: %s', embedding_name)
        embeddings = self.embedding_manager.get_embeddings_for_name(
            embedding_name
        )
        if embeddings.embed_size <= 0:
            raise AssertionError(
                'invalid embedding size, embeddings not loaded? %s' % embedding_name
            )
        return embeddings

    def get_meta(self):
        return {
            'training_config': vars(self.training_config)
        }

    def save(self, dir_path=None, weight_file: Optional[str] = None):
        # create subfolder for the model if not already exists
        directory = self._get_model_directory(dir_path)
        os.makedirs(directory, exist_ok=True)
        self.get_model_saver().save_to(
            directory,
            model=self.model,
            meta=self.get_meta(),
            weight_file=weight_file
        )

    def load(self, dir_path=None, weight_file: Optional[str] = None):
        directory = None
        try:
            directory = self._get_model_directory(dir_path)
            self.load_from(directory, weight_file=weight_file)
        except Exception as exc:
            self._load_exception = exc
            LOGGER.exception('failed to load model from %r', directory, exc_info=exc)
            raise

    def download_model(self, dir_path: str) -> str:
        if not dir_path.endswith('.tar.gz'):
            return dir_path
        local_dir_path = str(self.download_manager.get_local_file(
            dir_path, auto_uncompress=False
        )).replace('.tar.gz', '')
        copy_directory_with_source_meta(dir_path, local_dir_path)
        return local_dir_path

    def load_from(self, directory: str, weight_file: Optional[str] = None):
        model_loader = ModelLoader(download_manager=self.download_manager)
        directory = self.download_model(directory)
        self.model_path = directory
        self.p = model_loader.load_preprocessor_from_directory(directory)
        self.model_config = model_loader.load_model_config_from_directory(directory)
        self.model_config.batch_size = self.training_config.batch_size
        if self.stateful is not None:
            self.model_config.stateful = self.stateful

        # load embeddings
        LOGGER.info('loading embeddings: %s', self.model_config.embeddings_name)
        self.embeddings = self.get_embedding_for_model_config(self.model_config)
        self.update_model_config_word_embedding_size()

        self.model = get_model(self.model_config, self.p, ntags=len(self.p.vocab_tag))
        # update stateful flag depending on whether the model is actually stateful
        # (and supports that)
        self.model_config.stateful = is_model_stateful(self.model)
        # load weights
        model_loader.load_model_from_directory(
            directory,
            model=self.model,
            weight_file=weight_file
        )
        self.update_dataset_transformer_factor()
