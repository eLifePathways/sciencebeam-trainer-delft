# initially based on:
# https://github.com/kermitt2/delft/blob/master/grobidTagger.py

import logging
import time
import tempfile
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np

from sklearn.model_selection import train_test_split

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.utils.io import (
    write_text,
    auto_uploading_output_file
)
from sciencebeam_trainer_delft.utils.device import (
    get_device_info,
    get_device_summary
)

from sciencebeam_trainer_delft.embedding import EmbeddingManager

from sciencebeam_trainer_delft.sequence_labelling.utils.train_notify import (
    TrainNotificationManager,
    notify_train_start,
    notify_train_success,
    notify_train_error
)

from sciencebeam_trainer_delft.sequence_labelling.wrapper import (
    Sequence
)
from sciencebeam_trainer_delft.sequence_labelling.trainer_torch import (
    set_random_seed as set_torch_random_seed
)
from sciencebeam_trainer_delft.sequence_labelling.feature_lengths import (
    DEFAULT_FEATURE_LENGTH_MODE
)
from sciencebeam_trainer_delft.sequence_labelling.tools.grobid_trainer.data_loading import (
    DEFAULT_RANDOM_SEED,
    load_data_and_labels,
    load_data_and_labels_with_check_result
)

from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti_adapters import (
    WapitiModelAdapter,
    WapitiModelTrainAdapter
)

from sciencebeam_trainer_delft.sequence_labelling.tag_formatter import (
    TagOutputFormats,
    TagLabelFormats,
    get_tag_result,
    iter_format_tag_result
)

from sciencebeam_trainer_delft.sequence_labelling.evaluation import (
    EvaluationOutputFormats,
    ClassificationResult
)

from sciencebeam_trainer_delft.sequence_labelling.utils.checkpoints import (
    get_resume_train_model_params
)


LOGGER = logging.getLogger(__name__)


DEFAULT_TAG_OUTPUT_FORMAT = TagOutputFormats.XML


def set_random_seeds(random_seed: int):
    np.random.seed(random_seed)
    set_torch_random_seed(random_seed)


def get_default_training_data(model: str) -> str:
    return 'data/sequenceLabelling/grobid/' + model + '/' + model + '-060518.train'


def notify_model_train_start(
    model: Union[Sequence, WapitiModelTrainAdapter],
    train_notification_manager: Optional[TrainNotificationManager],
    output_path: Optional[str]
):
    notify_train_start(
        train_notification_manager,
        model_path=model.get_model_output_path(output_path),
        checkpoints_path=model.log_dir,
        resume_train_model_path=model.model_path,
        initial_epoch=model.training_config.initial_epoch,
        device=get_device_summary(get_device_info())
    )


def do_train(
    model: Union[Sequence, WapitiModelTrainAdapter],
    input_paths: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    shuffle_input: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    feature_length_mode: str = DEFAULT_FEATURE_LENGTH_MODE,
    train_notification_manager: Optional[TrainNotificationManager] = None,
    download_manager: Optional[DownloadManager] = None
):
    x_all, y_all, features_all = load_data_and_labels(
        input_paths=input_paths, limit=limit, shuffle_input=shuffle_input,
        random_seed=random_seed,
        feature_length_mode=feature_length_mode,
        read_feature_indices=model.get_read_feature_indices(),
        download_manager=download_manager
    )
    x_train, x_valid, y_train, y_valid, features_train, features_valid = train_test_split(
        x_all, y_all, features_all, test_size=0.1, shuffle=False
    )

    LOGGER.info('%d train sequences', len(x_train))
    LOGGER.info('%d validation sequences', len(x_valid))

    notify_model_train_start(
        model,
        train_notification_manager,
        output_path=output_path
    )

    start_time = time.time()
    model.train(
        x_train, y_train, x_valid, y_valid,
        features_train=features_train, features_valid=features_valid
    )
    runtime = round(time.time() - start_time, 3)
    LOGGER.info("training runtime: %s seconds ", runtime)

    # saving the model
    if output_path:
        LOGGER.info('saving model to: %s', output_path)
        model.save(output_path)
    else:
        model.save()

    notify_train_success(
        train_notification_manager,
        model_path=model.get_model_output_path(output_path),
        last_checkpoint_path=model.last_checkpoint_path
    )


def do_train_with_error_notification(
    model: Union[Sequence, WapitiModelTrainAdapter],
    output_path: Optional[str] = None,
    train_notification_manager: Optional[TrainNotificationManager] = None,
    **kwargs
):
    model_path = model.get_model_output_path(output_path)
    try:
        do_train(
            model=model,
            output_path=output_path,
            train_notification_manager=train_notification_manager,
            **kwargs
        )
    except BaseException as error:  # pylint: disable=broad-except
        notify_train_error(
            train_notification_manager,
            model_path=model_path,
            error=repr(error)
        )
        raise


def process_resume_train_model_params(
    model: Sequence,
    auto_resume: bool,
    resume_train_model_path: Optional[str]
):
    resume_train_model_params = get_resume_train_model_params(
        log_dir=model.log_dir,
        auto_resume=auto_resume,
        resume_train_model_path=resume_train_model_path,
        initial_epoch=model.training_config.initial_epoch
    )
    if resume_train_model_params:
        model.load_from(resume_train_model_params.model_path)
        model.training_config.initial_epoch = resume_train_model_params.initial_epoch
        model.training_config.initial_meta = resume_train_model_params.initial_meta


# train a GROBID model with all available data
def train(
    model_name: str,
    embeddings_name: Optional[str],
    architecture='BidLSTM_CRF',
    input_paths: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    shuffle_input: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    feature_length_mode: str = DEFAULT_FEATURE_LENGTH_MODE,
    max_sequence_length: int = 100,
    max_epoch=100,
    resume_train_model_path: Optional[str] = None,
    auto_resume: bool = False,
    train_notification_manager: Optional[TrainNotificationManager] = None,
    download_manager: Optional[DownloadManager] = None,
    embedding_manager: Optional[EmbeddingManager] = None,
    **kwargs
):

    model_name = get_model_name(
        model_name, output_path=output_path
    )

    model = Sequence(
        model_name,
        max_epoch=max_epoch,
        embeddings_name=embeddings_name,
        embedding_manager=embedding_manager,
        max_sequence_length=max_sequence_length,
        architecture=architecture,
        **kwargs
    )
    process_resume_train_model_params(
        model,
        auto_resume=auto_resume,
        resume_train_model_path=resume_train_model_path
    )

    do_train_with_error_notification(
        model,
        input_paths=input_paths,
        output_path=output_path,
        limit=limit,
        shuffle_input=shuffle_input,
        random_seed=random_seed,
        feature_length_mode=feature_length_mode,
        train_notification_manager=train_notification_manager,
        download_manager=download_manager
    )


def wapiti_train(
    model_name: str,
    template_path: str,
    output_path: str,
    download_manager: DownloadManager,
    input_paths: Optional[List[str]] = None,
    limit: Optional[int] = None,
    shuffle_input: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    feature_length_mode: str = DEFAULT_FEATURE_LENGTH_MODE,
    max_epoch: int = 100,
    train_notification_manager: Optional[TrainNotificationManager] = None,
    gzip_enabled: bool = False,
    wapiti_binary_path: Optional[str] = None,
    wapiti_train_args: Optional[dict] = None
):
    with tempfile.TemporaryDirectory(suffix='-wapiti') as temp_dir:
        temp_model_path = os.path.join(temp_dir, 'model.wapiti')
        model = WapitiModelTrainAdapter(
            model_name=model_name,
            template_path=template_path,
            temp_model_path=temp_model_path,
            max_epoch=max_epoch,
            download_manager=download_manager,
            gzip_enabled=gzip_enabled,
            wapiti_binary_path=wapiti_binary_path,
            wapiti_train_args=wapiti_train_args
        )
        do_train_with_error_notification(
            model,
            input_paths=input_paths,
            output_path=output_path,
            limit=limit,
            shuffle_input=shuffle_input,
            random_seed=random_seed,
            feature_length_mode=feature_length_mode,
            train_notification_manager=train_notification_manager,
            download_manager=download_manager
        )


def output_classification_result(
    classification_result: ClassificationResult,
    eval_output_args: Optional[dict],
    eval_input_paths: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    model_summary_props: Optional[dict] = None
):
    eval_output_args = eval_output_args or dict()
    assert eval_output_args is not None
    output_format = eval_output_args.get('eval_output_args')
    output_path = eval_output_args.get('eval_output_path')
    eval_first_entity = eval_output_args.get('eval_first_entity')
    if not output_format:
        output_format = EvaluationOutputFormats.TEXT
    if eval_first_entity:
        classification_result = classification_result.with_first_entities()
    meta:  Dict[str, Any] = {}
    meta['eval_timestamp'] = datetime.now(timezone.utc).isoformat()
    if eval_input_paths:
        meta['eval_input_paths'] = eval_input_paths
    if model_path:
        meta['model_path'] = model_path
    if model_summary_props:
        meta.update(model_summary_props)
    if output_path:
        LOGGER.info('writing evaluation to: %s', output_path)
        write_text(output_path, classification_result.get_json_formatted_report(meta=meta))
    if output_format == EvaluationOutputFormats.TEXT:
        print("\nEvaluation:\n%s" % classification_result.get_text_formatted_report(
            digits=4
        ))
    elif output_format == EvaluationOutputFormats.JSON:
        print(classification_result.get_json_formatted_report(meta=meta))
    else:
        print(classification_result.get_formatted_report(
            output_format=output_format
        ))


def do_train_eval(
    model: Union[Sequence, WapitiModelTrainAdapter],
    input_paths: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    shuffle_input: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    feature_length_mode: str = DEFAULT_FEATURE_LENGTH_MODE,
    eval_input_paths: Optional[List[str]] = None,
    eval_limit: Optional[int] = None,
    eval_output_args: Optional[dict] = None,
    fold_count: int = 1,
    train_notification_manager: Optional[TrainNotificationManager] = None,
    download_manager: Optional[DownloadManager] = None
):
    read_feature_indices = model.get_read_feature_indices()
    x_all, y_all, features_all, check_result = load_data_and_labels_with_check_result(
        input_paths=input_paths, limit=limit, shuffle_input=shuffle_input,
        random_seed=random_seed,
        feature_length_mode=feature_length_mode,
        read_feature_indices=read_feature_indices,
        download_manager=download_manager
    )

    if eval_input_paths:
        # held to the feature length the training data settled on: a model evaluated on
        # features it was not trained on is the same defect
        x_eval, y_eval, features_eval = load_data_and_labels(
            input_paths=eval_input_paths, limit=eval_limit,
            feature_length_mode=feature_length_mode,
            read_feature_indices=read_feature_indices,
            expected_feature_count=check_result.adopted_feature_count,
            download_manager=download_manager
        )
        x_train_all, y_train_all, features_train_all = (
            x_all, y_all, features_all
        )
    else:
        x_train_all, x_eval, y_train_all, y_eval, features_train_all, features_eval = (
            train_test_split(x_all, y_all, features_all, test_size=0.1, shuffle=False)
        )
    x_train, x_valid, y_train, y_valid, features_train, features_valid = train_test_split(
        x_train_all, y_train_all, features_train_all, test_size=0.1, shuffle=False
    )

    LOGGER.info('%d train sequences', len(x_train))
    LOGGER.info('%d validation sequences', len(x_valid))
    LOGGER.info('%d evaluation sequences', len(x_eval))

    notify_model_train_start(
        model,
        train_notification_manager,
        output_path=output_path
    )

    start_time = time.time()

    if fold_count == 1:
        model.train(
            x_train, y_train, x_valid, y_valid,
            features_train=features_train, features_valid=features_valid
        )
    else:
        assert isinstance(model, Sequence), \
            'nfold evaluation currently only supported for DL models'
        model.train_nfold(
            x_train, y_train, x_valid, y_valid,
            features_train=features_train, features_valid=features_valid,
            fold_number=fold_count
        )

    runtime = round(time.time() - start_time, 3)
    LOGGER.info("training runtime: %s seconds ", runtime)

    # evaluation
    classification_result = model.get_evaluation_result(
        x_eval, y_eval, features=features_eval
    )
    output_classification_result(
        classification_result,
        eval_output_args=eval_output_args,
        eval_input_paths=eval_input_paths,
        model_path=model.get_model_output_path(output_path),
        model_summary_props=model.model_summary_props
    )

    # saving the model
    if output_path:
        model.save(output_path)
    else:
        model.save()

    notify_train_success(
        train_notification_manager,
        model_path=model.get_model_output_path(output_path),
        last_checkpoint_path=model.last_checkpoint_path,
        classification_result=classification_result
    )


def do_train_eval_with_error_notification(
    model: Union[Sequence, WapitiModelTrainAdapter],
    output_path: Optional[str] = None,
    train_notification_manager: Optional[TrainNotificationManager] = None,
    **kwargs
):
    model_path = model.get_model_output_path(output_path)
    try:
        do_train_eval(
            model=model,
            output_path=output_path,
            train_notification_manager=train_notification_manager,
            **kwargs
        )
    except BaseException as error:  # pylint: disable=broad-except
        notify_train_error(
            train_notification_manager,
            model_path=model_path,
            error=repr(error)
        )
        raise


# split data, train a GROBID model and evaluate it
def train_eval(
    model_name: str,
    embeddings_name,
    architecture='BidLSTM_CRF',
    input_paths: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    shuffle_input: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    feature_length_mode: str = DEFAULT_FEATURE_LENGTH_MODE,
    eval_input_paths: Optional[List[str]] = None,
    eval_limit: Optional[int] = None,
    eval_output_args: Optional[dict] = None,
    max_sequence_length: int = 100,
    fold_count=1,
    max_epoch=100,
    batch_size=20,
    resume_train_model_path: Optional[str] = None,
    auto_resume: bool = False,
    train_notification_manager: Optional[TrainNotificationManager] = None,
    download_manager: Optional[DownloadManager] = None,
    embedding_manager: Optional[EmbeddingManager] = None,
    **kwargs
):

    model_name = get_model_name(
        model_name,
        output_path=output_path
    )

    model = Sequence(
        model_name,
        max_epoch=max_epoch,
        embeddings_name=embeddings_name,
        embedding_manager=embedding_manager,
        max_sequence_length=max_sequence_length,
        architecture=architecture,
        batch_size=batch_size,
        fold_number=fold_count,
        **kwargs
    )

    process_resume_train_model_params(
        model,
        auto_resume=auto_resume,
        resume_train_model_path=resume_train_model_path
    )

    do_train_eval_with_error_notification(
        model,
        input_paths=input_paths,
        output_path=output_path,
        limit=limit,
        shuffle_input=shuffle_input,
        random_seed=random_seed,
        feature_length_mode=feature_length_mode,
        eval_input_paths=eval_input_paths,
        eval_limit=eval_limit,
        eval_output_args=eval_output_args,
        train_notification_manager=train_notification_manager,
        download_manager=download_manager
    )


def wapiti_train_eval(
    model_name: str,
    template_path: str,
    download_manager: DownloadManager,
    input_paths: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    shuffle_input: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    feature_length_mode: str = DEFAULT_FEATURE_LENGTH_MODE,
    eval_input_paths: Optional[List[str]] = None,
    eval_limit: Optional[int] = None,
    eval_output_args: Optional[dict] = None,
    fold_count: int = 1,
    max_epoch: int = 100,
    train_notification_manager: Optional[TrainNotificationManager] = None,
    gzip_enabled: bool = False,
    wapiti_binary_path: Optional[str] = None,
    wapiti_train_args: Optional[dict] = None
):
    assert fold_count == 1, 'only fold_count == 1 supported'
    with tempfile.TemporaryDirectory(suffix='-wapiti') as temp_dir:
        temp_model_path = os.path.join(temp_dir, 'model.wapiti')
        model = WapitiModelTrainAdapter(
            model_name=model_name,
            template_path=template_path,
            temp_model_path=temp_model_path,
            max_epoch=max_epoch,
            download_manager=download_manager,
            gzip_enabled=gzip_enabled,
            wapiti_binary_path=wapiti_binary_path,
            wapiti_train_args=wapiti_train_args
        )
        do_train_eval_with_error_notification(
            model,
            input_paths=input_paths,
            output_path=output_path,
            limit=limit,
            shuffle_input=shuffle_input,
            random_seed=random_seed,
            feature_length_mode=feature_length_mode,
            eval_input_paths=eval_input_paths,
            eval_limit=eval_limit,
            eval_output_args=eval_output_args,
            train_notification_manager=train_notification_manager,
            download_manager=download_manager
        )


def do_eval_model(
    model: Union[Sequence, WapitiModelAdapter],
    input_paths: Optional[List[str]] = None,
    limit: Optional[int] = None,
    shuffle_input: bool = False,
    split_input: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    feature_length_mode: str = DEFAULT_FEATURE_LENGTH_MODE,
    eval_output_args: Optional[dict] = None,
    download_manager: Optional[DownloadManager] = None
):
    x_all, y_all, features_all = load_data_and_labels(
        input_paths=input_paths, limit=limit, shuffle_input=shuffle_input,
        random_seed=random_seed,
        feature_length_mode=feature_length_mode,
        read_feature_indices=model.get_read_feature_indices(),
        download_manager=download_manager
    )

    if split_input:
        _, x_eval, _, y_eval, _, features_eval = train_test_split(
            x_all, y_all, features_all, test_size=0.1, shuffle=False
        )
    else:
        x_eval = x_all
        y_eval = y_all
        features_eval = features_all

    LOGGER.info('%d evaluation sequences', len(x_eval))

    # evaluation
    classification_result = model.get_evaluation_result(
        x_eval, y_eval, features=features_eval
    )
    output_classification_result(
        classification_result,
        eval_output_args=eval_output_args,
        eval_input_paths=input_paths,
        model_path=model.model_path,
        model_summary_props=model.model_summary_props
    )


def get_model_name(
    model_name: str,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None
):
    if output_path or model_path:
        pass
    else:
        model_name = 'grobid-' + model_name
    return model_name


def load_delft_model(
    model_name: str,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None,
    max_sequence_length: Optional[int] = 100,
    fold_count: int = 1,
    batch_size: int = 20,
    embedding_manager: Optional[EmbeddingManager] = None,
    **kwargs
):
    model = Sequence(
        get_model_name(
            model_name,
            output_path=output_path,
            model_path=model_path
        ),
        embeddings_name=None,
        embedding_manager=embedding_manager,
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
        fold_number=fold_count,
        **kwargs
    )

    assert model_path
    model.load_from(model_path)
    return model


def eval_model(
    model_name: str,
    input_paths: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None,
    limit: Optional[int] = None,
    shuffle_input: bool = False,
    split_input: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    feature_length_mode: str = DEFAULT_FEATURE_LENGTH_MODE,
    max_sequence_length: int = 100,
    fold_count: int = 1,
    batch_size: int = 20,
    eval_output_args: Optional[dict] = None,
    download_manager: Optional[DownloadManager] = None,
    embedding_manager: Optional[EmbeddingManager] = None,
    **kwargs
):

    model = load_delft_model(
        model_name=model_name,
        output_path=output_path,
        model_path=model_path,
        max_sequence_length=max_sequence_length,
        fold_count=fold_count,
        batch_size=batch_size,
        embedding_manager=embedding_manager,
        **kwargs
    )

    do_eval_model(
        model,
        input_paths=input_paths,
        limit=limit,
        shuffle_input=shuffle_input,
        random_seed=random_seed,
        split_input=split_input,
        feature_length_mode=feature_length_mode,
        eval_output_args=eval_output_args,
        download_manager=download_manager
    )


def wapiti_eval_model(
    model_path: str,
    download_manager: DownloadManager,
    input_paths: Optional[List[str]] = None,
    limit: Optional[int] = None,
    shuffle_input: bool = False,
    split_input: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    feature_length_mode: str = DEFAULT_FEATURE_LENGTH_MODE,
    fold_count: int = 1,
    eval_output_args: Optional[dict] = None,
    wapiti_binary_path: Optional[str] = None
):
    assert fold_count == 1, 'only fold_count == 1 supported'

    model = WapitiModelAdapter.load_from(
        model_path,
        download_manager=download_manager,
        wapiti_binary_path=wapiti_binary_path
    )
    do_eval_model(
        model,
        input_paths=input_paths,
        limit=limit,
        shuffle_input=shuffle_input,
        random_seed=random_seed,
        split_input=split_input,
        feature_length_mode=feature_length_mode,
        eval_output_args=eval_output_args,
        download_manager=download_manager
    )


def do_tag_input(
    model: Union[Sequence, WapitiModelAdapter],
    tag_output_format: str = DEFAULT_TAG_OUTPUT_FORMAT,
    tag_label_format: str = TagLabelFormats.GROBID,
    tag_output_path: Optional[str] = None,
    input_paths: Optional[List[str]] = None,
    limit: Optional[int] = None,
    shuffle_input: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    feature_length_mode: str = DEFAULT_FEATURE_LENGTH_MODE,
    download_manager: Optional[DownloadManager] = None
):
    x_all, y_all, features_all = load_data_and_labels(
        input_paths=input_paths, limit=limit, shuffle_input=shuffle_input,
        random_seed=random_seed,
        feature_length_mode=feature_length_mode,
        read_feature_indices=model.get_read_feature_indices(),
        download_manager=download_manager
    )

    LOGGER.info('%d input sequences', len(x_all))

    tag_result = model.iter_tag(
        x_all,
        output_format=None,
        features=features_all
    )
    if LOGGER.isEnabledFor(logging.DEBUG):
        if not isinstance(tag_result, dict):
            tag_result = list(tag_result)
        LOGGER.debug('actual raw tag_result: %s', tag_result)
    if isinstance(model, Sequence) and model.tag_transformed:
        dataset_transformer = model.dataset_transformer_factory()
        expected_x_all, expected_y_all, expected_features_all = dataset_transformer.fit_transform(
            x_all, y_all, features=features_all
        )
    else:
        expected_x_all = x_all
        expected_y_all = y_all
        expected_features_all = features_all
    expected_tag_result = get_tag_result(
        texts=expected_x_all,
        labels=expected_y_all
    )
    LOGGER.debug('actual raw expected_tag_result: %s', expected_tag_result)
    formatted_tag_result_iterable = iter_format_tag_result(
        tag_result,
        output_format=tag_output_format,
        expected_tag_result=expected_tag_result,
        texts=expected_x_all,
        features=expected_features_all,
        model_name=model._get_model_name(),  # pylint: disable=protected-access
        label_format=tag_label_format
    )
    if tag_output_path:
        LOGGER.info('writing tag results to: %r', tag_output_path)
        with auto_uploading_output_file(tag_output_path) as fp:
            for text in formatted_tag_result_iterable:
                fp.write(text)
        LOGGER.info('tag results written to: %r', tag_output_path)
    else:
        LOGGER.info('writing tag_result to stdout')
        try:
            for text in formatted_tag_result_iterable:
                print(text, end='')
        except BrokenPipeError:
            LOGGER.info('received broken pipe error')


def tag_input(
    model_name: str,
    tag_output_format: str = DEFAULT_TAG_OUTPUT_FORMAT,
    tag_label_format: str = TagLabelFormats.GROBID,
    tag_output_path: Optional[str] = None,
    input_paths: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None,
    limit: Optional[int] = None,
    shuffle_input: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    feature_length_mode: str = DEFAULT_FEATURE_LENGTH_MODE,
    max_sequence_length: Optional[int] = None,
    input_window_stride: Optional[int] = None,
    stateful: Optional[bool] = None,
    fold_count: int = 1,
    batch_size: int = 20,
    download_manager: Optional[DownloadManager] = None,
    embedding_manager: Optional[EmbeddingManager] = None,
    **kwargs
):

    model = load_delft_model(
        model_name=model_name,
        output_path=output_path,
        model_path=model_path,
        max_sequence_length=max_sequence_length,
        input_window_stride=input_window_stride,
        stateful=stateful,
        fold_count=fold_count,
        batch_size=batch_size,
        embedding_manager=embedding_manager,
        **kwargs
    )

    do_tag_input(
        model,
        tag_output_format=tag_output_format,
        tag_label_format=tag_label_format,
        tag_output_path=tag_output_path,
        input_paths=input_paths,
        limit=limit,
        shuffle_input=shuffle_input,
        random_seed=random_seed,
        feature_length_mode=feature_length_mode,
        download_manager=download_manager
    )


def wapiti_tag_input(
    model_path: str,
    download_manager: DownloadManager,
    tag_output_format: str = DEFAULT_TAG_OUTPUT_FORMAT,
    tag_label_format: str = TagLabelFormats.GROBID,
    tag_output_path: Optional[str] = None,
    input_paths: Optional[List[str]] = None,
    limit: Optional[int] = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    shuffle_input: bool = False,
    feature_length_mode: str = DEFAULT_FEATURE_LENGTH_MODE,
    wapiti_binary_path: Optional[str] = None
):
    model: WapitiModelAdapter = WapitiModelAdapter.load_from(
        model_path,
        download_manager=download_manager,
        wapiti_binary_path=wapiti_binary_path
    )
    do_tag_input(
        model=model,
        tag_output_format=tag_output_format,
        tag_label_format=tag_label_format,
        tag_output_path=tag_output_path,
        input_paths=input_paths,
        limit=limit,
        shuffle_input=shuffle_input,
        random_seed=random_seed,
        feature_length_mode=feature_length_mode,
        download_manager=download_manager
    )
