import argparse
from unittest.mock import MagicMock

import pytest

from sciencebeam_trainer_delft.sequence_labelling.evaluation import (
    ClassificationResult
)

from sciencebeam_trainer_delft.sequence_labelling.utils.train_notify import (
    get_rendered_notification_message,
    add_train_notification_arguments,
    get_train_notification_manager,
    check_required_gpu,
    RequiredGpuNotAvailable,
    DEFAULT_TRAIN_START_MESSAGE_FORMAT,
    DEFAULT_TRAIN_EVAL_SUCCESS_MESSAGE_FORMAT,
    DEFAULT_TRAIN_SUCCESS_MESSAGE_FORMAT,
    DEFAULT_TRAIN_ERROR_MESSAGE_FORMAT
)


MODEL_NAME_1 = 'model1'

TF_INFO_WITH_GPU: dict = {
    'tf_version': '2.17.1',
    'tf_device_lib': [MagicMock(device_type='GPU', physical_device_desc='name: Tesla T4')]
}

TF_INFO_WITHOUT_GPU: dict = {
    'tf_version': '2.17.1',
    'tf_device_lib': [MagicMock(device_type='CPU')]
}


class TestGetRenderedNotificationMessage:
    def test_should_return_static_message(self):
        assert get_rendered_notification_message('test') == 'test'

    def test_should_return_replace_placeholder(self):
        classification_result = ClassificationResult(['B-DUMMY'], ['B-DUMMY'])
        assert get_rendered_notification_message(
            'f1: {classification_result.f1}',
            classification_result=classification_result
        ) == 'f1: 1.0'

    def test_should_not_fail_using_default_train_start_message(self):
        get_rendered_notification_message(
            DEFAULT_TRAIN_START_MESSAGE_FORMAT,
            model_path='model_path',
            checkpoints_path=None,
            resume_train_model_path=None,
            initial_epoch=0,
            device='CPU only [tf: 2.17.1]'
        )

    def test_should_not_fail_using_default_train_message(self):
        get_rendered_notification_message(
            DEFAULT_TRAIN_SUCCESS_MESSAGE_FORMAT,
            last_checkpoint_path=None,
            model_path='model_path'
        )

    def test_should_not_fail_using_default_train_eval_message(self):
        classification_result = ClassificationResult(['B-DUMMY'], ['B-DUMMY'])
        get_rendered_notification_message(
            DEFAULT_TRAIN_EVAL_SUCCESS_MESSAGE_FORMAT,
            model_path='model_path',
            last_checkpoint_path=None,
            classification_result=classification_result
        )

    def test_should_not_fail_using_default_train_error_message(self):
        get_rendered_notification_message(
            DEFAULT_TRAIN_ERROR_MESSAGE_FORMAT,
            model_path='model_path',
            error='error'
        )


class TestGetTrainNotificationManager:
    def test_should_be_able_to_get_train_notification_manager_with_defaults(self):
        parser = argparse.ArgumentParser()
        add_train_notification_arguments(parser)
        args = parser.parse_args([])
        train_notification_manager = get_train_notification_manager(args)
        assert train_notification_manager is not None
        train_notification_manager.notify_start(
            model_path='model_path',
            checkpoints_path=None,
            resume_train_model_path=None,
            initial_epoch=0,
            device='CPU only [tf: 2.17.1]'
        )
        train_notification_manager.notify_success(model_path='model_path')
        train_notification_manager.notify_error(model_path='model_path', error='error')


class TestCheckRequiredGpu:
    def test_should_not_fail_if_gpu_not_required(self):
        check_required_gpu(
            require_gpu=False,
            tf_info=TF_INFO_WITHOUT_GPU,
            model_path=MODEL_NAME_1,
            train_notification_manager=None
        )

    def test_should_not_fail_if_gpu_required_and_available(self):
        check_required_gpu(
            require_gpu=True,
            tf_info=TF_INFO_WITH_GPU,
            model_path=MODEL_NAME_1,
            train_notification_manager=None
        )

    def test_should_raise_error_if_gpu_required_but_not_available(self):
        with pytest.raises(RequiredGpuNotAvailable):
            check_required_gpu(
                require_gpu=True,
                tf_info=TF_INFO_WITHOUT_GPU,
                model_path=MODEL_NAME_1,
                train_notification_manager=None
            )

    def test_should_notify_error_if_gpu_required_but_not_available(self):
        train_notification_manager = MagicMock()
        with pytest.raises(RequiredGpuNotAvailable):
            check_required_gpu(
                require_gpu=True,
                tf_info=TF_INFO_WITHOUT_GPU,
                model_path=MODEL_NAME_1,
                train_notification_manager=train_notification_manager
            )
        train_notification_manager.notify_error.assert_called_once()
        assert (
            train_notification_manager.notify_error.call_args.kwargs['model_path']
            == MODEL_NAME_1
        )
