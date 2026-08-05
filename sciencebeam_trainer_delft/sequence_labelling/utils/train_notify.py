import argparse
import logging
from typing import Optional

import requests

from sciencebeam_trainer_delft.sequence_labelling.evaluation import (
    ClassificationResult
)
from sciencebeam_trainer_delft.utils.device import (
    get_gpu_devices,
    get_device_summary
)


LOGGER = logging.getLogger(__name__)


DEFAULT_TRAIN_START_MESSAGE_FORMAT = '\n'.join([
    'Model training started',
    'model_path: `{model_path}`',
    'checkpoints_path: `{checkpoints_path}`',
    'resume_train_model_path: `{resume_train_model_path}`',
    'initial_epoch: `{initial_epoch}`',
    'device: `{device}`'
])


DEFAULT_TRAIN_SUCCESS_MESSAGE_FORMAT = '\n'.join([
    'Model training complete',
    'model_path: `{model_path}`',
    'last_checkpoint_path: `{last_checkpoint_path}`'
])

DEFAULT_TRAIN_EVAL_SUCCESS_MESSAGE_FORMAT = '\n'.join([
    'Model training complete, f1: `{classification_result.f1:.4f}`',
    'model_path: `{model_path}`',
    'last_checkpoint_path: `{last_checkpoint_path}`',
    '```\n{classification_result.text_formatted_report}\n```'
])

DEFAULT_TRAIN_ERROR_MESSAGE_FORMAT = (
    'Model training failed due to: `{error}`\nmodel_path: `{model_path}`'
)


def get_rendered_notification_message(message_format: str, **kwargs):
    return message_format.format(**kwargs)


def get_fallback_notification_message(message_format: str, conversion_error: str, args: dict):
    return 'failed to format %r due to %s (args: %s)' % (message_format, conversion_error, args)


def safe_rendered_notification_message(message_format: str, **kwargs):
    try:
        return get_rendered_notification_message(message_format, **kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning(
            'failed to convert message due to: %s', exc, exc_info=exc
        )
        return get_fallback_notification_message(message_format, str(exc), kwargs)


class TrainNotificationManager:
    def __init__(
            self,
            notification_url: str,
            notification_train_start_message: str,
            notification_train_success_message: str,
            notification_train_eval_success_message: str,
            notification_error_message: str):
        self.notification_url = notification_url
        self.notification_train_start_message = notification_train_start_message
        self.notification_train_success_message = notification_train_success_message
        self.notification_train_eval_success_message = notification_train_eval_success_message
        self.notification_error_message = notification_error_message

    def send_notification(self, message_format: str, **kwargs):
        message = safe_rendered_notification_message(message_format, **kwargs)
        if not message or not self.notification_url:
            LOGGER.info('not sending notification: %r (url: %r)', message, self.notification_url)
            return
        data = {
            'text': message
        }
        LOGGER.info('sending notification: %r (url: %r)', message, self.notification_url)
        requests.post(self.notification_url, json=data, timeout=60)

    def notify_error(self, model_path: str, error: str):
        self.send_notification(
            self.notification_error_message,
            model_path=model_path,
            error=error
        )

    def notify_start(
        self,
        model_path: str,
        checkpoints_path: Optional[str],
        resume_train_model_path: Optional[str],
        initial_epoch: int,
        device: Optional[str] = None
    ):
        self.send_notification(
            self.notification_train_start_message,
            model_path=model_path,
            checkpoints_path=checkpoints_path,
            resume_train_model_path=resume_train_model_path,
            initial_epoch=initial_epoch,
            device=device
        )

    def notify_success(
        self,
        model_path: str,
        last_checkpoint_path: Optional[str] = None,
        classification_result: Optional[ClassificationResult] = None
    ):
        if classification_result is None:
            self.send_notification(
                self.notification_train_success_message,
                model_path=model_path,
                last_checkpoint_path=last_checkpoint_path
            )
        else:
            self.send_notification(
                self.notification_train_eval_success_message,
                model_path=model_path,
                last_checkpoint_path=last_checkpoint_path,
                classification_result=classification_result
            )


def add_train_notification_arguments(parser: argparse.ArgumentParser):
    notification_group = parser.add_argument_group('notification')
    notification_group.add_argument(
        "--notification-url",
        help="A URL to post to on success error (e.g. a Slack Webhook URL)"
    )
    notification_group.add_argument(
        "--notification-train-start-message",
        default=DEFAULT_TRAIN_START_MESSAGE_FORMAT,
        help="Model training start notification message"
    )
    notification_group.add_argument(
        "--notification-train-success-message",
        default=DEFAULT_TRAIN_SUCCESS_MESSAGE_FORMAT,
        help="Model training success notification message"
    )
    notification_group.add_argument(
        "--notification-train-eval-success-message",
        default=DEFAULT_TRAIN_EVAL_SUCCESS_MESSAGE_FORMAT,
        help="Model training and evaluation success notification message"
    )
    notification_group.add_argument(
        "--notification-error-message",
        default=DEFAULT_TRAIN_ERROR_MESSAGE_FORMAT,
        help="Model training failed notification message"
    )


def get_train_notification_manager(args: argparse.Namespace) -> TrainNotificationManager:
    return TrainNotificationManager(
        notification_url=args.notification_url,
        notification_train_start_message=args.notification_train_start_message,
        notification_train_success_message=args.notification_train_success_message,
        notification_train_eval_success_message=args.notification_train_eval_success_message,
        notification_error_message=args.notification_error_message
    )


def notify_train_start(
    train_notification_manager: Optional[TrainNotificationManager] = None,
    **kwargs
):
    if train_notification_manager is not None:
        train_notification_manager.notify_start(**kwargs)


def notify_train_success(
    train_notification_manager: Optional[TrainNotificationManager] = None,
    **kwargs
):
    if train_notification_manager is not None:
        train_notification_manager.notify_success(**kwargs)


def notify_train_error(
    train_notification_manager: Optional[TrainNotificationManager] = None,
    **kwargs
):
    if train_notification_manager is not None:
        train_notification_manager.notify_error(**kwargs)


class RequiredGpuNotAvailable(RuntimeError):
    pass


def check_required_gpu(
    require_gpu: bool,
    device_info: dict,
    model_path: str,
    train_notification_manager: Optional[TrainNotificationManager]
):
    if not require_gpu or get_gpu_devices(device_info):
        return
    error_message = (
        'GPU required (--require-gpu) but none available (device: %s)'
        % get_device_summary(device_info)
    )
    notify_train_error(
        train_notification_manager,
        model_path=model_path,
        error=error_message
    )
    raise RequiredGpuNotAvailable(error_message)
