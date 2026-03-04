# initially based on:
# https://github.com/kermitt2/delft/blob/master/grobidTagger.py

import logging
import argparse
from abc import abstractmethod
from typing import List, Optional, Sequence

import sciencebeam_trainer_delft.utils.no_warn_if_disabled  # noqa, pylint: disable=unused-import
import sciencebeam_trainer_delft.utils.no_keras_backend_message  # noqa, pylint: disable=unused-import
# pylint: disable=wrong-import-order, ungrouped-imports

import tf_keras.backend as K  # type: ignore[import-untyped]

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.utils.cloud_support import patch_cloud_support
from sciencebeam_trainer_delft.utils.tf import get_tf_info
from sciencebeam_trainer_delft.utils.io import (
    copy_file,
    auto_uploading_output_file
)
from sciencebeam_trainer_delft.utils.logging import (
    tee_stdout_and_stderr_lines_to,
    tee_logging_lines_to
)

from sciencebeam_trainer_delft.embedding import EmbeddingManager

from sciencebeam_trainer_delft.sequence_labelling.utils.train_notify import (
    get_train_notification_manager
)

from sciencebeam_trainer_delft.sequence_labelling.models import patch_get_model

from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti_install import (
    install_wapiti_and_get_path_or_none
)

from sciencebeam_trainer_delft.utils.cli import (
    SubCommand,
    SubCommandProcessor
)

from sciencebeam_trainer_delft.sequence_labelling.transfer_learning import (
    get_transfer_learning_config_for_parsed_args
)
from sciencebeam_trainer_delft.sequence_labelling.tools.grobid_trainer.cli_args import (
    add_common_arguments,
    add_train_arguments,
    add_model_path_argument,
    add_wapiti_train_arguments,
    add_wapiti_install_arguments,
    get_wapiti_train_args,
    add_fold_count_argument,
    add_eval_input_arguments,
    add_eval_output_arguments,
    add_dl_eval_model_arguments,
    add_stateful_argument,
    add_input_window_stride_argument,
    add_tag_output_format_argument,
    add_tag_output_path_argument,
    add_tag_transformed_argument,
    add_model_positional_argument,
    create_argument_parser,
    process_args
)

from sciencebeam_trainer_delft.sequence_labelling.tools.grobid_trainer.utils import (
    set_random_seeds,
    train,
    wapiti_train,
    train_eval,
    wapiti_train_eval,
    eval_model,
    wapiti_eval_model,
    tag_input,
    wapiti_tag_input,
    print_input_info
)


LOGGER = logging.getLogger(__name__)


class Tasks:
    TRAIN = 'train'
    TRAIN_EVAL = 'train_eval'
    EVAL = 'eval'
    TAG = 'tag'
    WAPITI_TRAIN = 'wapiti_train'
    WAPITI_TRAIN_EVAL = 'wapiti_train_eval'
    WAPITI_EVAL = 'wapiti_eval'
    WAPITI_TAG = 'wapiti_tag'
    INPUT_INFO = 'input_info'


def save_input_to(input_paths: List[str], output_path: str):
    assert len(input_paths) == 1, "exactly one input path expected (got: %s)" % input_paths
    input_path = input_paths[0]
    LOGGER.info('saving input (%s) to: %s', input_path, output_path)
    copy_file(input_path, output_path)


def get_eval_input_args(args: argparse.Namespace) -> dict:
    return dict(
        eval_input_paths=args.eval_input,
        eval_limit=args.eval_limit,
    )


def get_eval_output_args(args: argparse.Namespace) -> dict:
    return dict(
        eval_output_format=args.eval_output_format,
        eval_first_entity=args.eval_first_entity,
        eval_output_path=args.eval_output_path
    )


def get_dl_eval_model_args(args: argparse.Namespace) -> dict:
    return dict(
        eval_max_sequence_length=args.eval_max_sequence_length,
        eval_input_window_stride=args.eval_input_window_stride,
        eval_batch_size=args.eval_batch_size
    )


class GrobidTrainerSubCommand(SubCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.download_manager = None
        self.embedding_manager = None

    @abstractmethod
    def do_run(self, args: argparse.Namespace):
        pass

    def preload_and_validate_embedding(
            self,
            embedding_name: str,
            use_word_embeddings: bool = True) -> Optional[str]:
        if not use_word_embeddings:
            return None
        embedding_name = self.embedding_manager.ensure_available(embedding_name)
        LOGGER.info('embedding_name: %s', embedding_name)
        self.embedding_manager.validate_embedding(embedding_name)
        return embedding_name

    def get_common_args(self, args: argparse.Namespace) -> dict:
        return dict(
            model_name=args.model,
            input_paths=args.input,
            limit=args.limit,
            shuffle_input=args.shuffle_input,
            random_seed=args.random_seed,
            batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length,
            multiprocessing=args.multiprocessing,
            embedding_manager=self.embedding_manager,
            download_manager=self.download_manager
        )

    def get_train_args(self, args: argparse.Namespace) -> dict:
        return dict(
            architecture=args.architecture,
            output_path=args.output,
            log_dir=args.checkpoint,
            char_emb_size=args.char_embedding_size,
            char_lstm_units=args.char_lstm_units,
            word_lstm_units=args.word_lstm_units,
            dropout=args.dropout,
            recurrent_dropout=args.recurrent_dropout,
            max_epoch=args.max_epoch,
            use_features=args.use_features,
            features_indices=args.features_indices,
            features_embedding_size=args.features_embedding_size,
            patience=args.early_stopping_patience,
            config_props=dict(
                max_char_length=args.max_char_length,
                char_input_mask_zero=args.char_input_mask_zero,
                char_input_dropout=args.char_input_dropout,
                char_lstm_dropout=args.char_lstm_dropout,
                additional_token_feature_indices=args.additional_token_feature_indices,
                text_feature_indices=args.text_feature_indices,
                unroll_text_feature_index=args.unroll_text_feature_index,
                concatenated_embeddings_token_count=args.concatenated_embeddings_token_count,
                use_word_embeddings=args.use_word_embeddings,
                use_features_indices_input=args.use_features_indices_input,
                continuous_features_indices=args.continuous_features_indices,
                features_lstm_units=args.features_lstm_units,
                stateful=args.stateful
            ),
            training_props=dict(
                initial_epoch=args.initial_epoch,
                input_window_stride=args.input_window_stride,
                checkpoint_epoch_interval=args.checkpoint_epoch_interval
            ),
            resume_train_model_path=args.resume_train_model_path,
            auto_resume=args.auto_resume,
            transfer_learning_config=get_transfer_learning_config_for_parsed_args(args),
            train_notification_manager=get_train_notification_manager(args),
            **self.get_common_args(args)
        )

    def run(self, args: argparse.Namespace):
        if args.save_input_to_and_exit:
            save_input_to(args.input, args.save_input_to_and_exit)
            return

        self.download_manager = DownloadManager()
        self.embedding_manager = EmbeddingManager(
            download_manager=self.download_manager
        )
        if args.no_use_lmdb:
            self.embedding_manager.disable_embedding_lmdb_cache()

        set_random_seeds(args.random_seed)

        self.do_run(args)

        # see https://github.com/tensorflow/tensorflow/issues/3388
        K.clear_session()


class TrainSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_train_arguments(parser)
        add_model_path_argument(parser, help='directory to the saved model')

    def do_run(self, args: argparse.Namespace):
        if not args.model:
            raise ValueError("model required")
        if args.preload_embedding:
            self.preload_and_validate_embedding(
                args.preload_embedding,
                use_word_embeddings=True
            )
        embedding_name = self.preload_and_validate_embedding(
            args.embedding,
            use_word_embeddings=args.use_word_embeddings and not args.resume_train_model_path
        )
        LOGGER.info('get_tf_info: %s', get_tf_info())
        train(
            embeddings_name=embedding_name,
            **self.get_train_args(args)
        )


class WapitiTrainSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_wapiti_train_arguments(parser)
        add_wapiti_install_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        if not args.model:
            raise ValueError("model required")
        wapiti_train(
            model_name=args.model,
            template_path=args.wapiti_template,
            input_paths=args.input,
            limit=args.limit,
            output_path=args.output,
            max_epoch=args.max_epoch,
            download_manager=self.download_manager,
            gzip_enabled=args.wapiti_gzip,
            wapiti_binary_path=install_wapiti_and_get_path_or_none(
                args.wapiti_install_source,
                download_manager=self.download_manager
            ),
            wapiti_train_args=get_wapiti_train_args(args),
            train_notification_manager=get_train_notification_manager(args)
        )


class TrainEvalSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_train_arguments(parser)
        add_model_path_argument(parser, help='directory to the saved model')
        add_fold_count_argument(parser)
        add_eval_input_arguments(parser)
        add_eval_output_arguments(parser)
        add_dl_eval_model_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        if not args.model:
            raise ValueError("model required")
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        if args.preload_embedding:
            self.preload_and_validate_embedding(
                args.preload_embedding,
                use_word_embeddings=True
            )
        embedding_name = self.preload_and_validate_embedding(
            args.embedding,
            use_word_embeddings=args.use_word_embeddings and not args.resume_train_model_path
        )
        LOGGER.info('get_tf_info: %s', get_tf_info())
        train_eval(
            fold_count=args.fold_count,
            embeddings_name=embedding_name,
            eval_output_args=get_eval_output_args(args),
            **get_eval_input_args(args),
            **get_dl_eval_model_args(args),
            **self.get_train_args(args)
        )


class WapitiTrainEvalSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_wapiti_train_arguments(parser)
        add_eval_input_arguments(parser)
        add_eval_output_arguments(parser)
        add_wapiti_install_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        if not args.model:
            raise ValueError("model required")
        wapiti_train_eval(
            model_name=args.model,
            template_path=args.wapiti_template,
            input_paths=args.input,
            limit=args.limit,
            eval_input_paths=args.eval_input,
            eval_limit=args.eval_limit,
            output_path=args.output,
            max_epoch=args.max_epoch,
            download_manager=self.download_manager,
            gzip_enabled=args.wapiti_gzip,
            wapiti_binary_path=install_wapiti_and_get_path_or_none(
                args.wapiti_install_source,
                download_manager=self.download_manager
            ),
            wapiti_train_args=get_wapiti_train_args(args),
            train_notification_manager=get_train_notification_manager(args),
            eval_output_args=get_eval_output_args(args)
        )


class EvalSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_model_path_argument(parser, required=True, help='directory to load the model from')
        parser.add_argument(
            "--use-eval-train-test-split",
            action="store_true",
            help=" ".join([
                "If enabled, split the input when running 'eval'",
                "(in the same way it is split for 'train_eval')"
            ])
        )
        add_eval_output_arguments(parser)
        add_stateful_argument(parser)
        add_dl_eval_model_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        eval_model(
            model_path=args.model_path,
            split_input=args.use_eval_train_test_split,
            eval_output_args=get_eval_output_args(args),
            stateful=args.stateful,
            **get_dl_eval_model_args(args),
            **self.get_common_args(args)
        )


class WapitiEvalSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_model_path_argument(parser, required=True, help='directory to load the model from')
        add_eval_output_arguments(parser)
        add_wapiti_install_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        wapiti_eval_model(
            model_path=args.model_path,
            input_paths=args.input,
            limit=args.limit,
            eval_output_args=get_eval_output_args(args),
            download_manager=self.download_manager,
            wapiti_binary_path=install_wapiti_and_get_path_or_none(
                args.wapiti_install_source,
                download_manager=self.download_manager
            )
        )


class TagSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser, max_sequence_length_default=None)
        add_stateful_argument(parser)
        add_input_window_stride_argument(parser)
        add_model_path_argument(parser, required=True, help='directory to load the model from')
        add_tag_output_format_argument(parser)
        add_tag_output_path_argument(parser)
        add_tag_transformed_argument(parser)

    def do_run(self, args: argparse.Namespace):
        tag_input(
            model_path=args.model_path,
            tag_output_format=args.tag_output_format,
            tag_output_path=args.tag_output_path,
            tag_transformed=args.tag_transformed,
            stateful=args.stateful,
            input_window_stride=args.input_window_stride,
            **self.get_common_args(args)
        )


class WapitiTagSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser, max_sequence_length_default=None)
        add_model_path_argument(parser, required=True, help='directory to load the model from')
        add_tag_output_format_argument(parser)
        add_tag_output_path_argument(parser)
        add_wapiti_install_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        wapiti_tag_input(
            model_path=args.model_path,
            tag_output_format=args.tag_output_format,
            tag_output_path=args.tag_output_path,
            input_paths=args.input,
            limit=args.limit,
            download_manager=self.download_manager,
            wapiti_binary_path=install_wapiti_and_get_path_or_none(
                args.wapiti_install_source,
                download_manager=self.download_manager
            )
        )


class InputInfoSubCommand(GrobidTrainerSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        print_input_info(
            input_paths=args.input,
            limit=args.limit,
            download_manager=self.download_manager
        )


SUB_COMMANDS = [
    TrainSubCommand(
        Tasks.TRAIN,
        'Train the model using the provided input(s)'
    ),
    TrainEvalSubCommand(
        Tasks.TRAIN_EVAL,
        'Train and reserve a slice of the input data for evaluation'
    ),
    EvalSubCommand(
        Tasks.EVAL,
        'Evaluate the already trained model on the provided input(s)'
    ),
    TagSubCommand(
        Tasks.TAG,
        'Tag inputs and show results. Optionally also show a diff to the expected labels'
    ),
    WapitiTrainSubCommand(
        Tasks.WAPITI_TRAIN,
        'Train the model using the provided input(s)'
    ),
    WapitiTrainEvalSubCommand(
        Tasks.WAPITI_TRAIN_EVAL,
        'Train and reserve a slice of the input data for evaluation'
    ),
    WapitiEvalSubCommand(
        Tasks.WAPITI_EVAL,
        'Evaluate the already trained model on the provided input(s)'
    ),
    WapitiTagSubCommand(
        Tasks.WAPITI_TAG,
        'Tag inputs and show results. Optionally also show a diff to the expected labels'
    ),
    InputInfoSubCommand(
        Tasks.INPUT_INFO,
        'Display input summary information relating to the passed in input(s)'
    )
]


def get_subcommand_processor():
    return SubCommandProcessor(SUB_COMMANDS, command_dest='action')


def parse_args(
    argv: Optional[Sequence[str]] = None,
    subcommand_processor: Optional[SubCommandProcessor] = None
):
    parser = create_argument_parser()
    if subcommand_processor is None:
        subcommand_processor = SubCommandProcessor(SUB_COMMANDS, command_dest='action')

    add_model_positional_argument(parser)
    subcommand_processor.add_sub_command_parsers(parser)

    args = parser.parse_args(argv)
    process_args(args)
    return args


def run(args: argparse.Namespace, subcommand_processor: Optional[SubCommandProcessor] = None):
    if subcommand_processor is None:
        subcommand_processor = SubCommandProcessor(SUB_COMMANDS, command_dest='action')
    try:
        subcommand_processor.run(args)
    except BaseException as exc:
        LOGGER.error('uncaught exception: %s', exc, exc_info=exc)
        raise


def main(argv: Optional[Sequence[str]] = None):
    subcommand_processor = get_subcommand_processor()
    args = parse_args(argv, subcommand_processor=subcommand_processor)
    if args.quiet:
        logging.root.setLevel('ERROR')
    elif args.debug:
        for name in [__name__, 'sciencebeam_trainer_delft', 'delft']:
            logging.getLogger(name).setLevel('DEBUG')
    if args.log_file:
        with auto_uploading_output_file(args.log_file, mode='w') as log_fp:
            try:
                with tee_stdout_and_stderr_lines_to(log_fp.write, append_line_feed=True):
                    with tee_logging_lines_to(log_fp.write, append_line_feed=True):
                        run(args, subcommand_processor=subcommand_processor)
            finally:
                logging.shutdown()
    else:
        run(args, subcommand_processor=subcommand_processor)


def main_setup():
    logging.root.handlers = []
    logging.basicConfig(level='INFO')
    patch_cloud_support()
    patch_get_model()


if __name__ == "__main__":
    main_setup()
    main()
