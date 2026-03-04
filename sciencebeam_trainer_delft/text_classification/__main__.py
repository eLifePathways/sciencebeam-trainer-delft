import argparse
import logging
import json
from abc import abstractmethod
from typing import List, Optional, Sequence

import sciencebeam_trainer_delft.utils.no_warn_if_disabled  # noqa, pylint: disable=unused-import
import sciencebeam_trainer_delft.utils.no_keras_backend_message  # noqa, pylint: disable=unused-import
# pylint: disable=wrong-import-order, ungrouped-imports

import tf_keras.backend as K

import pandas as pd

from sciencebeam_trainer_delft.utils.cli import (
    SubCommand,
    SubCommandProcessor
)

from sciencebeam_trainer_delft.utils.io import (
    auto_uploading_output_file
)
from sciencebeam_trainer_delft.utils.logging import (
    tee_stdout_and_stderr_lines_to,
    tee_logging_lines_to
)

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.embedding import EmbeddingManager

from sciencebeam_trainer_delft.text_classification.config import (
    AppConfig,
    ModelConfig,
    TrainingConfig
)
from sciencebeam_trainer_delft.text_classification.reader import (
    save_data_frame,
    get_texts_and_classes_from_data_frame
)
from sciencebeam_trainer_delft.text_classification.cli_utils import (
    load_input_data_frame,
    load_input_data,
    load_label_data,
    train,
    predict,
    evaluate
)


LOGGER = logging.getLogger(__name__)


DEFAULT_MODEL_PATH = 'data/models/textClassification/toxic'
DEFAULT_EMBEDDNGS_NAME = 'glove.6B.50d'


def add_common_arguments(
        parser: argparse.ArgumentParser):
    parser.add_argument("--quiet", action="store_true", help="Only log errors")
    parser.add_argument(
        "--model-path",
        required=True
    )
    parser.add_argument(
        "--embeddings",
        default=DEFAULT_EMBEDDNGS_NAME
    )
    parser.add_argument(
        "--embedding",
        dest="embeddings",
        help="Alias for --embeddings"
    )
    parser.add_argument(
        "--preload-embedding",
        help=" ".join([
            "Name or URL to embedding to preload.",
            "This can be useful in combination with resuming model training."
        ])
    )
    parser.add_argument(
        "--no-use-lmdb", action="store_true",
        help="Do not use LMDB embedding cache (load embeddings into memory instead)"
    )
    parser.add_argument(
        "--log-file",
        help=(
            "If set, saves the output to the specified log file."
            " This may also be a file in a bucket, in which case it will be uploaded at the end."
            " Add the .gz extension if you wish to compress the file."
        )
    )
    parser.add_argument(
        "--job-dir",
        help="job dir (only used when running via ai platform)"
    )


def add_train_arguments(
        parser: argparse.ArgumentParser):
    train_group = parser.add_argument_group('train')
    train_group.add_argument(
        "--train-input",
        nargs='+',
        default=[],
        action='append',
        required=True,
        help="provided training file"
    )
    train_group.add_argument(
        "--train-input-limit",
        type=int,
        help=(
            "limit the number of training samples."
            " With more than one input file, the limit will be applied to"
            " each of the input files individually"
        )
    )
    train_group.add_argument(
        "--architecture",
        default='bidLstm_simple',
        help="The desired architecture"
    )
    train_group.add_argument(
        "--max-epoch",
        type=int,
        default=100,
        help="max epoch to train to"
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="batch size"
    )
    train_group.add_argument(
        "--checkpoint",
        help="directory where to save a checkpoint model"
    )


def add_predict_arguments(
        parser: argparse.ArgumentParser):
    predict_group = parser.add_argument_group('predict')
    predict_group.add_argument(
        "--predict-input",
        nargs='+',
        required=True,
        action='append',
        help="provided predict file"
    )
    predict_group.add_argument(
        "--predict-input-limit",
        type=int,
        help=(
            "limit the number of predict samples."
            " With more than one input file, the limit will be applied to"
            " each of the input files individually"
        )
    )
    predict_group.add_argument(
        "--predict-output",
        help="save output as csv / tsv to"
    )


def add_eval_arguments(
        parser: argparse.ArgumentParser):
    eval_group = parser.add_argument_group('eval')
    eval_group.add_argument(
        "--eval-input",
        nargs='+',
        required=True,
        action='append',
        help="provided evaluation file"
    )
    eval_group.add_argument(
        "--eval-label-input",
        nargs='+',
        required=False,
        action='append',
        help="provided separate evaluation label file"
    )
    eval_group.add_argument(
        "--eval-input-limit",
        type=int,
        help=(
            "limit the number of evaluation samples."
            " With more than one input file, the limit will be applied to"
            " each of the input files individually"
        )
    )


def _flatten_input_paths(input_paths_list: List[List[str]]) -> List[str]:
    if not input_paths_list:
        return []
    return [input_path for input_paths in input_paths_list for input_path in input_paths]


class SubCommandNames:
    TRAIN = 'train'
    EVAL = 'eval'
    TRAIN_EVAL = 'train_eval'
    PREDICT = 'predict'


class BaseSubCommand(SubCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.download_manager = None
        self.embedding_manager = None
        self.app_config = None

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

    def run(self, args: argparse.Namespace):
        self.download_manager = DownloadManager()
        self.embedding_manager = EmbeddingManager(
            download_manager=self.download_manager
        )
        self.app_config = AppConfig(
            download_manager=self.download_manager,
            embedding_manager=self.embedding_manager
        )
        if args.no_use_lmdb:
            self.embedding_manager.disable_embedding_lmdb_cache()

        if args.preload_embedding:
            self.preload_and_validate_embedding(
                args.preload_embedding,
                use_word_embeddings=True
            )

        self.do_run(args)

        # see https://github.com/tensorflow/tensorflow/issues/3388
        K.clear_session()


class TrainSubCommand(BaseSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_train_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        LOGGER.info('train')
        download_manager = DownloadManager()
        train_input_paths = _flatten_input_paths(args.train_input)
        train_input_texts, train_input_labels, list_classes = load_input_data(
            train_input_paths,
            download_manager=download_manager,
            limit=args.train_input_limit
        )
        LOGGER.info('list_classes: %s', list_classes)
        embedding_name = self.preload_and_validate_embedding(
            args.embeddings,
            use_word_embeddings=True
        )
        train(
            app_config=self.app_config,
            model_config=ModelConfig(
                embeddings_name=embedding_name,
                architecture=args.architecture,
                list_classes=list_classes
            ),
            training_config=TrainingConfig(
                batch_size=args.batch_size,
                max_epoch=args.max_epoch,
                log_dir=args.checkpoint
            ),
            train_input_texts=train_input_texts,
            train_input_labels=train_input_labels,
            model_path=args.model_path
        )


class EvalSubCommand(BaseSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_eval_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        LOGGER.info('eval')
        download_manager = DownloadManager()
        eval_input_paths = _flatten_input_paths(args.eval_input)
        eval_label_input_paths = _flatten_input_paths(args.eval_label_input)
        eval_input_texts, eval_input_labels, list_classes = load_input_data(
            eval_input_paths,
            download_manager=download_manager,
            limit=args.eval_input_limit
        )
        if eval_label_input_paths:
            eval_input_labels, _ = load_label_data(
                eval_label_input_paths,
                download_manager=download_manager,
                limit=args.eval_input_limit
            )
        LOGGER.info('list_classes: %s', list_classes)
        result = evaluate(
            app_config=self.app_config,
            eval_input_texts=eval_input_texts,
            eval_input_labels=eval_input_labels,
            model_path=args.model_path
        )
        print(result.text_formatted_report)


class TrainEvalSubCommand(BaseSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_train_arguments(parser)
        add_eval_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        LOGGER.info('train eval')
        download_manager = DownloadManager()

        train_input_paths = _flatten_input_paths(args.train_input)
        train_input_texts, train_input_labels, list_classes = load_input_data(
            train_input_paths,
            download_manager=download_manager,
            limit=args.train_input_limit
        )

        eval_input_paths = _flatten_input_paths(args.eval_input)
        eval_label_input_paths = _flatten_input_paths(args.eval_label_input)
        eval_input_texts, eval_input_labels, _ = load_input_data(
            eval_input_paths,
            download_manager=download_manager,
            limit=args.eval_input_limit
        )
        if eval_label_input_paths:
            eval_input_labels, _ = load_label_data(
                eval_label_input_paths,
                download_manager=download_manager,
                limit=args.eval_input_limit
            )

        LOGGER.info('list_classes: %s', list_classes)

        embedding_name = self.preload_and_validate_embedding(
            args.embeddings,
            use_word_embeddings=True
        )

        train(
            app_config=self.app_config,
            model_config=ModelConfig(
                embeddings_name=embedding_name,
                architecture=args.architecture,
                list_classes=list_classes
            ),
            training_config=TrainingConfig(
                batch_size=args.batch_size,
                max_epoch=args.max_epoch,
                log_dir=args.checkpoint
            ),
            train_input_texts=train_input_texts,
            train_input_labels=train_input_labels,
            model_path=args.model_path
        )

        result = evaluate(
            app_config=self.app_config,
            eval_input_texts=eval_input_texts,
            eval_input_labels=eval_input_labels,
            model_path=args.model_path
        )
        print(result.text_formatted_report)


class PredictSubCommand(BaseSubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_predict_arguments(parser)

    def do_run(self, args: argparse.Namespace):
        LOGGER.info('train')
        download_manager = DownloadManager()
        predict_input_paths = _flatten_input_paths(args.predict_input)
        predict_df = load_input_data_frame(
            predict_input_paths,
            download_manager=download_manager,
            limit=args.predict_input_limit
        )
        predict_input_texts, _, _ = get_texts_and_classes_from_data_frame(
            predict_df
        )
        result = predict(
            app_config=self.app_config,
            eval_input_texts=predict_input_texts,
            model_path=args.model_path
        )
        list_classes = result['labels']
        prediction = result['prediction']
        LOGGER.info('list_classes: %s', list_classes)
        result_df = pd.concat([
            predict_df[predict_df.columns[:2]],
            pd.DataFrame(
                prediction,
                columns=list_classes,
                index=predict_df.index
            )
        ], axis=1)
        if args.predict_output:
            LOGGER.info('writing output to: %s', args.predict_output)
            save_data_frame(result_df, args.predict_output)
        else:
            print(json.dumps(
                result_df.to_dict(orient='records'),
                indent=2
            ))


SUB_COMMANDS = [
    TrainSubCommand(
        SubCommandNames.TRAIN,
        'Train the model using the provided input(s)'
    ),
    EvalSubCommand(
        SubCommandNames.EVAL,
        'Evaluate the model using the provided input(s)'
    ),
    TrainEvalSubCommand(
        SubCommandNames.TRAIN_EVAL,
        'Train and then evaluate the model using the provided input(s)'
    ),
    PredictSubCommand(
        SubCommandNames.PREDICT,
        'Predict the model using the provided input(s)'
    ),
]


def create_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Trainer for GROBID models"
    )


def get_subcommand_processor():
    return SubCommandProcessor(SUB_COMMANDS, command_dest='command')


def parse_args(
    argv: Optional[Sequence[str]] = None,
    subcommand_processor: Optional[SubCommandProcessor] = None
):
    parser = create_parser()
    if subcommand_processor is None:
        subcommand_processor = SubCommandProcessor(SUB_COMMANDS, command_dest='command')

    subcommand_processor.add_sub_command_parsers(parser)

    args = parser.parse_args(argv)
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


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()
