# initially based on:
# https://github.com/kermitt2/delft/blob/master/grobidTagger.py

import logging
import argparse
from argparse import _ActionsContainer as ArgParseActionsContainer
from typing import List, Optional

from sciencebeam_trainer_delft.utils.misc import parse_number_ranges

from sciencebeam_trainer_delft.sequence_labelling.utils.train_notify import (
    add_train_notification_arguments
)

from sciencebeam_trainer_delft.sequence_labelling.wrapper import (
    get_default_batch_size,
    get_default_stateful
)
from sciencebeam_trainer_delft.sequence_labelling.config import (
    DEFAULT_CHAR_INPUT_DROPOUT,
    DEFAULT_CHAR_LSTM_DROPOUT
)
from sciencebeam_trainer_delft.sequence_labelling.models import get_model_names

from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti import (
    DEFAULT_STOP_EPSILON_VALUE,
    DEFAULT_STOP_WINDOW_SIZE
)

from sciencebeam_trainer_delft.sequence_labelling.tag_formatter import (
    TagOutputFormats,
    TAG_OUTPUT_FORMATS
)

from sciencebeam_trainer_delft.sequence_labelling.evaluation import (
    EvaluationOutputFormats,
    EVALUATION_OUTPUT_FORMATS
)

from sciencebeam_trainer_delft.sequence_labelling.transfer_learning import (
    add_transfer_learning_arguments
)


LOGGER = logging.getLogger(__name__)


GROBID_MODEL_NAMES = [
    'affiliation-address', 'citation', 'date', 'figure', 'fulltext', 'header',
    'name', 'name-citation', 'name-header', 'patent', 'reference-segmenter',
    'segmentation', 'software', 'table'
]


DEFAULT_RANDOM_SEED = 42

DEFAULT_TAG_OUTPUT_FORMAT = TagOutputFormats.XML


def add_common_arguments(
    parser: argparse.ArgumentParser,
    max_sequence_length_default: Optional[int] = None
):
    input_group = parser.add_argument_group('input')
    input_group.add_argument(
        "--input",
        nargs='+',
        action='append',
        help="provided training file"
    )
    input_group.add_argument(
        "--shuffle-input",
        action="store_true",
        help="Shuffle the input before splitting"
    )
    input_group.add_argument(
        "--limit",
        type=int,
        help=(
            "limit the number of training samples."
            " With more than one input file, the limit will be applied to"
            " each of the input files individually"
        )
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Set the random seed for reproducibility"
    )
    parser.add_argument(
        "--batch-size", type=int, default=get_default_batch_size(),
        help="batch size"
    )
    parser.add_argument(
        "--max-sequence-length", type=int,
        default=max_sequence_length_default,
        help="maximum sequence length"
    )
    parser.add_argument(
        "--no-use-lmdb", action="store_true",
        help="Do not use LMDB embedding cache (load embeddings into memory instead)"
    )

    parser.add_argument("--multiprocessing", action="store_true", help="Use multiprocessing")

    parser.add_argument("--quiet", action="store_true", help="Only log errors")

    parser.add_argument(
        "--save-input-to-and-exit",
        help=(
            "If set, saves the input to the specified path and exits."
            " This can be useful to retrain the model outside GROBID."
        )
    )

    parser.add_argument(
        "--log-file",
        help=(
            "If set, saves the output to the specified log file."
            " This may also be a file in a bucket, in which case it will be uploaded at the end."
            " Add the .gz extension if you wish to compress the file."
        )
    )

    parser.add_argument("--job-dir", help="job dir (only used when running via ai platform)")


def add_model_path_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument("--model-path", **kwargs)


def add_fold_count_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument("--fold-count", type=int, default=1, **kwargs)


def add_eval_output_format_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--eval-output-format",
        choices=EVALUATION_OUTPUT_FORMATS,
        default=EvaluationOutputFormats.TEXT
    )


def add_eval_first_entity_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--eval-first-entity",
        action="store_true",
        help=''.join([
            'If set, additional evaluates the first entity (e.g. first_<author>).'
        ])
    )


def add_eval_output_path_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--eval-output-path",
        help='If specified, saves the evaluation to the specified path in the JSON format'
    )


def add_eval_output_arguments(parser: argparse.ArgumentParser):
    add_eval_output_format_argument(parser)
    add_eval_first_entity_argument(parser)
    add_eval_output_path_argument(parser)


def add_eval_input_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--eval-input",
        nargs='+',
        action='append',
        help=' '.join([
            "Evaluation data at the end of training. If not specified,",
            "it will use a slice of the training data"
        ])
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        help=' '.join([
            "Limit the number of documents to use for evaluation.",
            "This is mostly for testing to make evaluation faster."
        ])
    )


def add_dl_eval_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--eval-max-sequence-length",
        type=int,
        help=' '.join([
            "Maximum sequence length to use for evaluation.",
            "If not specified, no limit will be applied."
        ])
    )
    parser.add_argument(
        "--eval-input-window-stride",
        type=int,
        help="Should be equal or less than eval max sequence length"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        help=' '.join([
            "The batch size to be used for evaluation.",
            "If not specified, the training batch size is used.",
            "This may be useful to evaluate on longer sequences",
            "that could require a smaller batch size."
        ])
    )


def add_tag_output_format_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument(
        "--tag-output-format",
        default=DEFAULT_TAG_OUTPUT_FORMAT,
        choices=TAG_OUTPUT_FORMATS,
        help="output format for tag results",
        **kwargs
    )


def add_tag_output_path_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--tag-output-path",
        help='If specified, saves the tag result to the specified path'
    )


def add_tag_transformed_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--tag-transformed",
        action='store_true',
        help=(
            'If enabled, the output will contain the transformed dataset (if any).'
            ' More specifically, that will for example contain the "unrolled" data.'
        )
    )


def add_output_argument(parser: ArgParseActionsContainer, **kwargs):
    parser.add_argument("--output", help="directory where to save a trained model", **kwargs)


def add_max_epoch_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument(
        "--max-epoch", type=int, default=10,
        help="max epoch to train to",
        **kwargs
    )


def add_stateful_argument(parser: argparse.ArgumentParser, **kwargs):
    default_value = get_default_stateful()
    parser.add_argument(
        "--stateful",
        dest="stateful",
        default=default_value,
        action="store_true",
        help="Make RNNs stateful (required for truncated BPTT)",
        **kwargs
    )
    parser.add_argument(
        "--no-stateful",
        dest="stateful",
        default=default_value,
        action="store_false",
        help="Disable statefulness (default)",
        **kwargs
    )


def add_input_window_stride_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument(
        "--input-window-stride",
        type=int,
        help="Should be equal or less than max sequence length",
        **kwargs
    )


def add_train_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--architecture", default='BidLSTM_CRF',
        choices=get_model_names(),
        help="type of model architecture to be used"
    )

    parser.add_argument(
        "--max-char-length",
        type=int,
        default=30,
        help="The maximum number of chars used by the model"
    )

    parser.add_argument(
        "--additional-token-feature-indices",
        type=parse_number_ranges,
        help="".join([
            "Additional feature values that should be used as tokens.",
            " e.g. 0 or 0-3.",
            " If blank, no additional token features will be used."
        ])
    )

    parser.add_argument(
        "--text-feature-indices",
        type=parse_number_ranges,
        help="".join([
            "Feature values that should be treated as text input.",
            " e.g. 0 or 0-3.",
            " If blank, no additext features will be used.",
            " Cannot be used together with --additional-token-feature-indices.",
            " Text features will get tokenized."
            " If word embeddings are used, then the number of tokens will depend on",
            " --concatenated-embeddings-token-count.",
            " Tokens from text features replace regular tokens from the training data."
        ])
    )

    parser.add_argument(
        "--unroll-text-feature-index",
        type=int,
        help="".join([
            "Tokenizes the text at the specified index.",
            " Each token will become a separate token.",
            " The features will be duplicated for each token.",
            " Labels will also be duplicated for each token.",
            " Where a label refers to the beginning of a tag,",
            " this will only be used for the first token.",
            " All other labels will be the intermediate version of the tag."
            " The max sequence length will get applied to the unrolled tokens."
            " Additionally a new token will be added, with the values:"
            " LINESTART, LINEIN, LINEEND"
        ])
    )

    parser.add_argument(
        "--concatenated-embeddings-token-count",
        type=int,
        help="".join([
            "The number of tokens to concatenate as word embeddings.",
            " If not specified, it concatenate the main token with any",
            " --additional-token-feature-indices (if any).",
            " This option is mainly useful in combination with --text-feature-indices.",
            " It has no effect, if no word embeddings are used."
        ])
    )

    features_group = parser.add_argument_group('features')
    features_group.add_argument("--use-features", action="store_true", help="Use features")
    features_group.add_argument(
        "--features-indices", "--feature-indices",
        type=parse_number_ranges,
        help="The feature indices to use. e.g. 7-10. If blank, all of the features will be used."
    )
    features_group.add_argument(
        "--continuous-features-indices",
        type=parse_number_ranges,
        help=(
            "The feature indices to use that are continous. e.g. 7-10."
            " If blank, features will be assumed to be categorical."
        )
    )
    features_group.add_argument(
        "--features-embedding-size", "--feature-embedding-size",
        type=int,
        help="size of feature embedding, use 0 to disable embedding"
    )
    features_group.add_argument(
        "--use-features-indices-input", action="store_true",
        help="Use features indices values (should be inferred from the model)"
    )
    features_group.add_argument(
        "--features-lstm-units", type=int,
        help="Number of LSTM units used by the features"
    )

    add_stateful_argument(parser)
    add_input_window_stride_argument(parser)

    output_group = parser.add_argument_group('output')
    add_output_argument(output_group)
    output_group.add_argument("--checkpoint", help="directory where to save a checkpoint model")
    output_group.add_argument(
        "--checkpoint-epoch-interval",
        type=int,
        default=1,
        help="save checkpoints every n epochs"
    )

    parser.add_argument(
        "--embedding", default="glove-6B-50d",
        help="name of word embedding"
    )
    parser.add_argument(
        "--preload-embedding",
        help=" ".join([
            "Name or URL to embedding to preload.",
            "This can be useful in combination with resuming model training."
        ])
    )
    features_group.add_argument(
        "--no-embedding",
        dest="use_word_embeddings",
        default=True,
        action="store_false",
        help="Disable the use of word embedding"
    )
    parser.add_argument(
        "--char-embedding-size", type=int, default=25,
        help="size of char embedding"
    )
    parser.add_argument(
        "--char-lstm-units", type=int, default=25,
        help="number of list units for chars"
    )
    parser.add_argument(
        "--char-input-mask-zero", action='store_true',
        help="enables masking of zero for the char input"
    )
    parser.add_argument(
        "--char-input-dropout", type=float, default=DEFAULT_CHAR_INPUT_DROPOUT,
        help="dropout for char input"
    )
    parser.add_argument(
        "--char-lstm-dropout", type=float, default=DEFAULT_CHAR_LSTM_DROPOUT,
        help="dropout for char lstm"
    )
    parser.add_argument(
        "--word-lstm-units", type=int, default=100,
        help="number of lstm units for words"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5,
        help="main dropout"
    )
    parser.add_argument(
        "--recurrent-dropout", type=float, default=0.5,
        help="recurrent dropout"
    )
    add_max_epoch_argument(parser)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=10,
        help="how many epochs to continue training after the f1 score hasn't improved"
    )
    parser.add_argument(
        "--resume-train-model-path",
        help="path to the model training should be resumed from (e.g. path to checkpoint)"
    )
    parser.add_argument(
        "--initial-epoch",
        type=int,
        default=0,
        help="Sets the initial epoch for model training."
    )
    parser.add_argument(
        "--auto-resume", action='store_true',
        help="enables auto-resuming training using checkpoints"
    )
    add_transfer_learning_arguments(parser)
    add_train_notification_arguments(parser)


def add_wapiti_train_arguments(parser: argparse.ArgumentParser):
    add_output_argument(parser)
    add_max_epoch_argument(parser)
    parser.add_argument("--wapiti-template", required=True)
    parser.add_argument(
        "--wapiti-gzip",
        action="store_true",
        help="whether to gzip wapiti models before saving"
    )
    parser.add_argument(
        "--wapiti-stop-epsilon-value",
        default=DEFAULT_STOP_EPSILON_VALUE
    )
    parser.add_argument(
        "--wapiti-stop-window-size",
        type=int,
        default=DEFAULT_STOP_WINDOW_SIZE
    )
    add_train_notification_arguments(parser)


def get_wapiti_train_args(args: argparse.Namespace) -> dict:
    return dict(
        stop_epsilon_value=args.wapiti_stop_epsilon_value,
        stop_window_size=args.wapiti_stop_window_size
    )


def add_wapiti_install_arguments(parser: argparse.ArgumentParser):
    example_url = "https://github.com/kermitt2/Wapiti/archive/master.tar.gz"
    parser.add_argument(
        "--wapiti-install-source",
        help="source file to install wapiti from, e.g. %s" % example_url
    )


def add_all_non_positional_arguments(parser: argparse.ArgumentParser):
    add_common_arguments(parser)
    add_train_arguments(parser)


def add_model_positional_argument(parser: argparse.ArgumentParser):
    parser.add_argument("model", nargs='?', choices=GROBID_MODEL_NAMES)


def _flatten_input_paths(input_paths_list: List[List[str]]) -> List[str]:
    if not input_paths_list:
        return []
    return [input_path for input_paths in input_paths_list for input_path in input_paths]


def process_args(args: argparse.Namespace) -> None:
    args.input = _flatten_input_paths(args.input)
    try:
        args.eval_input = _flatten_input_paths(args.eval_input)
    except AttributeError:
        pass


def create_argument_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Trainer for GROBID models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
