"""Writes a converted copy of a TF-era model directory.

Loading a TF-era directory already converts it (see
``sequence_labelling.tf_weight_conversion``), so nothing needs this to use one.
It exists for doing the work once rather than on every load, and is a thin
wrapper over that same mapping: it loads the model exactly as inference does and
saves what it gets, so there is no second conversion to keep in step.

Usage::

    python -m sciencebeam_trainer_delft.sequence_labelling.tools.convert_tf_model \\
        --source-model-path=path/or/url/to/model \\
        --output-path=path/to/converted/model
"""
import argparse
import logging
import os
from typing import Optional, Sequence

from sciencebeam_trainer_delft.utils.cli import (
    add_default_arguments,
    initialize_and_call_main,
    process_default_args
)
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager

from sciencebeam_trainer_delft.sequence_labelling.models import get_model
from sciencebeam_trainer_delft.sequence_labelling.saving import ModelLoader, ModelSaver
from sciencebeam_trainer_delft.sequence_labelling.wrapper import get_vocab_size


LOGGER = logging.getLogger(__name__)


def convert_model(
    source_model_path: str,
    output_path: str,
    download_manager: Optional[DownloadManager] = None
) -> None:
    if download_manager is None:
        download_manager = DownloadManager()
    model_loader = ModelLoader(download_manager=download_manager)
    local_source_path = model_loader.download_model(source_model_path)
    if os.path.abspath(local_source_path) == os.path.abspath(output_path):
        raise ValueError(
            'the output path must differ from the source: conversion writes a'
            ' new directory rather than replacing the model it read'
        )

    model_config = model_loader.load_model_config_from_directory(local_source_path)
    preprocessor = model_loader.load_preprocessor_from_directory(local_source_path)
    ntags = get_vocab_size(preprocessor.vocab_tag)
    LOGGER.info(
        'loading %s: architecture=%s, ntags=%d',
        local_source_path, model_config.architecture, ntags
    )

    # the same call inference makes, so a weight file in either format is read
    # by the same code path and no second mapping exists to diverge from it
    model = get_model(model_config, preprocessor, ntags=ntags)
    model_loader.load_model_from_directory(local_source_path, model=model)

    meta = {
        'converted_from': source_model_path,
        'architecture': model_config.architecture
    }
    ModelSaver(preprocessor=preprocessor, model_config=model_config).save_to(
        output_path, model=model, meta=meta
    )
    LOGGER.info('converted model written to %s', output_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Write a converted copy of a TF-era model directory'
    )
    parser.add_argument(
        '--source-model-path', required=True,
        help='the model directory or .tar.gz to read, local or a URL'
    )
    parser.add_argument(
        '--output-path', required=True,
        help='the directory to write the converted model to'
    )
    add_default_arguments(parser)
    return parser.parse_args(list(argv) if argv is not None else None)


def run(args: argparse.Namespace) -> None:
    convert_model(
        source_model_path=args.source_model_path,
        output_path=args.output_path
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    process_default_args(args)
    run(args)


if __name__ == '__main__':
    initialize_and_call_main(main)
