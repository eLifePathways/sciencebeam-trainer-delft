import gzip
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from sciencebeam_trainer_delft.sequence_labelling.wrapper import (
    EnvironmentVariables
)
from sciencebeam_trainer_delft.sequence_labelling.tools.grobid_trainer.cli import (
    parse_args,
    main
)

from ....test_utils import log_on_exception


LOGGER = logging.getLogger(__name__)


DELFT_MODELS_ARE_KERAS_REASON = (
    'the published delft models are Keras hdf5 and there is no TensorFlow to'
    ' load them with; spec 002 converts them to PyTorch and restores this case'
)

INPUT_PATH_1 = '/path/to/dataset1'
INPUT_PATH_2 = '/path/to/dataset2'

GROBID_HEADER_MODEL_URL = (
    'https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/'
    'delft-grobid-header-biorxiv-no-word-embedding-2020-05-05.tar.gz'
)

GROBID_HEADER_TEST_DATA_URL = (
    'https://github.com/elifesciences/sciencebeam-datasets/releases/download/'
    'grobid-0.6.1/delft-grobid-0.6.1-header.test.gz'
)

GROBID_HEADER_TEST_DATA_TITLE_1 = (
    'Projections : A Preliminary Performance Tool for Charm'
)


class TestGrobidTrainer:
    class TestParseArgs:
        def test_should_require_arguments(self):
            with pytest.raises(SystemExit):
                parse_args([])

        def test_should_allow_multiple_input_files_via_single_input_param(self):
            opt = parse_args([
                'header',
                'train',
                '--input', '/path/to/dataset1', '/path/to/dataset2'
            ])
            assert opt.input == ['/path/to/dataset1', '/path/to/dataset2']

        def test_should_allow_multiple_input_files_via_multiple_input_params(self):
            opt = parse_args([
                'header',
                'train',
                '--input', INPUT_PATH_1,
                '--input', INPUT_PATH_2
            ])
            assert opt.input == [INPUT_PATH_1, INPUT_PATH_2]

        def test_should_use_stateful_env_variable_true_by_default(self, env_mock):
            env_mock[EnvironmentVariables.STATEFUL] = 'true'
            opt = parse_args([
                'tag',
                '--input', INPUT_PATH_1,
                '--model-path', INPUT_PATH_2
            ])
            assert opt.stateful is True

        def test_should_use_stateful_env_variable_false_by_default(self, env_mock):
            env_mock[EnvironmentVariables.STATEFUL] = 'false'
            opt = parse_args([
                'tag',
                '--input', INPUT_PATH_1,
                '--model-path', INPUT_PATH_2
            ])
            assert opt.stateful is False

        def test_should_fallback_to_none_statefulness(self, env_mock):
            env_mock[EnvironmentVariables.STATEFUL] = ''
            opt = parse_args([
                'tag',
                '--input', INPUT_PATH_1,
                '--model-path', INPUT_PATH_2
            ])
            assert opt.stateful is None

        def test_should_allow_to_set_stateful(self, env_mock):
            env_mock[EnvironmentVariables.STATEFUL] = 'false'
            opt = parse_args([
                'tag',
                '--input', INPUT_PATH_1,
                '--model-path', INPUT_PATH_2,
                '--stateful'
            ])
            assert opt.stateful is True

        def test_should_allow_to_unset_stateful(self, env_mock):
            env_mock[EnvironmentVariables.STATEFUL] = 'true'
            opt = parse_args([
                'tag',
                '--input', INPUT_PATH_1,
                '--model-path', INPUT_PATH_2,
                '--no-stateful'
            ])
            assert opt.stateful is False

    @pytest.mark.slow
    class TestEndToEndMain:
        @log_on_exception
        def test_should_be_able_capture_train_input_data(
                self, temp_dir: Path):
            input_path = temp_dir.joinpath('input.train')
            input_path.write_text('some training data')

            output_path = temp_dir.joinpath('captured-input.train')

            main([
                'header',
                'train',
                f'--input={input_path}',
                f'--save-input-to-and-exit={output_path}'
            ])

            assert output_path.read_text() == 'some training data'

        @log_on_exception
        def _test_should_be_able_capture_train_input_data_gzipped(
                self, temp_dir: Path):
            input_path = temp_dir.joinpath('input.train')
            input_path.write_text('some training data')

            output_path = temp_dir.joinpath('captured-input.train.gz')

            main([
                'header',
                'train',
                f'--input={input_path}',
                f'--save-input-to-and-exit={output_path}'
            ])

            with gzip.open(str(output_path), mode='rb') as fp:
                assert fp.read() == 'some training data'

        @pytest.mark.skip(reason=DELFT_MODELS_ARE_KERAS_REASON)
        @log_on_exception
        def test_should_be_able_tag_using_existing_grobid_model(
                self, capsys):
            main([
                'tag',
                f'--input={GROBID_HEADER_TEST_DATA_URL}',
                f'--model-path={GROBID_HEADER_MODEL_URL}',
                '--limit=1',
                '--tag-output-format=xml'
            ])
            captured = capsys.readouterr()
            output_text = captured.out
            LOGGER.debug('output_text: %r', output_text)
            assert output_text
            root = ET.fromstring(output_text)
            title = ' '.join(node.text for node in root.findall('.//title'))
            assert title == GROBID_HEADER_TEST_DATA_TITLE_1

        @pytest.mark.skip(reason=DELFT_MODELS_ARE_KERAS_REASON)
        @log_on_exception
        def test_should_be_able_eval_using_existing_grobid_model(
                self, temp_dir: Path):
            eval_output_path = temp_dir / 'eval.json'
            main([
                'eval',
                f'--input={GROBID_HEADER_TEST_DATA_URL}',
                f'--model-path={GROBID_HEADER_MODEL_URL}',
                '--limit=100',
                '--eval-output-format=json',
                f'--eval-output-path={eval_output_path}'
            ])
            eval_data = json.loads(eval_output_path.read_text())
            assert eval_data['scores']['<title>']['f1'] >= 0.5
