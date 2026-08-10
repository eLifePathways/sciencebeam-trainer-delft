import logging
from pathlib import Path
import re
from typing import Any, Sequence, TypedDict, Union
import xml.etree.ElementTree as ET

# from typing_extensions import NotRequired

import pytest
import yaml

from sciencebeam_trainer_delft.sequence_labelling.tools.grobid_trainer.utils import (
    tag_input,
    wapiti_tag_input
)
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager

from tests.test_utils import log_on_exception


LOGGER = logging.getLogger(__name__)


def load_test_data(filepath: Union[str, Path]) -> dict:
    with open(filepath, encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_test_cases(
    test_data: dict,
    test_name: str,
    id_key: str = 'id'
) -> Sequence[Any]:  # -> Sequence[ParameterSet]:
    return [
        pytest.param(tc, id=tc.get(id_key))
        for tc in test_data[test_name]
    ]


TEST_DATA = load_test_data(
    Path(__file__).parent / 'tag_using_existing_models_test.yaml'
)


class ExpectedTagTypedDict(TypedDict):
    tag: str
    regex: str


class TagUsingExistingModelsTestCaseTypedDict(TypedDict):
    id: str
    model_path: str
    input_path: str
    expected_tags: Sequence[ExpectedTagTypedDict]
    # engine: NotRequired[Literal['wapiti', 'delft']]


@pytest.fixture(name='download_manager', scope='session')
def _download_manager() -> DownloadManager:
    return DownloadManager()


@pytest.mark.slow
@pytest.mark.very_slow
class TestTagUsingExistingModels:

    @pytest.mark.parametrize(
        'test_case',
        get_test_cases(TEST_DATA, 'test_tag_using_existing_model')
    )
    @log_on_exception
    def test_tag_using_existing_model(
        self,
        test_case: TagUsingExistingModelsTestCaseTypedDict,
        download_manager: DownloadManager,
        tmp_path: Path
    ):
        model_path = test_case['model_path']
        input_path = test_case['input_path']
        engine = test_case.get('engine', 'delft')
        LOGGER.info('testing model: %s with input: %s', model_path, input_path)
        tag_output_path = tmp_path / 'tagged_output.xml'
        if engine == 'wapiti':
            wapiti_tag_input(
                model_path=model_path,
                input_paths=[input_path],
                download_manager=download_manager,
                limit=1,
                tag_output_path=str(tag_output_path),
                tag_output_format='xml'
            )
        else:
            tag_input(
                model_name='dummy-model-name',
                model_path=model_path,
                input_paths=[input_path],
                download_manager=download_manager,
                limit=1,
                tag_output_path=str(tag_output_path),
                tag_output_format='xml'
            )
        tag_output_xml_str = tag_output_path.read_text(encoding='utf-8')
        assert tag_output_xml_str
        LOGGER.info('tagged output: %s', tag_output_xml_str[:500])
        tag_output_xml_root = ET.fromstring(tag_output_xml_str)
        for expected_tag in test_case['expected_tags']:
            tag = expected_tag['tag']
            regex = expected_tag['regex']
            element = tag_output_xml_root.find(f'.//{tag}')
            assert element is not None, f'expected tag <{tag}> to be present in output'
            value = element.text
            assert value is not None, f'expected tag <{tag}> to have text content'
            LOGGER.debug('checking tag %r using regex %r, value: %r', tag, regex, value)
            m = re.match(regex, value)
            if not m:
                raise AssertionError(
                    f'expected tag <{tag}> to match regex'
                    f' {repr(regex)} but got value: {repr(value)}'
                )
            LOGGER.debug('check passed for tag %r', tag)
        LOGGER.info('test passed for model: %s with input: %s', model_path, input_path)
