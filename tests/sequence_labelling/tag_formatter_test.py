
import json
import logging
from typing import Iterable

import numpy as np

from sciencebeam_trainer_delft.sequence_labelling.tag_formatter import (
    TagOutputFormats,
    TagLabelFormats,
    get_tag_result,
    get_xml_tag_for_annotation_label,
    translate_tag_label_iob_to_grobid,
    format_tag_result as _format_tag_result
)


LOGGER = logging.getLogger(__name__)


TEXTS_1 = np.array([
    ['token1', 'token2']
])

FEATURES_1 = np.array([
    [['feat1.1', 'feat1.2'], ['feat2.1', 'feat2.2']]
])

ANNOTATIONS_1 = [
    [['token1', 'B-tag1'], ['token2', 'B-tag2']]
]

DATA_LINES_1 = [
    'token1 feat1.1 feat1.2 B-tag1',
    'token2 feat2.1 feat2.2 B-tag2'
]

FLAT_TEXT_1 = 'token1 token2'

XML_1 = '\n'.join([
    '<xml>',
    '  <p>',
    '    <tag1>token1</tag1>',
    '    <tag2>token2</tag2>',
    '  </p>',
    '</xml>'
])

MODEL_1 = 'model1'


def format_tag_result(*args, **kwargs):
    result = _format_tag_result(*args, **kwargs)
    LOGGER.debug('result: %s', result)
    return result


def to_iterable(some_list: list) -> Iterable:
    return (value for value in some_list)


class TestGetTagResult:
    def test_should_combine_text_with_labels(self):
        assert get_tag_result(
            texts=[['token1', 'token2']],
            labels=[['label1', 'label2']]
        ) == [
            [('token1', 'label1'), ('token2', 'label2')]
        ]


class TestGetXmlTagForAnnotationLabel:
    def test_should_remove_greater_less_signs(self):
        assert get_xml_tag_for_annotation_label('<tag>') == 'tag'

    def test_should_remove_prefix(self):
        assert get_xml_tag_for_annotation_label('B-<tag>') == 'tag'


class TestTranslateTagLabelIobToGrobid:
    def test_should_convert_begin_tag(self):
        assert translate_tag_label_iob_to_grobid('B-header') == 'I-header'

    def test_should_convert_inside_tag(self):
        assert translate_tag_label_iob_to_grobid('I-header') == 'header'

    def test_should_convert_outside_tag(self):
        assert translate_tag_label_iob_to_grobid('O') == '<other>'

    def test_should_pass_through_unknown_tag(self):
        assert translate_tag_label_iob_to_grobid('<other>') == '<other>'


class TestFormatTagResult:
    def test_should_handle_numpy_arrays_in_text(self):
        tag_result = {
            'texts': [{'text': np.array(['token1', 'token2', 'token3'])}]
        }
        result = json.loads(format_tag_result(
            tag_result, output_format=TagOutputFormats.JSON
        ))
        assert result['texts'][0]['text'] == ['token1', 'token2', 'token3']

    def test_should_format_tag_list_result_as_json(self):
        result = json.loads(format_tag_result(
            tag_result=to_iterable(ANNOTATIONS_1),
            output_format=TagOutputFormats.JSON,
            texts=TEXTS_1,
            features=FEATURES_1,
            model_name=MODEL_1
        ))
        assert result['model'] == MODEL_1
        assert result['texts'] == TEXTS_1.tolist()
        assert result['features'] == FEATURES_1.tolist()
        assert result['annotations'] == ANNOTATIONS_1

    def test_should_format_tag_list_result_as_data(self):
        result = format_tag_result(
            tag_result=to_iterable(ANNOTATIONS_1),
            output_format=TagOutputFormats.DATA,
            texts=TEXTS_1,
            features=FEATURES_1,
            model_name=MODEL_1
        )
        assert result.splitlines() == DATA_LINES_1

    def test_should_separate_document_data_lines_using_blank_line(self):
        result = format_tag_result(
            tag_result=to_iterable(ANNOTATIONS_1 * 2),
            output_format=TagOutputFormats.DATA,
            texts=np.asarray(TEXTS_1.tolist() * 2),
            features=np.asarray(FEATURES_1.tolist() * 2),
            model_name=MODEL_1
        )
        assert result.splitlines() == DATA_LINES_1 + [''] + DATA_LINES_1

    def test_should_format_tag_list_result_as_data_with_grobid_label_format(self):
        result = format_tag_result(
            tag_result=to_iterable(
                [[('token1', 'B-header'), ('token2', 'I-header'), ('token3', 'O')]]
            ),
            output_format=TagOutputFormats.DATA,
            features=np.array([[['f1'], ['f2'], ['f3']]]),
            label_format=TagLabelFormats.GROBID
        )
        assert result.splitlines() == [
            'token1 f1 I-header',
            'token2 f2 header',
            'token3 f3 <other>'
        ]

    def test_should_format_tag_list_result_as_data_unidiff_with_grobid_label_format(self):
        result = format_tag_result(
            tag_result=to_iterable(
                [[('token1', 'B-tag1'), ('token2', 'I-tag1'), ('token3', 'B-tag2')]]
            ),
            expected_tag_result=[
                [('token1', 'B-tag1'), ('token2', 'I-tag1'), ('token3', 'B-tag3')]
            ],
            features=np.array([[['f1'], ['f2'], ['f3']]]),
            output_format=TagOutputFormats.DATA_UNIDIFF,
            label_format=TagLabelFormats.GROBID
        )
        assert result.splitlines() == [
            '--- document_000001.expected',
            '+++ document_000001.actual',
            '@@ -1,3 +1,3 @@',
            ' token1 f1 I-tag1',
            ' token2 f2 tag1',
            '-token3 f3 I-tag3',
            '+token3 f3 I-tag2'
        ]

    def test_should_format_tag_list_result_as_data_unidiff_and_combined_tags(self):
        result = format_tag_result(
            tag_result=to_iterable(
                [[['token1', 'B-tag1'], ['token2', 'I-tag1'], ['token3', 'B-tag2']]]
            ),
            expected_tag_result=[
                [['token1', 'B-tag1'], ['token2', 'I-tag1'], ['token3', 'B-tag3']]
            ],
            features=np.array([
                [['feat1.1', 'feat1.2'], ['feat2.1', 'feat2.2'], ['feat3.1', 'feat3.2']]
            ]),
            output_format=TagOutputFormats.DATA_UNIDIFF
        )
        LOGGER.debug('result:\n%s', result)
        assert result.splitlines() == [
            '--- document_000001.expected',
            '+++ document_000001.actual',
            '@@ -1,3 +1,3 @@',
            ' token1 feat1.1 feat1.2 B-tag1',
            ' token2 feat2.1 feat2.2 I-tag1',
            '-token3 feat3.1 feat3.2 B-tag3',
            '+token3 feat3.1 feat3.2 B-tag2'
        ]

    def test_should_format_tag_data_unidiff_with_multiple_changes(self):
        result = format_tag_result(
            tag_result=to_iterable([
                [['token1.1', 'B-tag1'], ['token1.2', 'I-tag1'], ['token1.3', 'B-tag2']],
                [['token2.1', 'B-tag1'], ['token2.2', 'I-tag1'], ['token2.3', 'B-tag2']]
            ]),
            expected_tag_result=[
                [['token1.1', 'B-tag1'], ['token1.2', 'I-tag1'], ['token1.3', 'B-tag3']],
                [['token2.1', 'B-tag1'], ['token2.2', 'I-tag1'], ['token2.3', 'B-tag3']]
            ],
            features=np.array([
                [['feat1.1.1'], ['feat1.2.1'], ['feat1.3.1']],
                [['feat2.1.1'], ['feat2.2.1'], ['feat2.3.1']]
            ]),
            output_format=TagOutputFormats.DATA_UNIDIFF
        )
        LOGGER.debug('result:\n%s', result)
        assert result.splitlines() == [
            '--- document_000001.expected',
            '+++ document_000001.actual',
            '@@ -1,3 +1,3 @@',
            ' token1.1 feat1.1.1 B-tag1',
            ' token1.2 feat1.2.1 I-tag1',
            '-token1.3 feat1.3.1 B-tag3',
            '+token1.3 feat1.3.1 B-tag2',
            '--- document_000002.expected',
            '+++ document_000002.actual',
            '@@ -1,3 +1,3 @@',
            ' token2.1 feat2.1.1 B-tag1',
            ' token2.2 feat2.2.1 I-tag1',
            '-token2.3 feat2.3.1 B-tag3',
            '+token2.3 feat2.3.1 B-tag2'
        ]

    def test_should_format_tag_list_result_as_data_unidiff_without_difference(self):
        result = format_tag_result(
            tag_result=[[['token1', 'B-tag1'], ['token2', 'I-tag1'], ['token3', 'B-tag2']]],
            expected_tag_result=[
                [['token1', 'B-tag1'], ['token2', 'I-tag1'], ['token3', 'B-tag2']]
            ],
            features=np.array([
                [['feat1.1', 'feat1.2'], ['feat2.1', 'feat2.2'], ['feat3.1', 'feat3.2']]
            ]),
            output_format=TagOutputFormats.DATA_UNIDIFF
        )
        LOGGER.debug('result:\n%s', result)
        assert result.splitlines() == []

    def test_should_format_tag_list_result_as_text(self):
        result = format_tag_result(
            tag_result=to_iterable(ANNOTATIONS_1),
            output_format=TagOutputFormats.TEXT,
            texts=TEXTS_1,
            features=FEATURES_1,
            model_name=MODEL_1
        )
        assert result == FLAT_TEXT_1

    def test_should_format_tag_list_result_as_xml(self):
        result = format_tag_result(
            tag_result=to_iterable(ANNOTATIONS_1),
            output_format=TagOutputFormats.XML,
            texts=TEXTS_1,
            features=FEATURES_1,
            model_name=MODEL_1
        )
        assert result == XML_1

    def test_should_format_tag_list_result_as_xml_and_combined_tags(self):
        result = format_tag_result(
            tag_result=to_iterable(
                [[['token1', 'B-tag1'], ['token2', 'I-tag1'], ['token3', 'B-tag2']]]
            ),
            output_format=TagOutputFormats.XML
        )
        assert result.splitlines() == [
            '<xml>',
            '  <p>',
            '    <tag1>token1 token2</tag1>',
            '    <tag2>token3</tag2>',
            '  </p>',
            '</xml>'
        ]

    def test_should_format_tag_list_result_as_xml_and_include_text_without_tags(self):
        result = format_tag_result(
            tag_result=to_iterable(
                [[['token1', 'O'], ['token2', 'O'], ['token3', 'B-tag2']]]
            ),
            output_format=TagOutputFormats.XML
        )
        assert result.splitlines() == [
            '<xml>',
            '  <p>',
            '    token1 token2',
            '    <tag2>token3</tag2>',
            '  </p>',
            '</xml>'
        ]

    def test_should_format_tag_list_result_as_xml_diff_and_combined_tags(self):
        result = format_tag_result(
            tag_result=to_iterable(
                [[['token1', 'B-tag1'], ['token2', 'I-tag1'], ['token3', 'B-tag2']]]
            ),
            expected_tag_result=[
                [['token1', 'B-tag1'], ['token2', 'I-tag1'], ['token3', 'B-tag3']]
            ],
            output_format=TagOutputFormats.XML_DIFF
        )
        LOGGER.debug('result:\n%s', result)
        assert result.splitlines() == [
            '  <xml>',
            '    <p>',
            '      <tag1>token1 token2</tag1>',
            '-     <tag3>token3</tag3>',
            '?         ^            ^',
            '+     <tag2>token3</tag2>',
            '?         ^            ^',
            '    </p>',
            '  </xml>'
        ]
