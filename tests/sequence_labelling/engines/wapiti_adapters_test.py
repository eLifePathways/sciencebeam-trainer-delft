import logging
from io import StringIO

import numpy as np

from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti_adapters import (
    translate_tags_IOB_to_grobid,
    write_wapiti_train_data,
    iter_read_tagged_result,
    convert_wapiti_model_result_to_document_tagged_result
)


LOGGER = logging.getLogger(__name__)


class TestTranslateTagsIOBToGrobid:
    def test_should_convert_B_prefix_to_I(self):
        assert translate_tags_IOB_to_grobid('B-<label>') == 'I-<label>'

    def test_should_convert_I_prefix_to_no_prefix(self):
        assert translate_tags_IOB_to_grobid('I-<label>') == '<label>'

    def test_should_convert_O_to_other(self):
        assert translate_tags_IOB_to_grobid('O') == '<other>'


class TestWriteWapitiTrainData:
    def test_should_write_single_document_and_translate_label(self):
        buffer = StringIO()
        write_wapiti_train_data(
            buffer,
            x=[['token1', 'token2']],
            y=[['B-<label1>', 'I-<label1>']],
            features=[[['f1.1', 'f1.2'], ['f2.1', 'f2.2']]]
        )
        LOGGER.debug('buffer:\n%s', buffer.getvalue())
        assert buffer.getvalue().splitlines() == [
            'token1\tf1.1\tf1.2\tI-<label1>',
            'token2\tf2.1\tf2.2\t<label1>',
            '',
            ''
        ]

    def test_should_write_features_held_in_a_multi_dimensional_array(self):
        # the reader returns a multi-dimensional array where the documents have equal row
        # counts, and adding a list to a numpy row concatenates the strings elementwise
        buffer = StringIO()
        write_wapiti_train_data(
            buffer,
            x=np.array([['token1', 'token2']], dtype='object'),
            y=np.array([['B-<label1>', 'I-<label1>']], dtype='object'),
            features=np.array([[['f1.1', 'f1.2'], ['f2.1', 'f2.2']]], dtype='object')
        )
        assert buffer.getvalue().splitlines() == [
            'token1\tf1.1\tf1.2\tI-<label1>',
            'token2\tf2.1\tf2.2\t<label1>',
            '',
            ''
        ]

    def test_should_write_multiple_documents_and_translate_labels(self):
        buffer = StringIO()
        write_wapiti_train_data(
            buffer,
            x=[
                ['token1.1', 'token1.2'],
                ['token2.1', 'token2.2']
            ],
            y=[
                ['B-<label1.1>', 'I-<label1.2>'],
                ['B-<label2.1>', 'I-<label2.2>']
            ],
            features=[
                [['f1.1.1', 'f1.1.2'], ['f1.2.1', 'f1.2.2']],
                [['f2.1.1', 'f2.1.2'], ['f2.2.1', 'f2.2.2']]
            ]
        )
        LOGGER.debug('buffer:\n%s', buffer.getvalue())
        assert buffer.getvalue().splitlines() == [
            'token1.1\tf1.1.1\tf1.1.2\tI-<label1.1>',
            'token1.2\tf1.2.1\tf1.2.2\t<label1.2>',
            '',
            '',
            'token2.1\tf2.1.1\tf2.1.2\tI-<label2.1>',
            'token2.2\tf2.2.1\tf2.2.2\t<label2.2>',
            '',
            ''
        ]


class TestIterReadTaggedResult:
    def test_should_read_single_document_and_translate_label(self):
        tagged_data = '\n'.join([
            'token1\tf1.1\tf1.2\tI-<label1>',
            'token2\tf2.1\tf2.2\t<label1>',
            '',
            ''
        ])
        tagged_result = list(iter_read_tagged_result(StringIO(tagged_data)))
        LOGGER.debug('tagged_result:\n%s', tagged_result)
        assert tagged_result == [[
            ('token1', 'B-<label1>'),
            ('token2', 'I-<label1>')
        ]]

    def test_should_read_multiple_documents_and_translate_label(self):
        tagged_data = '\n'.join([
            'token1.1\tf1.1.1\tf1.1.2\tI-<label1.1>',
            'token1.2\tf1.2.1\tf1.2.2\t<label1.2>',
            '',
            '',
            'token2.1\tf2.1.1\tf2.1.2\tI-<label2.1>',
            'token2.2\tf2.2.1\tf2.2.2\t<label2.2>',
            '',
            ''
        ])
        tagged_result = list(iter_read_tagged_result(StringIO(tagged_data)))
        LOGGER.debug('tagged_result:\n%s', tagged_result)
        assert tagged_result == [[
            ('token1.1', 'B-<label1.1>'),
            ('token1.2', 'I-<label1.2>')
        ], [
            ('token2.1', 'B-<label2.1>'),
            ('token2.2', 'I-<label2.2>')
        ]]


class TestConvertWapitiModelResultToDocumentTaggedResult:
    def test_should_convert_single_document_and_translate_label(self):
        x_doc = [
            'token1',
            'token2'
        ]
        wapiti_model_result = [
            ['dummy', 'I-<label1>'],
            ['dummy', '<label1>']
        ]
        doc_tagged_result = convert_wapiti_model_result_to_document_tagged_result(
            x_doc,
            wapiti_model_result
        )
        LOGGER.debug('doc_tagged_result:\n%s', doc_tagged_result)
        assert doc_tagged_result == [
            ('token1', 'B-<label1>'),
            ('token2', 'I-<label1>')
        ]
