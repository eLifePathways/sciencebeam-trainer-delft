import json
import difflib
import logging
from xml.sax.saxutils import escape as xml_escape
from typing import Optional, Sequence, Union, Iterable, List, Tuple

import numpy as np

from delft.sequenceLabelling.evaluation import get_entities

from sciencebeam_trainer_delft.sequence_labelling.typing import (
    T_Batch_Features_Array,
    T_Batch_Label_Array_Or_List,
    T_Batch_Token_Array_Or_List
)


LOGGER = logging.getLogger(__name__)


class TagOutputFormats:
    JSON = 'json'
    DATA = 'data'
    DATA_UNIDIFF = 'data_unidiff'
    TEXT = 'text'
    XML = 'xml'
    XML_DIFF = 'xml_diff'


TAG_OUTPUT_FORMATS = [
    TagOutputFormats.JSON,
    TagOutputFormats.DATA,
    TagOutputFormats.DATA_UNIDIFF,
    TagOutputFormats.TEXT,
    TagOutputFormats.XML,
    TagOutputFormats.XML_DIFF,
]


class TagLabelFormats:
    IOB = 'iob'
    GROBID = 'grobid'


TAG_LABEL_FORMATS = [
    TagLabelFormats.IOB,
    TagLabelFormats.GROBID,
]

T_Token_Label_Tuple = Tuple[str, str]
T_Document_Token_Label_Tuple_List = Sequence[T_Token_Label_Tuple]
T_Batch_Token_Label_Tuple_List = Sequence[T_Document_Token_Label_Tuple_List]


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):  # pylint: disable=arguments-renamed, method-hidden
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def translate_tag_label_iob_to_grobid(tag: str) -> str:
    if tag == 'O':
        return '<other>'
    if tag.startswith('B-'):
        return 'I-' + tag[2:]
    if tag.startswith('I-'):
        return tag[2:]
    return tag


def translate_tag_label(tag: str, label_format: str) -> str:
    if label_format == TagLabelFormats.GROBID:
        return translate_tag_label_iob_to_grobid(tag)
    return tag


def get_tag_result(
    texts: T_Batch_Token_Array_Or_List,
    labels: T_Batch_Label_Array_Or_List
) -> T_Batch_Token_Label_Tuple_List:
    return [
        list(zip(doc_texts, doc_labels))
        for doc_texts, doc_labels in zip(texts, labels)
    ]


def format_json_tag_result_as_json(tag_result: dict) -> str:
    return json.dumps(tag_result, indent=2, cls=CustomJsonEncoder)


def format_list_tag_result_as_json(
    tag_result: Iterable[List[Tuple[str, str]]],
    texts: Optional[np.ndarray] = None,
    features: Optional[np.ndarray] = None,
    model_name: Optional[str] = None
) -> str:
    output_props = {
        'model': model_name,
        'texts': np.array(texts).tolist(),
        'features': np.array(features).tolist() if features is not None else None,
        'annotations': list(tag_result)
    }
    return json.dumps(output_props, indent=2, cls=CustomJsonEncoder)


def iter_to_data_lines(
    features: np.ndarray,
    annotations: Iterable[List[Tuple[str, str]]],
    label_format: str = TagLabelFormats.IOB
) -> Iterable[str]:
    for document_lindex, (line_annotations, line_features) in enumerate(
        zip(annotations, features.tolist())
    ):
        if document_lindex > 0:
            yield ''  # blank line separator
        yield from (
            ' '.join(
                [token_annotation[0]]
                + list(token_features)
                + [translate_tag_label(token_annotation[1], label_format)]
            )
            for token_annotation, token_features in zip(line_annotations, line_features)
        )


def to_data_lines(*args, **kwargs) -> List[str]:
    return list(iter_to_data_lines(*args, **kwargs))


def iter_format_list_tag_result_as_data(
    tag_result: Iterable[List[Tuple[str, str]]],
    texts: Optional[np.ndarray] = None,  # pylint: disable=unused-argument
    features: Optional[np.ndarray] = None,
    model_name: Optional[str] = None,  # pylint: disable=unused-argument
    label_format: str = TagLabelFormats.IOB
) -> Iterable[str]:
    assert features is not None
    data_text_iterable = iter_to_data_lines(
        features=features,
        annotations=tag_result,
        label_format=label_format
    )
    for line_index, data_text in enumerate(data_text_iterable):
        if line_index > 0:
            yield '\n'
        yield data_text


def format_list_tag_result_as_data(*args, **kwargs) -> str:
    return ''.join(iter_format_list_tag_result_as_data(*args, **kwargs))


def iter_simple_unidiff(
    a, b, fromfile='', tofile='', lineterm='\n',
    force_output: bool = False
) -> Iterable[str]:
    if len(a) > len(b):
        # truncate expected, as predicted sequences may have truncuated
        a = a[:len(b)]
    assert len(a) == len(b)
    line_count = len(a)
    is_diff_list = [
        value_1 != value_2
        for value_1, value_2 in zip(a, b)
    ]
    LOGGER.debug('is_diff_list: %s', is_diff_list)
    diff_count = sum(is_diff_list)
    if not diff_count and not force_output:
        return
    if fromfile:
        yield f'--- {fromfile}{lineterm}'
    if tofile:
        yield f'+++ {tofile}{lineterm}'
    removed_with_prefix = f'-{diff_count}' if diff_count else '-0'
    added_with_prefix = f'+{diff_count}' if diff_count else '+0'
    yield f'@@ {removed_with_prefix},{line_count} {added_with_prefix},{line_count} @@{lineterm}'
    for is_diff, value_1, value_2 in zip(is_diff_list, a, b):
        if is_diff:
            yield f'-{value_1}'
            yield f'+{value_2}'
        else:
            yield f' {value_1}'


def split_lines_with_line_feed(text: str, line_feed: str = '\n') -> List[str]:
    # Note: similar to .splitlines(keepends=True), but always adds the line feed
    return [
        line + line_feed
        for line in text.splitlines()
    ]


def iter_format_document_tag_result_as_data_unidiff(
    document_tag_result: List[Tuple[str, str]],
    document_expected_tag_result: T_Document_Token_Label_Tuple_List,
    document_features: List[List[str]],
    document_name: str,
    label_format: str = TagLabelFormats.IOB
) -> Iterable[str]:
    actual_data = format_list_tag_result_as_data(
        [document_tag_result],
        features=np.expand_dims(document_features, axis=0),
        label_format=label_format
    )
    expected_data = format_list_tag_result_as_data(
        [document_expected_tag_result],
        features=np.expand_dims(document_features, axis=0),
        label_format=label_format
    )
    LOGGER.debug('actual_data: %r', actual_data)
    LOGGER.debug('expected_data: %r', expected_data)
    yield from iter_simple_unidiff(
        split_lines_with_line_feed(expected_data),
        split_lines_with_line_feed(actual_data),
        fromfile=f'{document_name}.expected',
        tofile=f'{document_name}.actual'
    )


def iter_format_document_list_tag_result_as_data_unidiff(
    tag_result: Iterable[List[Tuple[str, str]]],
    expected_tag_result: T_Batch_Token_Label_Tuple_List,
    features: T_Batch_Features_Array,
    document_name_prefix: str,
    label_format: str = TagLabelFormats.IOB
) -> Iterable[str]:
    for document_index, document_tag_result in enumerate(tag_result):
        yield from iter_format_document_tag_result_as_data_unidiff(
            document_tag_result=document_tag_result,
            document_expected_tag_result=expected_tag_result[document_index],
            document_features=features[document_index],
            document_name='%s%06d' % (document_name_prefix, 1 + document_index),
            label_format=label_format
        )


def iter_format_list_tag_result_as_data_unidiff(
    tag_result: Iterable[List[Tuple[str, str]]],
    expected_tag_result: T_Batch_Token_Label_Tuple_List,
    texts: Optional[np.ndarray] = None,  # pylint: disable=unused-argument
    features: Optional[T_Batch_Features_Array] = None,
    model_name: Optional[str] = None,
    label_format: str = TagLabelFormats.IOB
) -> Iterable[str]:
    assert expected_tag_result
    assert features is not None
    document_name_prefix = 'document_'
    if model_name:
        document_name_prefix = model_name + '_' + document_name_prefix
    yield from iter_format_document_list_tag_result_as_data_unidiff(
        tag_result=tag_result,
        expected_tag_result=expected_tag_result,
        features=features,
        document_name_prefix=document_name_prefix,
        label_format=label_format
    )


def iter_to_flat_text(texts: np.ndarray) -> Iterable[str]:
    for document_index, line_tokens in enumerate(texts):
        if document_index > 0:
            yield '\n'
        yield ' '.join(line_tokens)


def iter_format_list_tag_result_as_text(
    tag_result: Iterable[List[Tuple[str, str]]],  # pylint: disable=unused-argument
    texts: Optional[np.ndarray] = None,
    features: Optional[np.ndarray] = None,  # pylint: disable=unused-argument
    model_name: Optional[str] = None  # pylint: disable=unused-argument
) -> Iterable[str]:
    assert texts is not None
    yield from iter_to_flat_text(texts=texts)


def get_xml_tag_for_annotation_label(annotation_label: str) -> str:
    return annotation_label.replace('<', '').replace('>', '').split('-', maxsplit=1)[-1]


def iter_add_untagged_token_spans(
    entity_chunks: Iterable[Tuple[str, int, int]],
    token_count: int,
    untagged_chunk_type: Optional[str] = None
) -> Iterable[Tuple[Optional[str], int, int]]:
    prev_chunk_end_excl = 0
    for chunk_type, chunk_start, chunk_end in entity_chunks:
        if chunk_start > prev_chunk_end_excl:
            yield untagged_chunk_type, prev_chunk_end_excl, (chunk_start - 1)
        yield chunk_type, chunk_start, chunk_end
        prev_chunk_end_excl = chunk_end + 1
    if token_count > prev_chunk_end_excl:
        yield untagged_chunk_type, prev_chunk_end_excl, (token_count - 1)


def iter_doc_annotations_xml_text(
        doc_annotations: List[Tuple[str, str]]) -> Iterable[str]:
    LOGGER.debug('doc_annotations: %s', doc_annotations)
    text_tokens = [token_text for token_text, _ in doc_annotations]
    token_labels = [token_label for _, token_label in doc_annotations]
    entity_chunks = list(iter_add_untagged_token_spans(
        get_entities(token_labels),
        len(token_labels)
    ))
    LOGGER.debug('text_tokens: %s', text_tokens)
    LOGGER.debug('token_labels: %s', token_labels)
    LOGGER.debug('entity_chunks: %s', entity_chunks)
    return '\n'.join((
        (
            '    <{tag}>{text}</{tag}>'.format(
                tag=get_xml_tag_for_annotation_label(chunk_type),
                text=xml_escape(' '.join(text_tokens[chunk_start:chunk_end + 1]))
            )
            if chunk_type
            else
            '    {text}'.format(
                text=xml_escape(' '.join(text_tokens[chunk_start:chunk_end + 1]))
            )
        )
        for chunk_type, chunk_start, chunk_end in entity_chunks
    )) + '\n'


def iter_annotations_xml_text(
    annotations: Iterable[List[Tuple[str, str]]]
) -> Iterable[str]:
    for doc_index, doc_annotations in enumerate(annotations):
        if doc_index > 0:
            yield '\n\n'
        yield '  <p>\n'
        yield from iter_doc_annotations_xml_text(doc_annotations)
        yield '  </p>\n'


def iter_format_list_tag_result_as_xml(
    tag_result: Iterable[List[Tuple[str, str]]],
    texts: Optional[np.ndarray] = None,  # pylint: disable=unused-argument
    features: Optional[np.ndarray] = None,  # pylint: disable=unused-argument
    model_name: Optional[str] = None  # pylint: disable=unused-argument
) -> Iterable[str]:
    yield '<xml>\n'
    yield from iter_annotations_xml_text(
        annotations=tag_result
    )
    yield '</xml>'


def format_list_tag_result_as_xml(*args, **kwargs) -> str:
    return ''.join(iter_format_list_tag_result_as_xml(*args, **kwargs))


def iter_format_list_tag_result_as_xml_diff(
    tag_result: Iterable[List[Tuple[str, str]]],
    expected_tag_result: T_Batch_Token_Label_Tuple_List,
    texts: Optional[np.ndarray] = None,  # pylint: disable=unused-argument
    features: Optional[np.ndarray] = None,  # pylint: disable=unused-argument
    model_name: Optional[str] = None  # pylint: disable=unused-argument
) -> Iterable[str]:
    assert expected_tag_result
    actual_xml = format_list_tag_result_as_xml(tag_result)
    expected_xml = format_list_tag_result_as_xml(expected_tag_result)
    yield from difflib.ndiff(
        expected_xml.splitlines(keepends=True),
        actual_xml.splitlines(keepends=True)
    )


def iter_format_list_tag_result(
        *args,
        output_format: str,
        expected_tag_result: Optional[T_Batch_Token_Label_Tuple_List] = None,
        label_format: str = TagLabelFormats.IOB,
        **kwargs) -> Iterable[str]:
    if output_format == TagOutputFormats.JSON:
        yield format_list_tag_result_as_json(*args, **kwargs)
        return
    if output_format == TagOutputFormats.DATA:
        yield from iter_format_list_tag_result_as_data(  # type: ignore[misc]
            *args, label_format=label_format, **kwargs
        )
        return
    if output_format == TagOutputFormats.DATA_UNIDIFF:
        assert expected_tag_result
        yield from iter_format_list_tag_result_as_data_unidiff(  # type: ignore
            *args,
            expected_tag_result=expected_tag_result,
            label_format=label_format,
            **kwargs
        )
        return
    if output_format == TagOutputFormats.TEXT:
        yield from iter_format_list_tag_result_as_text(*args, **kwargs)
        return
    if output_format == TagOutputFormats.XML:
        yield from iter_format_list_tag_result_as_xml(*args, **kwargs)
        return
    if output_format == TagOutputFormats.XML_DIFF:
        assert expected_tag_result
        yield from iter_format_list_tag_result_as_xml_diff(  # type: ignore
            *args,
            expected_tag_result=expected_tag_result,
            **kwargs
        )
        return
    raise ValueError('unrecognised output format: %s' % output_format)


def iter_format_tag_result(
    tag_result: Union[dict, list, Iterable],
    output_format: str,
    expected_tag_result: Optional[T_Batch_Token_Label_Tuple_List] = None,
    texts: Optional[np.ndarray] = None,
    features: Optional[np.ndarray] = None,
    model_name: Optional[str] = None,
    label_format: str = TagLabelFormats.IOB
) -> Iterable[str]:
    if isinstance(tag_result, dict):
        assert output_format == TagOutputFormats.JSON
        yield format_json_tag_result_as_json(tag_result)
        return
    yield from iter_format_list_tag_result(
        tag_result,
        output_format=output_format,
        expected_tag_result=expected_tag_result,
        texts=texts,
        features=features,
        model_name=model_name,
        label_format=label_format
    )


def format_tag_result(*args, **kwargs) -> str:
    return ''.join(iter_format_tag_result(*args, **kwargs))
