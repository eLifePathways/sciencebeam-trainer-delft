import datetime
import logging
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from delft.utilities.Embeddings import Embeddings
from delft.utilities.Tokenizer import tokenizeAndFilter
from delft.sequenceLabelling.preprocess import Preprocessor

from sciencebeam_trainer_delft.utils.progress_logger import logging_tqdm

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.data_generator import DataGenerator

from sciencebeam_trainer_delft.sequence_labelling.dataset_transform import (
    T_DatasetTransformerFactory,
    DummyDatasetTransformer
)


LOGGER = logging.getLogger(__name__)


def iter_batch_window_indices_and_offsets(
        data_generator: DataGenerator) -> Iterable[List[Tuple[int, int]]]:
    return (
        data_generator.get_batch_window_indices_and_offsets(batch_index)
        for batch_index in range(len(data_generator))
    )


def iter_predict_texts_with_sliding_window_if_enabled(
    texts: List[Union[str, List[str]]],
    model_config: ModelConfig,
    preprocessor: Preprocessor,
    max_sequence_length: Optional[int],
    model,
    input_window_stride: Optional[int] = None,
    embeddings: Optional[Embeddings] = None,
    features: Optional[List[List[List[str]]]] = None
):
    if not texts:
        LOGGER.info('passed in empty texts, model: %s', model_config.model_name)
        return
    should_tokenize = (
        len(texts) > 0  # pylint: disable=len-as-condition
        and isinstance(texts[0], str)
    )

    if not should_tokenize and max_sequence_length:
        max_actual_sequence_length = max(len(text) for text in texts)
        if max_actual_sequence_length <= max_sequence_length:
            LOGGER.info(
                'all text sequences below max sequence length: %d <= %d (model: %s)',
                max_actual_sequence_length, max_sequence_length,
                model_config.model_name
            )
        elif model_config.stateful:
            LOGGER.info(
                (
                    'some text sequences exceed max sequence length (using sliding windows):'
                    ' %d > %d (model: %s)'
                ),
                max_actual_sequence_length, max_sequence_length,
                model_config.model_name
            )
        else:
            LOGGER.info(
                (
                    'some text sequences exceed max sequence length'
                    ' (truncate, model is not stateful): %d > %d (model: %s)'
                ),
                max_actual_sequence_length, max_sequence_length,
                model_config.model_name
            )

    predict_generator = DataGenerator(
        x=texts,
        y=None,
        batch_size=model_config.batch_size,
        preprocessor=preprocessor,
        additional_token_feature_indices=model_config.additional_token_feature_indices,
        text_feature_indices=model_config.text_feature_indices,
        concatenated_embeddings_token_count=(
            model_config.concatenated_embeddings_token_count
        ),
        char_embed_size=model_config.char_embedding_size,
        is_deprecated_padded_batch_text_list_enabled=(
            model_config.is_deprecated_padded_batch_text_list_enabled
        ),
        max_sequence_length=max_sequence_length,
        input_window_stride=input_window_stride,
        stateful=model_config.stateful,
        embeddings=embeddings,
        tokenize=should_tokenize,
        shuffle=False,
        features=features,
        use_chain_crf=model_config.use_chain_crf,
        name='%s.predict_generator' % model_config.model_name
    )

    prediction_list_list: List[List[np.ndarray]] = [[] for _ in texts]
    batch_window_indices_and_offsets_iterable = logging_tqdm(
        iter_batch_window_indices_and_offsets(
            predict_generator
        ),
        logger=LOGGER,
        total=len(predict_generator),
        desc='%s: ' % predict_generator.name,
        unit='batch'
    )
    completed_curser = 0
    for batch_window_indices_and_offsets in batch_window_indices_and_offsets_iterable:
        LOGGER.debug(
            'predict batch_window_indices_and_offsets: %s',
            batch_window_indices_and_offsets
        )
        generator_output = predict_generator.get_window_batch_data(
            batch_window_indices_and_offsets
        )
        LOGGER.debug('predict on batch: %d', len(batch_window_indices_and_offsets))
        batch_predictions = model.predict_on_batch(generator_output[0])
        LOGGER.debug('preds.shape: %s', batch_predictions.shape)
        for window_indices_and_offsets, seq_predictions in zip(
            batch_window_indices_and_offsets, batch_predictions
        ):
            text_index, text_offset = window_indices_and_offsets
            current_prediction_list = prediction_list_list[text_index]
            LOGGER.debug(
                'prediction_list_list[%d]: %s',
                text_index,
                current_prediction_list
            )
            current_offset = sum((len(a) for a in current_prediction_list))
            if current_offset > text_offset:
                # skip over the overlapping window
                seq_predictions = seq_predictions[(current_offset - text_offset):]
                text_offset = current_offset
            assert (
                current_offset == text_offset
            ), "expected %d to be %d" % (
                current_offset, text_offset
            )
            current_prediction_list.append(seq_predictions)
            next_offset = sum((len(a) for a in current_prediction_list))
            is_complete = (next_offset >= len(texts[text_index]))
            LOGGER.debug(
                'is_complete: %s, text_index=%d, completed_curser=%d, next_offset=%d, textlen=%d',
                is_complete, text_index, completed_curser, next_offset, len(texts[text_index])
            )
            if (is_complete and text_index == completed_curser):
                yield np.concatenate(current_prediction_list, axis=0)
                completed_curser += 1

    for prediction_list in prediction_list_list[completed_curser:]:
        yield np.concatenate(prediction_list, axis=0)


class Tagger:
    def __init__(
        self,
        model,
        model_config,
        preprocessor: Preprocessor,
        embeddings: Optional[Embeddings] = None,
        dataset_transformer_factory: Optional[T_DatasetTransformerFactory] = None,
        max_sequence_length: Optional[int] = None,
        input_window_stride: Optional[int] = None
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.model_config = model_config
        self.embeddings = embeddings
        self.max_sequence_length = max_sequence_length
        self.input_window_stride = input_window_stride
        if dataset_transformer_factory is None:
            self.dataset_transformer_factory: T_DatasetTransformerFactory = (
                DummyDatasetTransformer
            )
        else:
            self.dataset_transformer_factory = dataset_transformer_factory
        LOGGER.debug('Model config: %r', self.model_config)

    def iter_tag(
        self,
        texts: Sequence[str],
        output_format: Optional[str],
        features=None,
        tag_transformed: bool = False
    ) -> Iterable[Union[dict, Sequence[Tuple[str, str]]]]:
        assert isinstance(texts, list)

        dataset_transformer = self.dataset_transformer_factory()
        transformed_texts, transformed_features = dataset_transformer.fit_transform_x_and_features(
            texts, features
        )

        preds_concatenated_iterable = iter_predict_texts_with_sliding_window_if_enabled(
            texts=transformed_texts,
            features=transformed_features,
            model=self.model,
            model_config=self.model_config,
            preprocessor=self.preprocessor,
            max_sequence_length=self.max_sequence_length,
            input_window_stride=self.input_window_stride,
            embeddings=self.embeddings
        )
        for i, pred_item in enumerate(preds_concatenated_iterable):
            LOGGER.debug('pred_item.shape: %s', pred_item.shape)
            LOGGER.debug('pred_item=%r', pred_item)

            pred = [pred_item]
            text = texts[i]
            if tag_transformed:
                text = transformed_texts[i]

            if isinstance(text, str):
                tokens, offsets = tokenizeAndFilter(text)
            else:
                # it is a list of string, so a string already tokenized
                # note that in this case, offset are not present and json output is impossible
                tokens = text
                offsets = []

            LOGGER.debug('tokens: %s', tokens)

            is_sparse = self.model_config.use_crf and not self.model_config.use_chain_crf
            tags = self._get_tags(pred, is_sparse=is_sparse)
            if not tag_transformed:
                tags = dataset_transformer.inverse_transform_y([tags])[0]
            LOGGER.debug('tags: %s', tags)

            if output_format == 'json':
                prob = self._get_prob(pred, is_sparse=is_sparse)
                piece = {}
                piece["text"] = text
                piece["entities"] = self._build_json_response(
                    tokens, tags, prob, offsets
                )["entities"]
                yield piece
            else:
                the_tags = list(zip(tokens, tags))
                yield the_tags

    def tag(
        self, texts, output_format, features=None, **kwargs
    ) -> Union[dict, Sequence[Sequence[Tuple[str, str]]]]:
        result = list(self.iter_tag(texts, output_format, features, **kwargs))
        if output_format == 'json':
            return {
                "software": "ScienceBeam Trainer DeLFT",
                "date": datetime.datetime.now().isoformat(),
                "model": self.model.config.model_name,
                "texts": result
            }
        else:
            return result  # type: ignore

    def _get_tags_dense(self, pred):
        pred = np.argmax(pred, -1)
        tags = self.preprocessor.inverse_transform(pred[0])
        return tags

    def _get_tags_sparse(self, pred):
        tags = self.preprocessor.inverse_transform(pred[0])
        return tags

    def _get_tags(self, pred, is_sparse: bool = False):
        if is_sparse:
            return self._get_tags_sparse(pred)
        return self._get_tags_dense(pred)

    def _get_prob_dense(self, pred):
        prob = np.max(pred, -1)[0]
        return prob

    def _get_prob_sparse(self, pred):
        return [1.0] * len(pred[0])

    def _get_prob(self, pred, is_sparse: bool = False):
        if is_sparse:
            return self._get_prob_sparse(pred)
        return self._get_prob_dense(pred)

    def _build_json_response(self, tokens, tags, prob, offsets):
        res = {
            "entities": []
        }
        chunks = get_entities_with_offsets(tags, offsets)
        LOGGER.debug('entity chunks: %s', chunks)
        for chunk_type, chunk_start, chunk_end, pos_start, pos_end in chunks:
            # TODO: get the original string rather than regenerating it from tokens
            entity = {
                "text": ' '.join(tokens[chunk_start: chunk_end]),
                "class": chunk_type,
                "score": float(np.average(prob[chunk_start:chunk_end])),
                "beginOffset": pos_start,
                "endOffset": pos_end
            }
            res["entities"].append(entity)

        return res


def get_entities_with_offsets(seq, offsets):
    """
    Gets entities from sequence

    Args:
        seq (list): sequence of labels.
        offsets (list of integer pair): sequence of offset position

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end, pos_start, pos_end)

    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> offsets = [(0,10), (11, 15), (16, 29), (30, 41)]
        >>> print(get_entities(seq))
        [('PER', 0, 2, 0, 15), ('LOC', 3, 4, 30, 41)]
    """
    i = 0
    chunks = []
    seq = seq + ['O']  # add sentinel
    types = [tag.split('-')[-1] for tag in seq]
    max_length = min(len(seq)-1, len(offsets))
    while i < max_length:
        if seq[i].startswith('B'):
            # if we are at the end of the offsets, we can stop immediatly
            j = max_length
            if i+2 != max_length:
                for j in range(i+1, max_length):
                    if seq[j].startswith('I') and types[j] == types[i]:
                        continue
                    break
            start_pos = offsets[i][0]
            end_pos = offsets[j-1][1]-1
            chunks.append((types[i], i, j, start_pos, end_pos))
            i = j
        else:
            i += 1
    return chunks
