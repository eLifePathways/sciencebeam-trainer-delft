import logging
import itertools
from functools import partial
from typing import Any, Dict, List, Iterable, Optional, Set, Tuple, Union

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer

from delft.sequenceLabelling.preprocess import (
    FeaturesPreprocessor as DelftFeaturesPreprocessor,
    Preprocessor as DelftWordPreprocessor,
    PAD,
    UNK
)

from sciencebeam_trainer_delft.utils.typing import T
from sciencebeam_trainer_delft.utils.progress_logger import logging_tqdm

import sciencebeam_trainer_delft.utils.compat.sklearn  # noqa pylint: disable=unused-import


LOGGER = logging.getLogger(__name__)


def to_dict(
    value_list_batch: List[list],
    feature_indices: Optional[Set[int]] = None,
    exclude_features_indices: Optional[Set[int]] = None
) -> Iterable[dict]:
    # Note: keeping `feature_indices` name for pickle compatibility
    #   (also matches upstream for `to_dict`)
    return (
        {
            index: value
            for index, value in enumerate(value_list)
            if (
                (not feature_indices or index in feature_indices)
                and (not exclude_features_indices or index not in exclude_features_indices)
            )
        }
        for value_list in value_list_batch
    )


def to_float_features(
    value_list_batch: List[list],
    features_indices: Set[int]
) -> Iterable[List[float]]:
    return (
        [
            float(value)
            for index, value in enumerate(value_list)
            if index in features_indices
        ]
        for value_list in value_list_batch
    )


def faster_preprocessor_fit(self: DelftWordPreprocessor, X, y):
    chars = {PAD: 0, UNK: 1}
    tags = {PAD: 0}

    if self.return_chars:
        temp_chars = {
            c
            for w in set(itertools.chain(*X))
            for c in w
        }

        sorted_chars = sorted(temp_chars)
        sorted_chars_dict = {
            c: idx + 2
            for idx, c in enumerate(sorted_chars)
        }
        chars = {**chars, **sorted_chars_dict}

    temp_tags = set(itertools.chain(*y))
    sorted_tags = sorted(temp_tags)
    sorted_tags_dict = {
        tag: idx + 1
        for idx, tag in enumerate(sorted_tags)
    }
    tags = {**tags, **sorted_tags_dict}

    self.vocab_char = chars
    self.vocab_tag = tags


class WordPreprocessor(DelftWordPreprocessor):
    # keeping class for pickle compatibility
    pass


def iter_batch(iterable: Iterable[T], n: int = 1) -> Iterable[List[T]]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


class IterableMinMaxScaler(MinMaxScaler):
    def fit(self, X, y=None):
        batch_size = 1000
        for batch in iter_batch(X, batch_size):
            self.partial_fit(batch)

    def transform(self, X):
        return super().transform(list(X))


STATE_ATTRIBUTE_NAMES_BY_TYPE = {
    DictVectorizer: ['feature_names_', 'vocabulary_'],
    StandardScaler: ['scale_', 'mean_', 'var_', 'n_samples_seen_'],
    MinMaxScaler: ['min_', 'scale_', 'data_min_', 'data_max_', 'data_range_', 'n_samples_seen_']
}


STATE_ATTRIBUTE_NAMES_BY_TYPE[IterableMinMaxScaler] = STATE_ATTRIBUTE_NAMES_BY_TYPE[MinMaxScaler]


def _iter_nested_pipeline_steps(steps: List[Tuple[str, Any]]) -> Iterable[Tuple[str, Any]]:
    for step_name, step_value in steps:
        yield step_name, step_value
        if isinstance(step_value, Pipeline):
            yield from _iter_nested_pipeline_steps(step_value.steps)
        if isinstance(step_value, FeatureUnion):
            yield from _iter_nested_pipeline_steps(step_value.transformer_list)
            continue


def _find_step_by_name(steps: List[Tuple[str, Any]], name: str):
    for step_name, step_value in _iter_nested_pipeline_steps(steps):
        if step_name == name:
            return step_value
    raise ValueError(f'step with name {repr(name)} not found')


def _get_dict_vectorizer_state(vectorizer: DictVectorizer) -> dict:
    return {
        'vectorizer.feature_names': vectorizer.feature_names_,
        'vectorizer.vocabulary': vectorizer.vocabulary_
    }


def _get_attributes_state(obj, attribute_names: List[str]) -> dict:
    result = {}
    for attribute_name in attribute_names:
        value = getattr(obj, attribute_name)
        if isinstance(value, np.ndarray):
            result[attribute_name] = value.tolist()
            result[attribute_name + '.is_numpy'] = True
        else:
            result[attribute_name] = value
    return result


def _restore_attributes_state(obj, state: Dict[str, Any]):
    for attribute_name, value in state.items():
        if '.' in attribute_name:
            continue
        if state.get(attribute_name + '.is_numpy'):
            value = np.asarray(value)
        setattr(obj, attribute_name, value)


def _get_pipeline_steps_state(steps: List[Tuple[str, Any]]) -> dict:
    result = {}
    for step_name, step_value in _iter_nested_pipeline_steps(steps):
        state_attribute_names = STATE_ATTRIBUTE_NAMES_BY_TYPE.get(type(step_value))
        if not state_attribute_names:
            continue
        result[step_name] = _get_attributes_state(step_value, state_attribute_names)
    return result


def _restore_pipeline_steps_state(steps: List[Tuple[str, Any]], state: dict):
    for step_name, step_value in _iter_nested_pipeline_steps(steps):
        step_state = state.get(step_name)
        if not step_state:
            continue
        _restore_attributes_state(step_value, step_state)


def _migrate_legacy_pipeline_steps_if_necessary(steps: List[Tuple[str, Any]]):
    for _, step in _iter_nested_pipeline_steps(steps):
        if isinstance(step, FunctionTransformer) and not hasattr(step, 'feature_names_out'):
            step.feature_names_out = None
            LOGGER.info('migrated legacy FunctionTransformer to add feature_names_out=None')


def _fit_transformer_with_progress_logging(
    transformer: TransformerMixin,
    X,
    logger: logging.Logger,
    message_prefix: str,
    unit: str,
    message_suffx: str = ': '
):
    if isinstance(transformer, Pipeline):
        steps = transformer.steps
        if len(steps) == 1 and isinstance(steps[0][1], FeatureUnion):
            feature_union = steps[0][1]
            for name, union_transformer in feature_union.transformer_list:
                X = logging_tqdm(
                    iterable=X,
                    logger=logger,
                    desc=f'{message_prefix}.{name}{message_suffx}',
                    unit=unit
                )
                union_transformer.fit(X)
            return
    X = logging_tqdm(iterable=X, logger=logger, desc=message_prefix + message_suffx, unit=unit)
    transformer.fit(X)


class FeaturesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        features_indices: Optional[Iterable[int]] = None,
        continuous_features_indices: Optional[Iterable[int]] = None
    ):
        self.features_indices = features_indices
        self.continuous_features_indices = continuous_features_indices
        self.features_map_to_index = None
        self.pipeline = FeaturesPreprocessor._create_pipeline(
            features_indices=features_indices,
            continuous_features_indices=continuous_features_indices
        )

    @staticmethod
    def _create_pipeline(
        features_indices: Optional[Iterable[int]] = None,
        continuous_features_indices: Optional[Iterable[int]] = None
    ):
        features_indices_set = None
        if features_indices:
            features_indices_set = set(features_indices)
        continuous_features_indices_set = set(
            continuous_features_indices or []
        )
        to_dict_fn = partial(
            to_dict,
            feature_indices=features_indices_set,
            exclude_features_indices=continuous_features_indices_set
        )
        pipeline = Pipeline(steps=[
            ('to_dict', FunctionTransformer(to_dict_fn, validate=False)),
            ('vectorize', DictVectorizer(sparse=False))
        ])
        if continuous_features_indices_set:
            to_float_features_fn = partial(
                to_float_features,
                features_indices=continuous_features_indices_set
            )
            continuous_features_pipeline = Pipeline(steps=[
                ('to_float_features', FunctionTransformer(to_float_features_fn, validate=False)),
                ('min_max_scalar', IterableMinMaxScaler()),
            ])
            pipeline = Pipeline(steps=[
                ('union', FeatureUnion([
                    ('continuous', continuous_features_pipeline),
                    ('discreet', pipeline)
                ]))
            ])
        LOGGER.info('pipeline=%s', pipeline)
        return pipeline

    @property
    def vectorizer(self) -> DictVectorizer:
        return _find_step_by_name(self.pipeline.steps, 'vectorize')

    @property
    def standard_scalar(self) -> StandardScaler:
        return _find_step_by_name(self.pipeline.steps, 'standard_scalar')

    def __getstate__(self):
        return {
            **_get_pipeline_steps_state(self.pipeline.steps),
            'features_indices': self.features_indices,
            'continuous_features_indices': self.continuous_features_indices
        }

    def __setstate__(self, state):
        try:
            if 'pipeline' in state:
                # original pickle
                super().__setstate__(state)
                _migrate_legacy_pipeline_steps_if_necessary(self.pipeline.steps)
                return self
            self.features_indices = state['features_indices']
            self.continuous_features_indices = state.get('continuous_features_indices')
            self.pipeline = FeaturesPreprocessor._create_pipeline(
                features_indices=self.features_indices,
                continuous_features_indices=self.continuous_features_indices
            )
            _restore_pipeline_steps_state(self.pipeline.steps, state)
            vectorizer_feature_names = state.get('vectorizer.feature_names')
            vectorizer_vocabulary = state.get('vectorizer.vocabulary')
            if vectorizer_feature_names is not None:
                # restore deprecated state
                vectorizer = self.vectorizer
                vectorizer.feature_names_ = vectorizer_feature_names
                vectorizer.vocabulary_ = vectorizer_vocabulary
        except KeyError as exc:
            raise KeyError('%r: found %s' % (exc, state.keys())) from exc
        return self

    def fit(self, X):
        flattened_features = [
            word_features
            for sentence_features in X
            for word_features in sentence_features
        ]
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug('flattened_features: %s', flattened_features)
        _fit_transformer_with_progress_logging(
            self.pipeline,
            flattened_features,
            logger=LOGGER,
            message_prefix='FeaturesPreprocessor.fit',
            unit='token-features'
        )
        # self.pipeline.fit(flattened_features)
        vectorizer = self.vectorizer
        LOGGER.info('vectorizer.feature_names: %r', vectorizer.feature_names_)
        LOGGER.info('vectorizer.vocabulary size: %r', len(vectorizer.vocabulary_))
        return self

    def transform(self, X, **_):
        LOGGER.debug('transform, X: %s', X)
        transformed = [
            self.pipeline.transform(sentence_features)
            for sentence_features in X
        ]
        try:
            return np.asarray(transformed)
        except ValueError:
            return np.asarray(transformed, dtype=object)


T_FeaturesPreprocessor = Union[FeaturesPreprocessor, DelftFeaturesPreprocessor]


class Preprocessor(WordPreprocessor):
    # keeping class for pickle compatibility
    pass
