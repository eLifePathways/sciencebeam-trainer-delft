import json
import logging
from pathlib import Path
from typing import Optional

from torch import nn

from delft.sequenceLabelling.preprocess import (
    FeaturesPreprocessor as DelftFeaturesPreprocessor,
    Preprocessor as DelftWordPreprocessor
)

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.preprocess import (
    Preprocessor as ScienceBeamPreprocessor,
    FeaturesPreprocessor as ScienceBeamFeaturesPreprocessor
)
from sciencebeam_trainer_delft.sequence_labelling.saving import (
    ModelSaver,
    ModelLoader,
    get_preprocessor_json,
    get_preprocessor_for_json
)

from ..test_utils import log_on_exception


LOGGER = logging.getLogger(__name__)

SAMPLE_X = [['Word1']]
SAMPLE_FEATURES = [[['F1', 'F2']]]
SAMPLE_Y = [['label1']]


class DummyModel(nn.Module):
    """Stands in for a real architecture; these tests are about the preprocessor."""

    def __init__(self, config, ntags: Optional[int] = None):
        super().__init__()
        self.config = config
        self.ntags = ntags
        self.dense = nn.Linear(2, 2)

    def forward(self, inputs):
        return self.dense(inputs)


def get_vars(obj) -> dict:
    try:
        return obj.__getstate__()
    except TypeError:
        try:
            return vars(obj)
        except TypeError:
            attr_map = {
                attr_name: getattr(obj, attr_name)
                for attr_name in dir(obj)
                if not attr_name.startswith('_')
            }
            return {
                key: value
                for key, value in attr_map.items()
                if not callable(value)
            }


def get_normalized_vars_with_type(obj) -> dict:
    variables = get_vars(obj)
    normalized_vars = {
        key: (
            get_normalized_vars_with_type(value)
            if isinstance(value, ScienceBeamFeaturesPreprocessor)
            else value
        )
        for key, value in variables.items()
        if value is not None
    }
    return {
        'type': type(obj).__qualname__,
        'vars': normalized_vars
    }


class TestJsonSerializePreprocessors:
    def test_should_serialize_features_preprocessor(self):
        features_preprocessor = ScienceBeamFeaturesPreprocessor(features_indices=[0])
        features_preprocessor.fit(SAMPLE_FEATURES)
        preprocessor = DelftWordPreprocessor(feature_preprocessor=features_preprocessor)
        LOGGER.debug('original params: %s', features_preprocessor.vectorizer.get_params())
        LOGGER.debug('original feature_names_: %s', features_preprocessor.vectorizer.feature_names_)
        LOGGER.debug('original vocabulary_: %s', features_preprocessor.vectorizer.vocabulary_)
        output_json = json.dumps(get_preprocessor_json(preprocessor))
        LOGGER.debug('output_json: %s', output_json)
        loaded_preprocessor = get_preprocessor_for_json(json.loads(output_json))
        loaded_features_preprocessor = loaded_preprocessor.feature_preprocessor
        LOGGER.debug('type: %s', type(loaded_features_preprocessor))
        assert isinstance(loaded_features_preprocessor, ScienceBeamFeaturesPreprocessor)
        LOGGER.debug('params: %s', loaded_features_preprocessor.vectorizer.get_params())
        assert (
            loaded_features_preprocessor.vectorizer.get_params()
            == features_preprocessor.vectorizer.get_params()
        )
        assert (
            loaded_features_preprocessor.vectorizer.feature_names_
            == features_preprocessor.vectorizer.feature_names_
        )
        assert (
            loaded_features_preprocessor.vectorizer.vocabulary_
            == features_preprocessor.vectorizer.vocabulary_
        )

    def test_should_serialize_features_indices_input_preprocessor(self):
        features_preprocessor = DelftFeaturesPreprocessor(features_indices=[0])
        features_preprocessor.fit(SAMPLE_FEATURES)
        preprocessor = DelftWordPreprocessor(feature_preprocessor=features_preprocessor)
        LOGGER.debug(
            'original features_vocabulary_size: %s', features_preprocessor.features_vocabulary_size
        )
        LOGGER.debug(
            'original features_indices: %s', features_preprocessor.features_indices
        )
        LOGGER.debug(
            'original features_map_to_index: %s', features_preprocessor.features_map_to_index
        )
        output_json = json.dumps(get_preprocessor_json(preprocessor))
        LOGGER.debug('output_json: %s', output_json)
        loaded_preprocessor = get_preprocessor_for_json(json.loads(output_json))
        LOGGER.debug('type: %s', type(loaded_preprocessor))
        assert isinstance(loaded_preprocessor, DelftWordPreprocessor)
        loaded_features_preprocessor = loaded_preprocessor.feature_preprocessor
        LOGGER.debug('type: %s', type(loaded_features_preprocessor))
        assert isinstance(loaded_features_preprocessor, DelftFeaturesPreprocessor)
        assert (
            loaded_features_preprocessor.features_vocabulary_size
            == features_preprocessor.features_vocabulary_size
        )
        assert (
            loaded_features_preprocessor.features_indices
            == features_preprocessor.features_indices
        )
        assert (
            loaded_features_preprocessor.features_map_to_index
            == features_preprocessor.features_map_to_index
        )

    def test_should_serialize_delft_word_preprocessor(self):
        preprocessor = DelftWordPreprocessor()
        preprocessor.fit(SAMPLE_X, SAMPLE_Y)
        output_json = json.dumps(get_preprocessor_json(preprocessor))
        LOGGER.debug('original vocab_char: %s', preprocessor.vocab_char)
        LOGGER.debug('output_json: %s', output_json)
        loaded_preprocessor = get_preprocessor_for_json(
            json.loads(output_json)
        )
        LOGGER.debug('type: %s', type(loaded_preprocessor))
        assert isinstance(loaded_preprocessor, DelftWordPreprocessor)
        LOGGER.debug('loaded vocab_char: %s', loaded_preprocessor.vocab_char)
        assert (
            loaded_preprocessor.vocab_char
            == preprocessor.vocab_char
        )

    def test_should_serialize_delft_word_preprocessor_to_plain_json(self):
        preprocessor = DelftWordPreprocessor()
        preprocessor.fit(SAMPLE_X, SAMPLE_Y)
        output_json_dict = get_preprocessor_json(preprocessor)
        assert output_json_dict.keys() == preprocessor.__dict__.keys()
        output_json = json.dumps(output_json_dict)
        LOGGER.debug(
            'original features_vocabulary_size: %s', preprocessor.vocab_char
        )
        LOGGER.debug('output_json: %s', output_json)
        loaded_preprocessor = get_preprocessor_for_json(
            json.loads(output_json)
        )
        LOGGER.debug('type: %s', type(loaded_preprocessor))
        assert isinstance(loaded_preprocessor, DelftWordPreprocessor)
        LOGGER.debug('loaded vocab_char: %s', loaded_preprocessor.vocab_char)
        assert (
            loaded_preprocessor.vocab_char
            == preprocessor.vocab_char
        )
        assert (
            json.dumps(get_preprocessor_json(loaded_preprocessor))
            == output_json
        )


class TestModelSaverLoader:
    @log_on_exception
    def test_should_save_and_load_delft_preprocessor_from_json_or_pickle(self, temp_dir: Path):
        model_config = ModelConfig()
        preprocessor = DelftWordPreprocessor()
        preprocessor.fit(SAMPLE_X, SAMPLE_Y)
        model = DummyModel(model_config)
        saver = ModelSaver(preprocessor=preprocessor, model_config=model_config)
        loader = ModelLoader()

        saver.save_to(str(temp_dir), model)

        assert (temp_dir / saver.preprocessor_json_file).exists()
        assert (temp_dir / saver.preprocessor_pickle_file).exists()

        (temp_dir / saver.preprocessor_json_file).rename(
            temp_dir / 'preprocessor_hidden.json'
        )

        loaded_preprocessor = loader.load_preprocessor_from_directory(str(temp_dir))
        assert (
            get_normalized_vars_with_type(loaded_preprocessor)
            == get_normalized_vars_with_type(preprocessor)
        )

        (temp_dir / 'preprocessor_hidden.json').rename(
            temp_dir / saver.preprocessor_json_file
        )

        (temp_dir / saver.preprocessor_pickle_file).rename(
            temp_dir / 'preprocessor_hidden.pickle'
        )

        loaded_preprocessor = loader.load_preprocessor_from_directory(str(temp_dir))
        assert (
            get_normalized_vars_with_type(loaded_preprocessor)
            == get_normalized_vars_with_type(preprocessor)
        )

    @log_on_exception
    def test_should_save_and_load_sciencebeam_preprocessor_from_json_or_pickle(
            self, temp_dir: Path):
        model_config = ModelConfig()
        feature_preprocessor = ScienceBeamFeaturesPreprocessor([0])
        preprocessor = ScienceBeamPreprocessor(
            feature_preprocessor=feature_preprocessor
        )
        preprocessor.fit(SAMPLE_X, SAMPLE_Y)
        preprocessor.fit_features(SAMPLE_X)
        model = DummyModel(model_config)
        saver = ModelSaver(preprocessor=preprocessor, model_config=model_config)
        loader = ModelLoader()

        saver.save_to(str(temp_dir), model)

        assert (temp_dir / saver.preprocessor_json_file).exists()
        assert (temp_dir / saver.preprocessor_pickle_file).exists()

        LOGGER.debug(
            'preprocessor_json: %s',
            (temp_dir / saver.preprocessor_json_file).read_text()
        )

        (temp_dir / saver.preprocessor_json_file).rename(
            temp_dir / 'preprocessor_hidden.json'
        )

        loaded_preprocessor = loader.load_preprocessor_from_directory(str(temp_dir))
        assert (
            get_normalized_vars_with_type(loaded_preprocessor)
            == get_normalized_vars_with_type(preprocessor)
        )

        (temp_dir / 'preprocessor_hidden.json').rename(
            temp_dir / saver.preprocessor_json_file
        )

        (temp_dir / saver.preprocessor_pickle_file).rename(
            temp_dir / 'preprocessor_hidden.pickle'
        )

        loaded_preprocessor = loader.load_preprocessor_from_directory(str(temp_dir))
        assert (
            get_normalized_vars_with_type(loaded_preprocessor)
            == get_normalized_vars_with_type(preprocessor)
        )
