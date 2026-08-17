import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.models import CustomBidLSTM_CRF
from sciencebeam_trainer_delft.sequence_labelling.saving import ModelLoader, ModelSaver
from sciencebeam_trainer_delft.sequence_labelling.tools.convert_tf_model import (
    convert_model,
    main
)

from tests.sequence_labelling.keras_weights_helper import write_keras_weights_for_model


NTAGS = 5
CHAR_VOCAB_SIZE = 12
MAX_CHAR_LENGTH = 5
MAX_FEATURE_SIZE = 7
WORD_EMBEDDING_SIZE = 3

VOCAB_TAG = {'<PAD>': 0, 'B-<title>': 1, 'I-<title>': 2, 'O': 3, 'B-<author>': 4}


@pytest.fixture(name='model_config')
def _model_config() -> ModelConfig:
    return ModelConfig(
        model_name='test-model',
        architecture='CustomBidLSTM_CRF',
        char_vocab_size=CHAR_VOCAB_SIZE,
        char_embedding_size=5,
        num_char_lstm_units=4,
        max_char_length=MAX_CHAR_LENGTH,
        num_word_lstm_units=6,
        word_embedding_size=WORD_EMBEDDING_SIZE,
        dropout=0.0,
        use_features=True,
        max_feature_size=MAX_FEATURE_SIZE,
        features_embedding_size=0
    )


class _Preprocessor:
    """Enough of a preprocessor for the loader and saver to round-trip."""

    def __init__(self):
        self.vocab_tag = dict(VOCAB_TAG)
        self.indice_tag = {index: tag for tag, index in VOCAB_TAG.items()}
        self.return_features = True
        self.return_casing = False
        self.return_bert_embeddings = False
        self.feature_preprocessor = None


@pytest.fixture(name='tf_model_directory')
def _tf_model_directory(model_config: ModelConfig, temp_dir: Path) -> Path:
    """A directory in the shape a pre-migration release wrote."""
    directory = temp_dir / 'source'
    directory.mkdir()
    model = CustomBidLSTM_CRF(model_config, NTAGS)
    ModelSaver(
        preprocessor=cast(Any, _Preprocessor()), model_config=model_config
    ).save_to(str(directory), model=model)
    (directory / 'model_weights.pt').unlink()
    write_keras_weights_for_model(directory / 'model_weights.hdf5', model)
    return directory


class TestConvertModel:
    def test_should_write_a_torch_state_dict_and_the_other_artifacts(
        self, tf_model_directory: Path, temp_dir: Path
    ):
        output_path = temp_dir / 'converted'
        convert_model(str(tf_model_directory), str(output_path))
        written = {path.name for path in output_path.iterdir()}
        assert 'model_weights.pt' in written
        assert 'config.json' in written
        assert 'preprocessor.json' in written
        assert 'model_weights.hdf5' not in written

    def test_should_leave_the_source_directory_untouched(
        self, tf_model_directory: Path, temp_dir: Path
    ):
        before = {path.name for path in tf_model_directory.iterdir()}
        convert_model(str(tf_model_directory), str(temp_dir / 'converted'))
        assert {path.name for path in tf_model_directory.iterdir()} == before

    def test_should_record_where_the_model_was_converted_from(
        self, tf_model_directory: Path, temp_dir: Path
    ):
        output_path = temp_dir / 'converted'
        convert_model(str(tf_model_directory), str(output_path))
        meta = json.loads((output_path / 'meta.json').read_text(encoding='utf-8'))
        assert meta['converted_from'] == str(tf_model_directory)
        assert meta['architecture'] == 'CustomBidLSTM_CRF'

    def test_should_produce_weights_the_loader_reads_back_unchanged(
        self, model_config: ModelConfig, tf_model_directory: Path, temp_dir: Path
    ):
        output_path = temp_dir / 'converted'
        convert_model(str(tf_model_directory), str(output_path))

        from_hdf5 = CustomBidLSTM_CRF(model_config, NTAGS)
        ModelLoader().load_model_from_directory(str(tf_model_directory), model=from_hdf5)
        from_converted = CustomBidLSTM_CRF(model_config, NTAGS)
        ModelLoader().load_model_from_directory(str(output_path), model=from_converted)
        for key, value in from_hdf5.state_dict().items():
            assert torch.equal(value, from_converted.state_dict()[key]), key

    def test_should_refuse_to_write_over_the_model_it_read(
        self, tf_model_directory: Path
    ):
        with pytest.raises(ValueError, match='must differ'):
            convert_model(str(tf_model_directory), str(tf_model_directory))


class TestMain:
    def test_should_convert_via_the_command_line(
        self, tf_model_directory: Path, temp_dir: Path
    ):
        output_path = temp_dir / 'converted'
        main([
            f'--source-model-path={tf_model_directory}',
            f'--output-path={output_path}'
        ])
        assert (output_path / 'model_weights.pt').exists()
