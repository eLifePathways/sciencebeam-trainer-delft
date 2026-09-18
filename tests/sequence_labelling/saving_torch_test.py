from pathlib import Path
from typing import Any, Iterator, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from delft.utilities.weights import SAFETENSORS_WEIGHT_FILE_NAME, save_weights

import sciencebeam_trainer_delft.sequence_labelling.saving as saving_module
from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.models import CustomBidLSTM_CRF
from sciencebeam_trainer_delft.sequence_labelling.saving import ModelLoader, ModelSaver
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager

from tests.sequence_labelling.keras_weights_helper import write_keras_weights_for_model


NTAGS = 5
CHAR_VOCAB_SIZE = 12
MAX_CHAR_LENGTH = 5
MAX_FEATURE_SIZE = 7
WORD_EMBEDDING_SIZE = 3


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
        features_embedding_size=0,
        # the TensorFlow era models are all chain CRF, so that is what their
        # weights convert into
        use_chain_crf=True
    )


def _crf_transitions(model: CustomBidLSTM_CRF) -> torch.Tensor:
    """The CRF parameters are registered dynamically, so not statically typed."""
    return cast(torch.Tensor, model.crf.U)


def _inputs():
    torch.manual_seed(7)
    return {
        'word_input': torch.randn(2, 4, WORD_EMBEDDING_SIZE),
        'char_input': torch.randint(0, CHAR_VOCAB_SIZE, (2, 4, MAX_CHAR_LENGTH)),
        'features_input': torch.randn(2, 4, MAX_FEATURE_SIZE)
    }


class TestModelSaverLoader:
    def test_should_write_the_documented_layout(
        self, model_config: ModelConfig, temp_dir: Path
    ):
        model = CustomBidLSTM_CRF(model_config, NTAGS)
        saver = ModelSaver(preprocessor=cast(Any, None), model_config=model_config)
        saver.save_to(str(temp_dir), model=model, meta={'epoch': 3})
        written = {path.name for path in temp_dir.iterdir()}
        assert 'config.json' in written
        assert 'model_weights.pt' in written
        assert 'meta.json' in written

    def test_should_write_a_torch_state_dict(
        self, model_config: ModelConfig, temp_dir: Path
    ):
        model = CustomBidLSTM_CRF(model_config, NTAGS)
        saver = ModelSaver(preprocessor=cast(Any, None), model_config=model_config)
        saver.save_to(str(temp_dir), model=model)
        state_dict = torch.load(temp_dir / 'model_weights.pt', map_location='cpu')
        assert 'crf.U' in state_dict
        assert 'dense_ntags.weight' in state_dict

    def test_should_round_trip_the_weights_into_a_fresh_model(
        self, model_config: ModelConfig, temp_dir: Path
    ):
        model = CustomBidLSTM_CRF(model_config, NTAGS)
        model.eval()
        inputs = _inputs()
        expected_logits = model(inputs)['logits']

        ModelSaver(preprocessor=cast(Any, None), model_config=model_config).save_to(
            str(temp_dir), model=model
        )

        loaded_model = CustomBidLSTM_CRF(model_config, NTAGS)
        loaded_model.eval()
        # a fresh model starts from different weights
        assert not torch.allclose(loaded_model(inputs)['logits'], expected_logits)

        ModelLoader(download_manager=DownloadManager()).load_model_from_directory(
            str(temp_dir), model=loaded_model
        )
        assert torch.equal(loaded_model(inputs)['logits'], expected_logits)

    def test_should_round_trip_the_model_config(
        self, model_config: ModelConfig, temp_dir: Path
    ):
        model = CustomBidLSTM_CRF(model_config, NTAGS)
        ModelSaver(preprocessor=cast(Any, None), model_config=model_config).save_to(
            str(temp_dir), model=model
        )
        loaded_config = ModelLoader(
            download_manager=DownloadManager()
        ).load_model_config_from_directory(str(temp_dir))
        assert loaded_config.architecture == 'CustomBidLSTM_CRF'
        assert loaded_config.num_word_lstm_units == model_config.num_word_lstm_units
        assert loaded_config.max_feature_size == model_config.max_feature_size

    def test_should_reject_weights_that_do_not_match_the_architecture(
        self, model_config: ModelConfig, temp_dir: Path
    ):
        model = CustomBidLSTM_CRF(model_config, NTAGS)
        ModelSaver(preprocessor=cast(Any, None), model_config=model_config).save_to(
            str(temp_dir), model=model
        )
        wider_config = ModelConfig(
            **{**vars(model_config), 'num_word_lstm_units': 8}
        )
        mismatched_model = CustomBidLSTM_CRF(wider_config, NTAGS)
        with pytest.raises(RuntimeError):
            ModelLoader(download_manager=DownloadManager()).load_model_from_directory(
                str(temp_dir), model=mismatched_model
            )


class TestModelLoaderLegacyWeights:
    def test_should_convert_tf_era_weights_when_there_is_no_torch_state_dict(
        self, model_config: ModelConfig, temp_dir: Path
    ):
        model = CustomBidLSTM_CRF(model_config, NTAGS)
        write_keras_weights_for_model(temp_dir / 'model_weights.hdf5', model)

        loaded_model = CustomBidLSTM_CRF(model_config, NTAGS)
        ModelLoader(download_manager=DownloadManager()).load_model_from_directory(
            str(temp_dir), model=loaded_model
        )
        assert torch.equal(_crf_transitions(loaded_model), _crf_transitions(model))
        assert torch.equal(loaded_model.dense_ntags.weight, model.dense_ntags.weight)

    def test_should_not_write_to_the_directory_it_loaded_from(
        self, model_config: ModelConfig, temp_dir: Path
    ):
        model = CustomBidLSTM_CRF(model_config, NTAGS)
        write_keras_weights_for_model(temp_dir / 'model_weights.hdf5', model)
        before = {path.name for path in temp_dir.iterdir()}

        ModelLoader(download_manager=DownloadManager()).load_model_from_directory(
            str(temp_dir), model=CustomBidLSTM_CRF(model_config, NTAGS)
        )
        assert {path.name for path in temp_dir.iterdir()} == before

    def test_should_prefer_the_torch_state_dict_where_both_are_present(
        self, model_config: ModelConfig, temp_dir: Path
    ):
        torch_model = CustomBidLSTM_CRF(model_config, NTAGS)
        ModelSaver(preprocessor=cast(Any, None), model_config=model_config).save_to(
            str(temp_dir), model=torch_model
        )
        keras_model = CustomBidLSTM_CRF(model_config, NTAGS)
        write_keras_weights_for_model(temp_dir / 'model_weights.hdf5', keras_model)
        assert not torch.equal(_crf_transitions(keras_model), _crf_transitions(torch_model))

        loaded_model = CustomBidLSTM_CRF(model_config, NTAGS)
        ModelLoader(download_manager=DownloadManager()).load_model_from_directory(
            str(temp_dir), model=loaded_model
        )
        assert torch.equal(_crf_transitions(loaded_model), _crf_transitions(torch_model))


class TestModelLoaderSafetensorsWeights:
    def test_should_load_the_safetensors_weights_delft_saves(
        self, model_config: ModelConfig, temp_dir: Path
    ):
        model = CustomBidLSTM_CRF(model_config, NTAGS)
        save_weights(model, str(temp_dir / SAFETENSORS_WEIGHT_FILE_NAME))

        loaded_model = CustomBidLSTM_CRF(model_config, NTAGS)
        ModelLoader(download_manager=DownloadManager()).load_model_from_directory(
            str(temp_dir), model=loaded_model
        )
        assert torch.equal(_crf_transitions(loaded_model), _crf_transitions(model))
        assert torch.equal(loaded_model.dense_ntags.weight, model.dense_ntags.weight)

    def test_should_prefer_safetensors_weights_to_the_tf_era_ones(
        self, model_config: ModelConfig, temp_dir: Path
    ):
        safetensors_model = CustomBidLSTM_CRF(model_config, NTAGS)
        save_weights(safetensors_model, str(temp_dir / SAFETENSORS_WEIGHT_FILE_NAME))
        keras_model = CustomBidLSTM_CRF(model_config, NTAGS)
        write_keras_weights_for_model(temp_dir / 'model_weights.hdf5', keras_model)

        loaded_model = CustomBidLSTM_CRF(model_config, NTAGS)
        ModelLoader(download_manager=DownloadManager()).load_model_from_directory(
            str(temp_dir), model=loaded_model
        )
        assert torch.equal(
            _crf_transitions(loaded_model), _crf_transitions(safetensors_model)
        )


@pytest.fixture(name='resolve_model_mock')
def _resolve_model_mock() -> Iterator[MagicMock]:
    with patch.object(saving_module, 'resolve_model') as mock:
        yield mock


class TestModelLoaderDownloadModel:
    @pytest.mark.parametrize('model_path', [
        'hf://owner/repository/model-name',
        'hf://owner/repository@v1.1.0/model-name',
        'https://huggingface.co/owner/repository/tree/main/model-name',
        'https://example.org/models/model-name.zip',
        'https://example.org/models/model-name.tar.gz'
    ])
    def test_should_leave_what_delft_calls_remote_to_delft(
        self, resolve_model_mock: MagicMock, temp_dir: Path, model_path: str
    ):
        model_loader = ModelLoader(download_manager=DownloadManager(str(temp_dir)))
        assert model_loader.download_model(model_path) == resolve_model_mock.return_value
        resolve_model_mock.assert_called_once_with(
            model_path, cache_dir=str(temp_dir / 'models')
        )

    def test_should_return_a_local_directory_as_it_is(
        self, resolve_model_mock: MagicMock, temp_dir: Path
    ):
        model_loader = ModelLoader(download_manager=DownloadManager(str(temp_dir)))
        assert model_loader.download_model(str(temp_dir)) == str(temp_dir)
        resolve_model_mock.assert_not_called()

    def test_should_not_leave_a_cloud_storage_archive_to_delft(
        self, resolve_model_mock: MagicMock, temp_dir: Path
    ):
        model_loader = ModelLoader(download_manager=DownloadManager(str(temp_dir)))
        with patch.object(saving_module, 'copy_directory_with_source_meta') as copy_mock:
            local_path = model_loader.download_model('gs://bucket/models/model-name.tar.gz')
        resolve_model_mock.assert_not_called()
        copy_mock.assert_called_once_with('gs://bucket/models/model-name.tar.gz', local_path)
        assert local_path.startswith(str(temp_dir))
