from pathlib import Path
from typing import Any, cast

import pytest
import torch

from delft.utilities.crf_pytorch import ChainCRF

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.models import CustomBidLSTM_CRF
from sciencebeam_trainer_delft.sequence_labelling.saving import ModelLoader, ModelSaver
from sciencebeam_trainer_delft.sequence_labelling.upstream_patches import (
    patch_chain_crf_eager_build
)
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager


NTAGS = 5
CHAR_VOCAB_SIZE = 12
MAX_CHAR_LENGTH = 5
MAX_FEATURE_SIZE = 7
WORD_EMBEDDING_SIZE = 3


@pytest.fixture(name='patched_chain_crf', autouse=True)
def _patched_chain_crf():
    original_init = ChainCRF.__init__
    patch_chain_crf_eager_build()
    yield
    ChainCRF.__init__ = original_init  # type: ignore[method-assign]


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
