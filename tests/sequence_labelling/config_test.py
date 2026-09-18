import json

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig


FEATURE_INDICES_1 = [9, 10, 11]

FEATURES_EMBEDDING_SIZE_1 = 13


class TestModelConfig:
    def test_should_be_able_to_pass_in_feature_indices(self):
        model_config = ModelConfig(feature_indices=FEATURE_INDICES_1)
        assert model_config.feature_indices == FEATURE_INDICES_1
        assert model_config.features_indices == FEATURE_INDICES_1

    def test_should_be_able_to_pass_in_features_indices(self):
        model_config = ModelConfig(features_indices=FEATURE_INDICES_1)
        assert model_config.feature_indices == FEATURE_INDICES_1
        assert model_config.features_indices == FEATURE_INDICES_1

    def test_should_be_able_to_pass_in_feature_embedding_size(self):
        model_config = ModelConfig(feature_embedding_size=FEATURES_EMBEDDING_SIZE_1)
        assert model_config.feature_embedding_size == FEATURES_EMBEDDING_SIZE_1
        assert model_config.features_embedding_size == FEATURES_EMBEDDING_SIZE_1

    def test_should_be_able_to_pass_in_features_embedding_size(self):
        model_config = ModelConfig(features_embedding_size=FEATURES_EMBEDDING_SIZE_1)
        assert model_config.feature_embedding_size == FEATURES_EMBEDDING_SIZE_1
        assert model_config.features_embedding_size == FEATURES_EMBEDDING_SIZE_1


# what the deployed models carry: neither `architecture` nor `use_chain_crf`,
# both of which postdate them
DEPLOYED_MODEL_CONFIG = {
    'model_type': 'CustomBidLSTM_CRF',
    'use_crf': True,
    'use_features': True,
    'features_indices': FEATURE_INDICES_1,
    'features_embedding_size': 0,
    'max_feature_size': 53
}


class TestModelConfigLoad:
    def test_should_read_a_deployed_model_config_as_chain_crf(self, tmp_path):
        # the deployed models were trained before the architecture could take
        # either CRF, so their weights are the chain CRF's whatever the current
        # default is
        path = tmp_path / 'config.json'
        path.write_text(json.dumps(DEPLOYED_MODEL_CONFIG))
        with path.open() as fp:
            model_config = ModelConfig.load(fp)
        assert model_config.architecture == 'CustomBidLSTM_CRF'
        assert model_config.use_chain_crf is True

    def test_should_keep_an_explicit_use_chain_crf(self, tmp_path):
        path = tmp_path / 'config.json'
        path.write_text(json.dumps({
            **DEPLOYED_MODEL_CONFIG,
            'architecture': 'CustomBidLSTM_CRF',
            'use_chain_crf': False
        }))
        with path.open() as fp:
            model_config = ModelConfig.load(fp)
        assert model_config.use_chain_crf is False

    def test_should_default_a_new_config_to_the_plain_crf(self):
        assert ModelConfig().use_chain_crf is False
