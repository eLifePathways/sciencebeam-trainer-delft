import json
from typing import List, Optional

from delft.sequenceLabelling.config import (
    ModelConfig as _ModelConfig,
    TrainingConfig as _TrainingConfig
)


FIRST_MODEL_VERSION = 1
MODEL_VERSION = 2

DEFAULT_CHAR_INPUT_DROPOUT = 0.0
DEFAULT_CHAR_LSTM_DROPOUT = 0.0

NOT_SET = -1


class ModelConfig(_ModelConfig):
    def __init__(
        self,
        *args,
        use_word_embeddings: bool = True,
        use_features: bool = False,
        continuous_features_indices: Optional[List[int]] = None,
        max_feature_size: int = 50,
        additional_token_feature_indices: Optional[List[int]] = None,
        text_feature_indices: Optional[List[int]] = None,
        unroll_text_feature_index: Optional[int] = None,
        concatenated_embeddings_token_count: Optional[int] = None,
        use_features_indices_input: bool = False,
        char_input_mask_zero: bool = False,
        char_input_dropout: float = DEFAULT_CHAR_INPUT_DROPOUT,
        char_lstm_dropout: float = DEFAULT_CHAR_LSTM_DROPOUT,
        stateful: bool = False,
        masked_crf_loss: bool = False,
        model_version: int = MODEL_VERSION,
        # deprecated
        feature_indices: Optional[List[int]] = None,
        feature_embedding_size: int = NOT_SET,
        **kwargs
    ):
        if feature_indices:
            kwargs['features_indices'] = feature_indices
        if feature_embedding_size != NOT_SET:
            kwargs['features_embedding_size'] = feature_embedding_size
        super().__init__(*args)
        self.use_word_embeddings = use_word_embeddings
        self.additional_token_feature_indices = additional_token_feature_indices
        self.text_feature_indices = text_feature_indices
        self.unroll_text_feature_index = unroll_text_feature_index
        self.concatenated_embeddings_token_count = concatenated_embeddings_token_count
        self.use_features = use_features
        self.continuous_features_indices = continuous_features_indices
        self.max_feature_size = max_feature_size
        self.use_features_indices_input = use_features_indices_input
        self.char_input_mask_zero = char_input_mask_zero
        self.char_input_dropout = char_input_dropout
        self.char_lstm_dropout = char_lstm_dropout
        self.stateful = stateful
        self.masked_crf_loss = masked_crf_loss
        self.model_version = model_version
        for key, val in kwargs.items():
            setattr(self, key, val)

    @property
    def is_deprecated_padded_batch_text_list_enabled(self):
        return bool(
            self.model_version < 2
            and self.text_feature_indices
        )

    def save(self, file):
        try:
            super().save(file)
        except TypeError:
            json.dump(vars(self), file, sort_keys=False, indent=4)

    @classmethod
    def load(cls, file):
        variables = json.load(file)
        if 'use_chain_crf' not in variables and variables.get('use_crf', False):
            variables['use_chain_crf'] = True
        if 'architecture' not in variables and variables.get('model_type'):
            variables['architecture'] = variables.get('model_type')
            del variables['model_type']
        if variables.get('use_chain_crf') and variables['architecture'] == 'BidLSTM_CRF':
            variables['architecture'] = 'BidLSTM_ChainCRF'
        self = cls()
        # model version is assumed to the first version if not saved
        self.model_version = FIRST_MODEL_VERSION
        for key, val in variables.items():
            setattr(self, key, val)
        return self

    # alias due to properties having been renamed in upstream implementation
    @property
    def feature_indices(self) -> Optional[List[int]]:
        features_indices = self.features_indices
        if not features_indices:
            features_indices = self.__dict__.get('feature_indices', [])
        return features_indices

    @feature_indices.setter
    def feature_indices(self, feature_indices: List[int]):
        self.features_indices = feature_indices

    @property
    def feature_embedding_size(self) -> Optional[int]:
        return (
            self.features_embedding_size
            or self.__dict__.get('feature_embedding_size')
        )

    @feature_embedding_size.setter
    def feature_embedding_size(self, feature_embedding_size: int):
        self.features_embedding_size = feature_embedding_size


class TrainingConfig(_TrainingConfig):
    def __init__(
        self,
        *args,
        learning_rate=0.001,
        initial_epoch: Optional[int] = None,
        input_window_stride: Optional[int] = None,
        checkpoint_epoch_interval: int = 1,
        initial_meta: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(*args, learning_rate=learning_rate, **kwargs)
        self.initial_epoch = initial_epoch
        self.input_window_stride = input_window_stride
        self.checkpoint_epoch_interval = checkpoint_epoch_interval
        self.initial_meta = initial_meta
