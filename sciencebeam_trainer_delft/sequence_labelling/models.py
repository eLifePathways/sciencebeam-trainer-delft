"""Model architectures and the registry that builds them.

Every model published from this repo is `CustomBidLSTM_CRF` with a features
embedding size of 0, which feeds the raw feature matrix straight into the word
LSTM. Upstream's `BidLSTM_CRF_FEATURES` embeds features through `nn.Embedding`
and a features LSTM instead and has no size-0 path, so it cannot represent
these models at any configuration.

Layer and parameter names mirror the Keras implementation so that weights can
be mapped across mechanically.
"""
import logging
from typing import Dict, List, Optional, Type

import torch
from torch import nn

import delft.sequenceLabelling.wrapper
from delft.sequenceLabelling.models import (
    BidLSTM_CRF_FEATURES,
    get_model as _get_model
)
from delft.utilities.crf_pytorch import ChainCRF

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.upstream_patches import (
    patch_chain_crf_eager_build
)


LOGGER = logging.getLogger(__name__)


class CharacterEncoder(nn.Module):
    """Encodes the characters of each token with a bidirectional LSTM.

    The Keras model wraps an `Embedding` and a `Bidirectional(LSTM)` in
    `TimeDistributed`; here the token axis is flattened into the batch instead.
    """

    def __init__(
        self,
        char_vocab_size: int,
        char_embedding_size: int,
        num_char_lstm_units: int,
        char_input_mask_zero: bool = False,
        char_input_dropout: float = 0.0
    ):
        super().__init__()
        self.char_embeddings = nn.Embedding(
            char_vocab_size,
            char_embedding_size,
            padding_idx=0 if char_input_mask_zero else None
        )
        self.char_input_dropout = nn.Dropout(char_input_dropout)
        self.char_lstm = nn.LSTM(
            char_embedding_size,
            num_char_lstm_units,
            batch_first=True,
            bidirectional=True
        )
        self.output_size = num_char_lstm_units * 2

    def forward(self, char_input: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, max_char_length = char_input.shape
        flattened = char_input.reshape(batch_size * sequence_length, max_char_length)
        char_embeddings = self.char_input_dropout(self.char_embeddings(flattened))
        _, (hidden, _) = self.char_lstm(char_embeddings)
        # concatenate the final state of each direction, as Keras does for
        # Bidirectional(LSTM(return_sequences=False))
        encoded = torch.cat([hidden[0], hidden[1]], dim=-1)
        return encoded.view(batch_size, sequence_length, self.output_size)


class CustomBidLSTM_CRF(nn.Module):  # pylint: disable=invalid-name
    """BiLSTM-CRF over word embeddings, character encodings and features.

    Features are passed through unchanged when `features_embedding_size` is 0,
    which is what every published model does, and through a dense projection
    otherwise.
    """

    name = 'CustomBidLSTM_CRF'
    use_crf = True
    use_chain_crf = True
    supports_features = True

    def __init__(self, config: ModelConfig, ntags: int):
        super().__init__()
        self.config = config
        self.ntags = ntags

        self.char_encoder = CharacterEncoder(
            char_vocab_size=config.char_vocab_size,
            char_embedding_size=config.char_embedding_size,
            num_char_lstm_units=config.num_char_lstm_units,
            char_input_mask_zero=config.char_input_mask_zero,
            char_input_dropout=config.char_input_dropout
        )

        word_lstm_input_size = config.word_embedding_size + self.char_encoder.output_size

        self.features_embeddings_dense: Optional[nn.Linear] = None
        if config.use_features:
            assert config.max_feature_size > 0, 'config.max_feature_size required'
            if config.features_embedding_size:
                self.features_embeddings_dense = nn.Linear(
                    config.max_feature_size, config.features_embedding_size
                )
                word_lstm_input_size += config.features_embedding_size
            else:
                # pass the feature matrix through unchanged
                word_lstm_input_size += config.max_feature_size

        self.dropout = nn.Dropout(config.dropout)
        self.word_lstm = nn.LSTM(
            word_lstm_input_size,
            config.num_word_lstm_units,
            batch_first=True,
            bidirectional=True
        )
        self.word_lstm_dense = nn.Linear(
            config.num_word_lstm_units * 2, config.num_word_lstm_units
        )
        self.dense_ntags = nn.Linear(config.num_word_lstm_units, ntags)
        self.crf = ChainCRF(ntags)

    def get_word_lstm_input(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        lstm_inputs: List[torch.Tensor] = []
        word_input = inputs.get('word_input')
        if word_input is not None and word_input.shape[-1]:
            lstm_inputs.append(word_input)
        lstm_inputs.append(self.char_encoder(inputs['char_input']))
        if self.config.use_features:
            features_input = inputs['features_input']
            if self.features_embeddings_dense is not None:
                features_input = self.features_embeddings_dense(features_input)
            lstm_inputs.append(features_input)
        if len(lstm_inputs) == 1:
            return lstm_inputs[0]
        return torch.cat(lstm_inputs, dim=-1)

    def get_logits(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.dropout(self.get_word_lstm_input(inputs))
        lstm_output, _ = self.word_lstm(x)
        x = self.dropout(lstm_output)
        x = torch.tanh(self.word_lstm_dense(x))
        return self.dense_ntags(x)

    def get_crf_mask(self, labels: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.config.masked_crf_loss:
            return None
        # PAD has label index 0; without masking those positions dominate the
        # gradient on padded batches
        return labels != 0

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        logits = self.get_logits(inputs)
        outputs = {'logits': logits}
        if labels is not None:
            outputs['loss'] = self.crf(logits, labels, mask=self.get_crf_mask(labels))
        return outputs

    def decode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            return self.crf.decode(self.get_logits(inputs))


MODEL_MAP: Dict[str, Type[nn.Module]] = {
    CustomBidLSTM_CRF.name: CustomBidLSTM_CRF
}

DEFAULT_MODEL_NAMES = [
    'BidLSTM_CRF', 'BidLSTM_ChainCRF', 'BidLSTM_CNN', 'BidLSTM_CNN_CRF', 'BidGRU_CRF',
    'BidLSTM_CRF_CASING', BidLSTM_CRF_FEATURES.name
]

# architectures that only work with features, whatever the config asked for
IMPLICIT_MODEL_CONFIG_PROPS_MAP = {
    BidLSTM_CRF_FEATURES.name: {
        'use_features': True,
        'use_features_indices_input': True
    },
    'CustomBidLSTM_CRF_FEATURES': {
        'use_features': True,
        'use_features_indices_input': True
    }
}


def register_model(name: str, model_class: Type[nn.Module]):
    MODEL_MAP[name] = model_class


def updated_implicit_model_config_props(model_config: ModelConfig):
    implicit_model_config_props = IMPLICIT_MODEL_CONFIG_PROPS_MAP.get(
        model_config.architecture
    )
    if not implicit_model_config_props:
        return
    for key, value in implicit_model_config_props.items():
        setattr(model_config, key, value)


def is_model_stateful(model: nn.Module) -> bool:
    """Reports whether the model carries LSTM state across batches.

    Nothing does: Keras exposed this directly and torch does not, so the
    architectures here are stateless.
    """
    return getattr(model, 'stateful', False)


def get_model(config: ModelConfig, preprocessor, ntags: Optional[int] = None):
    LOGGER.info('get_model, architecture=%s, ntags=%s', config.architecture, ntags)
    # the CRF has to register its transition parameters before the model is
    # constructed, or they are absent from the optimizer and the state dict
    patch_chain_crf_eager_build()
    model_class = MODEL_MAP.get(config.architecture)
    if not model_class:
        assert ntags is not None, 'ntags required'
        return _get_model(config, ntags)
    model = model_class(config, ntags)
    config.use_crf = model.use_crf
    config.use_chain_crf = model.use_chain_crf
    preprocessor.return_casing = getattr(model, 'require_casing', False)
    if config.use_features and not getattr(model, 'supports_features', False):
        LOGGER.warning('features enabled but not supported by model (disabling)')
        config.use_features = False
    preprocessor.return_features = config.use_features
    return model


def get_model_names() -> List[str]:
    return sorted(set(DEFAULT_MODEL_NAMES) | set(MODEL_MAP.keys()))


def patch_get_model():
    delft.sequenceLabelling.wrapper.get_model = get_model
