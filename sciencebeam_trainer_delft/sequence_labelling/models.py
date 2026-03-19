import logging
import json
from typing import Dict, List, Type, Union

from tf_keras.models import Model
from tf_keras.layers import (
    Concatenate,
    Dense, LSTM, Bidirectional, Embedding, Input, Dropout,
    TimeDistributed
)

import delft.sequenceLabelling.wrapper
from delft.utilities.crf_wrapper_default import CRFModelWrapperDefault
from delft.utilities.crf_layer import ChainCRF
from delft.sequenceLabelling.models import BaseModel
from delft.sequenceLabelling.models import get_model as _get_model, BidLSTM_CRF_FEATURES

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig


LOGGER = logging.getLogger(__name__)


class CustomModel(BaseModel):
    def __init__(
        self,
        config,
        ntags,
        require_casing: bool = False,
        use_crf: bool = False,
        use_chain_crf: bool = False,
        supports_features: bool = False,
        require_features_indices_input: bool = False,
        stateful: bool = False
    ):
        super().__init__(config, ntags)
        self.require_casing = require_casing
        self.use_crf = use_crf
        self.use_chain_crf = use_chain_crf
        self.supports_features = supports_features
        self.require_features_indices_input = require_features_indices_input
        self.stateful = stateful


def _concatenate_inputs(inputs: list, **kwargs):
    if len(inputs) == 1:
        return inputs[0]
    return Concatenate(**kwargs)(inputs)


# renamed copy of BidLSTM_CRF to demonstrate a custom model
class CustomBidLSTM_CRF(CustomModel):
    """
    A Keras implementation of BidLSTM-CRF for sequence labelling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    def __init__(self, config: ModelConfig, ntags=None):
        super().__init__(
            config,
            ntags,
            require_casing=False,
            use_crf=True,
            use_chain_crf=True,
            supports_features=True,
            stateful=config.stateful
        )

        stateful = self.stateful
        # stateful RNNs require the batch size to be passed in
        input_batch_size = config.batch_size if stateful else None

        model_inputs = []
        lstm_inputs = []
        # build input, directly feed with word embedding by the data generator
        word_input = Input(
            # shape=(None, config.word_embedding_size),
            batch_shape=(input_batch_size, None, config.word_embedding_size),
            name='word_input'
        )
        model_inputs.append(word_input)
        lstm_inputs.append(word_input)

        # build character based embedding
        char_input = Input(
            # shape=(None, config.max_char_length),
            batch_shape=(input_batch_size, None, config.max_char_length),
            dtype='int32',
            name='char_input'
        )
        model_inputs.append(char_input)

        if config.char_embedding_size:
            assert config.char_vocab_size, 'config.char_vocab_size required'
            char_embeddings = TimeDistributed(Embedding(
                input_dim=config.char_vocab_size,
                output_dim=config.char_embedding_size,
                mask_zero=config.char_input_mask_zero,
                name='char_embeddings_embedding'
            ), name='char_embeddings')(char_input)

            chars = TimeDistributed(
                Bidirectional(LSTM(
                    config.num_char_lstm_units,
                    dropout=config.char_input_dropout,
                    recurrent_dropout=config.char_lstm_dropout,
                    return_sequences=False
                )),
                name='char_lstm'
            )(char_embeddings)
            lstm_inputs.append(chars)

        # length of sequence not used for the moment (but used for f1 communication)
        length_input = Input(batch_shape=(None, 1), dtype='int32', name='length_input')

        # combine characters and word embeddings
        LOGGER.debug('model, config.use_features: %s', config.use_features)
        if config.use_features:
            LOGGER.info('model using features')
            assert config.max_feature_size > 0
            features_input = Input(
                batch_shape=(input_batch_size, None, config.max_feature_size),
                name='features_input'
            )
            model_inputs.append(features_input)
            features = features_input
            if config.features_embedding_size:
                features = TimeDistributed(Dense(
                    config.features_embedding_size,
                    name='features_embeddings_dense'
                ), name='features_embeddings')(features)
            LOGGER.info(
                'word_input=%s, chars=%s, features=%s',
                word_input, chars, features
            )
            lstm_inputs.append(features)

        x = _concatenate_inputs(lstm_inputs, name='word_lstm_input')
        x = Dropout(config.dropout, name='word_lstm_input_dropout')(x)

        x = Bidirectional(LSTM(
            units=config.num_word_lstm_units,
            return_sequences=True,
            recurrent_dropout=config.recurrent_dropout,
            stateful=stateful,
        ), name='word_lstm')(x)
        x = Dropout(config.dropout, name='word_lstm_output_dropout')(x)
        x = Dense(
            config.num_word_lstm_units, name='word_lstm_dense', activation='tanh'
        )(x)
        x = Dense(ntags, name='dense_ntags')(x)
        self.crf = ChainCRF(name='crf')
        pred = self.crf(x)

        model_inputs.append(length_input)

        self.model = Model(inputs=model_inputs, outputs=[pred])
        self.config = config


# copied from
# https://github.com/kermitt2/delft/blob/d2f8390ac01779cab959f57aa6e1a8f1d2723505/
# delft/sequenceLabelling/models.py
class CustomBidLSTM_CRF_FEATURES(CustomModel):
    """
    A Keras implementation of BidLSTM-CRF for sequence labelling which create features
    from additional orthogonal information generated by GROBID.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    name = 'CustomBidLSTM_CRF_FEATURES'

    def __init__(self, config, ntags=None):
        super().__init__(
            config,
            ntags,
            require_casing=False,
            use_crf=True,
            use_chain_crf=False,
            supports_features=True,
            require_features_indices_input=True
        )

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(Embedding(
            input_dim=config.char_vocab_size,
            output_dim=config.char_embedding_size,
            mask_zero=True,
            name='char_embeddings'
        ))(char_input)

        chars = TimeDistributed(Bidirectional(LSTM(
            config.num_char_lstm_units,
            return_sequences=False
        )))(char_embeddings)

        # layout features input and embeddings
        features_input = Input(
            shape=(None, len(config.features_indices)),
            dtype='float32',
            name='features_input'
        )

        assert config.features_vocabulary_size, "config.features_vocabulary_size required"
        assert config.features_embedding_size, "config.features_embedding_size required"
        # features_vocabulary_size (default 12) * number_of_features + 1
        # (the zero is reserved for masking / padding)
        features_embedding = TimeDistributed(
            Embedding(
                input_dim=config.features_vocabulary_size * len(config.features_indices) + 1,
                output_dim=config.features_embedding_size,
                mask_zero=True,
                trainable=True,
                name='features_embedding'),
            name="features_embedding_td_1"
        )(features_input)

        assert config.features_lstm_units, "config.features_lstm_units required"
        features_embedding_bd = TimeDistributed(
            Bidirectional(LSTM(config.features_lstm_units, return_sequences=False)),
            name="features_embedding_td_2"
        )(features_embedding)

        features_embedding_out = Dropout(config.dropout)(features_embedding_bd)
        # length of sequence not used for the moment (but used for f1 communication)
        length_input = Input(batch_shape=(None, 1), dtype='int32', name='length_input')

        # combine characters and word embeddings
        x = Concatenate()([word_input, chars, features_embedding_out])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(
            units=config.num_word_lstm_units,
            return_sequences=True,
            recurrent_dropout=config.recurrent_dropout
        ))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        x = Dense(ntags)(x)

        base_model = Model(
            inputs=[word_input, char_input, features_input, length_input],
            outputs=[x]
        )

        self.model = CRFModelWrapperDefault(base_model, ntags)

        input_shapes = [
            (None, None, config.word_embedding_size),      # word_input
            (None, None, config.max_char_length),          # char_input
            (None, None, len(config.features_indices)),    # features_input
            (None, None, 1),                               # length_input
        ]
        self.model.build(input_shape=input_shapes)

        self.config = config


DEFAULT_MODEL_NAMES = [
    'BidLSTM_CRF', 'BidLSTM_CNN', 'BidLSTM_CNN_CRF', 'BidGRU_CRF', 'BidLSTM_CRF_CASING',
    BidLSTM_CRF_FEATURES.name
]

MODEL_MAP: Dict[str, Type[CustomModel]] = {
    'CustomBidLSTM_CRF': CustomBidLSTM_CRF,
    CustomBidLSTM_CRF_FEATURES.name: CustomBidLSTM_CRF_FEATURES
}

IMPLICIT_MODEL_CONFIG_PROPS_MAP = {
    BidLSTM_CRF_FEATURES.name: dict(
        use_features=True,
        use_features_indices_input=True
    ),
    CustomBidLSTM_CRF_FEATURES.name: dict(
        use_features=True,
        use_features_indices_input=True
    )
}


def register_model(name: str, model_class: Type[CustomModel]):
    MODEL_MAP[name] = model_class


def updated_implicit_model_config_props(model_config: ModelConfig):
    implicit_model_config_props = IMPLICIT_MODEL_CONFIG_PROPS_MAP.get(model_config.architecture)
    if not implicit_model_config_props:
        return
    for key, value in implicit_model_config_props.items():
        setattr(model_config, key, value)


def _create_model(
        model_class: Type[CustomModel],
        config: ModelConfig,
        ntags=None) -> CustomModel:
    return model_class(config, ntags=ntags)


def is_model_stateful(model: Union[BaseModel, CustomModel]) -> bool:
    try:
        return model.stateful
    except AttributeError:
        return False


def get_model(config: ModelConfig, preprocessor, ntags=None):
    LOGGER.info(
        'get_model, config: %s, ntags=%s',
        json.dumps(vars(config), indent=4),
        ntags
    )

    model_class = MODEL_MAP.get(config.architecture)
    if not model_class:
        return _get_model(config, preprocessor, ntags=ntags)

    model = _create_model(model_class, config, ntags=ntags)
    config.use_crf = model.use_crf
    config.use_chain_crf = model.use_chain_crf
    preprocessor.return_casing = model.require_casing
    if config.use_features and not model.supports_features:
        LOGGER.warning('features enabled but not supported by model (disabling)')
        config.use_features = False
    preprocessor.return_features = config.use_features
    return model


def get_model_names() -> List[str]:
    return sorted(set(DEFAULT_MODEL_NAMES) | set(MODEL_MAP.keys()))


def patch_get_model():
    delft.sequenceLabelling.wrapper.get_model = get_model
