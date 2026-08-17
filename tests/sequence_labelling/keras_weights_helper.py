"""Writes a Keras-layout weights file from a torch model, for tests.

This is the inverse of `tf_weight_conversion`, so that a test can produce a
TF-era file whose contents it already knows, without carrying a published model
as a fixture.
"""
from pathlib import Path
from typing import Any, Dict, Union, cast

import h5py
import numpy as np
import torch
from torch import nn

from sciencebeam_trainer_delft.sequence_labelling.models import CustomBidLSTM_CRF


def _lstm_arrays(prefix: str, lstm: nn.LSTM) -> Dict[str, np.ndarray]:
    arrays = {}
    for direction, suffix in (('forward', ''), ('backward', '_reverse')):
        weight_ih = getattr(lstm, f'weight_ih_l0{suffix}')
        weight_hh = getattr(lstm, f'weight_hh_l0{suffix}')
        bias_ih = getattr(lstm, f'bias_ih_l0{suffix}')
        bias_hh = getattr(lstm, f'bias_hh_l0{suffix}')
        path = f'{prefix}/{direction}_lstm'
        arrays[f'{path}/kernel'] = _numpy(weight_ih).T.copy()
        arrays[f'{path}/recurrent_kernel'] = _numpy(weight_hh).T.copy()
        # torch adds its two biases, so the single Keras bias is their sum
        arrays[f'{path}/bias'] = _numpy(bias_ih + bias_hh)
    return arrays


def _dense_arrays(prefix: str, dense: nn.Linear) -> Dict[str, np.ndarray]:
    return {
        f'{prefix}/kernel': _numpy(dense.weight).T.copy(),
        f'{prefix}/bias': _numpy(dense.bias),
    }


def _numpy(value: Any) -> np.ndarray:
    """The CRF parameters are registered dynamically, so not statically typed."""
    return cast(torch.Tensor, value).detach().numpy().copy()


def get_keras_arrays_for_model(model: CustomBidLSTM_CRF) -> Dict[str, np.ndarray]:
    """Builds the Keras weights a `CustomBidLSTM_CRF` of this shape would hold."""
    arrays: Dict[str, np.ndarray] = {
        'char_embeddings/char_embeddings/embeddings':
            _numpy(model.char_encoder.char_embeddings.weight),
        'chain_crf_1/chain_crf_1/U': _numpy(model.crf.U),
        'chain_crf_1/chain_crf_1/b_start': _numpy(model.crf.b_start),
        'chain_crf_1/chain_crf_1/b_end': _numpy(model.crf.b_end),
    }
    arrays.update(_lstm_arrays('char_lstm/char_lstm', model.char_encoder.char_lstm))
    arrays.update(_lstm_arrays('bidirectional_2/bidirectional_2', model.word_lstm))
    arrays.update(_dense_arrays('dense_1/dense_1', model.word_lstm_dense))
    arrays.update(_dense_arrays('dense_ntags/dense_ntags', model.dense_ntags))
    return arrays


def write_keras_weights_for_model(
    filepath: Union[str, Path], model: CustomBidLSTM_CRF
) -> str:
    with h5py.File(str(filepath), 'w') as h5_file:
        for name, value in get_keras_arrays_for_model(model).items():
            h5_file.create_dataset(f'model_weights/{name}:0', data=value)
    return str(filepath)
