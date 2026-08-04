"""Adapts the batch-producing data generator to the tensors the model takes.

The generator yields a list of numpy arrays, in the order the Keras model
declared its inputs. The torch model takes a dict, so the arrays are named here
from the same flags that decided whether they were produced.

Batches are produced in the calling process. The alternative, a torch Dataset
with worker processes, would move word-embedding lookup into forked children,
where the LMDB environment the embeddings are read from is not fork-safe.
"""
import logging
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from sciencebeam_trainer_delft.sequence_labelling.data_generator import DataGenerator


LOGGER = logging.getLogger(__name__)


WORD_INPUT = 'word_input'
CHAR_INPUT = 'char_input'
CASING_INPUT = 'casing_input'
FEATURES_INPUT = 'features_input'
LENGTH_INPUT = 'length_input'

LONG_INPUTS = {CHAR_INPUT, CASING_INPUT, LENGTH_INPUT}


def get_input_names(data_generator: DataGenerator) -> List[str]:
    """Names the arrays the generator produces, in the order it appends them."""
    names = [WORD_INPUT, CHAR_INPUT]
    if data_generator.preprocessor.return_casing:
        names.append(CASING_INPUT)
    if data_generator.preprocessor.return_features:
        names.append(FEATURES_INPUT)
    names.append(LENGTH_INPUT)
    return names


def to_input_tensor(name: str, array: np.ndarray) -> torch.Tensor:
    if name in LONG_INPUTS:
        return torch.as_tensor(np.asarray(array), dtype=torch.long)
    return torch.as_tensor(np.asarray(array), dtype=torch.float32)


def to_label_tensor(labels: np.ndarray) -> torch.Tensor:
    """Returns tag indices, whichever representation the generator produced.

    Labels come back one-hot when the model uses a chain CRF, and as indices
    otherwise. The CRF takes indices either way.
    """
    label_array = np.asarray(labels)
    if label_array.ndim == 3:
        label_array = label_array.argmax(axis=-1)
    return torch.as_tensor(label_array, dtype=torch.long)


def to_model_inputs(
    input_names: List[str],
    arrays: List[np.ndarray],
    device: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """Names and converts the arrays of one batch."""
    if len(arrays) != len(input_names):
        raise AssertionError(
            f'expected {len(input_names)} inputs {input_names}, got {len(arrays)}'
        )
    inputs = {
        name: to_input_tensor(name, array)
        for name, array in zip(input_names, arrays)
    }
    if device:
        inputs = {name: value.to(device) for name, value in inputs.items()}
    return inputs


class DataLoader:
    """Yields `(inputs, labels)` for each batch of a data generator."""

    def __init__(self, data_generator: DataGenerator, device: Optional[str] = None):
        self.data_generator = data_generator
        self.device = device
        self.input_names = get_input_names(data_generator)

    def __len__(self) -> int:
        return len(self.data_generator)

    def get_batch(
        self, index: int
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        arrays, labels = self.data_generator[index]
        inputs = to_model_inputs(self.input_names, arrays, device=self.device)
        label_tensor = None if labels is None else to_label_tensor(labels)
        if self.device and label_tensor is not None:
            label_tensor = label_tensor.to(self.device)
        return inputs, label_tensor

    def __iter__(
        self
    ) -> Iterator[Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]]:
        for index in range(len(self)):
            yield self.get_batch(index)
        self.data_generator.on_epoch_end()
