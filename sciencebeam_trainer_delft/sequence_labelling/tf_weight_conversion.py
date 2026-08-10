"""Maps TF-era Keras weights onto the torch modules that replaced them.

Model directories published before the PyTorch migration hold a Keras
``model_weights.hdf5`` where the current code writes a torch state dict. Reading
one needs ``h5py`` and nothing else -- no TensorFlow is involved, so this runs in
the normal environment.

The mapping is keyed on **role and shape, never on layer names**. Keras layer
names are auto-generated in most published files (``time_distributed_1``,
``dense_2``, ``lstm_cell_5``) and only partly meaningful in the rest, so they
differ between model generations that are otherwise identical. Roles come from
the weight names *within* a layer, which the Keras layer classes set and which
are therefore stable: an LSTM direction has ``kernel``, ``recurrent_kernel`` and
``bias``; a dense has ``kernel`` and ``bias``; a CRF has either
``U``/``b_start``/``b_end`` or ``chain_kernel``/``left_boundary``/
``right_boundary``. Each source tensor is then matched to the destination
parameter whose shape it fits, and an ambiguous or unmatched tensor raises.

Because destinations are read off an already-constructed module, this needs no
knowledge of which architecture it is converting, and every check is a
comparison against a known-correct shape.

What is deliberately not read: the Keras ``model_config`` that a fully saved
file carries. It records ``recurrent_activation: hard_sigmoid`` from Keras
2.2.4, while these models have been served under ``sigmoid`` since the
``tf_keras`` migration, so honouring it would diverge from the behaviour the
recorded scores describe. Optimizer state is skipped for the same reason it is
not saved: inference-equivalent weights are the whole goal.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from torch import nn


LOGGER = logging.getLogger(__name__)


LEGACY_WEIGHT_FILE_SUFFIX = '.hdf5'
LEGACY_WEIGHT_FILE = 'model_weights' + LEGACY_WEIGHT_FILE_SUFFIX

# Keras prefixes the weight groups with this when the whole model was saved,
# rather than just its weights
_MODEL_WEIGHTS_PREFIX = 'model_weights/'
_OPTIMIZER_WEIGHTS_PREFIX = 'optimizer_weights'

_LSTM_WEIGHT_NAMES = frozenset({'kernel', 'recurrent_kernel', 'bias'})
_DENSE_WEIGHT_NAMES = frozenset({'kernel', 'bias'})
_CHAIN_CRF_WEIGHT_NAMES = frozenset({'U', 'b_start', 'b_end'})
_CRF_WEIGHT_NAMES = frozenset({'chain_kernel', 'left_boundary', 'right_boundary'})


class TfWeightConversionError(RuntimeError):
    """Raised when the weights cannot be mapped faithfully.

    A converted model that silently holds a wrong tensor is worse than one that
    failed to convert, because the end-to-end assertions downstream check that
    output is structurally plausible rather than that it is right.
    """


def get_legacy_weight_file_path(directory: str) -> str:
    return os.path.join(directory, LEGACY_WEIGHT_FILE)


class KerasLayerWeights:
    """The weights of one Keras layer, keyed by their role within it."""

    def __init__(self, path: str, weights: Dict[str, np.ndarray]):
        self.path = path
        self.weights = weights

    def __repr__(self) -> str:
        shapes = {name: value.shape for name, value in self.weights.items()}
        return f'KerasLayerWeights({self.path!r}, {shapes!r})'

    @property
    def names(self) -> frozenset:
        return frozenset(self.weights)

    @property
    def is_lstm_direction(self) -> bool:
        return self.names == _LSTM_WEIGHT_NAMES

    @property
    def is_dense(self) -> bool:
        return self.names == _DENSE_WEIGHT_NAMES

    @property
    def is_embedding(self) -> bool:
        return self.names == frozenset({'embeddings'})

    @property
    def is_chain_crf(self) -> bool:
        return self.names == _CHAIN_CRF_WEIGHT_NAMES

    @property
    def is_crf(self) -> bool:
        return self.names == _CRF_WEIGHT_NAMES

    @property
    def is_reverse_direction(self) -> Optional[bool]:
        """Whether this is the backward half of a `Bidirectional` wrapper.

        Direction is the one thing shape cannot recover -- the two halves are
        the same shape -- so it is read from the path, where Keras always spells
        it out. Anything else is not a direction of a bidirectional layer.
        """
        if 'backward' in self.path:
            return True
        if 'forward' in self.path:
            return False
        return None


def read_keras_layer_weights(filepath: str) -> List[KerasLayerWeights]:
    """Reads the weight datasets out of a Keras hdf5 file.

    Optimizer state is skipped, and the `model_weights/` prefix a fully saved
    model carries is stripped, so that both save shapes produce the same paths.
    """
    try:
        import h5py  # noqa pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise TfWeightConversionError(
            'reading TF-era model weights requires h5py, which is part of the'
            ' "delft" extra: install this package with [delft]'
        ) from exc

    grouped: Dict[str, Dict[str, np.ndarray]] = {}

    def visit(name: str, obj) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if name.startswith(_OPTIMIZER_WEIGHTS_PREFIX):
            return
        path = name
        if path.startswith(_MODEL_WEIGHTS_PREFIX):
            path = path[len(_MODEL_WEIGHTS_PREFIX):]
        layer_path, _, weight_name = path.rpartition('/')
        # Keras suffixes the variable name with its output index
        weight_name = weight_name.split(':')[0]
        grouped.setdefault(layer_path, {})[weight_name] = np.array(obj)

    with h5py.File(filepath, 'r') as h5_file:
        h5_file.visititems(visit)

    return [
        KerasLayerWeights(path, weights)
        for path, weights in sorted(grouped.items())
    ]


def _shape_of(value: Any) -> Tuple[int, ...]:
    return tuple(cast(torch.Tensor, value).shape)


def _describe(modules: Sequence[Tuple[str, nn.Module]]) -> str:
    return ', '.join(name or '<root>' for name, _ in modules) or 'none'


def _get_matching_module(
    candidates: Sequence[Tuple[str, nn.Module]],
    matches,
    description: str,
    source: KerasLayerWeights
) -> Tuple[str, nn.Module]:
    matching = [(name, module) for name, module in candidates if matches(module)]
    if len(matching) == 1:
        return matching[0]
    if not matching:
        raise TfWeightConversionError(
            f'no {description} matches {source!r};'
            f' available: {_describe(candidates)}'
        )
    raise TfWeightConversionError(
        f'{description} for {source!r} is ambiguous between'
        f' {_describe(matching)} -- shapes alone do not identify it'
    )


def _lstm_state(
    prefix: str, source: KerasLayerWeights, reverse: bool
) -> Dict[str, torch.Tensor]:
    """Maps one Keras LSTM direction onto the torch parameters of that half.

    Keras concatenates the gates as (i, f, c, o) and torch as (i, f, g, o),
    which is the same order, so only a transpose is needed. Keras carries a
    single bias where torch has two that it adds together, so all of it goes on
    the input side and the hidden side is zeroed.
    """
    suffix = '_reverse' if reverse else ''
    bias = source.weights['bias']
    return {
        f'{prefix}.weight_ih_l0{suffix}': torch.tensor(source.weights['kernel'].T.copy()),
        f'{prefix}.weight_hh_l0{suffix}': torch.tensor(
            source.weights['recurrent_kernel'].T.copy()
        ),
        f'{prefix}.bias_ih_l0{suffix}': torch.tensor(bias.copy()),
        f'{prefix}.bias_hh_l0{suffix}': torch.zeros(bias.shape[0]),
    }


def _add_lstm_weights(
    state: Dict[str, torch.Tensor],
    sources: Sequence[KerasLayerWeights],
    candidates: Sequence[Tuple[str, nn.Module]]
) -> None:
    for source in sources:
        reverse = source.is_reverse_direction
        if reverse is None:
            raise TfWeightConversionError(
                f'cannot tell the direction of {source!r}: the path names'
                ' neither "forward" nor "backward"'
            )
        weight_ih_shape = source.weights['kernel'].T.shape
        weight_hh_shape = source.weights['recurrent_kernel'].T.shape

        def matches(module: nn.Module, ih=weight_ih_shape, hh=weight_hh_shape) -> bool:
            if not isinstance(module, nn.LSTM):
                return False
            # nn.LSTM registers its parameters dynamically, so they are not
            # statically known attributes
            return (
                _shape_of(module.weight_ih_l0) == ih
                and _shape_of(module.weight_hh_l0) == hh
            )

        name, _ = _get_matching_module(candidates, matches, 'LSTM', source)
        state.update(_lstm_state(name, source, reverse=reverse))


def _add_dense_weights(
    state: Dict[str, torch.Tensor],
    sources: Sequence[KerasLayerWeights],
    candidates: Sequence[Tuple[str, nn.Module]]
) -> None:
    for source in sources:
        weight_shape = source.weights['kernel'].T.shape

        def matches(module: nn.Module, shape=weight_shape) -> bool:
            return isinstance(module, nn.Linear) and tuple(module.weight.shape) == shape

        name, _ = _get_matching_module(candidates, matches, 'dense layer', source)
        state[f'{name}.weight'] = torch.tensor(source.weights['kernel'].T.copy())
        state[f'{name}.bias'] = torch.tensor(source.weights['bias'].copy())


def _add_embedding_weights(
    state: Dict[str, torch.Tensor],
    sources: Sequence[KerasLayerWeights],
    candidates: Sequence[Tuple[str, nn.Module]]
) -> None:
    for source in sources:
        embeddings = source.weights['embeddings']

        def matches(module: nn.Module, shape=embeddings.shape) -> bool:
            return isinstance(module, nn.Embedding) and tuple(module.weight.shape) == shape

        name, _ = _get_matching_module(candidates, matches, 'embedding', source)
        state[f'{name}.weight'] = torch.tensor(embeddings.copy())


def _add_crf_weights(
    state: Dict[str, torch.Tensor],
    sources: Sequence[KerasLayerWeights],
    state_dict_keys: Sequence[str]
) -> None:
    """Maps whichever CRF parameterisation the file holds.

    The Keras `ChainCRF` and delft's torch port share parameter names, so that
    side is a copy. The `CRFModelWrapper` layout, which GROBID 0.8.2 uses, names
    the same three tensors differently and lands on the `pytorch-crf` module
    delft wraps.
    """
    for source in sources:
        if source.is_chain_crf:
            mapping = {'U': 'U', 'b_start': 'b_start', 'b_end': 'b_end'}
        else:
            mapping = {
                'chain_kernel': 'transitions',
                'left_boundary': 'start_transitions',
                'right_boundary': 'end_transitions',
            }
        for keras_name, torch_name in mapping.items():
            matching = [key for key in state_dict_keys if key.endswith('.' + torch_name)]
            if len(matching) != 1:
                raise TfWeightConversionError(
                    f'expected exactly one destination ending in {torch_name!r}'
                    f' for {source!r}, found {matching}'
                )
            state[matching[0]] = torch.tensor(source.weights[keras_name].copy())


def get_state_dict_for_keras_layer_weights(
    layers: Sequence[KerasLayerWeights],
    model: nn.Module
) -> Dict[str, torch.Tensor]:
    """Builds a torch state dict from Keras weights, or raises.

    Every layer in the file has to be recognised and consumed -- an
    unrecognised one means the file holds something this mapping does not know
    about, which is exactly the case that must not pass silently.
    """
    unrecognised = [
        layer for layer in layers
        if not (
            layer.is_lstm_direction or layer.is_dense or layer.is_embedding
            or layer.is_chain_crf or layer.is_crf
        )
    ]
    if unrecognised:
        raise TfWeightConversionError(
            f'unrecognised Keras layers: {unrecognised!r}'
        )

    modules = list(model.named_modules())
    state: Dict[str, torch.Tensor] = {}
    _add_embedding_weights(
        state, [layer for layer in layers if layer.is_embedding], modules
    )
    _add_lstm_weights(
        state, [layer for layer in layers if layer.is_lstm_direction], modules
    )
    _add_dense_weights(
        state, [layer for layer in layers if layer.is_dense], modules
    )
    _add_crf_weights(
        state,
        [layer for layer in layers if layer.is_chain_crf or layer.is_crf],
        list(model.state_dict())
    )
    return state


def _check_state_dict_is_complete(
    state: Dict[str, torch.Tensor], model: nn.Module
) -> None:
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    if missing:
        raise TfWeightConversionError(
            f'no converted weights for: {missing}'
        )
    unexpected = sorted(set(state) - set(expected))
    if unexpected:
        raise TfWeightConversionError(
            f'converted weights the model has no place for: {unexpected}'
        )
    mismatched = {
        key: (tuple(state[key].shape), tuple(value.shape))
        for key, value in expected.items()
        if tuple(state[key].shape) != tuple(value.shape)
    }
    if mismatched:
        raise TfWeightConversionError(
            f'converted weights of the wrong shape (converted, expected):'
            f' {mismatched}'
        )


def load_keras_weights_into_model(filepath: str, model: nn.Module) -> None:
    """Loads TF-era Keras weights into an already-constructed torch model."""
    LOGGER.info('converting Keras weights from %s', filepath)
    layers = read_keras_layer_weights(filepath)
    if not layers:
        raise TfWeightConversionError(f'no weights found in {filepath}')
    state = get_state_dict_for_keras_layer_weights(layers, model)
    _check_state_dict_is_complete(state, model)
    model.load_state_dict(state, strict=True)
    LOGGER.info('converted %d Keras layers from %s', len(layers), filepath)
