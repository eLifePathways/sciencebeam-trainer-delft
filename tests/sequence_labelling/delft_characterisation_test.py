"""Pins what the installed delft computes for this repo's architecture.

`models_test.py` checks relationships that hold whatever delft computes -- the
same tags whatever the padding, the same score from either CRF -- and those stay
true across an upstream change that moves every number. This module fixes the
numbers themselves, for both CRF paths of `CustomBidLSTM_CRF`: a model whose
weights are a deterministic function of its shapes, one fixed padded batch, and
the logits, loss and tags it produces, recorded in
`test_data/custom_bid_lstm_crf_characterisation.json`.

A delft upgrade that changes what a trained model predicts fails here first,
which is the only way to tell an upgrade apart from a retraining.

The weights come from `random.Random`, a documented and stable generator, rather
than from `torch.manual_seed`: torch's RNG is not what is under test, and an
expectation derived from it would break on a torch upgrade for a reason that has
nothing to do with delft.

Regenerate deliberately, never to make a failure go away:

    python tests/sequence_labelling/delft_characterisation_test.py
"""
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch

from sciencebeam_trainer_delft.sequence_labelling.config import ModelConfig
from sciencebeam_trainer_delft.sequence_labelling.models import CustomBidLSTM_CRF


FIXTURE_PATH = (
    Path(__file__).parent / 'test_data' / 'custom_bid_lstm_crf_characterisation.json'
)

WEIGHT_SEED = 20260918

# wide enough that the emissions, rather than the transitions alone, decide the
# tags: a narrower range decodes almost everything to tag 0, which would pin a
# result that survives most ways of getting the decoding wrong
WEIGHT_RANGE = (-1.0, 1.0)

NTAGS = 4

# the published shape -- no word embeddings, continuous features passed through
# unchanged, padded tokens masked -- at a size whose output fits in a fixture
CHARACTERISATION_CONFIG: Dict[str, Any] = {
    'char_vocab_size': 12,
    'char_embedding_size': 4,
    'num_char_lstm_units': 3,
    'max_char_length': 5,
    'num_word_lstm_units': 5,
    'word_embedding_size': 0,
    'dropout': 0.0,
    'use_features': True,
    'max_feature_size': 3,
    'features_embedding_size': 0,
    'mask_padded_tokens': True
}

# two documents of different length in one batch, each token with its own
# character padding: the shape every defect this repo patched shows up in
CHAR_INPUT = torch.tensor([
    [
        [1, 2, 3, 0, 0], [4, 5, 0, 0, 0], [6, 7, 8, 9, 0], [10, 11, 0, 0, 0],
        [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
    ],
    [
        [2, 3, 0, 0, 0], [5, 6, 7, 0, 0], [8, 9, 10, 11, 1], [3, 4, 0, 0, 0],
        [7, 8, 9, 0, 0], [1, 1, 0, 0, 0]
    ]
])

REAL_LENGTHS = [4, 6]

LABELS = torch.tensor([
    [1, 2, 3, 2, 0, 0],
    [1, 3, 2, 1, 2, 3]
])


def _model_config(**kwargs) -> ModelConfig:
    values: Dict[str, Any] = {**CHARACTERISATION_CONFIG, **kwargs}
    return ModelConfig(architecture='CustomBidLSTM_CRF', **values)


def _inputs(model_config: ModelConfig) -> Dict[str, torch.Tensor]:
    batch_size, sequence_length = CHAR_INPUT.shape[:2]
    feature_count = model_config.max_feature_size
    features = torch.linspace(
        -1.0, 1.0, batch_size * sequence_length * feature_count
    ).reshape(batch_size, sequence_length, feature_count)
    for index, real_length in enumerate(REAL_LENGTHS):
        features[index, real_length:] = 0.0
    return {
        'word_input': torch.zeros(batch_size, sequence_length, 0),
        'char_input': CHAR_INPUT,
        'features_input': features
    }


def _with_fixed_weights(model: CustomBidLSTM_CRF) -> CustomBidLSTM_CRF:
    """Fills every parameter from one stable generator, in sorted name order.

    Sorted rather than declaration order, so that reordering the modules does not
    silently redistribute the weights and invalidate the fixture for no reason.
    """
    rng = random.Random(WEIGHT_SEED)
    state_dict = model.state_dict()
    for name in sorted(state_dict):
        value = state_dict[name]
        assert value.is_floating_point(), name
        value.copy_(torch.tensor(
            [rng.uniform(*WEIGHT_RANGE) for _ in range(value.numel())]
        ).reshape(value.shape))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _capture(use_chain_crf: bool) -> Dict[str, Any]:
    model_config = _model_config(use_chain_crf=use_chain_crf)
    model = _with_fixed_weights(CustomBidLSTM_CRF(model_config, NTAGS))
    inputs = _inputs(model_config)
    with torch.no_grad():
        outputs = model(inputs, LABELS)
        tags = model.decode(inputs)
    return {
        'logits': outputs['logits'].tolist(),
        'loss': outputs['loss'].item(),
        'tags': torch.as_tensor(tags).tolist()
    }


def _load_fixture() -> Dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding='utf-8'))


def _write_fixture():
    FIXTURE_PATH.write_text(
        json.dumps(
            {
                'chain_crf': _capture(use_chain_crf=True),
                'plain_crf': _capture(use_chain_crf=False)
            },
            indent=2
        ) + '\n',
        encoding='utf-8'
    )


class TestCustomBidLSTMCRFCharacterisation:
    def test_should_produce_the_recorded_chain_crf_output(self):
        expected = _load_fixture()['chain_crf']
        actual = _capture(use_chain_crf=True)
        assert actual['tags'] == expected['tags']
        assert torch.allclose(
            torch.tensor(actual['logits']), torch.tensor(expected['logits']), atol=1e-6
        )
        assert abs(actual['loss'] - expected['loss']) < 1e-5

    def test_should_produce_the_recorded_plain_crf_output(self):
        expected = _load_fixture()['plain_crf']
        actual = _capture(use_chain_crf=False)
        assert actual['tags'] == expected['tags']
        assert torch.allclose(
            torch.tensor(actual['logits']), torch.tensor(expected['logits']), atol=1e-6
        )
        assert abs(actual['loss'] - expected['loss']) < 1e-5

    def test_should_have_recorded_an_output_worth_pinning(self):
        # guards the fixture rather than the model: an all-zero tagging would
        # survive most ways of getting the decoding wrong, so a regenerated
        # fixture that degenerates to one has stopped testing anything
        for path in ('chain_crf', 'plain_crf'):
            tags: List[List[int]] = _load_fixture()[path]['tags']
            real_tags = [
                row[:real_length] for row, real_length in zip(tags, REAL_LENGTHS)
            ]
            assert len({tag for row in real_tags for tag in row}) >= 3, path
            # the padded positions are filled back in, never decoded
            for row, real_length in zip(tags, REAL_LENGTHS):
                assert row[real_length:] == [0] * (len(row) - real_length), path


if __name__ == '__main__':
    _write_fixture()
