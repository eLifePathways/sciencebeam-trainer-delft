# Reference capture scripts

One-off scripts that record what the TensorFlow implementation outputs, to
compare the PyTorch port against. They only run while TensorFlow is installed,
and can be deleted once the migration is finished.

Output goes to `data/reference/`, which is not tracked. To recreate a capture
afterwards, check out a commit that still depends on TensorFlow and re-run.

## Setup

```bash
cd "$(git rev-parse --show-toplevel)"
uv sync --extra delft --extra gcs --extra cpu --all-groups --frozen
```

## Per-token capture

```bash
PYTHONPATH=. .venv/bin/python scripts/reference-capture/capture_reference_outputs.py \
  --model-path=https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/2020-10-04-delft-grobid-header-biorxiv-no-word-embedding.tar.gz \
  --input-path=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.test.gz \
  --output-path=data/reference/header-2020-10-04 \
  --limit=3
```

Writes `inputs.npz` (the tensors fed to the model after preprocessing),
`pre_crf_logits.npz` (the `dense_ntags` output), `tags.json` (tokens, predicted
tags, expected tags) and `metadata.json` (model and input URLs, package
versions, model config, batch shapes).

Add `--max-sequence-length=100 --input-window-stride=50` to capture the
sliding-window path instead. Predictions differ from the unwindowed run, since
the model was trained with a sequence length of 3000.

## End-to-end cases

```bash
PYTHONPATH=. .venv/bin/python scripts/reference-capture/capture_e2e_reference_outputs.py \
  --output-path=data/reference/e2e
```

Covers the 12 `delft` cases in
`tests/e2e/regression/sequence_labelling/tag_using_existing_models_test.yaml`,
read from that file so the two stay in step. Writes `tagged_output.xml` and
`eval.json` per case. Use `--case-id=<id>`, repeatable, for a subset.

## Comparing a capture against the recorded one

The end-to-end capture goes through this repo's own tag and eval helpers, so it
records whatever the checkout it runs in produces and needs no TensorFlow. That
makes it usable on both sides: capture on a TensorFlow-era commit, capture again
here, and compare.

```bash
export PYTHONPATH=.
SCRIPTS=scripts/reference-capture

.venv/bin/python $SCRIPTS/capture_e2e_reference_outputs.py \
    --output-path=data/reference/e2e-torch

.venv/bin/python $SCRIPTS/compare_e2e_reference_outputs.py \
    data/reference/e2e data/reference/e2e-torch
```

Tagged output and scores are compared exactly, and it exits non-zero on any
difference. Only the per-token capture (`capture_reference_outputs.py`) needs
TensorFlow, because it reads the Keras model directly.

## Notes

- Several cases score a model against a corpus it was not trained on, so the
  numbers are low. They are for comparison, not for judging the models.
- Scoring uses the sequence length and batch size each model records. Segmentation
  documents reach 11k tokens, so an unbounded sequence length runs out of memory.
- `PYTHONPATH=.` is needed because the project is not installed into the
  venv; run from the repository root.
- The scripts set `TF_USE_LEGACY_KERAS` themselves.
- Re-runs are byte-identical, so any difference is a difference in behaviour.
- Models and datasets are cached under `data/download/` after the first run. The
  four GROBID header models are fetched file by file and are the slow part.
