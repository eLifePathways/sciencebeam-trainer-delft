# Reference capture scripts

Records what the sequence labelling models output, so a change can be compared
against what was recorded before it. Output goes to `data/reference/`, which is
not tracked.

## Setup

```bash
cd "$(git rev-parse --show-toplevel)"
uv sync --extra delft --extra gcs --extra cpu --all-groups --frozen
```

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
difference.

## The per-token capture

There was a second script, `capture_reference_outputs.py`, recording per-token
inputs, pre-CRF logits and tags from a TensorFlow model. It read the Keras model
directly, so it only ran while TensorFlow was installed, and it is not in this
checkout. It is in the history, alongside the TensorFlow it needs:

```bash
git log --all --diff-filter=D -- scripts/reference-capture/capture_reference_outputs.py
git checkout <commit>~1 -- scripts/reference-capture/capture_reference_outputs.py
```

Regenerating that capture means checking out such a commit anyway, and the
script is in it, so carrying a copy here that cannot run would serve nothing.

## Notes

- Several cases score a model against a corpus it was not trained on, so the
  numbers are low. They are for comparison, not for judging the models.
- Scoring uses the sequence length and batch size each model records. Segmentation
  documents reach 11k tokens, so an unbounded sequence length runs out of memory.
- `PYTHONPATH=.` is needed because the project is not installed into the
  venv; run from the repository root.
- Re-runs are byte-identical, so any difference is a difference in behaviour.
- Models and datasets are cached under `data/download/` after the first run. The
  four GROBID header models are fetched file by file and are the slow part.
