# Contributing to IndianLanguageSummarization

This repository contains a research record, not just application code.

## Before changing research artifacts

Read:

- `docs/research_protocol.md`
- `docs/reproducibility.md`
- `docs/evaluation_protocol.md`

## Experiment changes

Any new experiment should have:

- a unique experiment ID
- an exact model/revision
- a dataset version
- explicit train/validation/test boundaries
- recorded generation settings
- recorded environment versions
- artifact hashes where applicable
- a short experiment README
- reproducible analysis outputs

## Frozen data

Never modify:

`data/processed/marathi_v1/06_test_frozen.csv`

after baseline visibility.

Never use frozen-test references for:

- training
- prompt tuning
- hyperparameter tuning
- checkpoint selection

## Git and large files

Use Git LFS for approved large CSV/JSON/ZIP artifacts covered by `.gitattributes`.

Do not commit:

- model checkpoints
- optimizer states
- RNG state files
- large temporary archives
- local environment caches
- secrets/tokens

## Review standard

A research change should be:

1. reproducible
2. testable
3. documented
4. traceable to an experiment ID
5. explicit about whether a number is measured, assumed, or planned
