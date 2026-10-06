# Artifact Registry

This file is the human-readable registry of canonical research artifacts.

## Frozen evaluation

| Artifact | Purpose | Status | SHA-256 |
|---|---|---|---|
| `data/processed/marathi_v1/06_test_frozen.csv` | Immutable E0/E1 test set | ✅ Frozen | `D93B3EE80E6AE306C1C1F02855A244AE2C38F46D587F3C39C8C415F0011A117D` |

## Baseline — MR-BM-001

| Artifact | Purpose | Status | SHA-256 |
|---|---|---|---|
| `experiments/baseline/MR-BM-001/baseline_predictions.csv` | Official E0 predictions | ✅ Complete | `c67d873a6f762ac75e6708d3f686407059abae85a5c9b558836b054bf1beefd0` |

## QLoRA — MR-FT-001

| Artifact | Purpose | Status | SHA-256 |
|---|---|---|---|
| `best_adapter` / checkpoint-1750 | Official E1 adapter | ✅ Selected | `6828bff5f35f384cb6c2a3736f1a0b8c2960b65cac70a874dd4d5230e45bde91` |
| `MR-FT-001_predictions.csv` | Official E1 frozen-test predictions | ✅ Generated + verified | `2A258ADB8F87AB810B3365F65919E0617E87449D07004A9CA9EDE89A1B521E` |
| `MR-FT-001_predictions_checkpoint_0794.csv` | Final cumulative E1 checkpoint | ✅ Generated + verified | `2A258ADB8F87AB810B3365F65919E0617E87449D07004A9CA9EDE89A1B521E` |

## Inference verification

MR-FT-001 final and checkpoint-0794 CSVs were verified against the frozen test on 2026-10-06:

- 794 rows
- 794 unique `pair_id`
- exact frozen-test ID order
- exact frozen-test source text
- no empty predictions
- final CSV and checkpoint-0794 CSV are byte-identical

Frozen-test SHA-256 used for verification:

`D93B3EE80E6AE306C1C1F02855A244AE2C38F46D587F3C39C8C415F0011A117D`

## Analysis

Canonical training analysis is under:

`experiments/qlora/MR-FT-001/analysis/`

Preparation evidence is under:

`experiments/qlora/MR-FT-001/preparation/`

## Storage policy

- Model weights/checkpoints: external/cloud storage
- Approved datasets and prediction artifacts: Git LFS when committed
- Small text/metadata/plots: normal Git
- Hashes: this registry + experiment README/metadata
