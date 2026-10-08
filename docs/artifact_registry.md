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

## Core paper result summary

| Artifact | Purpose | Status | SHA-256 |
|---|---|---|---|
| `results/MR-FT-001/paper_summary/MR-FT-001_paper_results.csv` | Canonical five-metric E0/E1 table | ✅ Complete | `41B4ED255AC4DBFA148A4FB4329907B62BA463A8287892B7F7B9F889BC6D4F2E` |
| `results/MR-FT-001/paper_summary/MR-FT-001_paper_results.json` | Statistical configuration + canonical metric results | ✅ Complete | `AEEC801DEB7FA012C5B8C025848EBF13861E5A553E16CCA214335A9110B67B00` |
| `results/MR-FT-001/paper_summary/README.md` | Human-readable paper-results interpretation | ✅ Complete | `7A12F95FD2CD960561BEAAF2C31B5248D9E66B246AAD046FB5486F1C5E80D27B` |


## Supplementary blind human evaluation

| Artifact | Purpose | Status | SHA-256 |
|---|---|---|---|
| `results/MR-FT-001/human_evaluation/MR-FT-001_human_eval_summary.csv` | Final audited human-preference summary | ✅ Complete | `29e68141daae11836bc9cb8c551b57f08d1c6be6eb480eac8ffbc863554df94f` |
| `results/MR-FT-001/human_evaluation/MR-FT-001_human_eval_verdicts.csv` | Final audited per-example model-labeled verdicts | ✅ Complete | `42c461c8bd45022edac4a08fb2e881b6c10ed9443d2d402506094ad6d9ddfc46` |
| `results/MR-FT-001/human_evaluation/MR-FT-001_human_eval_metadata.json` | Human-evaluation configuration and result metadata | ✅ Complete | (metadata generated from the final audited record) |

Final outcome counts:

- E0 = 19
- E1 = 27
- Tie = 54

Private A/B mapping keys are intentionally excluded from the public repository.

## Storage policy

- Model weights/checkpoints: external/cloud storage
- Approved datasets and prediction artifacts: Git LFS when committed
- Small text/metadata/plots: normal Git
- Hashes: this registry + experiment README/metadata
