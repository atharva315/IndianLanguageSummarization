# Paper Artifact Checklist

This document defines the minimum artifacts required for the core Marathi results section of the multilingual research paper.

## Canonical experiment inputs

- `data/processed/marathi_v1/06_test_frozen.csv`
- `experiments/baseline/MR-BM-001/baseline_predictions.csv`
- `experiments/qlora/MR-FT-001/inference/MR-FT-001_predictions.csv`
- `experiments/qlora/MR-FT-001/MR-FT-001_config.json`
- `experiments/qlora/MR-FT-001/MR-FT-001_final_manifest.json`

## Canonical core evaluation outputs

The final repository should contain the complete metric-level evidence alongside the compact paper summary:

- ROUGE-1/2/L per-example results, summary, and metadata
- chrF++ per-example results, summary, length-bucket results, and metadata
- BERTScore per-example results, summary, and metadata
- paired statistical-analysis outputs
- length-bucket comparison outputs
- data-leakage audit outputs
- the compact paper result table in `results/MR-FT-001/paper_summary/`

## Supplementary evaluation

The 100-example blind A/B package and AI/LLM-judge analysis are supplementary Marathi evidence. They are not required for the standardized cross-language core comparison.

## Storage policy

Large CSV/JSON/ZIP artifacts are stored through Git LFS according to `.gitattributes`. Full training checkpoints, optimizer state, and RNG state remain outside normal Git history.

## Paper-writing rule

The IEEE paper should cite the compact paper summary for headline values and use the underlying per-example/statistical artifacts for detailed tables, figures, and auditability. No headline result should be transcribed manually without checking the canonical artifact.
