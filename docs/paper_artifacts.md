# Paper Artifact Checklist

## Research cutoff

The canonical research study ends at the **paired statistical analysis of the automatic summarization metrics**.

The core research evidence is:

`E0 vs E1 → ROUGE-1/2/L → chrF++ → BERTScore → paired bootstrap/sign-flip statistics → length analysis → leakage/integrity audit`

AI/LLM-judge results, human-evaluation results, and the unfinished external XL-Sum inference are **post-cutoff/supplementary artifacts and are not part of the core research claim**.

## Canonical experiment inputs

- `data/processed/marathi_v1/06_test_frozen.csv`
- `experiments/baseline/MR-BM-001/baseline_predictions.csv`
- `experiments/qlora/MR-FT-001/inference/MR-FT-001_predictions.csv`
- `experiments/qlora/MR-FT-001/MR-FT-001_config.json`
- `experiments/qlora/MR-FT-001/MR-FT-001_final_manifest.json`
- `experiments/qlora/MR-FT-001/MR-FT-001_runtime_config.json`
- `experiments/qlora/MR-FT-001/MR-FT-001_training_start_manifest.json`

## Canonical training evidence

- `experiments/qlora/MR-FT-001/analysis/`
- `experiments/qlora/MR-FT-001/preparation/`

These document checkpoint selection, training/validation behavior, sequence-length coverage, and model architecture.

## Canonical automatic evaluation evidence

The detailed automatic-evaluation package should be stored under:

`results/MR-FT-001/automatic_evaluation/`

### ROUGE

- `rouge/MR-FT-001_ROUGE_unicode_per_example.csv`
- `rouge/MR-FT-001_ROUGE_unicode_summary.csv`
- `rouge/MR-FT-001_ROUGE_unicode_metadata.json`
- `rouge/MR-FT-001_E1_non_exact_cases.csv`

### chrF++

- `chrfpp/MR-FT-001_chrFpp_per_example.csv`
- `chrfpp/MR-FT-001_chrFpp_summary.csv`
- `chrfpp/MR-FT-001_chrFpp_length_bucket.csv`
- `chrfpp/MR-FT-001_chrFpp_metadata.json`

### BERTScore

- `bertscore/MR-FT-001_BERTScore_per_example.csv`
- `bertscore/MR-FT-001_BERTScore_summary.csv`
- `bertscore/MR-FT-001_BERTScore_metadata.json`

### Statistical analysis

- `statistical_analysis/MR-FT-001_statistics_input.csv`
- `statistical_analysis/MR-FT-001_bootstrap_distributions.csv`
- `statistical_analysis/MR-FT-001_randomization_distributions.csv`
- `statistical_analysis/MR-FT-001_paired_statistical_results.csv`
- `statistical_analysis/MR-FT-001_statistical_analysis_metadata.json`

### Leakage audit

- `data_leakage_audit/MR-FT-001_data_leakage_summary.json`
- `data_leakage_audit/MR-FT-001_E1_exact_match_audit.csv`

### Length analysis

- `length_analysis/MR-FT-001_ROUGE_length_bucket.csv`
- `length_analysis/MR-FT-001_chrFpp_length_bucket.csv`

## Canonical paper summary

- `results/MR-FT-001/paper_summary/MR-FT-001_paper_results.csv`
- `results/MR-FT-001/paper_summary/MR-FT-001_paper_results.json`
- `results/MR-FT-001/paper_summary/README.md`

These are the headline values; the detailed files above remain the validation evidence.

## Validation requirements

A paper-ready repository must be able to verify:

- frozen test SHA-256
- E0 prediction SHA-256
- E1 prediction SHA-256
- 794 rows for frozen test, E0, and E1
- unique IDs
- exact E0/E1/frozen-test ID order
- exact source-text alignment
- no empty predictions
- five primary automatic metrics
- 10,000 paired bootstrap resamples
- 10,000 paired sign-flip randomization resamples
- seed `20261007`
- Holm correction, alpha = 0.05
- delta definition `E1 - E0`

## Storage policy

Large CSV/JSON artifacts are stored through Git LFS according to `.gitattributes`. Full training checkpoints, optimizer state, RNG state, and private evaluation mappings remain outside normal Git history.

The selected checkpoint-1750 adapter is identified by hash and configuration; its full learned model state is not committed to this repository.


## Supplementary blind human evaluation

The human study is supplementary and does not alter the core research endpoint.

- `results/MR-FT-001/human_evaluation/MR-FT-001_human_eval_summary.csv`
- `results/MR-FT-001/human_evaluation/MR-FT-001_human_eval_verdicts.csv`
- `results/MR-FT-001/human_evaluation/MR-FT-001_human_eval_metadata.json`

Final audited model mapping:

- E0 = untouched base Gemma
- E1 = Gemma + checkpoint-1750 Marathi QLoRA

Final example-level outcome counts:

- E0: 19
- E1: 27
- Tie: 54

The private A/B assignment key is intentionally excluded from the public repository.

## Excluded from the core paper

Do not promote these into the core research evidence:

- AI/LLM-judge results
- private human-evaluation keys/mappings
- preliminary ROUGE results
- duplicate backup directories
- checkpoint-250 through checkpoint-2500 directories
- optimizer/RNG/scheduler state
- unfinished XL-Sum external inference

No headline paper result should be transcribed manually without checking the canonical detailed artifact and paper-summary table.
