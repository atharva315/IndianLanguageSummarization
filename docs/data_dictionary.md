# Marathi v1 Data Dictionary

This document defines the fields used by the canonical Marathi v1 CSV package.

## Core identifiers

| Field | Type | Role |
|---|---|---|
| `pair_id` | string | Stable example identifier used to align train/validation/test and prediction artifacts. |
| `source_file` | string | Provenance identifier for the source record. |
| `source_row` | integer/string | Original row reference in the source file. |
| `template_id` | string | Structural template-family identifier used for template-aware quality control and splitting. |

## Content fields

| Field | Type | Role |
|---|---|---|
| `text` | string | Marathi source article/document presented to the summarization model. |
| `reference_summary` | string | Gold/reference summary used only after generation for evaluation; never supplied to the model in E0/E1 inference. |

## Derived QA fields

| Field | Type | Role |
|---|---|---|
| `text_chars` | integer | Character count of the source text used by the data QA pipeline. |
| `summary_chars` | integer | Character count of the reference summary used by the data QA pipeline. |
| `compression_ratio` | numeric | Summary-to-source size ratio recorded during data analysis. |
| `length_bucket` | string | Source/summary length bucket used for split diversity and QA analysis. |

## Canonical split files

- `04_train_sft.csv`: QLoRA training data, 14,020 examples.
- `05_validation_sft.csv`: model-selection/validation data, 796 examples.
- `06_test_frozen.csv`: immutable E0/E1 evaluation data, 794 examples.
- `07_human_eval_100.csv`: prepared blind human-evaluation sample.
- `08_excluded_with_reasons.csv`: records excluded from the experimental pool with reasons.
- `09_split_summary.csv`: split-level summary/QA record.

## Integrity rules

`06_test_frozen.csv` is immutable after baseline visibility.

Prediction artifacts intentionally use only:

`pair_id,text,model_summary`

so evaluation references remain outside the prediction file and are joined only inside the evaluation workflow.
