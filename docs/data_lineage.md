# Data Lineage and Provenance

## Canonical pipeline

```
raw source pool
    |
    v
cleaning / normalization
    |
    v
exact normalized deduplication
    |
    v
structural QA + template analysis
    |
    v
template-family cap (12)
    |
    v
template-aware split
    |
    +--> 04_train_sft.csv
    |
    +--> 05_validation_sft.csv
    |
    +--> 06_test_frozen.csv  <-- immutable evaluation boundary
    |
    +--> 07_human_eval_100.csv
    |
    +--> 08_excluded_with_reasons.csv
    |
    +--> 09_split_summary.csv
```

## Versioned experimental pool

Canonical Marathi v1 counts:

- training: 14,020
- validation: 796
- frozen test: 794
- experimental pool: 15,610

The split is template-aware so a structural template family does not cross train/validation/test.

## Leakage boundary

The frozen test is used for:

- MR-BM-001 baseline inference
- MR-FT-001 final inference
- post-hoc automatic evaluation
- blind human-evaluation sampling where explicitly prepared

The frozen test is not used for:

- QLoRA training
- model checkpoint selection
- prompt tuning
- hyperparameter tuning

## Frozen identifier

`06_test_frozen.csv` SHA-256:

`D93B3EE80E6AE306C1C1F02855A244AE2C38F46D587F3C39C8C415F0011A117D`

## Dataset limitation

The current Marathi v1 corpus is template-heavy/synthetic-style. Template-aware splitting reduces structural leakage risk, but it does not prove semantic diversity or real-world generalization. An independent real-Marathi benchmark is therefore a separate planned experiment.
