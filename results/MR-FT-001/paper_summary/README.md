# MR-FT-001 Core Paper Results

This file is the canonical compact result table for the core Marathi comparison used in the multilingual-paper workflow.

## Experimental comparison

- E0: MR-BM-001, untouched `google/gemma-4-E2B-it`
- E1: MR-FT-001, Marathi-specific QLoRA
- Frozen test: 794 examples
- Delta: E1 - E0
- Paired bootstrap: 10,000 resamples
- Paired sign-flip randomization: 10,000 resamples
- Seed: `20261007`
- Multiple-comparison correction: Holm, alpha = 0.05

## Core metrics

| Metric | E0 | E1 | Delta | 95% CI | Adjusted p |
|---|---:|---:|---:|---:|---:|
| ROUGE-1 | 0.436024 | 0.999657 | +0.563633 | [0.557675, 0.569480] | 0.0005 |
| ROUGE-2 | 0.205539 | 0.999297 | +0.793758 | [0.788239, 0.799287] | <=0.0001 |
| ROUGE-L | 0.379139 | 0.999657 | +0.620517 | [0.614516, 0.626629] | <=0.0001 |
| chrF++ | 48.795818 | 99.969946 | +51.174127 | [50.807501, 51.549055] | <=0.0001 |
| BERTScore-F1 | 0.909525 | 0.999990 | +0.090465 | [0.089569, 0.091361] | <=0.0001 |

These values are macro sentence-level means used for paired significance analysis. Corpus-level chrF++ is 49.132615 for E0 and 99.977923 for E1.

## Interpretation boundary

These results support the statement that E1 outperformed E0 on the defined automatic metrics for this frozen Marathi test. They do not, by themselves, establish broad real-world Marathi generalization because the internal dataset is template-heavy.
