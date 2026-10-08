# Blind Human A/B Evaluation Protocol

## Purpose

This is a **supplementary** human evaluation for the Marathi MR-FT-001 comparison. The core research endpoint remains the paired statistical analysis of automatic metrics; the human study provides complementary evidence.

## Sample

- 100-example independent Marathi evaluation set.
- The examples were generated outside the MR-FT-001 training and frozen-test split.
- Model outputs were generated before the human judgments and then frozen for evaluation.

## Models

- **E0:** untouched `google/gemma-4-E2B-it`
- **E1:** the same base model with the selected Marathi-specific QLoRA adapter from checkpoint-1750.

Checkpoint-1750 adapter SHA-256:

`6828bff5f35f384cb6c2a3736f1a0b8c2960b65cac70a874dd4d5230e45bde91`

## Generation setup

Both model outputs used the same Marathi summarization prompt:

> `खालील मराठी मजकुराचा अचूक, संक्षिप्त आणि तथ्यसुसंगत सारांश लिहा.`

Decoding was deterministic with `do_sample=false`.

For this supplementary human-evaluation inference, `max_new_tokens=2048` was used only as a generous safety ceiling; it was not a target summary length.

## Blind design

For each example:

1. show the source Marathi article
2. show two summaries labeled only **A** and **B**
3. randomize which model is A and which is B
4. do not reveal model identity during judgment
5. ask only: **Which summary do you prefer?**
6. allow **A / B / Tie**

The private A/B-to-model key is not committed to the public repository.

## Final audited result

| Model / outcome | Count | Share |
|---|---:|---:|
| E0 — Base Gemma | 19 | 19% |
| E1 — Gemma + QLoRA | 27 | 27% |
| Tie | 54 | 54% |
| **Total** | **100** | **100%** |

Thus E1 received more preferences than E0, but ties were the largest outcome. This result is supplementary and should not be presented as statistically stronger than the core automatic evaluation.

## Mapping audit

The final audited interpretation is:

- **E0 = Base Gemma**
- **E1 = Gemma + QLoRA**

The per-example verdict artifact stored in this repository uses these final audited model labels.

## Integrity rules

- human responses do not change the frozen-test prompt
- human responses do not change the selected checkpoint
- the frozen-test automatic results remain unchanged
- the private A/B key is withheld
- the final model-labeled verdicts and aggregate result are versioned separately from any private participant information
