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

## Google Forms blind design

The study was conducted as a blind human-preference comparison using Google Forms. For each of the 100 examples:

1. Display the source Marathi article as the form question/prompt.
2. Use the generated E0 and E1 summaries as the first and second answer options. Each option displays only the summary text: there are no A/B tags, E0/E1 labels, or model-identifying tags beside either summary.
3. Shuffle the order of the two summary options per example, so the first option can be E0 or E1 and the second option is the other model's summary.
4. Use **Tie** as the third answer option.
5. Ask the evaluator: **Which summary do you prefer?**
6. Keep the per-example mapping from first/second option position to E0/E1 private, and use it after collecting responses to map each preference back to the correct model.

This was a human-response study through Google Forms, not an automated model-judge evaluation. The private option-position-to-model mapping is not committed to the public repository.

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
- the private per-example option-position-to-model mapping is withheld
- the final model-labeled verdicts and aggregate result are versioned separately from any private participant information
