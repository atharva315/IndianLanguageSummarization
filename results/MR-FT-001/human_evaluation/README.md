# MR-FT-001 Supplementary Human Evaluation

This directory contains the final audited supplementary blind human-evaluation artifacts. The preferences were collected from human evaluators through Google Forms.

## Collection method

For each of 100 independent Marathi examples, the source article appeared as the form question. The first and second response options each displayed only one summary's text, without A/B tags or E0/E1/model labels. The two summaries were shuffled between those positions per example. The third response option was **Tie**. The private option-position-to-model mapping was used to map responses to E0, E1, or Tie after collection.

## Final outcome

- E0 — Base Gemma: **19**
- E1 — Gemma + QLoRA: **27**
- Tie: **54**
- Total: **100**

## Public artifacts

- `MR-FT-001_human_eval_summary.csv`
- `MR-FT-001_human_eval_verdicts.csv`
- `MR-FT-001_human_eval_metadata.json`

The private per-example option-position-to-model mapping and participant-identifying information are intentionally not committed.

The 100-item model-output master is maintained in the research backup outside this commit; the public result package contains the audited per-example verdicts and aggregate summary.
