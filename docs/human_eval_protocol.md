# Blind Human A/B Evaluation Protocol

## Purpose

Automatic metrics measure different forms of similarity. The human study provides a complementary judgment of summary quality.

## Sample

Use the prepared `data/processed/marathi_v1/07_human_eval_100.csv` sample.

The sample must remain fixed once the human study begins.

## Blind design

For each item:

1. show the source Marathi article
2. show two summaries labeled only **A** and **B**
3. randomize whether A is E0 or E1
4. do not reveal model identity to the evaluator
5. collect one preference
6. optionally collect a reason/quality score

## Recommended criteria

Ask evaluators to consider:

- factual accuracy
- coverage of important information
- Marathi fluency/readability
- conciseness
- overall preference

## Analysis

Report:

- E1 preference rate
- E0 preference rate
- ties/invalid responses, if allowed
- confidence interval
- number of evaluators/items
- randomized A/B mapping procedure

For a paired binary preference outcome, use an explicitly documented significance test (for example, an exact binomial test or a paired/bootstrap analysis) and record the random seed.

## Integrity rules

- evaluators must not see the E0/E1 identity during judgment
- no human responses may be used to tune the frozen-test prompt
- no human responses may change the already selected E1 checkpoint
- any post-hoc filtering must be disclosed
