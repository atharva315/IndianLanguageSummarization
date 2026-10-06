# Evaluation Protocol

## Objective

Measure whether Marathi-specific QLoRA improves summarization relative to the untouched Gemma baseline under a fixed, contamination-controlled evaluation setup.

## Systems

- **E0:** MR-BM-001 baseline
- **E1:** MR-FT-001 QLoRA adapter

## Evaluation input

For every example, the model receives:

- source Marathi text
- the fixed summarization instruction/prompt

The model does **not** receive:

- `reference_summary`
- evaluation labels
- any field derived from the reference summary

## Frozen test

- 794 examples
- same example IDs for E0 and E1
- same source text
- same reference summary
- immutable after experiment visibility

Frozen SHA-256:

`D93B3EE80E6AE306C1C1F02855A244AE2C38F46D587F3C39C8C415F0011A117D`

## Automatic metrics

### ROUGE

Report:

- ROUGE-1
- ROUGE-2
- ROUGE-L

Use the same implementation/version and normalization for E0 and E1.

ROUGE is a lexical-overlap family introduced for automatic summary evaluation by Lin (2004).

### chrF++

Use chrF++ as a character/word n-gram similarity metric. This is useful for morphologically rich languages and does not depend on exact word segmentation as strongly as purely word-based overlap metrics.

Record the exact metric implementation/version and configuration used for the paper.

### BERTScore

Use BERTScore as a contextual semantic-similarity metric.

Record:

- scorer model
- package/version
- language/model setting
- aggregation strategy

Do not interpret BERTScore as a direct factuality guarantee; report it as one complementary automatic signal.

## Aggregation

Primary report should contain corpus-level E0 vs E1 scores and absolute deltas:

`delta = E1 - E0`

Report sentence-level distributions where useful.

## Statistical comparison

For paired E0/E1 comparisons, use paired bootstrap resampling or another documented paired test where appropriate.

The statistical method, number of resamples, random seed, confidence level, and multiple-comparison handling must be recorded before final claims are made.

## Human A/B evaluation

Use the prepared 100-example human-evaluation set.

Design:

1. show the source article
2. show two anonymized summaries A/B
3. randomize which system appears as A/B
4. do not reveal E0/E1 identity to the evaluator
5. collect a preference and optional quality reason
6. retain the mapping privately for analysis

Recommended preference dimensions:

- factual accuracy
- important-content coverage
- fluency/readability
- conciseness
- overall preference

Do not use human preferences to tune the frozen-test prompt after the comparison.

## Error analysis

Sample representative failures from both systems and classify:

- factual error / hallucination
- omitted important information
- unsupported addition
- over-compression
- under-compression
- repetition
- awkward Marathi
- template copying / memorization behavior

Error categories should be defined before final aggregate interpretation.

## External generalization

Because the current Marathi v1 set is template-heavy, an independent real-Marathi summarization benchmark is recommended.

The external set must be kept outside:

- training
- checkpoint selection
- prompt tuning
- hyperparameter tuning

Its results should be labeled as a separate generalization experiment, not merged with the frozen-test result.

## Reproducibility

Commit:

- evaluation script
- metric configuration
- package/version information
- input artifact hashes
- output metric table
- statistical-test configuration
- final E0/E1 prediction hashes
