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

The canonical E0-versus-E1 automatic-metric analysis uses the same 794 frozen-test examples for both systems and performs the following procedures separately for each of the five primary metrics (ROUGE-1, ROUGE-2, ROUGE-L, chrF++, and BERTScore-F1).

### Paired bootstrap confidence interval

- Calculate each paired difference as `d_i = E1_i - E0_i`.
- Draw 10,000 bootstrap samples, each resampling the 794 paired differences with replacement.
- Use the 2.5th and 97.5th percentiles of the bootstrap distribution of mean differences as the 95% confidence interval.

### Paired sign-flip randomization test

The committed analysis notebook implements a **two-sided paired sign-flip randomization test** (not a one-sided test):

- Number of randomization draws: **10,000**.
- NumPy random generator: `numpy.random.default_rng(20261007)`.
- For each draw, independently multiply each paired difference by a randomly selected sign, `-1` or `+1`, then calculate the mean of the sign-flipped differences.
- A draw is counted as at least as extreme when `abs(randomized_mean) >= abs(observed_mean_difference)`; the absolute values make the test two-sided.
- The Monte Carlo p-value uses the **add-one correction**:

  `p = (1 + number_of_extreme_draws) / (10,000 + 1)`

This avoids reporting a zero p-value merely because no sampled randomization was as extreme as the observed statistic. The raw p-values are then adjusted using the Holm step-down procedure across the five primary metrics. The documented significance threshold is `alpha = 0.05`, applied to the Holm-adjusted p-values.

The notebook's code is the source of truth for the test tail and p-value convention. Do not describe this core automatic-metric test as one-sided. A separate AI/LLM-judge analysis, if present in an exploratory notebook, is not the canonical human evaluation and is not part of this core automatic-metric procedure.

## Supplementary Google Forms blind human evaluation

A separate 100-example sample was evaluated through Google Forms. This was a human preference study, not an automated model-judge evaluation.

For each example:

1. Display the source Marathi article as the form question/prompt.
2. Provide two answer options containing only the generated E0 and E1 summary text. Do not show A/B tags, model names, E0/E1 labels, or other origin tags beside the summaries.
3. Shuffle the E0/E1 order for each example so either model's summary can appear as the first option.
4. Set the third option to **Tie**.
5. Ask: **Which summary do you prefer?**
6. Retain the per-example option-position-to-model mapping privately and apply it to the collected responses to calculate E0, E1, and Tie counts.

The reported audited outcome counts are E0 = 19, E1 = 27, and Tie = 54. The private option-order mapping is not committed to the public repository. Do not use these human preferences to tune the frozen-test prompt after the comparison.

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
