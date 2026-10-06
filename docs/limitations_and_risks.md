# Limitations and Research Risks

## Dataset limitations

The Marathi v1 corpus is template-heavy and synthetic-style. Although template-family leakage was controlled, high automatic scores would not by themselves establish strong generalization to naturally occurring Marathi news, government, educational, or web documents.

**Action:** evaluate an independent real-Marathi benchmark before making broad generalization claims.

## Automatic metric limitations

ROUGE measures lexical overlap and can undervalue valid paraphrases. chrF++ adds character/word n-gram similarity, which is useful as a complementary signal but is still reference-overlap based. BERTScore captures contextual semantic similarity, but semantic similarity is not equivalent to factual correctness.

**Action:** report multiple metrics and complement them with blind human A/B evaluation and targeted error analysis.

## Checkpoint overfitting risk

Training loss continued to decrease while validation loss reached a minimum at checkpoint 1750 and later increased.

**Action:** use the predefined validation-based best checkpoint and disclose the later degradation.

## Compute and environment risk

QLoRA training/inference depends on compatible versions of PyTorch, Transformers, PEFT, TRL, bitsandbytes, CUDA, and the target GPU environment.

**Action:** pin the model revision and record exact package/runtime versions.

## Reproducibility risk

Cloud runtimes are ephemeral and large model checkpoints are expensive to store.

**Action:** preserve experiment configuration, dataset hashes, prediction hashes, selected-adapter hash, analysis files, and reproducible scripts in Git/Git LFS.

## Human-evaluation risk

A human A/B study can be affected by evaluator inconsistency, presentation order, and small sample size.

**Action:** blind the systems, randomize A/B display order, keep the sample fixed, and report the study design and uncertainty.

## Interpretation rule

A higher score on a single metric is not sufficient evidence that Marathi QLoRA is universally better. Conclusions must be based on the full E0/E1 evidence set and explicitly bounded by the dataset and evaluation design.
