# Research Decision Log

This log records material methodological decisions that affect experiment reproducibility.

## D-001 — Freeze one common test set

**Decision:** use one 794-example frozen Marathi test set for both E0 and E1.

**Reason:** isolate the effect of Marathi-specific QLoRA from changes in evaluation data.

**Integrity constraint:** the test file is immutable after baseline visibility.

## D-002 — Template-aware split

**Decision:** final train/validation/test construction is template-aware.

**Reason:** the corpus contains repeated structural templates. A naive random split could place near-identical structures across splits and inflate apparent generalization.

## D-003 — Template-family cap = 12

**Decision:** retain up to 12 examples per structural template family.

**Reason:** reduce domination by repeated synthetic structures while keeping enough examples for instruction tuning and preserving broad source/length diversity.

## D-004 — Sequence length = 768

**Decision:** use `max_length=768` for MR-FT-001 training.

**Evidence:** the train/validation token-length diagnostic showed 100% coverage of the 14,816 train+validation examples at 768 tokens, with maximum observed sequence length below the limit.

## D-005 — LoRA target modules

**Decision:** target language-model `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` modules.

**Reason:** the experiment is intended to adapt the language model while leaving vision/audio components outside the LoRA trainable set.

## D-006 — Best checkpoint selection

**Decision:** select checkpoint-1750 using validation loss.

**Observed best validation loss:** 0.0001272337.

Later validation loss increased. This is retained as a possible overfitting signal rather than being hidden by selecting a later checkpoint.

## D-007 — Deterministic E0/E1 generation

**Decision:** use the same prompt and deterministic generation settings for E0 and E1:

- prompt: `खालील मराठी मजकुराचा अचूक, संक्षिप्त आणि तथ्यसुसंगत सारांश लिहा.`
- `max_new_tokens=224`
- `do_sample=False`

## D-008 — Reference isolation

**Decision:** `reference_summary` is never supplied to the model.

**Reason:** prevent reference leakage and preserve a valid prediction-vs-reference evaluation boundary.

## D-009 — Store adapter, not full model state

**Decision:** archive the best LoRA adapter hash and configuration, but keep full base-model/checkpoint state out of normal Git history.

**Reason:** the base model is externally resolved by its pinned revision, while the adapter is the experiment-specific learned artifact.

## D-010 — Separate training and evaluation environments

**Decision:** record local development and Kaggle cloud training/inference environments separately.

**Reason:** cloud GPU packages and local development packages serve different purposes; exact experiment identity is anchored to model revision, dataset hash, code/configuration, and artifact hashes.

