# Reproducibility Guide

## Experiment identity

The canonical identifiers for the current Marathi study are:

- Baseline: `MR-BM-001`
- QLoRA: `MR-FT-001`
- Dataset: `marathi_v1`

## Immutable identifiers

Frozen test SHA-256:

`D93B3EE80E6AE306C1C1F02855A244AE2C38F46D587F3C39C8C415F0011A117D`

MR-BM-001 prediction SHA-256:

`c67d873a6f762ac75e6708d3f686407059abae85a5c9b558836b054bf1beefd0`

MR-FT-001 best adapter SHA-256:

`6828bff5f35f384cb6c2a3736f1a0b8c2960b65cac70a874dd4d5230e45bde91`

## Model identity

- Model: `google/gemma-4-E2B-it`
- Revision: `b515064b63ff28985d549455f7709f112e8a5e39`

## Inference identity

- Prompt: `खालील मराठी मजकुराचा अचूक, संक्षिप्त आणि तथ्यसुसंगत सारांश लिहा.`
- max_new_tokens: 224
- do_sample: false
- Reference summary provided to model: **No**
- Frozen examples: 794

## Training identity

- Train: 14,020
- Validation: 796
- max_length: 768
- effective global batch: 16
- epochs: 3
- lr: 1e-4
- warmup: 263
- LoRA: r=16, alpha=32, dropout=0.05
- targets: q/k/v/o/gate/up/down
- NF4 + double quantization
- FP16 QLoRA compute
- paged AdamW 8-bit
- gradient checkpointing
- best checkpoint: 1750

## Artifact policy

1. Frozen test data is immutable.
2. Predictions are separate from evaluation references.
3. Model checkpoints are kept out of normal Git history.
4. Large approved artifacts use Git LFS.
5. Every official artifact gets a SHA-256 recorded in metadata or an experiment README.
6. Generated temporary files and failed experimental runs are not promoted to canonical artifacts.

## Verification workflow

A reproducible rerun should:

1. verify the frozen-test hash
2. verify the exact model revision
3. verify the experiment configuration
4. generate predictions without reference summaries in the input
5. verify 794 unique IDs and exact frozen-test order
6. calculate the same metrics using the versioned evaluation configuration
7. retain prediction and metric hashes
