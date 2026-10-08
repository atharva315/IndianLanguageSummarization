# MR-FT-001 — Marathi QLoRA Experiment

## Status

**Training: complete**  
**Frozen-test inference: complete (794/794)**  
**Automatic evaluation: complete**

## Research purpose

Measure the effect of Marathi-specific QLoRA SFT on the same base model used by MR-BM-001.

Base model:

`google/gemma-4-E2B-it`

Pinned revision:

`b515064b63ff28985d549455f7709f112e8a5e39`

## Data

- Train: 14,020
- Validation: 796
- Frozen test: 794
- Total experimental pool: 15,610
- Frozen test SHA-256: `D93B3EE80E6AE306C1C1F02855A244AE2C38F46D587F3C39C8C415F0011A117D`

The model never receives `reference_summary` during inference.

## Training configuration

- 4-bit NF4
- double quantization
- FP16 QLoRA compute
- LoRA r=16
- alpha=32
- dropout=0.05
- bias=none
- target projections: q/k/v/o + gate/up/down
- max length: 768
- per-device batch: 1
- gradient accumulation: 8
- 2-GPU effective batch: 16
- epochs: 3
- learning rate: 1e-4
- cosine schedule
- warmup: 263
- weight decay: 0.01
- max grad norm: 1.0
- gradient checkpointing: enabled
- paged AdamW 8-bit
- final AMP mode: disabled

## Checkpoint selection

Best validation checkpoint: **1750**

Best validation loss: **0.0001272337**

Best adapter SHA-256:

`6828bff5f35f384cb6c2a3736f1a0b8c2960b65cac70a874dd4d5230e45bde91`

Later validation loss increased after the best checkpoint; this is retained as a possible overfitting signal, not removed from the record.

## Inference

Generation settings:

- prompt: `खालील मराठी मजकुराचा अचूक, संक्षिप्त आणि तथ्यसुसंगत सारांश लिहा.`
- max_new_tokens: 224
- do_sample: false
- same frozen 794 examples as MR-BM-001
- source text only

See [inference/README.md](inference/README.md) for checkpointing and artifact rules.

## Repository contents

- `preparation/`: token-length and architecture evidence
- `analysis/`: training history, curves, checkpoint summary, post-training analysis
- `inference/`: prediction/checkpoint provenance
- `MR-FT-001_config.json`: training configuration
- `MR-FT-001_final_manifest.json`: final training manifest
- `MR-FT-001_runtime_config.json`: validated runtime configuration
- `MR-FT-001_training_start_manifest.json`: training start record

Model weights and full training checkpoints are intentionally not committed to Git.

  
### Supplementary human evaluation

A blind pairwise human evaluation was completed on 100 independent Marathi examples.

- **E0 (Base Gemma): 19 preferences**
- **E1 (Gemma + checkpoint-1750 QLoRA): 27 preferences**
- **Tie: 54**

For this supplementary inference, the shared prompt and deterministic decoding were retained; `max_new_tokens=2048` served only as a safety ceiling rather than a target length.

The public repository stores the final audited verdicts and aggregate result; the private A/B assignment key is not committed.
