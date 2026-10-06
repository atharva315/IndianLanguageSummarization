# MR-FT-001 Inference Artifacts

## Status

**Complete: 794/794 frozen-test examples**

## Exact inference setup

- Base model: `google/gemma-4-E2B-it`
- Revision: `b515064b63ff28985d549455f7709f112e8a5e39`
- Adapter: checkpoint-1750 / `best_adapter`
- Adapter SHA-256: `6828bff5f35f384cb6c2a3736f1a0b8c2960b65cac70a874dd4d5230e45bde91`
- Frozen test SHA-256: `D93B3EE80E6AE306C1C1F02855A244AE2C38F46D587F3C39C8C415F0011A117D`
- Prompt: `खालील मराठी मजकुराचा अचूक, संक्षिप्त आणि तथ्यसुसंगत सारांश लिहा.`
- max_new_tokens: 224
- do_sample: false
- reference_summary supplied to model: **No**

## Checkpoint strategy

Official cumulative checkpoints were saved every 25 examples:

`0025, 0050, 0075, ..., 0725, 0750, 0775, 0794`

A live recovery file was also maintained during inference.

The 0725 checkpoint was the last completed checkpoint before a controlled Kaggle interruption. Example 726 was separately smoke-tested after the session restart and explicitly incorporated into the continuation state. The final run then generated 727–794. The final cumulative 0794 artifact therefore contains all examples 1–794 exactly once.

## Required final validation

The canonical prediction file must satisfy:

- 794 rows
- 794 unique `pair_id`
- exact frozen-test `pair_id` order
- exact frozen-test source-text order
- no empty `model_summary`
- schema:
  `pair_id,text,model_summary`

## Canonical files

- `MR-FT-001_predictions.csv`
- `MR-FT-001_predictions_checkpoint_0794.csv`

Final SHA-256 values should be recorded in [docs/artifact_registry.md](../../../docs/artifact_registry.md) after the exact GitHub archive copy is verified.

## Storage policy

Prediction/checkpoint CSVs are approved for Git LFS. They must remain separate from `06_test_frozen.csv` and must never contain reference summaries as model input.

Temporary/live recovery files should not be treated as primary published results.
