MR-BM-001 — OFFICIAL MARATHI GEMMA BASELINE
================================================

Status:
COMPLETE AND FROZEN

Model:
google/gemma-4-E2B-it

Model revision:
b515064b63ff28985d549455f7709f112e8a5e39

Quantization:
4-bit NF4

Compute dtype:
float16

Generation:
max_new_tokens = 224
do_sample = False
single-example generation

Dataset:
Marathi v1

Frozen test:
794 examples

Baseline predictions:
794 examples

Verification:
- 794/794 predictions
- 794 unique pair IDs
- 0 empty summaries
- prediction pair IDs match frozen test
- prediction source texts match frozen test
- reference_summary was NOT provided to model

IMPORTANT:
This is the official MR-BM-001 baseline.
Do not mix it with earlier experimental/batch inference outputs.

The same frozen 794-example test set must be used for
MR-FT-001 QLoRA evaluation.

SHA-256
-------

baseline_predictions.csv:
c67d873a6f762ac75e6708d3f686407059abae85a5c9b558836b054bf1beefd0

06_test_frozen.csv:
d93b3ee80e6ae306c1c1f02855a244ae2c38f46d587f3c39c8c415f0011a117d
