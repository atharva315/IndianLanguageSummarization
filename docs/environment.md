# Development and Training Environment

## Local development

The local machine is used for data preparation, code development, evaluation, tokenizer analysis, documentation, and Git/GitHub workflows.

- Python: 3.13
- GPU: NVIDIA GeForce RTX 2050
- VRAM: 4 GB
- Driver: 560.94
- CUDA reported by `nvidia-smi`: 12.6

Verified local packages:

- PyTorch: 2.14.0+cu126
- Transformers: 5.17.0
- Datasets: 5.0.1
- Accelerate: 1.15.0
- PEFT: 0.20.0
- TRL: 1.13.0
- bitsandbytes: 0.50.2
- Evaluate: 0.4.6
- Pandas: 3.0.5
- Safetensors: 0.8.0

The local package snapshot is stored in [requirements-local.txt](../requirements-local.txt).

## Kaggle cloud environment used for MR-FT-001

The final validated training/inference runtime used:

- Python: 3.13.15
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- GPU: 2 × NVIDIA Tesla T4 (~14.56 GB each)
- Transformers: 5.17.0 during the validated training run
- PEFT: 0.20.0
- Datasets: 4.8.5
- Accelerate: 1.14.0
- TRL: 1.13.0
- bitsandbytes: 0.50.2

The inference recovery session also verified:

- CUDA available
- Tesla T4 available
- bitsandbytes 0.50.2
- pinned Gemma revision
- NF4 + double quantization + FP16 compute
- best adapter loaded successfully
- 1-example resume smoke test passed before resuming

## Important distinction

The local and cloud environments are intentionally recorded separately. The local RTX 2050 was not used as the target hardware for Gemma QLoRA training.

## Reproducibility note

The exact model revision, dataset hash, training configuration, adapter hash, and inference settings are more important to experiment identity than matching every incidental package version between local and cloud machines.
