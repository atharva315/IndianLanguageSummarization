MR-FT-001 — POST-TRAINING ANALYSIS
===================================

Experiment:
MR-FT-001

Model:
google/gemma-4-E2B-it

Task:
Marathi abstractive summarization

Training:
- Train examples: 14,020
- Validation examples: 796
- Epochs: 3
- Max sequence length: 768
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Optimizer: bitsandbytes AdamW 8-bit paged
- Best checkpoint: checkpoint-1750

Best validation loss:
0.0001272336955492864

This folder contains:
- complete training history
- checkpoint summary
- post-training analysis JSON
- training-loss graph
- validation-loss graph
- combined train/validation-loss graph
- learning-rate graph
- gradient-norm graph
