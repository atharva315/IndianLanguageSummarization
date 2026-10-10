# Model Selection and QLoRA Rationale

## Purpose

The controlled Marathi experiment compares one fixed base model in two states:

- **E0 / MR-BM-001:** untouched `google/gemma-4-E2B-it`
- **E1 / MR-FT-001:** the same base model with Marathi-specific QLoRA SFT

The experimental question is whether adapting the selected base model improves its Marathi summarization results on the same frozen test. This design isolates the effect of the fine-tuning intervention more clearly than comparing different model families at the same time.

## Why Gemma 4 E2B IT?

The recorded project uses model ID `google/gemma-4-E2B-it`, pinned to revision:

`b515064b63ff28985d549455f7709f112e8a5e39`

The practical rationale is:

1. **Instruction-tuned variant:** the `-it` version is instruction-tuned, making it suitable for a prompt-driven summarization task without first having to instruction-tune a base-only checkpoint.
2. **Multilingual intent:** Gemma 4 is documented by Google as supporting more than 140 languages. This makes it a reasonable candidate for Marathi adaptation, though multilingual support alone does not prove good Marathi summarization.
3. **Practical size:** E2B is the small/efficient family member. Google describes approximately 2.3B effective parameters and 5.1B parameters including its per-layer embedding tables. This is a better starting point for resource-constrained experimentation than a much larger model, especially when quantized.
4. **Open weights and reproducibility:** the model can be downloaded and loaded at a pinned revision, rather than treating a changing hosted API as the experimental system. The model card currently lists Apache 2.0; re-check the exact revision's terms when distributing the model or adapter.
5. **Controlled before/after comparison:** using the same exact model revision for E0 and E1 means any measured change is associated with the fine-tuning pipeline, subject to data quality, leakage, and evaluation limitations.

These are methodological and practical reasons for choosing the model. The repository does **not** record a full head-to-head summarization benchmark of every alternative model. Therefore, do not claim that Gemma 4 E2B IT was experimentally proven to be the best available Marathi summarizer.

## Alternatives considered conceptually

| Alternative | Potential advantage | Why it was not the selected primary system for this experiment |
|---|---|---|
| A larger Gemma 4 variant, such as E4B, 12B, or 31B | May provide more capacity | Greater memory/runtime demands; a different model-size comparison would change the experimental question. These were not documented as full E0/E1 summarization baselines in this study. |
| Qwen-family instruction-tuned model | A multilingual open-weight alternative | Would be a valid future comparative baseline, but changing model family while testing fine-tuning would introduce another variable. Qwen3-4B was used in the repository's tokenizer-fertility analysis, not as a completed core summarization baseline. |
| IndicBART / MahaSum-associated BART models | Designed or adapted for Indic summarization; a useful task-specific reference | Different architecture and pretraining/training recipe; an interesting separate baseline, but not the matched same-model before/after experiment performed here. |
| mT5 or another multilingual encoder-decoder | Strong alternative architecture for summarization | Would require its own input/output formatting and training setup, so it should be treated as a separate experiment. |
| Full supervised fine-tuning | Updates all model weights and may adapt the model extensively | Much higher memory and storage requirements; less suitable for a constrained GPU budget than a parameter-efficient adapter. |
| LoRA without 4-bit base quantization | Trains a small adapter while keeping base weights frozen | Can be simpler in some environments but uses more base-weight memory than 4-bit QLoRA. |
| Prompting only / few-shot prompting | No training run or adapter storage needed | Useful as a low-cost baseline, but does not learn task-specific parameters from the project's training examples. It is not the intervention tested in MR-FT-001. |

Only E0 versus E1 is the official core comparison. The alternatives in this table are design choices and possible follow-up experiments, not systems that were proven inferior by this project's results.

## Why QLoRA?

### Full fine-tuning

Full fine-tuning updates all model parameters. For a model with billions of parameters, the training process also needs memory for gradients and optimizer state. This can make the setup expensive or impractical on limited GPUs.

### LoRA

LoRA freezes the original model weights and adds small trainable low-rank matrices to selected layers. The model's task behaviour is adapted by training those additional parameters instead of updating every original parameter.

### QLoRA

QLoRA adds 4-bit quantization of the frozen base weights while training LoRA adapters. The original weights are loaded in 4-bit NF4, double quantization is enabled, and adapter computation uses FP16 in the recorded run. The run used a paged AdamW 8-bit optimizer.

That combination reduces memory needs compared with conventional full fine-tuning, making parameter-efficient adaptation more practical on a constrained GPU notebook.

The selected configuration was:

- 4-bit NF4 quantization
- Double quantization: enabled
- Compute dtype: FP16
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Bias: none
- Targets: language-model q/k/v/o/gate/up/down projections
- Maximum sequence length: 768
- Per-device batch: 1
- Gradient accumulation: 8
- Two GPUs; effective global batch: 16
- Three epochs
- Learning rate: 1e-4 with cosine scheduling
- Warmup: 263 steps
- Weight decay: 0.01
- Maximum gradient norm: 1.0
- Gradient checkpointing: enabled
- Optimizer: paged AdamW 8-bit

The recorded run had 24,158,208 trainable parameters and the selected best adapter was about 92 MB.

The lowest recorded validation loss selected checkpoint 1750. The test set was not used to choose the checkpoint.

## Important interpretation limits

- QLoRA was chosen for resource efficiency and for a clear parameter-efficient adaptation experiment, not because this study compared every possible fine-tuning method.
- A high score after fine-tuning does not by itself prove broad Marathi generalization.
- The dataset's external source provenance, template balancing, duplicate controls, and possible template memorization must be examined alongside the metrics.
- Any future model comparison should keep the data split and evaluation protocol fixed and record the revision/configuration for each model.

## References

- [Google Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 E2B IT on Hugging Face](https://huggingface.co/google/gemma-4-E2B-it)
- [LoRA paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA paper (Dettmers et al., 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)
- [Hugging Face PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
