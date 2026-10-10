# Research Protocol

## 1. Objective

This project studies Marathi abstractive summarization using `google/gemma-4-E2B-it` and parameter-efficient fine-tuning.

The primary comparison is:

1. **MR-BM-001:** untouched Gemma 4 E2B IT baseline
2. **MR-FT-001:** Marathi-specific QLoRA fine-tuned Gemma

The same immutable 794-example Marathi test set is used for both.

## 2. Model

- Model: `google/gemma-4-E2B-it`
- Pinned revision: `b515064b63ff28985d549455f7709f112e8a5e39`
- Quantized inference/training load: 4-bit NF4
- Double quantization: enabled
- QLoRA compute dtype: FP16

Gemma 4 is an open-weight model family from Google DeepMind; the official model card lists Apache 2.0 licensing and multilingual support. See [docs/industry_references.md](industry_references.md).

## 3. Dataset

Marathi v1:

- Train: 14,020
- Validation: 796
- Frozen test: 794
- Total experimental pool: 15,610
- Template-family cap: 12
- Split: template-aware

Frozen test SHA-256:

`D93B3EE80E6AE306C1C1F02855A244AE2C38F46D587F3C39C8C415F0011A117D`

Pipeline:

`raw -> cleaned -> normalized/deduplicated -> template QA -> template-aware split -> frozen test`

The dataset was assembled from external websites and article-summary resources, including BBC/XL-Sum and additional sources. Preprocessing was applied before AI-assisted template/domain analysis; recurring structures were categorized, representation was balanced, and the final records were shuffled and split with template awareness. This means template patterns were analysed in externally sourced material; it does not mean the articles were synthetically generated. Template separation reduces structural leakage risk, but it does not by itself prove semantic diversity or broad real-world generalization. Preserve the original source URLs, collection metadata, licence/terms, and the template-labeling procedure as part of the data provenance record.

## 4. Leakage controls

The data pipeline checks or documents:

- exact duplicate source-summary pairs
- normalized duplicate pairs
- source leakage across splits
- summary leakage across splits
- duplicate/near-duplicate evaluation content
- template-family overlap

No frozen test example is used for training or model-selection tuning.

## 5. MR-BM-001 — baseline

Purpose: measure untouched Gemma performance before Marathi adaptation.

- Training: none
- Adapter: none
- Input: source text only
- Reference summary: evaluation only
- Generation: `max_new_tokens=224`, deterministic greedy generation
- Frozen test: 794 examples
- Prediction SHA-256: `c67d873a6f762ac75e6708d3f686407059abae85a5c9b558836b054bf1beefd0`

## 6. MR-FT-001 — QLoRA

Purpose: isolate the effect of Marathi-specific parameter-efficient adaptation.

Training:

- Train: 14,020
- Validation: 796
- Epochs: 3
- max length: 768
- per-device batch: 1
- gradient accumulation: 8
- 2-GPU global effective batch: 16
- learning rate: 1e-4
- cosine schedule
- warmup: 263
- weight decay: 0.01
- max grad norm: 1.0
- LoRA: r=16, alpha=32, dropout=0.05, bias=none
- optimizer: paged AdamW 8-bit
- gradient checkpointing: enabled
- AMP: disabled in final run; QLoRA compute stayed FP16

Best checkpoint: **1750**

Best validation loss: **0.0001272337**

Best adapter SHA-256:

`6828bff5f35f384cb6c2a3736f1a0b8c2960b65cac70a874dd4d5230e45bde91`

Observed training loss fell from 2.5597917557 to 2.26634205e-05. Validation loss was lowest at step 1750; later validation degradation is recorded as a possible overfitting signal rather than being hidden.

## 7. Fair E0 vs E1 comparison

Both experiments use:

- identical frozen test examples
- identical source texts
- identical reference summaries for scoring
- the same Gemma model family and pinned revision
- the same inference prompt
- the same generation settings
- the same evaluation implementation

Reference summaries are never passed to the model.

## 8. Inference

MR-FT-001 frozen-test inference is complete:

- 794/794 generated
- deterministic generation
- checkpointed at 25-example intervals
- final checkpoint: 794

Inference artifact provenance is documented in [experiments/qlora/MR-FT-001/inference/README.md](../experiments/qlora/MR-FT-001/inference/README.md).

## 9. Evaluation

Primary metrics:

- ROUGE-1 / ROUGE-2 / ROUGE-L
- chrF++
- BERTScore

Human evaluation:

- blind A/B comparison
- fixed prepared sample
- model identity hidden from evaluators

Any external/generalization benchmark must remain outside the Marathi v1 training pool.

## 10. Reproducibility

Each experiment records:

- experiment ID
- model and exact revision
- dataset version
- frozen test hash
- Git commit / repository state
- Python and package versions
- GPU/CUDA environment
- training hyperparameters
- generation settings
- evaluation configuration
- artifact hashes where available

## 11. Storage policy

Large model weights, optimizer states, and training checkpoints are not stored in normal Git history.

Git LFS is used for approved large datasets/results. Hashes and metadata are retained for provenance.

## 12. Research integrity

The repository distinguishes:

- measured results
- configuration facts
- methodological decisions
- hypotheses
- interpretations

No result should be edited to produce a preferred outcome. The frozen test remains immutable after baseline visibility.
