# IndianLanguageSummarization

Research project on multilingual abstractive summarization for Indian languages, beginning with a controlled Marathi study using an open-weight Gemma model and parameter-efficient fine-tuning.

## Current research question

For Marathi abstractive summarization:

> Does Marathi-specific QLoRA fine-tuning improve a fixed Gemma 4 E2B IT baseline on the same frozen evaluation set?

The Marathi comparison is:

- **E0 / MR-BM-001:** untouched `google/gemma-4-E2B-it`
- **E1 / MR-FT-001:** the same base model with Marathi-specific QLoRA SFT

The same frozen 794-example test set is used for both experiments. The model receives source text only; reference summaries are used only after generation for evaluation.

## Current status — 2026-10-08

| Stage | Status |
|---|---|
| Marathi dataset curation and split | ✅ Complete |
| Frozen test creation | ✅ Complete |
| Tokenizer diagnostic | ✅ Complete |
| MR-BM-001 baseline inference | ✅ Complete (794/794) |
| MR-FT-001 QLoRA training | ✅ Complete |
| MR-FT-001 training analysis | ✅ Complete |
| MR-FT-001 frozen-test inference | ✅ Complete (794/794) |
| ROUGE-1/2/L | ✅ Complete + corrected Unicode-aware evaluation |
| chrF++ | ✅ Complete |
| BERTScore | ✅ Complete |
| Paired statistical analysis | ✅ Complete (10,000 bootstrap + 10,000 sign-flip; Holm correction) |
| Length-bucket analysis | ✅ Complete |
| Data-leakage audit | ✅ Complete |
| Blind AI/LLM judge | ✅ Complete (supplementary Marathi analysis) |
| Independent real-Marathi benchmark | ⏸️ Deferred from core multilingual paper |
| Hindi / Tamil experiments | ⏳ Separate student work |

The core Marathi research checkpoint for the multilingual paper is now **automatic metrics + paired statistics + length-specific analysis + leakage/integrity audit**. The AI/LLM judge is retained as supplementary evidence and is not required for the other languages.

## Dataset: Marathi v1

- Train: **14,020**
- Validation: **796**
- Frozen test: **794**
- Total experimental pool: **15,610**
- Template-family cap: **12**
- Split: **template-aware**
- Frozen test SHA-256: `D93B3EE80E6AE306C1C1F02855A244AE2C38F46D587F3C39C8C415F0011A117D`

The dataset is synthetic/template-heavy. Structural leakage was explicitly controlled, but semantic quality is not assumed from the split alone. An independent real-Marathi benchmark is therefore recommended before making broad generalization claims.

Dataset documentation: [data/processed/marathi_v1/README.txt](data/processed/marathi_v1/README.txt)

## Model and training

Base model:

`google/gemma-4-E2B-it`

Pinned revision:

`b515064b63ff28985d549455f7709f112e8a5e39`

QLoRA configuration:

- 4-bit NF4
- double quantization
- FP16 compute
- LoRA rank `r=16`
- LoRA alpha `32`
- LoRA dropout `0.05`
- bias `none`
- language-model targets: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- max sequence length: **768**
- per-device batch size: **1**
- gradient accumulation: **8**
- 2-GPU global effective batch size: **16**
- epochs: **3**
- learning rate: **1e-4**, cosine schedule
- warmup steps: **263**
- weight decay: **0.01**
- max grad norm: **1.0**
- gradient checkpointing: enabled
- optimizer: paged AdamW 8-bit
- AMP: disabled in the final run because the Kaggle BF16 GradScaler path failed; QLoRA compute remained FP16

Best checkpoint:

- **checkpoint-1750**
- validation loss: **0.0001272337**
- adapter SHA-256: `6828bff5f35f384cb6c2a3736f1a0b8c2960b65cac70a874dd4d5230e45bde91`

The full training record and analysis are under [experiments/qlora/MR-FT-001/](experiments/qlora/MR-FT-001/).

## Inference protocol

Both E0 and E1 use:

- the exact frozen test set
- identical source ordering
- the same Marathi prompt
- `max_new_tokens=224`
- `do_sample=False`
- reference summary excluded from model input

E1 inference completed on all **794** frozen examples with cumulative checkpoints at 25-example intervals and a final checkpoint at 794.

The final prediction artifact is archived separately from the frozen test. Model checkpoints/optimizer state are intentionally not committed to this repository.

## Evaluation plan

Primary automatic metrics:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- chrF++
- BERTScore

Core statistical analysis:

- paired bootstrap confidence intervals
- paired sign-flip randomization
- Holm correction across the five primary metrics
- length-bucket analysis
- data-leakage/integrity audit

Supplementary Marathi analysis:

- blind A/B evaluation package on the prepared 100-example set
- AI/LLM-judge results and targeted qualitative/error analysis

An independent real-Marathi generalization benchmark is documented as a deferred supplementary direction rather than a core multilingual-paper requirement.

Metric definitions, normalization, aggregation, and statistical testing are documented in [docs/evaluation_protocol.md](docs/evaluation_protocol.md).

## Reproducibility

The repository tracks:

- experiment IDs
- exact model revision
- dataset version and frozen hash
- training and inference configuration
- environment information
- prediction provenance
- analysis artifacts
- research integrity checks
- external methodology references

See:

- [docs/research_protocol.md](docs/research_protocol.md)
- [docs/reproducibility.md](docs/reproducibility.md)
- [docs/artifact_registry.md](docs/artifact_registry.md)
- [docs/industry_references.md](docs/industry_references.md)

## Project structure

```
data/
  processed/
    marathi_v1/
experiments/
  baseline/
    MR-BM-001/
  qlora/
    MR-FT-001/
      analysis/
      preparation/
      inference/
docs/
scripts/
src/
.github/
  workflows/
```

## Team scope

- Marathi: current local research track
- Hindi: separate experiment
- Tamil: separate experiment
- Final multilingual integration/routing: later stage

## Research discipline

Every completed stage should be:

1. implemented
2. verified
3. documented
4. committed
5. pushed

Results must not be changed to fit a preferred conclusion. The frozen test and evaluation protocol remain fixed once the comparison begins.
