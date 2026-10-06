# Industry and Research References

These references anchor the project in publicly documented model, fine-tuning, evaluation, and software-engineering practices. A reference listed here does not imply that the project used every referenced tool.

## Model and fine-tuning

### Gemma 4 — Google DeepMind

Google's official Gemma 4 model card documents the open-weight family, Apache 2.0 licensing, multimodal capabilities, and broad multilingual support.

Project use:
- exact base model: `google/gemma-4-E2B-it`
- pinned revision recorded in experiment metadata

https://ai.google.dev/gemma/docs/core/model_card_4

### QLoRA

Dettmers et al. introduced QLoRA, combining 4-bit quantization with trainable low-rank adapters and innovations such as NF4, double quantization, and paged optimizers.

Project use:
- NF4
- double quantization
- LoRA adapters
- paged 8-bit optimizer

https://arxiv.org/abs/2305.14314

### Hugging Face Transformers — bitsandbytes quantization

The Transformers documentation describes `BitsAndBytesConfig`, 4-bit NF4 quantization, and nested/double quantization.

Project use:
- `BitsAndBytesConfig(load_in_4bit=True, ...)`
- NF4
- double quantization

https://huggingface.co/docs/transformers/en//quantization/bitsandbytes

### Hugging Face PEFT — LoRA

PEFT documents LoRA as a parameter-efficient method that freezes the pretrained model and trains low-rank updates.

Project use:
- LoRA rank 16
- alpha 32
- dropout 0.05
- targeted language-model projections

https://huggingface.co/docs/peft/package_reference/lora

### Hugging Face TRL — SFTTrainer

TRL documents supervised fine-tuning through `SFTTrainer` and associated training configuration.

Project use:
- Marathi SFT / QLoRA training pipeline

https://huggingface.co/docs/trl/main/trainer

## Evaluation

### ROUGE

Lin (2004), “ROUGE: A Package for Automatic Evaluation of Summaries.”

Project use:
- ROUGE-1
- ROUGE-2
- ROUGE-L

https://aclanthology.org/W04-1013/

### chrF++

Popović (2017), “chrF++: words helping character n-grams.”

Project use:
- chrF++ as a complementary character/word n-gram metric

https://aclanthology.org/W17-4770/

### BERTScore

Zhang et al., “BERTScore: Evaluating Text Generation with BERT.”

Project use:
- contextual semantic similarity as a complementary automatic metric

https://arxiv.org/abs/1904.09675

### SacreBLEU / reproducible metric signatures

SacreBLEU documents chrF/chrF++, output signatures, and paired statistical testing utilities.

Project relevance:
- metric configuration/version signatures
- reproducible evaluation metadata

https://github.com/mjpost/sacreBLEU

## Reproducible evaluation practice

### EleutherAI lm-evaluation-harness

The project documents shareable task configuration and code-version practices for reproducible evaluations.

Project relevance:
- explicit versioned evaluation configuration
- public, reproducible prompts/configuration
- benchmark/task separation

https://github.com/EleutherAI/lm-evaluation-harness

## Engineering / CI

### GitHub Actions

GitHub Actions workflows are version-controlled YAML automation under `.github/workflows/`.

Project use:
- repository integrity checks
- Python syntax checks
- dataset/prediction schema checks
- frozen-test hash protection
- prevention of accidental model-weight commits

https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax

## Research caution

Automatic metrics are evidence, not proof of factuality or real-world usefulness. Human evaluation and an independent real-Marathi test are retained in this project to reduce over-interpretation of a single metric or a template-heavy dataset.
