# IndianLanguageSummarization

Research project on multilingual abstractive summarization for Indian languages using open-weight language models and parameter-efficient fine-tuning.

## Project Scope

The project studies language-specific summarization for Marathi, Hindi, and Tamil, followed by integration into a shared-base multilingual system.

## Core Model

- Base model: `google/gemma-4-E2B-it`
- Fine-tuning method: QLoRA
- Training hardware: cloud GPU
- Local machine: data preparation, research scripts, evaluation, documentation, and Git/GitHub workflow

## Language Responsibilities

- Marathi: Student 1
- Hindi: Student 2
- Tamil: Student 3
- Final adapter integration and multilingual routing: Student 4

## Evaluation

The project uses fixed frozen test data for fair comparison between the untouched Gemma baseline and the language-specific QLoRA model. Evaluation will include automatic metrics and human evaluation.

## Research Principle

Each project stage is verified, documented, and committed to GitHub before the next stage begins.
