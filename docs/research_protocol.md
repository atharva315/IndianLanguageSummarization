\# Research Protocol



\## 1. Project Objective



This project studies Marathi abstractive text summarization using

`google/gemma-4-E2B-it` and parameter-efficient fine-tuning.



The main Marathi research comparison is:



1\. Untuned Gemma 4 E2B IT baseline

2\. Marathi-specific QLoRA fine-tuned model



The same frozen Marathi test set will be used for both experiments.



\---



\## 2. Main Model



Base model:



`google/gemma-4-E2B-it`



The exact model revision used for experiments will be recorded before

baseline inference.



The model will be loaded and evaluated on cloud GPU hardware.



The local RTX 2050 with 4 GB VRAM is not the target hardware for

Gemma training.



\---



\## 3. Marathi Research Scope



The user's contribution is restricted to Marathi.



The Marathi pipeline includes:



\- Marathi source-data preparation

\- Marathi dataset quality control

\- Marathi train/validation/test construction

\- Marathi tokenizer analysis

\- Gemma Marathi baseline evaluation

\- Marathi QLoRA fine-tuning

\- Marathi automatic evaluation

\- Marathi human evaluation preparation



Hindi and Tamil processing are outside the user's local research scope.



\---



\## 4. Experiment BM-001: Marathi Gemma Baseline



Experiment ID:



`MR-BM-001`



Purpose:



Measure the performance of the untouched Gemma model on the frozen

Marathi test set before Marathi-specific fine-tuning.



Configuration:



\- Model: `google/gemma-4-E2B-it`

\- Training: none

\- Adapter: none

\- Dataset: frozen Marathi test set

\- Input: source Marathi text only

\- Reference summary: used only for evaluation

\- Generation settings: fixed and recorded

\- Predictions: saved separately from the frozen test data



Metrics:



\- ROUGE

\- chrF++

\- BERTScore



Human evaluation will be performed separately.



\---



\## 5. Experiment FT-001: Marathi QLoRA



Experiment ID:



`MR-FT-001`



Purpose:



Measure the effect of Marathi-specific QLoRA fine-tuning on the same

Gemma base model.



Configuration:



\- Base model: `google/gemma-4-E2B-it`

\- Fine-tuning method: QLoRA

\- Training data: Marathi training split

\- Validation data: Marathi validation split

\- Test data: the exact same frozen test set used by MR-BM-001

\- Adapter: Marathi-specific LoRA adapter



The base model and tokenizer must remain consistent with the baseline.



The QLoRA experiment must not modify the frozen test set.



Checkpoint selection will be based on the predefined validation procedure,

which will be documented before training.



\---



\## 6. Fair Baseline vs QLoRA Comparison



MR-BM-001 and MR-FT-001 must use:



\- the same frozen test examples

\- the same source texts

\- the same reference summaries

\- the same tokenizer/model family

\- the same generation settings

\- the same evaluation implementation



The reference summary must never be provided to the model as input.



It is used only after prediction generation for metric calculation.



\---



\## 7. Dataset Principles



Original source datasets will be preserved separately from processed data.



The dataset pipeline will maintain:



`raw -> processed -> frozen`



Raw data:



\- original source files

\- never modified in place



Processed data:



\- cleaned

\- validated

\- deduplicated

\- checked for quality

\- prepared for model training and evaluation



Frozen data:



\- final evaluation data

\- immutable after finalization

\- used consistently across baseline and QLoRA experiments



No test example may be added to training data.



No test example may be used to tune model or generation settings.



\---



\## 8. Data Leakage Prevention



The dataset pipeline must check for:



\- exact duplicate source-summary pairs

\- normalized duplicate pairs

\- source leakage across train/validation/test

\- summary leakage across train/validation/test

\- duplicate or near-duplicate evaluation examples

\- inappropriate template overlap where relevant



The final split must be created before model training.



The frozen test set must not be modified after experiments begin.



\---



\## 9. Evaluation Principle



Automatic metrics will be calculated from:



`model prediction vs reference summary`



The model receives only:



`source text`



The reference summary is never included in the model input.



Prediction files and metric files will be stored separately from the frozen

test dataset.



\---



\## 10. Reproducibility



Each experiment must record:



\- experiment ID

\- model ID

\- model revision

\- dataset version

\- code version / Git commit

\- Python version

\- PyTorch version

\- Transformers version

\- PEFT version

\- TRL version

\- bitsandbytes version

\- Accelerate version

\- GPU hardware

\- CUDA environment

\- generation settings

\- training hyperparameters where applicable

\- evaluation configuration



\---



\## 11. Git and Experiment Discipline



Each completed research stage must be:



1\. implemented

2\. verified

3\. documented

4\. committed to Git

5\. pushed to GitHub



Generated model weights and checkpoints will not be committed directly to

the Git repository.



Large datasets and artifacts will be handled according to the project's

Git LFS and storage policy.



\---



\## 12. Research Integrity



The project will distinguish between:



\- measured experimental results

\- documented model capabilities

\- methodological decisions

\- hypotheses

\- interpretations



Experimental results will not be altered to achieve a desired outcome.



The frozen test set and evaluation procedure will remain fixed for fair

comparison between the baseline and Marathi QLoRA model.

