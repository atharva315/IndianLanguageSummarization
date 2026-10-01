\# Development Environment



\## Purpose



This document records the local development environment used for the Marathi

Gemma summarization research project.



The local machine is used for data preparation, code development,

evaluation, tokenizer analysis, documentation, and Git/GitHub workflows.



Gemma baseline inference and QLoRA training will be performed on cloud GPU

hardware rather than the local RTX 2050.



\## Python Environment



\- Python: 3.13

\- Environment: project-local `.venv`



\## GPU



\- GPU: NVIDIA GeForce RTX 2050

\- VRAM: 4 GB

\- NVIDIA Driver: 560.94

\- CUDA reported by `nvidia-smi`: 12.6



\## Python Packages



\- PyTorch: 2.14.0+cu126

\- Transformers: 5.17.0

\- Datasets: 5.0.1

\- Accelerate: 1.15.0

\- PEFT: 0.20.0

\- TRL: 1.13.0

\- bitsandbytes: 0.50.2

\- Evaluate: 0.4.6

\- Pandas: 3.0.5

\- Safetensors: 0.8.0



\## Reproducibility



The complete local package snapshot is stored in:



`requirements-local.txt`



This file represents the verified local development environment.

It is not automatically treated as the cloud training environment.



The cloud environment will be checked separately for Gemma 4 compatibility

before baseline inference or QLoRA training.



\## Important Constraint



The local RTX 2050 has 4 GB VRAM and is not the target hardware for

Gemma 4 training.



Model training will be performed on suitable cloud GPU hardware.

