Marathi Gemma QLoRA — FINAL CSV DATA PACKAGE

Status: FINAL / FROZEN TEST IMMUTABLE

Dataset counts:
- Train: 14,020
- Validation: 796
- Frozen test: 794
- Total experimental pool: 15,610

Pipeline:
MASTER CORPUS -> cleaning -> exact normalized deduplication -> structural QA ->
template analysis -> template-family cap (12) -> template-aware split -> frozen test.

Methodological correction:
The earlier train-ready workbook used a 90/5/5 split that was NOT template-aware.
The final package recomputes the split so a structural template family does not cross
train/validation/test.

Why the template cap exists:
The collected articles and summaries came from external websites and article-summary resources, including BBC/XL-Sum and other sources. An AI-assisted analysis was used to identify/categorize recurring templates and domains; the examples were then balanced and shuffled before a template-aware split. Some externally sourced records can share the same underlying structure even when their wording or named entities differ. Without a cap, a few repeated structures can dominate training or make evaluation appear deceptively strong.

A template-family cap of 12 means: retain no more than 12 records from any one identified template family. It does NOT mean there are only 12 template families, nor does it mean the whole dataset contains only 12 examples. Keep source URLs, collection dates, licence/terms, and the template-analysis procedure in the data provenance record.

Recommended use:
04_train_sft.csv        -> QLoRA training
05_validation_sft.csv   -> checkpoint / model-selection validation
06_test_frozen.csv      -> baseline AND final evaluation; NEVER train on this
07_human_eval_100.csv   -> blinded baseline-vs-QLoRA human evaluation

Frozen test SHA-256:
D93B3EE80E6AE306C1C1F02855A244AE2C38F46D587F3C39C8C415F0011A117D

The current dataset is assembled from external sources and contains recurring structural templates. Structural leakage controls are implemented, but source quality, semantic correctness, source diversity, and real-world generalization should still be evaluated on a separately sourced real-Marathi benchmark before broad claims are made.

Do NOT change 06_test_frozen.csv after baseline results are visible.
