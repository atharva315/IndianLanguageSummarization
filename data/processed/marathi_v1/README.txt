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
Many records are synthetic variations of the same underlying structure. Without a cap,
the model can overfit repeated templates while validation/test appear deceptively strong.
The cap retains diversity while preserving multiple examples per template family.

Recommended use:
04_train_sft.csv        -> QLoRA training
05_validation_sft.csv   -> checkpoint / model-selection validation
06_test_frozen.csv      -> baseline AND final evaluation; NEVER train on this
07_human_eval_100.csv   -> blinded baseline-vs-QLoRA human evaluation

Frozen test SHA-256:
D93B3EE80E6AE306C1C1F02855A244AE2C38F46D587F3C39C8C415F0011A117D

The current dataset is synthetic/template-heavy. Structural leakage controls are
implemented, but semantic quality and real-world generalization should be evaluated
on an independent real-Marathi benchmark before broad claims are made.

Do NOT change 06_test_frozen.csv after baseline results are visible.
