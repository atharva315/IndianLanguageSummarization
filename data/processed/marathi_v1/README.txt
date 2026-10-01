Marathi Gemma QLoRA — FINAL CSV DATA PACKAGE

Pipeline completed:
MASTER CORPUS -> cleaning -> exact normalized deduplication -> structural QA ->
template analysis -> template-family cap -> template-aware split -> frozen test.

Important methodological correction:
The earlier train_ready workbook had a 90/5/5 split, but that split was NOT template-aware.
This revised package recomputes the split so a structural template family never crosses
train/validation/test.

Why the template cap exists:
Many records are synthetic variations of the same underlying structure. Without a cap,
the model can overfit to repeated templates while validation/test appear deceptively strong.
The cap preserves diversity while retaining multiple examples per template family.

How this avoids underfitting:
We do not collapse near-duplicates into one example; we retain up to 12 examples per
template family and keep broad source/length diversity. The training set remains large
enough for instruction tuning.

Recommended use:
04_train_sft.csv        -> QLoRA training
05_validation_sft.csv   -> checkpoint / early-stopping selection
06_test_frozen.csv      -> baseline AND final evaluation; NEVER train on it
07_human_eval_100.csv   -> blinded baseline-vs-QLoRA human evaluation

Do NOT change 06_test_frozen.csv after baseline results are seen.

For final study, add an independent real Marathi summarization test set such as MahaSum
if licensing/usage permits; keep it completely outside this training pool.
