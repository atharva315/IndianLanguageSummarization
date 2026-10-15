# Marathi Summarization — 12 Domain Sample Pairs

This folder contains one Marathi article and its corresponding reference summary for each of the 12 domains listed in the project progress report.

## Files

- [`dataset_samples_12_domains.csv`](./dataset_samples_12_domains.csv) — machine-readable UTF-8 CSV with 12 complete article–summary pairs.
- Source dataset: [`data/processed/marathi_v1/04_train_sft.csv`](../../data/processed/marathi_v1/04_train_sft.csv)

## Columns

| Column | Meaning |
|---|---|
| `number` | Sample row number (1–12) |
| `domain` | Domain label used for this illustrative sample collection |
| `pair_id` | Pair identifier recorded with the sample |
| `article` | Marathi source article |
| `reference_summary` | Corresponding Marathi reference summary |
| `source_file` | Source dataset path noted for the samples |

## Included domains

1. News
2. History and culture
3. Politics and policy
4. Environment
5. Banking and finance
6. Legal documents
7. Education
8. Agriculture
9. Health
10. Science
11. Government reports
12. Patents and technology

## Important interpretation notes

- The CSV contains **sample examples only**, not the full training dataset.
- Domain labels in this sample collection are presentation labels based on the subject of each selected record; the original SFT CSV schema does not have a dedicated row-level `domain` column.
- The examples are preserved as supplied in the project's sample table. Some articles explicitly describe hypothetical scenarios, so they should not be treated as verified real-world news, banking, government, or scientific records.
- Article and reference-summary text are kept together under the same pair ID. For training or evaluation, use the canonical dataset files and their documented split boundaries rather than this small presentation sample.

Repository: [IndianLanguageSummarization](https://github.com/atharva315/IndianLanguageSummarization)
