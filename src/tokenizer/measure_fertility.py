from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


INPUT_FILE = Path(
    "data/processed/marathi_tokenizer_diagnostic/tokenizer_diagnostic.csv"
)

OUTPUT_DIR = Path(
    "data/processed/marathi_tokenizer_diagnostic"
)

MODELS = {
    "gemma4_e2b": "google/gemma-4-E2B-it",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "bloomz_3b": "bigscience/bloomz-3b",
}


def count_words(text: str) -> int:
    """Count whitespace-separated words."""
    return len(text.split())


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Diagnostic file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {"id", "category", "text"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if len(df) != 1000:
        raise ValueError(f"Expected 1000 diagnostic examples, found {len(df)}")

    if df["id"].duplicated().any():
        raise ValueError("Diagnostic IDs are not unique")

    if df["text"].isna().any():
        raise ValueError("Diagnostic text contains missing values")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_example = df[["id", "category", "text"]].copy()

    summary_rows = []

    for model_name, model_id in MODELS.items():
        print(f"\nLoading tokenizer: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        token_counts = []
        word_counts = []
        fertilities = []

        for text in df["text"]:
            words = count_words(text)

            if words == 0:
                raise ValueError("Encountered text with zero whitespace-separated words")

            tokens = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            token_count = len(tokens)
            fertility = token_count / words

            token_counts.append(token_count)
            word_counts.append(words)
            fertilities.append(fertility)

        per_example[f"{model_name}_tokens"] = token_counts
        per_example[f"{model_name}_words"] = word_counts
        per_example[f"{model_name}_fertility"] = fertilities

        for category, group in per_example.assign(
            fertility=fertilities
        ).groupby("category")["fertility"]:

            summary_rows.append(
                {
                    "model": model_name,
                    "model_id": model_id,
                    "category": category,
                    "n": int(group.shape[0]),
                    "mean_fertility": group.mean(),
                    "median_fertility": group.median(),
                    "p90_fertility": group.quantile(0.90),
                    "p95_fertility": group.quantile(0.95),
                    "mean_tokens": (
                        per_example.loc[group.index, f"{model_name}_tokens"].mean()
                    ),
                    "mean_words": (
                        per_example.loc[group.index, f"{model_name}_words"].mean()
                    ),
                }
            )

        print(f"{model_name}: measurement complete")

    per_example_file = OUTPUT_DIR / "fertility_per_example.csv"
    summary_file = OUTPUT_DIR / "fertility_summary.csv"

    per_example.to_csv(per_example_file, index=False)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["category", "model"]).reset_index(drop=True)
    summary.to_csv(summary_file, index=False)

    print("\nSaved:")
    print(per_example_file)
    print(summary_file)

    print("\nFertility summary:")
    print(summary.to_string(index=False))

    print("\nTokenizer fertility measurement: PASS")


if __name__ == "__main__":
    main()