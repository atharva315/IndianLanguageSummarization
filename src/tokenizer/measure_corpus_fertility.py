from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


INPUT_FILE = Path(
    "data/processed/marathi_v1/04_train_sft.csv"
)

OUTPUT_DIR = Path(
    "data/processed/marathi_tokenizer_diagnostic"
)

MODELS = {
    "gemma4_e2b": "google/gemma-4-E2B-it",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "bloomz_3b": "bigscience/bloomz-3b",
}


def extract_source_text(prompt: str) -> str:
    """Extract source text from the verified SFT prompt."""

    separator = "\n\n"
    position = prompt.find(separator)

    if position == -1:
        raise ValueError(
            f"Verified separator not found. Prompt preview: {repr(prompt[:300])}"
        )

    source = prompt[position + len(separator):].strip()

    if not source:
        raise ValueError("Extracted source text is empty")

    return source


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required = {"pair_id", "prompt", "completion"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if len(df) != 14020:
        raise ValueError(
            f"Expected 14020 training rows, found {len(df)}"
        )

    # Verify the prompt structure before measuring anything.
    separator_counts = df["prompt"].str.count(r"\n\n")

    if not (separator_counts == 1).all():
        raise ValueError(
            "Not every prompt contains exactly one verified \\n\\n separator"
        )

    df["source_text"] = df["prompt"].map(extract_source_text)

    if df["source_text"].isna().any():
        raise ValueError("Missing extracted source text")

    if (df["source_text"].str.len() == 0).any():
        raise ValueError("Empty extracted source text")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result_rows = []
    summary_rows = []

    for model_name, model_id in MODELS.items():

        print(f"\nLoading tokenizer: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        token_counts = []
        word_counts = []
        fertilities = []

        for text in df["source_text"]:

            words = len(text.split())

            if words == 0:
                raise ValueError("Source text has zero words")

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

        model_result = pd.DataFrame(
            {
                "pair_id": df["pair_id"],
                "source_text": df["source_text"],
                "words": word_counts,
                "tokens": token_counts,
                "fertility": fertilities,
            }
        )

        model_result.to_csv(
            OUTPUT_DIR / f"{model_name}_corpus_fertility.csv",
            index=False,
        )

        summary_rows.append(
            {
                "model": model_name,
                "model_id": model_id,
                "n": len(model_result),
                "mean_fertility": model_result["fertility"].mean(),
                "median_fertility": model_result["fertility"].median(),
                "p90_fertility": model_result["fertility"].quantile(0.90),
                "p95_fertility": model_result["fertility"].quantile(0.95),
                "mean_tokens": model_result["tokens"].mean(),
                "mean_words": model_result["words"].mean(),
            }
        )

        print(f"{model_name}: measurement complete")

    summary = pd.DataFrame(summary_rows)

    summary_file = OUTPUT_DIR / "corpus_fertility_summary.csv"

    summary.to_csv(summary_file, index=False)

    print("\nCorpus fertility summary:")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(summary_file)

    print("\nCorpus tokenizer fertility measurement: PASS")


if __name__ == "__main__":
    main()