from pathlib import Path
import random
import pandas as pd


SEED = 42
NATIVE_MARATHI = 500
ENGLISH = 200
ROMANIZED_MARATHI = 150
CODE_MIXED = 150

INPUT_FILE = Path("data/processed/marathi_v1/04_train_sft.csv")
OUTPUT_DIR = Path("data/processed/marathi_tokenizer_diagnostic")


def is_devanagari(text):
    chars = [c for c in text if c.isalpha()]
    if not chars:
        return False

    devanagari = sum(
        "\u0900" <= c <= "\u097F"
        for c in chars
    )

    return devanagari / len(chars) >= 0.80


def is_latin(text):
    chars = [c for c in text if c.isalpha()]
    if not chars:
        return False

    latin = sum(
        ("A" <= c <= "Z") or ("a" <= c <= "z")
        for c in chars
    )

    return latin / len(chars) >= 0.80


def transliterate_marathi(text):
    mapping = {
        "अ": "a", "आ": "aa", "इ": "i", "ई": "ee",
        "उ": "u", "ऊ": "oo", "ए": "e", "ऐ": "ai",
        "ओ": "o", "औ": "au",
        "क": "ka", "ख": "kha", "ग": "ga", "घ": "gha",
        "च": "cha", "छ": "chha", "ज": "ja", "झ": "jha",
        "ट": "ta", "ठ": "tha", "ड": "da", "ढ": "dha",
        "त": "ta", "थ": "tha", "द": "da", "ध": "dha",
        "न": "na", "प": "pa", "फ": "pha", "ब": "ba",
        "भ": "bha", "म": "ma", "य": "ya", "र": "ra",
        "ल": "la", "व": "va", "श": "sha", "ष": "sha",
        "स": "sa", "ह": "ha",
        "ळ": "la",
        "ा": "a", "ि": "i", "ी": "i", "ु": "u",
        "ू": "u", "े": "e", "ै": "ai", "ो": "o",
        "ौ": "au", "ं": "n", "ः": "h", "ँ": "n",
        "्": "",
    }

    return "".join(mapping.get(char, char) for char in text)


def main():
    random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    texts = (
        df["prompt"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    native = texts[texts.apply(is_devanagari)].tolist()

    if len(native) < NATIVE_MARATHI:
        raise ValueError(
            f"Only {len(native)} native Marathi examples available."
        )

    native_sample = random.sample(native, NATIVE_MARATHI)

    english_samples = [
        "The government announced a new policy for public services.",
        "The company reported higher revenue during the financial year.",
        "Researchers developed a new method for language processing.",
        "The project uses machine learning for automatic summarization.",
        "The report contains important information about the economy.",
    ]

    english = []
    while len(english) < ENGLISH:
        english.extend(english_samples)

    english = english[:ENGLISH]

    romanized = [
        transliterate_marathi(text)
        for text in native_sample[:ROMANIZED_MARATHI]
    ]

    code_mixed_devanagari = [
        f"{text} The project uses machine learning."
        for text in native_sample[:75]
    ]

    code_mixed_roman = [
        f"{transliterate_marathi(text)} The project uses machine learning."
        for text in native_sample[75:150]
    ]

    records = []

    for i, text in enumerate(native_sample):
        records.append(
            {
                "id": f"native_devanagari_{i:04d}",
                "category": "native_devanagari",
                "text": text,
            }
        )

    for i, text in enumerate(english):
        records.append(
            {
                "id": f"native_latin_{i:04d}",
                "category": "native_latin",
                "text": text,
            }
        )

    for i, text in enumerate(romanized):
        records.append(
            {
                "id": f"romanized_{i:04d}",
                "category": "romanized",
                "text": text,
            }
        )

    for i, text in enumerate(code_mixed_devanagari):
        records.append(
            {
                "id": f"code_mixed_devanagari_{i:04d}",
                "category": "code_mixed_devanagari",
                "text": text,
            }
        )

    for i, text in enumerate(code_mixed_roman):
        records.append(
            {
                "id": f"code_mixed_roman_{i:04d}",
                "category": "code_mixed_roman",
                "text": text,
            }
        )

    diagnostic = pd.DataFrame(records)

    output_file = OUTPUT_DIR / "tokenizer_diagnostic.csv"
    diagnostic.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Created: {output_file}")
    print(f"Total examples: {len(diagnostic)}")
    print("\nCategory counts:")
    print(diagnostic["category"].value_counts().sort_index())


if __name__ == "__main__":
    main()