import csv
import os
from collections import Counter

PROJECT = r"D:\Marathi-Gemma-Research"

CURATED = os.path.join(PROJECT, "data", "01_curated")
TOKENIZER = os.path.join(PROJECT, "data", "02_tokenizer")


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check_file(path, expected_rows=None, required_columns=None):
    print("\n" + "=" * 70)
    print(path)

    if not os.path.exists(path):
        print("[MISSING FILE]")
        return None

    rows = read_csv(path)
    print("Rows:", len(rows))

    if expected_rows is not None:
        print(
            "Expected rows:",
            expected_rows,
            "->",
            "PASS" if len(rows) == expected_rows else "FAIL"
        )

    if rows and required_columns:
        columns = list(rows[0].keys())
        missing = [c for c in required_columns if c not in columns]

        print("Columns:", columns)
        print(
            "Required columns:",
            "PASS" if not missing else f"FAIL -> missing {missing}"
        )

    return rows


print("\nMARATHI GEMMA RESEARCH DATASET VERIFICATION")
print("=" * 70)

train = check_file(
    os.path.join(CURATED, "04_train_sft.csv"),
    expected_rows=14020,
    required_columns=["pair_id", "prompt", "completion"],
)

validation = check_file(
    os.path.join(CURATED, "05_validation_sft.csv"),
    expected_rows=796,
    required_columns=["pair_id", "prompt", "completion"],
)

test = check_file(
    os.path.join(CURATED, "06_test_frozen.csv"),
    expected_rows=794,
    required_columns=[
        "pair_id",
        "source_file",
        "source_row",
        "template_id",
        "text",
        "reference_summary",
    ],
)

human = check_file(
    os.path.join(CURATED, "07_human_eval_100.csv"),
    expected_rows=100,
    required_columns=[
        "human_eval_id",
        "pair_id",
        "text",
        "reference_summary",
    ],
)

tokenizer = check_file(
    os.path.join(TOKENIZER, "tokenizer_fertility_diagnostic_1000.csv"),
    expected_rows=1000,
    required_columns=[
        "diag_id",
        "language",
        "variant",
        "text",
        "word_count_whitespace",
        "gemma4e2b_token_count",
        "gemma4e2b_fertility",
        "qwen3_1p7b_token_count",
        "qwen3_1p7b_fertility",
    ],
)


def check_missing(rows, fields):
    if not rows:
        return

    print("\nMissing-value check:")
    for field in fields:
        missing = sum(
            1 for r in rows
            if not str(r.get(field, "")).strip()
        )
        print(
            f"  {field}: {missing} missing ->",
            "PASS" if missing == 0 else "FAIL"
        )


def check_duplicate_ids(rows, field="pair_id"):
    if not rows:
        return

    ids = [r.get(field, "") for r in rows]
    counts = Counter(ids)
    duplicates = [x for x, n in counts.items() if x and n > 1]

    print(
        f"\nDuplicate {field}:",
        len(duplicates),
        "->",
        "PASS" if len(duplicates) == 0 else "FAIL"
    )


check_missing(train, ["pair_id", "prompt", "completion"])
check_duplicate_ids(train)

check_missing(validation, ["pair_id", "prompt", "completion"])
check_duplicate_ids(validation)

check_missing(test, ["pair_id", "text", "reference_summary"])
check_duplicate_ids(test)

check_missing(human, ["human_eval_id", "pair_id", "text", "reference_summary"])

check_missing(tokenizer, ["diag_id", "text"])

if train and validation and test:
    train_ids = {r["pair_id"] for r in train}
    val_ids = {r["pair_id"] for r in validation}
    test_ids = {r["pair_id"] for r in test}

    print("\nPAIR-ID LEAKAGE CHECK")
    print(
        "train ∩ validation:",
        len(train_ids & val_ids),
        "-> PASS" if len(train_ids & val_ids) == 0 else "-> FAIL",
    )
    print(
        "train ∩ test:",
        len(train_ids & test_ids),
        "-> PASS" if len(train_ids & test_ids) == 0 else "-> FAIL",
    )
    print(
        "validation ∩ test:",
        len(val_ids & test_ids),
        "-> PASS" if len(val_ids & test_ids) == 0 else "-> FAIL",
    )

print("\nVERIFICATION COMPLETE")