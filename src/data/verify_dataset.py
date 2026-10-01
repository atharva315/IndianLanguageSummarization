from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/processed/marathi_v1")

TRAIN_FILE = DATA_DIR / "04_train_sft.csv"
VAL_FILE = DATA_DIR / "05_validation_sft.csv"
TEST_FILE = DATA_DIR / "06_test_frozen.csv"
HUMAN_FILE = DATA_DIR / "07_human_eval_100.csv"


EXPECTED_COUNTS = {
    "train": 14020,
    "validation": 796,
    "test": 794,
    "human_eval": 100,
}


def require_columns(df, required, filename):
    missing = set(required) - set(df.columns)
    if missing:
        raise AssertionError(
            f"{filename}: missing columns: {sorted(missing)}"
        )


def check_nonempty(df, columns, filename):
    for column in columns:
        if df[column].isna().any():
            raise AssertionError(
                f"{filename}: missing values found in '{column}'"
            )

        if df[column].astype(str).str.strip().eq("").any():
            raise AssertionError(
                f"{filename}: empty values found in '{column}'"
            )


def check_unique_pair_ids(df, filename):
    if df["pair_id"].duplicated().any():
        duplicates = df.loc[
            df["pair_id"].duplicated(keep=False), "pair_id"
        ].tolist()

        raise AssertionError(
            f"{filename}: duplicate pair_id values found: {duplicates[:10]}"
        )


def main():
    print("Starting Marathi dataset verification...\n")

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VAL_FILE)
    test = pd.read_csv(TEST_FILE)
    human = pd.read_csv(HUMAN_FILE)

    print(f"Train rows:       {len(train)}")
    print(f"Validation rows:  {len(validation)}")
    print(f"Test rows:        {len(test)}")
    print(f"Human eval rows:  {len(human)}")

    assert len(train) == EXPECTED_COUNTS["train"]
    assert len(validation) == EXPECTED_COUNTS["validation"]
    assert len(test) == EXPECTED_COUNTS["test"]
    assert len(human) == EXPECTED_COUNTS["human_eval"]

    require_columns(
        train,
        ["pair_id", "prompt", "completion"],
        TRAIN_FILE.name,
    )

    require_columns(
        validation,
        ["pair_id", "prompt", "completion"],
        VAL_FILE.name,
    )

    require_columns(
        test,
        [
            "pair_id",
            "source_file",
            "source_row",
            "template_id",
            "text",
            "reference_summary",
            "compression_ratio",
            "length_bucket",
        ],
        TEST_FILE.name,
    )

    require_columns(
        human,
        [
            "human_eval_id",
            "pair_id",
            "source_file",
            "length_bucket",
            "text",
            "reference_summary",
            "baseline_output",
            "qlora_output",
            "preferred",
            "faithfulness",
            "coverage",
            "fluency",
            "conciseness",
            "entity_number_accuracy",
            "notes",
        ],
        HUMAN_FILE.name,
    )

    check_nonempty(
        train,
        ["pair_id", "prompt", "completion"],
        TRAIN_FILE.name,
    )

    check_nonempty(
        validation,
        ["pair_id", "prompt", "completion"],
        VAL_FILE.name,
    )

    check_nonempty(
        test,
        ["pair_id", "text", "reference_summary"],
        TEST_FILE.name,
    )

    check_nonempty(
        human,
        ["human_eval_id", "pair_id", "text", "reference_summary"],
        HUMAN_FILE.name,
    )

    check_unique_pair_ids(train, TRAIN_FILE.name)
    check_unique_pair_ids(validation, VAL_FILE.name)
    check_unique_pair_ids(test, TEST_FILE.name)

    train_ids = set(train["pair_id"])
    validation_ids = set(validation["pair_id"])
    test_ids = set(test["pair_id"])

    train_validation_overlap = train_ids & validation_ids
    train_test_overlap = train_ids & test_ids
    validation_test_overlap = validation_ids & test_ids

    assert not train_validation_overlap, (
        f"Train/validation pair_id leakage: "
        f"{list(train_validation_overlap)[:10]}"
    )

    assert not train_test_overlap, (
        f"Train/test pair_id leakage: "
        f"{list(train_test_overlap)[:10]}"
    )

    assert not validation_test_overlap, (
        f"Validation/test pair_id leakage: "
        f"{list(validation_test_overlap)[:10]}"
    )

    human_ids = set(human["pair_id"])

    if not human_ids.issubset(test_ids):
        unexpected = human_ids - test_ids
        raise AssertionError(
            "Human-evaluation pair_ids are not all present in the "
            f"frozen test set: {list(unexpected)[:10]}"
        )

    print("\nPair-ID leakage checks: PASS")
    print("Human-evaluation membership check: PASS")

    print("\nDataset verification: PASS")


if __name__ == "__main__":
    main()