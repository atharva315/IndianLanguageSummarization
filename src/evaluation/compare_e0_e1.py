#!/usr/bin/env python3
"""
Compute reproducible E0 vs E1 Marathi summarization metrics.

Inputs:
  - official MR-BM-001 predictions
  - official MR-FT-001 predictions
  - frozen test references

Outputs:
  - aggregate_metrics.csv
  - per_example_metrics.csv
  - metadata.json

The evaluator assumes both prediction files contain:
  pair_id,text,model_summary

The frozen test contains:
  pair_id,text,reference_summary

The evaluator never sends reference summaries to a model. It only compares
already-generated predictions with the frozen references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
from bert_score import score as bertscore
from evaluate import load as load_metric


EXPECTED_TEST_SHA256 = (
    "D93B3EE80E6AE306C1C1F02855A244AE2C38F46D587F3C39C8C415F0011A117D"
)

EXPECTED_COLUMNS = ["pair_id", "text", "model_summary"]
EXPECTED_TEST_COLUMNS = ["pair_id", "text", "reference_summary"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def bootstrap_ci(
    diffs: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        raise ValueError("No paired observations available.")

    indices = rng.integers(0, n, size=(resamples, n))
    means = diffs[indices].mean(axis=1)
    alpha = 1.0 - confidence_level

    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    observed = float(diffs.mean())
    return observed, lower, upper


def validate_prediction_file(
    path: Path,
    *,
    expected_ids: list[str],
    expected_texts: list[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)

    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{path}: schema mismatch: {list(frame.columns)}")

    if len(frame) != len(expected_ids):
        raise ValueError(
            f"{path}: expected {len(expected_ids)} rows, found {len(frame)}"
        )

    ids = frame["pair_id"].tolist()
    texts = frame["text"].tolist()

    if ids != expected_ids:
        raise ValueError(f"{path}: pair_id order does not match frozen test")

    if texts != expected_texts:
        raise ValueError(f"{path}: source text does not match frozen test")

    if frame["model_summary"].str.strip().eq("").any():
        raise ValueError(f"{path}: empty prediction found")

    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        default="data/processed/marathi_v1/06_test_frozen.csv",
    )
    parser.add_argument(
        "--baseline",
        default="experiments/baseline/MR-BM-001/baseline_predictions.csv",
    )
    parser.add_argument(
        "--qlora",
        default="experiments/qlora/MR-FT-001/inference/MR-FT-001_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/MR-FT-001/evaluation",
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--bertscore-model",
        default="xlm-roberta-large",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    test_path = Path(args.test)
    baseline_path = Path(args.baseline)
    qlora_path = Path(args.qlora)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_test_hash = sha256_file(test_path)
    if actual_test_hash != EXPECTED_TEST_SHA256:
        raise ValueError(
            f"Frozen test hash mismatch: {actual_test_hash}"
        )

    test = pd.read_csv(test_path, dtype=str, keep_default_na=False)
    if list(test.columns) != EXPECTED_TEST_COLUMNS:
        raise ValueError("Frozen test schema mismatch")
    if len(test) != 794:
        raise ValueError(f"Expected 794 test rows, found {len(test)}")

    test_ids = test["pair_id"].tolist()
    test_texts = test["text"].tolist()
    references = test["reference_summary"].tolist()

    baseline = validate_prediction_file(
        baseline_path,
        expected_ids=test_ids,
        expected_texts=test_texts,
    )
    qlora = validate_prediction_file(
        qlora_path,
        expected_ids=test_ids,
        expected_texts=test_texts,
    )

    rouge = load_metric("rouge")

    baseline_rouge = rouge.compute(
        predictions=baseline["model_summary"].tolist(),
        references=references,
        use_stemmer=False,
    )
    qlora_rouge = rouge.compute(
        predictions=qlora["model_summary"].tolist(),
        references=references,
        use_stemmer=False,
    )

    # Sentence-level ROUGE values for paired bootstrap deltas.
    baseline_rouge_sentence = rouge.compute(
        predictions=baseline["model_summary"].tolist(),
        references=references,
        use_stemmer=False,
        use_aggregator=False,
    )
    qlora_rouge_sentence = rouge.compute(
        predictions=qlora["model_summary"].tolist(),
        references=references,
        use_stemmer=False,
        use_aggregator=False,
    )

    rows = []

    def add_metric(
        metric: str,
        e0: float,
        e1: float,
        *,
        e0_values: np.ndarray | None = None,
        e1_values: np.ndarray | None = None,
    ) -> None:
        record = {
            "metric": metric,
            "E0_MR-BM-001": float(e0),
            "E1_MR-FT-001": float(e1),
            "delta_E1_minus_E0": float(e1 - e0),
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
            "bootstrap_resamples": None,
            "bootstrap_seed": None,
        }

        if e0_values is not None and e1_values is not None:
            diff = np.asarray(e1_values) - np.asarray(e0_values)
            observed, low, high = bootstrap_ci(
                diff,
                resamples=args.bootstrap_resamples,
                seed=args.seed,
            )
            record["delta_E1_minus_E0"] = observed
            record["bootstrap_ci_low"] = low
            record["bootstrap_ci_high"] = high
            record["bootstrap_resamples"] = args.bootstrap_resamples
            record["bootstrap_seed"] = args.seed

        rows.append(record)

    for metric in ["rouge1", "rouge2", "rougeL"]:
        add_metric(
            metric,
            baseline_rouge[metric],
            qlora_rouge[metric],
            e0_values=np.asarray(baseline_rouge_sentence[metric], dtype=float),
            e1_values=np.asarray(qlora_rouge_sentence[metric], dtype=float),
        )

    # chrF++ corpus-level and sentence-level scores.
    e0_chrf = sacrebleu.corpus_chrf(
        baseline["model_summary"].tolist(),
        [references],
        char_order=6,
        word_order=2,
        beta=2,
        whitespace=False,
    ).score
    e1_chrf = sacrebleu.corpus_chrf(
        qlora["model_summary"].tolist(),
        [references],
        char_order=6,
        word_order=2,
        beta=2,
        whitespace=False,
    ).score

    e0_chrf_sentence = np.array(
        [
            sacrebleu.sentence_chrf(
                pred,
                [ref],
                char_order=6,
                word_order=2,
                beta=2,
                whitespace=False,
            ).score
            for pred, ref in zip(
                baseline["model_summary"],
                references,
                strict=True,
            )
        ]
    )
    e1_chrf_sentence = np.array(
        [
            sacrebleu.sentence_chrf(
                pred,
                [ref],
                char_order=6,
                word_order=2,
                beta=2,
                whitespace=False,
            ).score
            for pred, ref in zip(
                qlora["model_summary"],
                references,
                strict=True,
            )
        ]
    )

    add_metric(
        "chrF++",
        e0_chrf,
        e1_chrf,
        e0_values=e0_chrf_sentence,
        e1_values=e1_chrf_sentence,
    )

    # BERTScore.
    e0_p, e0_r, e0_f = bertscore(
        baseline["model_summary"].tolist(),
        references,
        model_type=args.bertscore_model,
        lang="mr",
        device=args.device,
        rescale_with_baseline=False,
        verbose=True,
    )
    e1_p, e1_r, e1_f = bertscore(
        qlora["model_summary"].tolist(),
        references,
        model_type=args.bertscore_model,
        lang="mr",
        device=args.device,
        rescale_with_baseline=False,
        verbose=True,
    )

    for metric, e0_values, e1_values in [
        ("BERTScore_P", e0_p, e1_p),
        ("BERTScore_R", e0_r, e1_r),
        ("BERTScore_F1", e0_f, e1_f),
    ]:
        e0_arr = e0_values.detach().cpu().numpy()
        e1_arr = e1_values.detach().cpu().numpy()

        add_metric(
            metric,
            float(e0_arr.mean()),
            float(e1_arr.mean()),
            e0_values=e0_arr,
            e1_values=e1_arr,
        )

    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(
        output_dir / "aggregate_metrics.csv",
        index=False,
    )

    # Per-example metric table.
    per_example = pd.DataFrame(
        {
            "pair_id": test_ids,
            "E0_rouge1": baseline_rouge_sentence["rouge1"],
            "E1_rouge1": qlora_rouge_sentence["rouge1"],
            "E0_rouge2": baseline_rouge_sentence["rouge2"],
            "E1_rouge2": qlora_rouge_sentence["rouge2"],
            "E0_rougeL": baseline_rouge_sentence["rougeL"],
            "E1_rougeL": qlora_rouge_sentence["rougeL"],
            "E0_chrfpp": e0_chrf_sentence,
            "E1_chrfpp": e1_chrf_sentence,
            "E0_bertscore_f1": e0_f.detach().cpu().numpy(),
            "E1_bertscore_f1": e1_f.detach().cpu().numpy(),
        }
    )
    per_example.to_csv(
        output_dir / "per_example_metrics.csv",
        index=False,
    )

    metadata = {
        "experiment_id": "MR-FT-001",
        "baseline_experiment_id": "MR-BM-001",
        "dataset_version": "marathi_v1",
        "frozen_test_sha256": actual_test_hash,
        "n_examples": 794,
        "reference_column": "reference_summary",
        "prediction_column": "model_summary",
        "rouge_use_stemmer": False,
        "chrfpp": {
            "char_order": 6,
            "word_order": 2,
            "beta": 2,
            "whitespace": False,
        },
        "bertscore": {
            "model_type": args.bertscore_model,
            "device": args.device,
            "rescale_with_baseline": False,
        },
        "bootstrap": {
            "resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "confidence_level": 0.95,
        },
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(aggregate.to_string(index=False))
    print(f"\nWrote evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
