#!/usr/bin/env python3
"""Score private page-OCR text files against human-verified ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vlm_handwriting.artifacts import write_csv, write_json
from vlm_handwriting.metrics import (
    character_error_rate,
    corpus_metrics,
    normalize_for_evaluation,
    word_error_rate,
)

PAGE_SCORE_FIELDS = [
    "model",
    "relative_page",
    "ground_truth_chars",
    "prediction_chars",
    "sample_CER",
    "sample_WER",
    "exact_match",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score page-level OCR predictions without copying private text into artifacts"
    )
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="Prediction filename suffix, for example glm-ocr or paddleocr-vl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_ground_truth(root: Path) -> list[Path]:
    files = sorted(path for path in root.rglob("*.txt") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No ground-truth .txt files found under {root}")
    return files


def prediction_path_for(
    *, ground_truth_path: Path, ground_truth_root: Path, prediction_root: Path, model: str
) -> Path:
    relative = ground_truth_path.relative_to(ground_truth_root)
    return prediction_root / relative.parent / f"{relative.stem}__{model}.txt"


def read_ground_truth(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Ground truth is empty: {path}")
    if "<ILLEGIBLE>" in text:
        raise ValueError(
            f"Ground truth has unresolved <ILLEGIBLE> marker: {path}. "
            "Resolve or explicitly exclude it before scoring."
        )
    normalized = normalize_for_evaluation(text)
    if text != normalized:
        raise ValueError(f"Ground truth must already be NFC-normalized: {path}")
    return text


def main() -> None:
    args = parse_args()
    ground_truth_root = args.ground_truth_root.expanduser().resolve()
    prediction_root = args.prediction_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    ground_truth_files = discover_ground_truth(ground_truth_root)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty score directory: {output_dir}. "
            "Pass --overwrite only for an intentional rerun."
        )

    missing_predictions: list[str] = []
    for ground_truth_path in ground_truth_files:
        prediction_path = prediction_path_for(
            ground_truth_path=ground_truth_path,
            ground_truth_root=ground_truth_root,
            prediction_root=prediction_root,
            model=args.model,
        )
        if not prediction_path.is_file():
            missing_predictions.append(str(prediction_path))
    if missing_predictions:
        raise FileNotFoundError(
            "Missing prediction files for model "
            f"{args.model}: {missing_predictions}"
        )

    page_rows: list[dict[str, Any]] = []
    references: list[str] = []
    hypotheses: list[str] = []
    for ground_truth_path in ground_truth_files:
        prediction_path = prediction_path_for(
            ground_truth_path=ground_truth_path,
            ground_truth_root=ground_truth_root,
            prediction_root=prediction_root,
            model=args.model,
        )
        ground_truth = read_ground_truth(ground_truth_path)
        prediction = normalize_for_evaluation(prediction_path.read_text(encoding="utf-8"))
        relative_page = str(ground_truth_path.relative_to(ground_truth_root))
        page_rows.append(
            {
                "model": args.model,
                "relative_page": relative_page,
                "ground_truth_chars": len(ground_truth),
                "prediction_chars": len(prediction),
                "sample_CER": character_error_rate(ground_truth, prediction),
                "sample_WER": word_error_rate(ground_truth, prediction),
                "exact_match": ground_truth == prediction,
            }
        )
        references.append(ground_truth)
        hypotheses.append(prediction)

    metrics = corpus_metrics(references, hypotheses)
    metrics.update(
        {
            "model": args.model,
            "pages": len(page_rows),
            "ground_truth_root": str(ground_truth_root),
            "prediction_root": str(prediction_root),
            "total_ground_truth_chars": sum(row["ground_truth_chars"] for row in page_rows),
            "total_prediction_chars": sum(row["prediction_chars"] for row in page_rows),
            "privacy_note": "Text content is intentionally excluded from score artifacts.",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "page_scores.csv", page_rows, PAGE_SCORE_FIELDS)
    write_json(output_dir / "metrics.json", metrics)
    print(metrics)
    print(f"Score artifacts: {output_dir}")


if __name__ == "__main__":
    main()
