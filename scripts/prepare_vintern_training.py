#!/usr/bin/env python3
"""Convert the frozen GLM OCR mixture into official InternVL chat JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath


PROMPT = (
    "<image>\nHãy chép lại nguyên văn toàn bộ chữ nhìn thấy trong ảnh theo thứ tự đọc. "
    "Giữ nguyên dấu tiếng Việt, chữ in, chữ viết tay và xuống dòng. "
    "Chỉ trả về nội dung được chép, không giải thích và không thêm Markdown."
)
EXPECTED_INPUT_COUNTS = {"divorce": 220, "meddies": 132, "uit": 88}
MOJIBAKE_PATTERN = re.compile(
    r"[ÑÖÏÆ]|ï¿½|\b(?:COÂNG|ÑAÀU|ÑIEÀU|VOÁN|TÖ|XAÂY|DÖÏNG)\b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/workspace/vlm-handwriting/data/llamafactory/glm_ocr_divorce"),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("/workspace/vlm-handwriting/data/eval/divorce_petition"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/vlm-handwriting/data/internvl/divorce_ocr"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def source_name(image: str) -> str:
    first = PurePosixPath(image).parts[0]
    mapping = {"divorce_images": "divorce", "meddies_images": "meddies", "images": "uit"}
    if first not in mapping:
        raise ValueError(f"Unknown image prefix: {image}")
    return mapping[first]


def safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe image path: {value}")
    return path.as_posix()


def convert(row: dict, index: int) -> dict:
    messages = row.get("messages", [])
    images = row.get("images", [])
    if len(messages) != 2 or len(images) != 1:
        raise ValueError(f"Expected one image and two messages at row {index}")
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise ValueError(f"Unexpected roles at row {index}")
    answer = unicodedata.normalize("NFC", messages[1]["content"])
    if not answer.strip():
        raise ValueError(f"Empty target at row {index}")
    image = safe_relative_path(images[0])
    return {
        "id": f"ocr-{index:06d}",
        "image": image,
        "conversations": [
            {"from": "human", "value": PROMPT},
            {"from": "gpt", "value": answer},
        ],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def divorce_split_names(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["filename"] for row in csv.DictReader(handle)}


def main() -> None:
    args = parse_args()
    source_file = args.source_dir / "divorce_domain_train.json"
    validation_file = args.source_dir / "divorce_validation.json"
    source_rows = json.loads(source_file.read_text(encoding="utf-8"))
    validation_rows = json.loads(validation_file.read_text(encoding="utf-8"))

    converted_input = [convert(row, index) for index, row in enumerate(source_rows)]
    validation = [convert(row, index) for index, row in enumerate(validation_rows)]
    input_counts = Counter(source_name(row["image"]) for row in converted_input)
    if dict(input_counts) != EXPECTED_INPUT_COUNTS:
        raise ValueError(
            f"Unexpected input mixture: {dict(input_counts)} != {EXPECTED_INPUT_COUNTS}"
        )
    removed_mojibake = [
        row
        for row in converted_input
        if source_name(row["image"]) == "meddies"
        and MOJIBAKE_PATTERN.search(row["conversations"][1]["value"])
    ]
    converted = [row for row in converted_input if row not in removed_mojibake]
    counts = Counter(source_name(row["image"]) for row in converted)
    if counts != Counter({"divorce": 220, "meddies": 126, "uit": 88}):
        raise ValueError(f"Unexpected clean mixture after filtering: {dict(counts)}")

    all_images = [row["image"] for row in converted + validation]
    if len(all_images) != len(set(all_images)):
        raise ValueError("Duplicate image or train/validation overlap detected")
    missing = [image for image in all_images if not (args.source_dir / image).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first={missing[0]}")

    train_divorce = {
        Path(row["image"]).name for row in converted if source_name(row["image"]) == "divorce"
    }
    val_names = divorce_split_names(args.split_dir / "val.csv")
    test_names = divorce_split_names(args.split_dir / "test.csv")
    if train_divorce & val_names or train_divorce & test_names or val_names & test_names:
        raise ValueError("Document/page leakage detected across divorce train/val/test")

    rng = random.Random(args.seed)
    rng.shuffle(converted)
    smoke = []
    for source, wanted in (("divorce", 4), ("meddies", 2), ("uit", 2)):
        candidates = [row for row in converted if source_name(row["image"]) == source]
        smoke.extend(candidates[:wanted])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", converted)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    write_jsonl(args.output_dir / "smoke.jsonl", smoke)

    root = str(args.source_dir.resolve())
    for name, annotation, length in (
        ("train", "train.jsonl", len(converted)),
        ("validation", "validation.jsonl", len(validation)),
        ("smoke", "smoke.jsonl", len(smoke)),
    ):
        meta = {
            f"vintern_divorce_ocr_{name}": {
                "root": root,
                "annotation": str((args.output_dir / annotation).resolve()),
                "data_augment": False,
                "max_dynamic_patch": 6,
                "repeat_time": 1,
                "length": length,
            }
        }
        (args.output_dir / f"meta_{name}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    summary = {
        "seed": args.seed,
        "prompt": PROMPT,
        "train_rows": len(converted),
        "validation_rows": len(validation),
        "test_rows_untouched": len(test_names),
        "source_counts": dict(counts),
        "removed_meddies_mojibake": [row["image"] for row in removed_mojibake],
        "image_root": root,
        "test_used_for_training": False,
    }
    (args.output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
