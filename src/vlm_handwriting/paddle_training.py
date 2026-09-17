"""ERNIEKit dataset conversion for PaddleOCR-VL supervised fine-tuning."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from vlm_handwriting.data import ManifestValidationError, resolve_image_path

OCR_PROMPT = "OCR:"


def to_erniekit_record(row: Mapping[str, str], raw_root: Path) -> dict[str, Any]:
    """Convert one frozen-manifest row to the official OCR-VL SFT format."""
    relative_path = str(row.get("relative_path", "")).strip()
    text = str(row.get("text", ""))
    if not relative_path:
        raise ManifestValidationError("Training row has no relative_path")
    if not text.strip():
        raise ManifestValidationError("Training row has an empty label")

    image_path = resolve_image_path(raw_root, relative_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Training image does not exist: {image_path}")

    return {
        "image_info": [
            {
                "matched_text_index": 0,
                "image_url": str(image_path),
            }
        ],
        "text_info": [
            {"text": OCR_PROMPT, "tag": "mask"},
            {"text": text, "tag": "no_mask"},
        ],
    }


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Atomically write UTF-8 JSONL and return the number of records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            count += 1
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return count
