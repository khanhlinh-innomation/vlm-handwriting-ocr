from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_handwriting.data import ManifestValidationError
from vlm_handwriting.paddle_training import to_erniekit_record, write_jsonl


def test_to_erniekit_record_uses_official_mask_format(tmp_path: Path) -> None:
    image = tmp_path / "train_data" / "writer-1" / "1.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    row = {
        "relative_path": "train_data/writer-1/1.jpg",
        "text": "Tiếng Việt giữ nguyên dấu.",
    }

    record = to_erniekit_record(row, tmp_path)

    assert record == {
        "image_info": [{"matched_text_index": 0, "image_url": str(image.resolve())}],
        "text_info": [
            {"text": "OCR:", "tag": "mask"},
            {"text": "Tiếng Việt giữ nguyên dấu.", "tag": "no_mask"},
        ],
    }


def test_to_erniekit_record_rejects_empty_label(tmp_path: Path) -> None:
    with pytest.raises(ManifestValidationError, match="empty label"):
        to_erniekit_record({"relative_path": "x.jpg", "text": "  "}, tmp_path)


def test_write_jsonl_preserves_unicode(tmp_path: Path) -> None:
    output = tmp_path / "data.jsonl"
    records = [{"text": "thành công"}, {"text": "độc lập"}]

    assert write_jsonl(output, records) == 2
    lines = output.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == records
    assert "thành công" in lines[0]
