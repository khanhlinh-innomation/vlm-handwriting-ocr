import json
import zipfile
from pathlib import Path

import pytest

from vlm_handwriting.data import ManifestValidationError
from vlm_handwriting.meddies import (
    dataset_summary,
    extract_zip_safely,
    normalize_label,
    safe_relative_path,
    split_by_document,
    validate_rows,
)


def test_normalize_label_preserves_content_and_applies_nfc() -> None:
    text, reason = normalize_label("Tie\u0302\u0301ng Việt\n")
    assert text == "Tiếng Việt\n"
    assert reason is None


@pytest.mark.parametrize("value", ["", "   ", "bad\x00text", "bad\ufffdtext"])
def test_normalize_label_rejects_only_structural_failures(value: str) -> None:
    text, reason = normalize_label(value)
    assert text is None
    assert reason is not None


def test_safe_relative_path_rejects_absolute_and_traversal() -> None:
    assert safe_relative_path("images/page.jpg").as_posix() == "images/page.jpg"
    with pytest.raises(ManifestValidationError):
        safe_relative_path("../page.jpg")
    with pytest.raises(ManifestValidationError):
        safe_relative_path("C:\\secret\\page.jpg")


def test_safe_zip_extraction_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", "no")
    with pytest.raises(ManifestValidationError):
        extract_zip_safely(archive, tmp_path / "out")


def make_rows(document_count: int = 10) -> list[dict[str, object]]:
    return [
        {
            "id": f"row-{index}",
            "relative_path": f"images/{index}.jpg",
            "text": f"Trang số {index}",
            "doc_id": f"doc-{index}",
            "page_number": 0,
        }
        for index in range(document_count)
    ]


def test_document_split_is_deterministic_and_disjoint() -> None:
    rows = make_rows()
    first = split_by_document(rows, seed=42)
    second = split_by_document(rows, seed=42)
    assert first == second
    documents = {
        name: {row["doc_id"] for row in split_rows}
        for name, split_rows in first.items()
    }
    assert len(documents["train"]) == 8
    assert len(documents["validation"]) == 1
    assert len(documents["test"]) == 1
    assert not documents["train"] & documents["validation"]
    assert not documents["train"] & documents["test"]
    assert not documents["validation"] & documents["test"]


def test_validation_reports_missing_images_without_manual_guessing(tmp_path: Path) -> None:
    rows = [
        {
            "source_line": 1,
            "image_path": "images/missing.jpg",
            "text": "Tiếng Việt",
            "doc_id": "doc-1",
        }
    ]
    accepted, rejected = validate_rows(rows, image_root=tmp_path, check_images=False)
    assert accepted == []
    assert rejected[0]["reason"] == "missing_image"


def test_summary_is_json_serializable() -> None:
    splits = split_by_document(make_rows(), seed=42)
    summary = dataset_summary(splits, [{"reason": "missing_image"}])
    assert summary["accepted_rows"] == 10
    assert summary["rejection_reasons"] == {"missing_image": 1}
    json.dumps(summary)
