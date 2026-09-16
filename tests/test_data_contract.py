import csv
from pathlib import Path

import pytest

from vlm_handwriting.data import (
    ManifestValidationError,
    find_raw_root,
    resolve_image_path,
    validate_frozen_manifests,
)


def _write_split(root: Path, split: str, writer_id: str, filename: str) -> None:
    with (root / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["writer_id", "filename", "relative_path", "text", "split"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "writer_id": writer_id,
                "filename": filename,
                "relative_path": f"UIT_HWDB_line/{writer_id}/{filename}",
                "text": "Tiếng Việt",
                "split": split,
            }
        )


def test_small_writer_disjoint_contract(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", "1", "train.jpg")
    _write_split(tmp_path, "val", "2", "val.jpg")
    _write_split(tmp_path, "test", "3", "test.jpg")
    report = validate_frozen_manifests(
        tmp_path,
        expected_split_counts={"train": 1, "val": 1, "test": 1},
        expected_writer_counts={"train": 1, "val": 1, "test": 1},
    )
    assert report["total_rows"] == 3
    assert report["total_writers"] == 3


def test_writer_leakage_fails(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", "1", "train.jpg")
    _write_split(tmp_path, "val", "1", "val.jpg")
    _write_split(tmp_path, "test", "3", "test.jpg")
    with pytest.raises(ManifestValidationError, match="Writer leakage"):
        validate_frozen_manifests(
            tmp_path,
            expected_split_counts={"train": 1, "val": 1, "test": 1},
            expected_writer_counts={"train": 1, "val": 1, "test": 1},
        )


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestValidationError, match="traversal"):
        resolve_image_path(tmp_path, "../secret.txt")


def test_nested_raw_root_is_discovered(tmp_path: Path) -> None:
    raw_root = tmp_path / "download" / "UIT_HWDB_line"
    image = raw_root / "train_data" / "1" / "1.jpg"
    image.parent.mkdir(parents=True)
    image.touch()
    assert find_raw_root(tmp_path, "train_data/1/1.jpg") == raw_root


def test_line_root_is_selected_when_subsets_reuse_paths(tmp_path: Path) -> None:
    line_root = tmp_path / "UIT_HWDB_line" / "UIT_HWDB_line"
    word_root = tmp_path / "UIT_HWDB_word" / "UIT_HWDB_word"
    for root in (line_root, word_root):
        image = root / "train_data" / "1" / "1.jpg"
        image.parent.mkdir(parents=True)
        image.touch()

    assert (
        find_raw_root(
            tmp_path,
            "train_data/1/1.jpg",
            required_path_component="UIT_HWDB_line",
        )
        == line_root
    )
