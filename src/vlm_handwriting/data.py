"""Frozen-manifest loading, validation, and safe image path resolution."""

from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

EXPECTED_SPLIT_COUNTS = {"train": 6346, "val": 682, "test": 201}
EXPECTED_WRITER_COUNTS = {"train": 224, "val": 25, "test": 6}
REQUIRED_COLUMNS = {"writer_id", "filename", "relative_path", "text"}


class ManifestValidationError(ValueError):
    """Raised when the frozen dataset contract is violated."""


def find_manifest_root(base: Path) -> Path:
    """Find a directory containing train.csv, val.csv, and test.csv."""
    base = Path(base).expanduser().resolve()
    required = {"train.csv", "val.csv", "test.csv"}
    if base.is_dir() and required.issubset({path.name for path in base.iterdir()}):
        return base

    candidates = []
    if base.exists():
        for train_file in base.rglob("train.csv"):
            parent = train_file.parent
            if all((parent / name).is_file() for name in required):
                candidates.append(parent)
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No frozen split CSVs found under {base}")
    raise ManifestValidationError(f"Multiple manifest roots found under {base}: {candidates}")


def load_manifest(path: Path) -> list[dict[str, str]]:
    """Load a UTF-8 CSV manifest and validate its required columns."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ManifestValidationError(f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def find_raw_root(
    base: Path,
    relative_path: str,
    *,
    required_path_component: str | None = None,
) -> Path:
    """Find the raw directory for one known manifest-relative image path.

    ``required_path_component`` disambiguates datasets whose subsets reuse the
    same internal paths, such as UIT-HWDB line, word, and paragraph folders.
    """
    base = Path(base).expanduser().resolve()
    if base.is_file():
        base = base.parent
    portable = PurePosixPath(str(relative_path).replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts:
        raise ManifestValidationError(f"Unsafe relative image path: {relative_path}")

    def is_allowed(candidate: Path) -> bool:
        return required_path_component is None or required_path_component in candidate.parts

    if base.joinpath(*portable.parts).is_file() and is_allowed(base):
        return base

    candidates: set[Path] = set()
    if base.exists():
        for image_path in base.rglob(portable.name):
            candidate = image_path
            for _ in portable.parts:
                candidate = candidate.parent
            if candidate.joinpath(*portable.parts).is_file() and is_allowed(candidate):
                candidates.add(candidate)
    if len(candidates) == 1:
        return candidates.pop()
    if not candidates:
        raise FileNotFoundError(f"Cannot resolve {relative_path} under {base}")
    raise ManifestValidationError(f"Multiple raw roots found under {base}: {sorted(candidates)}")


def resolve_image_path(raw_root: Path, relative_path: str) -> Path:
    """Resolve a portable manifest path while preventing path traversal."""
    raw_root = Path(raw_root).expanduser().resolve()
    portable = str(relative_path).replace("\\", "/")
    posix_path = PurePosixPath(portable)
    if posix_path.is_absolute() or PureWindowsPath(relative_path).is_absolute():
        raise ManifestValidationError(f"Absolute image path is forbidden: {relative_path}")
    if ".." in posix_path.parts:
        raise ManifestValidationError(f"Image path traversal is forbidden: {relative_path}")

    candidate = raw_root.joinpath(*posix_path.parts).resolve()
    try:
        candidate.relative_to(raw_root)
    except ValueError as error:
        raise ManifestValidationError(f"Image path escapes raw root: {relative_path}") from error
    return candidate


def validate_frozen_manifests(
    manifest_root: Path,
    *,
    raw_root: Path | None = None,
    check_images: bool = False,
    expected_split_counts: dict[str, int] | None = None,
    expected_writer_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Validate counts, labels, unique paths, and writer-disjoint splits."""
    manifest_root = find_manifest_root(Path(manifest_root))
    expected_split_counts = expected_split_counts or EXPECTED_SPLIT_COUNTS
    expected_writer_counts = expected_writer_counts or EXPECTED_WRITER_COUNTS
    if check_images and raw_root is None:
        raise ValueError("raw_root is required when check_images=True")

    rows_by_split: dict[str, list[dict[str, str]]] = {}
    writers_by_split: dict[str, set[str]] = {}
    all_paths: set[str] = set()
    missing_images: list[str] = []

    for split in ("train", "val", "test"):
        rows = load_manifest(manifest_root / f"{split}.csv")
        expected_rows = expected_split_counts[split]
        if len(rows) != expected_rows:
            raise ManifestValidationError(
                f"{split} has {len(rows)} rows; frozen contract requires {expected_rows}"
            )

        writers = {str(row["writer_id"]).strip() for row in rows}
        expected_writers = expected_writer_counts[split]
        if len(writers) != expected_writers:
            raise ManifestValidationError(
                f"{split} has {len(writers)} writers; frozen contract requires {expected_writers}"
            )

        for row_number, row in enumerate(rows, start=2):
            if not str(row["text"]).strip():
                raise ManifestValidationError(f"{split}.csv row {row_number} has an empty label")
            relative_path = str(row["relative_path"]).strip()
            if not relative_path:
                raise ManifestValidationError(f"{split}.csv row {row_number} has no relative_path")
            if relative_path in all_paths:
                raise ManifestValidationError(
                    f"Duplicate relative_path across splits: {relative_path}"
                )
            all_paths.add(relative_path)

            accepted_split_names = {split, "validation" if split == "val" else split}
            if row.get("split") and row["split"].strip() not in accepted_split_names:
                raise ManifestValidationError(
                    f"{split}.csv row {row_number} declares split={row['split']!r}"
                )
            if check_images:
                image_path = resolve_image_path(Path(raw_root), relative_path)
                if not image_path.is_file():
                    missing_images.append(relative_path)

        rows_by_split[split] = rows
        writers_by_split[split] = writers

    overlap = {
        "train_val": sorted(writers_by_split["train"] & writers_by_split["val"]),
        "train_test": sorted(writers_by_split["train"] & writers_by_split["test"]),
        "val_test": sorted(writers_by_split["val"] & writers_by_split["test"]),
    }
    leaking = {pair: writers for pair, writers in overlap.items() if writers}
    if leaking:
        raise ManifestValidationError(f"Writer leakage detected: {leaking}")
    if missing_images:
        raise ManifestValidationError(
            f"Missing {len(missing_images)} labeled images; first paths: {missing_images[:10]}"
        )

    return {
        "manifest_root": str(manifest_root),
        "split_rows": {split: len(rows) for split, rows in rows_by_split.items()},
        "split_writers": {split: len(writers) for split, writers in writers_by_split.items()},
        "total_rows": sum(len(rows) for rows in rows_by_split.values()),
        "total_writers": len(set().union(*writers_by_split.values())),
        "writer_overlap": overlap,
        "images_checked": check_images,
    }
