"""Deterministic preparation helpers for the MeddiesOCR page dataset."""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
import warnings
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from vlm_handwriting.data import ManifestValidationError

REQUIRED_FIELDS = {"image_path", "text", "doc_id"}
SPLIT_NAMES = ("train", "validation", "test")
MEDDIES_REPO_ID = "Leo1903/meddiesOCR"
MEDDIES_REVISION = "584aa75cac164df05fa999453432d18ad3783f5c"
MAX_IMAGE_PIXELS = 100_000_000


def safe_relative_path(value: str) -> PurePosixPath:
    """Return a normalized portable path and reject absolute/traversing paths."""
    raw = str(value).strip()
    portable = PurePosixPath(raw.replace("\\", "/"))
    if not raw or portable.is_absolute() or PureWindowsPath(raw).is_absolute():
        raise ManifestValidationError(f"Unsafe relative image path: {value!r}")
    if ".." in portable.parts:
        raise ManifestValidationError(f"Unsafe relative image path: {value!r}")
    return portable


def extract_zip_safely(archive: Path, destination: Path) -> dict[str, int]:
    """Extract a dataset archive without permitting path traversal."""
    archive = Path(archive).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            portable = safe_relative_path(member.filename)
            target = destination.joinpath(*portable.parts).resolve()
            try:
                target.relative_to(destination)
            except ValueError as error:
                raise ManifestValidationError(
                    f"Archive member escapes destination: {member.filename}"
                ) from error
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_file() and target.stat().st_size == member.file_size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(member) as source, target.open("wb") as sink:
                while chunk := source.read(1024 * 1024):
                    sink.write(chunk)
            extracted += 1
    return {"extracted_files": extracted, "skipped_existing_files": skipped}


def load_annotations(path: Path) -> list[dict[str, Any]]:
    """Load the dataset JSONL while enforcing the documented core fields."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ManifestValidationError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ManifestValidationError(f"Non-object row at {path}:{line_number}")
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ManifestValidationError(
                    f"Missing fields at {path}:{line_number}: {sorted(missing)}"
                )
            row["source_line"] = line_number
            rows.append(row)
    return rows


def normalize_label(value: Any) -> tuple[str | None, str | None]:
    """Apply NFC and reject only deterministic structural/encoding failures."""
    text = unicodedata.normalize("NFC", str(value))
    if not text.strip():
        return None, "empty_label"
    if "\x00" in text:
        return None, "nul_character"
    if "\ufffd" in text:
        return None, "unicode_replacement_character"
    if any(unicodedata.category(char) == "Cs" for char in text):
        return None, "surrogate_codepoint"
    return text, None


def verify_image(path: Path) -> tuple[bool, str | None, tuple[int, int] | None]:
    """Verify one image with Pillow and return its dimensions."""
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError("Pillow is required for MeddiesOCR image validation") from error

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                dimensions = image.size
                if dimensions[0] * dimensions[1] > MAX_IMAGE_PIXELS:
                    return False, "oversized_image", dimensions
                image.verify()
    except Exception as error:  # Pillow exposes format-specific exception classes
        return False, f"invalid_image:{type(error).__name__}", None
    if dimensions[0] <= 0 or dimensions[1] <= 0:
        return False, "invalid_dimensions", dimensions
    return True, None, dimensions


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    image_root: Path,
    check_images: bool = True,
    progress_every: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate annotations and return accepted rows plus deterministic rejections."""
    image_root = Path(image_root).expanduser().resolve()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for row_index, source in enumerate(rows, start=1):
        if progress_every and row_index % progress_every == 0:
            print(f"Validated {row_index}/{len(rows)} MeddiesOCR rows")
        reason: str | None = None
        try:
            portable = safe_relative_path(str(source.get("image_path", "")))
        except ManifestValidationError:
            portable = None
            reason = "unsafe_image_path"

        text, text_reason = normalize_label(source.get("text", ""))
        reason = reason or text_reason
        doc_id = str(source.get("doc_id", "")).strip()
        if not doc_id:
            reason = reason or "missing_doc_id"

        dimensions: tuple[int, int] | None = None
        if portable is not None:
            portable_string = portable.as_posix()
            if portable_string in seen_paths:
                reason = reason or "duplicate_image_path"
            image_path = image_root.joinpath(*portable.parts).resolve()
            try:
                image_path.relative_to(image_root)
            except ValueError:
                reason = reason or "image_path_escape"
            if not image_path.is_file():
                reason = reason or "missing_image"
            elif check_images and reason is None:
                valid, image_reason, dimensions = verify_image(image_path)
                if not valid:
                    reason = image_reason
        else:
            portable_string = ""

        if reason is not None:
            rejected.append(
                {
                    "source_line": source.get("source_line"),
                    "image_path": source.get("image_path"),
                    "doc_id": source.get("doc_id"),
                    "reason": reason,
                }
            )
            continue

        seen_paths.add(portable_string)
        accepted.append(
            {
                "id": f"meddies_{len(accepted):07d}",
                "relative_path": portable_string,
                "text": text,
                "doc_id": doc_id,
                "page_number": source.get("page_number"),
                "total_pages": source.get("total_pages"),
                "source_url": source.get("source_url"),
                "processor": source.get("processor"),
                "source_line": source.get("source_line"),
                "width": dimensions[0] if dimensions else None,
                "height": dimensions[1] if dimensions else None,
            }
        )
    return accepted, rejected


def split_by_document(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic document-disjoint train/validation/test splits."""
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("Split ratios must be between zero and one")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train + validation ratios must leave a non-empty test ratio")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        doc_id = str(row["doc_id"])
        grouped[doc_id].append(row)
    documents = sorted(grouped)
    if len(documents) < 3:
        raise ManifestValidationError("At least three documents are required for 3-way split")
    random.Random(seed).shuffle(documents)

    validation_count = max(1, round(len(documents) * validation_ratio))
    test_count = max(1, len(documents) - round(len(documents) * (train_ratio + validation_ratio)))
    train_count = len(documents) - validation_count - test_count
    if train_count < 1:
        raise ManifestValidationError("Split ratios leave no training documents")

    doc_splits = {
        "train": set(documents[:train_count]),
        "validation": set(documents[train_count : train_count + validation_count]),
        "test": set(documents[train_count + validation_count :]),
    }
    split_pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    if any(doc_splits[left] & doc_splits[right] for left, right in split_pairs):
        raise AssertionError("Document leakage detected after split")

    result: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for split_name, split_documents in doc_splits.items():
        for doc_id in sorted(split_documents):
            for row in sorted(
                grouped[doc_id], key=lambda item: (int(item.get("page_number") or 0), item["id"])
            ):
                result[split_name].append({**row, "split": split_name})
    return result


def percentile(values: list[int], quantile: float) -> float | None:
    """Return a dependency-free linearly interpolated percentile."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def token_length_summary(texts: list[str], tokenizer_json: Path) -> dict[str, Any]:
    """Profile target-only token lengths with the exact fast-tokenizer artifact."""
    try:
        from tokenizers import Tokenizer
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError("tokenizers is required for exact target-length profiling") from error

    tokenizer = Tokenizer.from_file(str(Path(tokenizer_json).expanduser().resolve()))
    lengths: list[int] = []
    batch_size = 256
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        lengths.extend(len(encoded.ids) for encoded in tokenizer.encode_batch(batch))
    lengths.sort()
    return {
        "scope": "target_text_only; visual and prompt tokens are excluded",
        "rows": len(lengths),
        "min": min(lengths) if lengths else None,
        "p50": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths) if lengths else None,
        "over_cutoff": {
            str(cutoff): sum(length > cutoff for length in lengths)
            for cutoff in (2048, 4096, 8192)
        },
    }


def dataset_summary(
    splits: dict[str, list[dict[str, Any]]], rejected: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build the preparation report persisted beside generated manifests."""
    all_rows = [row for split in SPLIT_NAMES for row in splits[split]]
    lengths = [len(str(row["text"])) for row in all_rows]
    rejection_counts = Counter(str(row["reason"]) for row in rejected)
    split_report = {}
    for name in SPLIT_NAMES:
        rows = splits[name]
        split_report[name] = {
            "rows": len(rows),
            "documents": len({str(row["doc_id"]) for row in rows}),
            "characters": sum(len(str(row["text"])) for row in rows),
        }
    fingerprint = hashlib.sha256()
    for row in sorted(all_rows, key=lambda item: str(item["relative_path"])):
        fingerprint.update(str(row["relative_path"]).encode())
        fingerprint.update(b"\0")
        fingerprint.update(str(row["text"]).encode())
        fingerprint.update(b"\n")
    return {
        "accepted_rows": len(all_rows),
        "accepted_documents": len({str(row["doc_id"]) for row in all_rows}),
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "split": split_report,
        "label_characters": {
            "min": min(lengths) if lengths else None,
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": max(lengths) if lengths else None,
        },
        "manifest_sha256": fingerprint.hexdigest(),
        "document_overlap": {
            "train_validation": [],
            "train_test": [],
            "validation_test": [],
        },
    }
