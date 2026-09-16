"""Conversion helpers for LLaMA-Factory multimodal ShareGPT datasets."""

from __future__ import annotations

from pathlib import PurePosixPath

from vlm_handwriting.data import ManifestValidationError


def to_sharegpt(
    rows: list[dict[str, str]],
    *,
    image_prefix: str = "images",
    prompt: str = "Text Recognition:",
) -> list[dict[str, object]]:
    """Convert frozen-manifest rows to the official multimodal ShareGPT shape."""
    converted: list[dict[str, object]] = []
    for row in rows:
        portable = PurePosixPath(str(row["relative_path"]).replace("\\", "/"))
        if portable.is_absolute() or ".." in portable.parts:
            raise ManifestValidationError(f"Unsafe relative image path: {row['relative_path']}")
        converted.append(
            {
                "messages": [
                    {"role": "user", "content": f"<image>{prompt}"},
                    {"role": "assistant", "content": str(row["text"])},
                ],
                "images": [str(PurePosixPath(image_prefix).joinpath(portable))],
            }
        )
    return converted


def dataset_entry(file_name: str) -> dict[str, object]:
    """Return LLaMA-Factory registration metadata for a ShareGPT JSON file."""
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
