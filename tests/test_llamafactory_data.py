import json
from pathlib import Path

import pytest

from vlm_handwriting.artifacts import write_json
from vlm_handwriting.data import ManifestValidationError
from vlm_handwriting.llamafactory_data import dataset_entry, to_sharegpt


def test_manifest_rows_convert_to_multimodal_sharegpt() -> None:
    rows = [
        {
            "relative_path": "train_data/7/17.jpg",
            "text": "Tiếng Việt",
        }
    ]
    assert to_sharegpt(rows) == [
        {
            "messages": [
                {"role": "user", "content": "<image>Text Recognition:"},
                {"role": "assistant", "content": "Tiếng Việt"},
            ],
            "images": ["images/train_data/7/17.jpg"],
        }
    ]


def test_sharegpt_conversion_rejects_path_traversal() -> None:
    with pytest.raises(ManifestValidationError, match="Unsafe"):
        to_sharegpt([{"relative_path": "../secret.jpg", "text": "secret"}])


def test_dataset_entry_and_list_artifact_are_json_serializable(tmp_path: Path) -> None:
    entry = dataset_entry("train.json")
    assert entry["formatting"] == "sharegpt"
    output = tmp_path / "rows.json"
    write_json(output, [{"text": "Tiếng Việt"}])
    assert json.loads(output.read_text(encoding="utf-8")) == [{"text": "Tiếng Việt"}]
