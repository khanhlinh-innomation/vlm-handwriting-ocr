import json
from pathlib import Path


def test_all_legacy_notebooks_are_valid_json() -> None:
    notebook_root = Path(__file__).parents[1] / "notebooks" / "legacy_colab"
    notebooks = sorted(notebook_root.glob("*.ipynb"))
    assert len(notebooks) == 9
    for notebook in notebooks:
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert isinstance(payload["cells"], list)
