from pathlib import Path

import pytest

from vlm_handwriting.glm import validate_adapter_path


def test_adapter_path_requires_config_and_weights(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="adapter_config"):
        validate_adapter_path(tmp_path)

    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="adapter weights"):
        validate_adapter_path(tmp_path)

    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    assert validate_adapter_path(tmp_path) == tmp_path.resolve()
