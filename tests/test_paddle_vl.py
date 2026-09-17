from pathlib import Path

import pytest
import yaml

from vlm_handwriting.paddle_vl import validate_paddle_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_paddle_base_config_uses_official_element_prompt_and_is_test_free() -> None:
    path = REPO_ROOT / "configs" / "paddle" / "base.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["model"]["id"] == "PaddlePaddle/PaddleOCR-VL-1.6"
    assert config["model"]["engine"] == "transformers"
    assert config["model"]["prompt"] == "OCR:"
    assert config["generation"]["do_sample"] is False
    assert config["evaluation"]["smoke_size"] == 20
    assert config["evaluation"]["smoke_seed"] == 42
    assert config["evaluation"]["test"] is False


def test_validate_paddle_checkpoint_requires_complete_hf_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing files"):
        validate_paddle_checkpoint(tmp_path)

    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="safetensors"):
        validate_paddle_checkpoint(tmp_path)

    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    assert validate_paddle_checkpoint(tmp_path) == tmp_path.resolve()
