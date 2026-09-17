from pathlib import Path

import pytest
import yaml

from vlm_handwriting.teleocr import validate_adapter_path
from vlm_handwriting.teleocr_training import is_language_lora_target

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_language_lora_target_guard() -> None:
    suffixes = ("q_proj", "v_proj")
    assert is_language_lora_target("model.language_model.layers.0.self_attn.q_proj", suffixes)
    assert not is_language_lora_target("model.visual.blocks.0.attn.q_proj", suffixes)
    assert not is_language_lora_target("model.projector.language_model.q_proj", suffixes)
    assert not is_language_lora_target("model.language_model.layers.0.lm_head", suffixes)


def test_teleocr_smoke_config_is_small_and_test_free() -> None:
    path = REPO_ROOT / "configs" / "teleocr" / "lora_smoke.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["data"] == {"train_size": 32, "validation_size": 8, "seed": 42}
    assert config["training"]["max_steps"] == 2
    assert config["training"]["gradient_accumulation_steps"] == 16
    assert config["training"]["overwrite_output_dir"] is False
    assert "test" not in str(config).lower()


def test_adapter_path_requires_config_and_weights(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="adapter_config"):
        validate_adapter_path(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="adapter weights"):
        validate_adapter_path(tmp_path)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    assert validate_adapter_path(tmp_path) == tmp_path.resolve()
