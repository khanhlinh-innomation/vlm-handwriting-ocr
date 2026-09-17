from pathlib import Path

import yaml

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
