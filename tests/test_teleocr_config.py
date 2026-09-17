from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_teleocr_base_config_preserves_official_prompt_and_frozen_smoke() -> None:
    with (REPO_ROOT / "configs" / "teleocr" / "base.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["id"] == "StarDoc-AI/TeleOCR"
    assert config["model"]["system_prompt"] == "You are a helpful assistant."
    assert config["model"]["prompt"] == "Please output the text content from the image."
    assert config["generation"] == {
        "do_sample": False,
        "use_cache": True,
        "max_new_tokens": 512,
    }
    assert config["evaluation"]["smoke_size"] == 20
    assert config["evaluation"]["smoke_seed"] == 42
