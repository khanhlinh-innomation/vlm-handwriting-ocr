from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str) -> dict[str, object]:
    with (REPO_ROOT / "configs" / "glm" / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_smoke_config_is_short_and_never_uses_test_data() -> None:
    config = load_config("lora_smoke.yaml")
    assert config["model_name_or_path"] == "zai-org/GLM-OCR"
    assert config["template"] == "glm_ocr"
    assert config["dataset"] == "uit_hwdb_line_train_smoke"
    assert config["eval_dataset"] == "uit_hwdb_line_validation_smoke"
    assert "test" not in str(config).lower()
    assert config["max_steps"] == 2
    assert config["overwrite_output_dir"] is False


def test_full_config_preserves_official_lora_recipe() -> None:
    config = load_config("lora.yaml")
    assert config["dataset"] == "uit_hwdb_line_train"
    assert config["eval_dataset"] == "uit_hwdb_line_validation"
    assert "test" not in str(config).lower()
    assert config["finetuning_type"] == "lora"
    assert config["lora_rank"] == 8
    assert config["lora_target"] == "all"
    assert config["learning_rate"] == 1.0e-4
    assert config["num_train_epochs"] == 3.0
    assert config["bf16"] is True
    assert config["overwrite_output_dir"] is False
