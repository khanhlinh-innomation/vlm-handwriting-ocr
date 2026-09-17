from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_paddle_smoke_config_is_two_step_and_never_uses_test_data() -> None:
    path = REPO_ROOT / "configs" / "paddle" / "erniekit_smoke.yaml"
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model_name_or_path"] == (
        "/workspace/vlm-handwriting/models/PaddleOCR-VL-1.6"
    )
    assert config["fine_tuning"] == "Full"
    assert config["train_dataset_path"].endswith("smoke_train.jsonl")
    assert config["eval_dataset_path"].endswith("smoke_validation.jsonl")
    assert "test" not in str(config).lower()
    assert config["max_steps"] == 2
    assert config["packing_size"] == 2
    assert config["gradient_accumulation_steps"] == 32
    assert config["compute_type"] == "bf16"
    assert config["save_steps"] == 2
