#!/usr/bin/env python3
"""Reload a saved GLM-OCR LoRA adapter and verify one validation inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from vlm_handwriting.artifacts import write_csv, write_json
from vlm_handwriting.data import find_manifest_root, load_manifest, resolve_image_path
from vlm_handwriting.environment import collect_environment
from vlm_handwriting.glm import load_glm_adapter, predict_one, validate_adapter_path
from vlm_handwriting.metrics import (
    character_error_rate,
    normalize_for_evaluation,
    word_error_rate,
)
from vlm_handwriting.smoke import select_smoke_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/glm/base.yaml"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def main() -> None:
    args = parse_args()
    adapter_path = validate_adapter_path(args.adapter_path)

    config = load_config(args.config)
    manifest_root = find_manifest_root(args.manifest_root)
    validation_rows = load_manifest(manifest_root / "val.csv")
    row = select_smoke_rows(validation_rows, size=20, seed=42)[0]
    image_path = resolve_image_path(args.raw_root, row["relative_path"])

    torch, processor, model, device, dtype = load_glm_adapter(config, adapter_path)
    prediction, latency = predict_one(
        torch=torch,
        processor=processor,
        model=model,
        device=device,
        image_path=image_path,
        prompt=str(config["model"]["prompt"]),
        generation=config["generation"],
    )
    ground_truth = normalize_for_evaluation(row["text"])
    result = {
        "model": "GLM-OCR LoRA smoke",
        "split": "validation_one",
        "writer_id": row["writer_id"],
        "filename": row["filename"],
        "relative_path": row["relative_path"],
        "ground_truth": ground_truth,
        "prediction": prediction,
        "sample_CER": character_error_rate(ground_truth, prediction),
        "sample_WER": word_error_rate(ground_truth, prediction),
        "exact_match": ground_truth == prediction,
        "latency_sec": latency,
    }
    output_dir = args.output_dir.expanduser().resolve()
    fields = list(result)
    write_csv(output_dir / "adapter_reload_prediction.csv", [result], fields)
    write_json(
        output_dir / "adapter_reload_metrics.json",
        {
            "adapter_path": str(adapter_path),
            "dtype": str(dtype),
            "gpu_name": torch.cuda.get_device_name(0),
            "peak_gpu_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            **result,
        },
    )
    write_json(output_dir / "environment.json", collect_environment())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Adapter reload verification passed: {output_dir}")


if __name__ == "__main__":
    main()
