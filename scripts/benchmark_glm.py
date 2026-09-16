#!/usr/bin/env python3
"""Run gated GLM-OCR base benchmarks on the frozen UIT-HWDB-line split."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vlm_handwriting.artifacts import write_csv, write_json
from vlm_handwriting.benchmark import (
    BENCHMARK_MODES,
    latency_summary,
    prediction_metrics,
    select_benchmark_rows,
)
from vlm_handwriting.data import find_manifest_root, load_manifest, resolve_image_path
from vlm_handwriting.environment import collect_environment
from vlm_handwriting.metrics import (
    character_error_rate,
    corpus_metrics,
    normalize_for_evaluation,
    word_error_rate,
)

PREDICTION_FIELDS = [
    "model",
    "split",
    "writer_id",
    "filename",
    "relative_path",
    "ground_truth",
    "prediction",
    "sample_CER",
    "sample_WER",
    "exact_match",
    "latency_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gated GLM-OCR base benchmark: one -> smoke -> frozen test"
    )
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/glm/base.yaml"))
    parser.add_argument("--mode", choices=BENCHMARK_MODES, default="one")
    parser.add_argument("--run-id", help="Output directory name; defaults to an UTC timestamp")
    parser.add_argument("--allow-test", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    for section in ("model", "generation", "evaluation"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing config section: {section}")
    return config


def load_glm(config: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    import torch
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("GLM-OCR benchmark requires a CUDA GPU")

    seed = int(config["evaluation"]["smoke_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_id = str(config["model"]["id"])
    processor = AutoProcessor.from_pretrained(model_id)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = GlmOcrForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
    ).eval()
    device = next(model.parameters()).device
    return torch, processor, model, (device, dtype)


def predict_one(
    *,
    torch: Any,
    processor: Any,
    model: Any,
    device: Any,
    image_path: Path,
    prompt: str,
    generation: dict[str, Any],
) -> tuple[str, float]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(device)

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(generation["max_new_tokens"]),
            do_sample=bool(generation["do_sample"]),
            repetition_penalty=float(generation["repetition_penalty"]),
        )
    torch.cuda.synchronize()
    latency = time.perf_counter() - started
    prompt_tokens = inputs["input_ids"].shape[-1]
    prediction = processor.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip()
    return normalize_for_evaluation(prediction), latency


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest_root = find_manifest_root(args.manifest_root)
    validation_rows = load_manifest(manifest_root / "val.csv")
    test_rows = load_manifest(manifest_root / "test.csv")
    evaluation = config["evaluation"]
    selected_rows, split_name, artifact_tag = select_benchmark_rows(
        validation_rows,
        test_rows,
        args.mode,
        smoke_size=int(evaluation["smoke_size"]),
        smoke_seed=int(evaluation["smoke_seed"]),
        allow_test=args.allow_test,
    )

    for row in selected_rows:
        image_path = resolve_image_path(args.raw_root, row["relative_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / "glm_ocr" / "base" / run_id
    prediction_path = run_dir / f"glm_ocr_base_{artifact_tag}_predictions.csv"
    metrics_path = run_dir / f"glm_ocr_base_{artifact_tag}_metrics.json"

    torch, processor, model, (device, dtype) = load_glm(config)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    predictions: list[dict[str, Any]] = []

    for index, row in enumerate(selected_rows, start=1):
        image_path = resolve_image_path(args.raw_root, row["relative_path"])
        ground_truth = normalize_for_evaluation(row["text"])
        prediction, latency = predict_one(
            torch=torch,
            processor=processor,
            model=model,
            device=device,
            image_path=image_path,
            prompt=str(config["model"]["prompt"]),
            generation=config["generation"],
        )
        result = {
            "model": str(config["model"]["name"]),
            "split": split_name,
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
        predictions.append(result)
        print(
            f"[{index}/{len(selected_rows)}] CER={result['sample_CER']:.4f} "
            f"latency={latency:.3f}s file={row['filename']}",
            flush=True,
        )

    runtime = time.perf_counter() - started
    references, hypotheses = prediction_metrics(predictions)
    metrics: dict[str, Any] = corpus_metrics(references, hypotheses)
    metrics.update(latency_summary([float(row["latency_sec"]) for row in predictions]))
    metrics.update(
        {
            "model": str(config["model"]["name"]),
            "model_id": str(config["model"]["id"]),
            "mode": args.mode,
            "split": split_name,
            "n": len(predictions),
            "total_runtime_sec": runtime,
            "throughput_samples_per_sec": len(predictions) / runtime,
            "peak_gpu_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "gpu_name": torch.cuda.get_device_name(0),
            "dtype": str(dtype),
            "prompt": str(config["model"]["prompt"]),
            "generation": config["generation"],
            "smoke_seed": int(evaluation["smoke_seed"]),
        }
    )

    write_csv(prediction_path, predictions, PREDICTION_FIELDS)
    write_json(metrics_path, metrics)
    write_json(run_dir / "environment.json", collect_environment())
    write_json(
        run_dir / "run_config.json",
        {
            "config_file": str(args.config.resolve()),
            "manifest_root": str(manifest_root),
            "raw_root": str(args.raw_root.resolve()),
            "output_directory": str(run_dir.resolve()),
            "config": config,
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
