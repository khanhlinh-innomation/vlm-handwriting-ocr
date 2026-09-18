#!/usr/bin/env python3
"""Run one selected OCR checkpoint over page images and save per-page text."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from vlm_handwriting.artifacts import write_csv, write_json
from vlm_handwriting.benchmark import latency_summary

MODEL_CONFIGS = {
    "glm": Path("configs/glm/base.yaml"),
    "teleocr": Path("configs/teleocr/base.yaml"),
    "paddle": Path("configs/paddle/base.yaml"),
}

MODEL_SLUGS = {
    "glm": "glm-ocr",
    "teleocr": "teleocr",
    "paddle": "paddleocr-vl",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

INDEX_FIELDS = [
    "model",
    "checkpoint_path",
    "input_image",
    "output_text",
    "latency_sec",
    "prediction_chars",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Page-image OCR preview with per-page text and throughput artifacts"
    )
    parser.add_argument("--model", choices=tuple(MODEL_CONFIGS), required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
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


def discover_images(input_root: Path) -> list[Path]:
    images = sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No page images found under {input_root}")
    return images


def output_path_for(
    *, image_path: Path, input_root: Path, output_root: Path, model_slug: str
) -> Path:
    relative = image_path.relative_to(input_root)
    return output_root / relative.parent / f"{relative.stem}__{model_slug}.txt"


def load_model(
    *, model_name: str, config: dict[str, Any], checkpoint_path: Path
) -> tuple[Any, Any, Any, Any, Any]:
    if model_name == "glm":
        from vlm_handwriting.glm import load_glm_adapter

        return load_glm_adapter(config, checkpoint_path)
    if model_name == "teleocr":
        from vlm_handwriting.teleocr import load_teleocr_adapter

        return load_teleocr_adapter(config, checkpoint_path)
    if model_name == "paddle":
        from vlm_handwriting.paddle_vl import load_paddle_vl_checkpoint

        return load_paddle_vl_checkpoint(config, checkpoint_path)
    raise ValueError(f"Unsupported model: {model_name}")


def predict_page(
    *,
    model_name: str,
    config: dict[str, Any],
    generation: dict[str, Any],
    torch: Any,
    processor: Any,
    model: Any,
    device: Any,
    dtype: Any,
    image_path: Path,
) -> tuple[str, float]:
    if model_name == "glm":
        from vlm_handwriting.glm import predict_one

        return predict_one(
            torch=torch,
            processor=processor,
            model=model,
            device=device,
            image_path=image_path,
            prompt=str(config["model"]["prompt"]),
            generation=generation,
        )
    if model_name == "teleocr":
        from vlm_handwriting.teleocr import predict_one

        return predict_one(
            torch=torch,
            processor=processor,
            model=model,
            device=device,
            dtype=dtype,
            image_path=image_path,
            system_prompt=str(config["model"]["system_prompt"]),
            prompt=str(config["model"]["prompt"]),
            generation=generation,
        )
    if model_name == "paddle":
        from vlm_handwriting.paddle_vl import predict_one

        return predict_one(
            torch=torch,
            processor=processor,
            model=model,
            device=device,
            image_path=image_path,
            prompt=str(config["model"]["prompt"]),
            generation=generation,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    config_path = (args.config or MODEL_CONFIGS[args.model]).expanduser().resolve()
    config = load_config(config_path)
    generation = dict(config["generation"])
    generation["max_new_tokens"] = int(args.max_new_tokens)
    model_slug = MODEL_SLUGS[args.model]
    images = discover_images(input_root)

    summary_path = output_root / f"perf__{model_slug}.json"
    index_path = output_root / f"index__{model_slug}.csv"
    planned_outputs = [
        output_path_for(
            image_path=image,
            input_root=input_root,
            output_root=output_root,
            model_slug=model_slug,
        )
        for image in images
    ]
    existing = [path for path in [summary_path, index_path, *planned_outputs] if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing artifacts; pass --overwrite only for an intentional rerun: "
            f"{existing}"
        )

    load_started = time.perf_counter()
    torch, processor, model, device, dtype = load_model(
        model_name=args.model,
        config=config,
        checkpoint_path=checkpoint_path,
    )
    model_load_sec = time.perf_counter() - load_started
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    rows: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()
    for index, (image_path, text_path) in enumerate(
        zip(images, planned_outputs, strict=True), start=1
    ):
        prediction, latency = predict_page(
            model_name=args.model,
            config=config,
            generation=generation,
            torch=torch,
            processor=processor,
            model=model,
            device=device,
            dtype=dtype,
            image_path=image_path,
        )
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(prediction.rstrip() + "\n", encoding="utf-8")
        row = {
            "model": model_slug,
            "checkpoint_path": str(checkpoint_path),
            "input_image": str(image_path),
            "output_text": str(text_path),
            "latency_sec": latency,
            "prediction_chars": len(prediction),
        }
        rows.append(row)
        print(
            f"[{model_slug} {index}/{len(images)}] latency={latency:.3f}s "
            f"chars={len(prediction)} file={image_path.name}",
            flush=True,
        )
    benchmark_runtime_sec = time.perf_counter() - benchmark_started

    latencies = [float(row["latency_sec"]) for row in rows]
    generation_time_sec = sum(latencies)
    metrics: dict[str, Any] = latency_summary(latencies)
    metrics.update(
        {
            "model": model_slug,
            "checkpoint_path": str(checkpoint_path),
            "config_file": str(config_path),
            "input_root": str(input_root),
            "output_root": str(output_root),
            "pages": len(rows),
            "batch_size": 1,
            "warmup_pages": 0,
            "model_load_sec": model_load_sec,
            "benchmark_runtime_sec": benchmark_runtime_sec,
            "generation_time_sec": generation_time_sec,
            "pages_per_sec": len(rows) / benchmark_runtime_sec,
            "generation_pages_per_sec": len(rows) / generation_time_sec,
            "peak_gpu_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "gpu_name": torch.cuda.get_device_name(0),
            "dtype": str(dtype),
            "prompt": str(config["model"]["prompt"]),
            "generation": generation,
            "note": "Four-page quality pilot; no warmup. Do not use as the final 100-form perf score.",
        }
    )

    write_csv(index_path, rows, INDEX_FIELDS)
    write_json(summary_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Text artifacts: {output_root}")


if __name__ == "__main__":
    main()
