#!/usr/bin/env python3
"""Benchmark and rank TeleOCR LoRA adapters on a gated frozen split."""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from vlm_handwriting.artifacts import write_csv, write_json
from vlm_handwriting.benchmark import (
    FULL_EVALUATION_SPLITS,
    enforce_frozen_test_single_adapter,
    evaluation_manifest_name,
    latency_summary,
    prediction_metrics,
)
from vlm_handwriting.data import find_manifest_root, load_manifest, resolve_image_path
from vlm_handwriting.environment import collect_environment
from vlm_handwriting.metrics import (
    character_error_rate,
    corpus_metrics,
    normalize_for_evaluation,
    word_error_rate,
)
from vlm_handwriting.teleocr import (
    load_teleocr_adapter,
    predict_one,
    validate_adapter_path,
)

PREDICTION_FIELDS = [
    "model",
    "adapter",
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

SUMMARY_FIELDS = [
    "rank",
    "adapter",
    "adapter_path",
    "split",
    "n",
    "cer",
    "wer",
    "exact_line_accuracy",
    "latency_mean_sec",
    "latency_p50_sec",
    "latency_p90_sec",
    "throughput_samples_per_sec",
    "peak_gpu_vram_gb",
    "total_runtime_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank TeleOCR LoRA checkpoints on validation before opening test"
    )
    parser.add_argument("--adapter-path", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/teleocr/base.yaml"))
    parser.add_argument("--split", choices=FULL_EVALUATION_SPLITS, default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--run-id", help="Output directory name; defaults to a UTC timestamp")
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


def benchmark_adapter(
    *,
    adapter_path: Path,
    rows: list[dict[str, str]],
    split_name: str,
    raw_root: Path,
    config: dict[str, Any],
    adapter_dir: Path,
) -> dict[str, Any]:
    torch, processor, model, device, dtype = load_teleocr_adapter(config, adapter_path)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    predictions: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        image_path = resolve_image_path(raw_root, row["relative_path"])
        ground_truth = normalize_for_evaluation(row["text"])
        prediction, latency = predict_one(
            torch=torch,
            processor=processor,
            model=model,
            device=device,
            dtype=dtype,
            image_path=image_path,
            system_prompt=str(config["model"]["system_prompt"]),
            prompt=str(config["model"]["prompt"]),
            generation=config["generation"],
        )
        result = {
            "model": "TeleOCR LoRA",
            "adapter": adapter_path.name,
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
            f"[{adapter_path.name} {index}/{len(rows)}] "
            f"CER={result['sample_CER']:.4f} latency={latency:.3f}s "
            f"file={row['filename']}",
            flush=True,
        )

    runtime = time.perf_counter() - started
    references, hypotheses = prediction_metrics(predictions)
    metrics: dict[str, Any] = corpus_metrics(references, hypotheses)
    metrics.update(latency_summary([float(row["latency_sec"]) for row in predictions]))
    metrics.update(
        {
            "adapter": adapter_path.name,
            "adapter_path": str(adapter_path),
            "split": split_name,
            "n": len(predictions),
            "total_runtime_sec": runtime,
            "throughput_samples_per_sec": len(predictions) / runtime,
            "peak_gpu_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "gpu_name": torch.cuda.get_device_name(0),
            "dtype": str(dtype),
            "system_prompt": str(config["model"]["system_prompt"]),
            "prompt": str(config["model"]["prompt"]),
            "generation": config["generation"],
        }
    )
    write_csv(adapter_dir / "predictions.csv", predictions, PREDICTION_FIELDS)
    write_json(adapter_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest_root = find_manifest_root(args.manifest_root)
    manifest_name, split_name = evaluation_manifest_name(
        args.split, allow_test=args.allow_test
    )
    rows = load_manifest(manifest_root / manifest_name)
    raw_root = args.raw_root.expanduser().resolve()
    for row in rows:
        image_path = resolve_image_path(raw_root, row["relative_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

    adapters = [validate_adapter_path(path) for path in args.adapter_path]
    enforce_frozen_test_single_adapter(split_name, len(adapters))
    adapter_names = [path.name for path in adapters]
    if len(set(adapter_names)) != len(adapter_names):
        raise ValueError(f"Adapter directory names must be unique: {adapter_names}")

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / "teleocr" / "lora" / split_name / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "environment.json", collect_environment())
    write_json(
        run_dir / "run_config.json",
        {
            "adapter_paths": [str(path) for path in adapters],
            "config_file": str(args.config.resolve()),
            "manifest_file": str((manifest_root / manifest_name).resolve()),
            "raw_root": str(raw_root),
            "split": split_name,
            "output_directory": str(run_dir.resolve()),
            "config": config,
        },
    )

    completed: list[dict[str, Any]] = []
    for adapter_path in adapters:
        metrics = benchmark_adapter(
            adapter_path=adapter_path,
            rows=rows,
            split_name=split_name,
            raw_root=raw_root,
            config=config,
            adapter_dir=run_dir / adapter_path.name,
        )
        completed.append(metrics)
        ranked = sorted(completed, key=lambda item: float(item["cer"]))
        summary = [{"rank": rank, **item} for rank, item in enumerate(ranked, start=1)]
        write_json(run_dir / "checkpoint_ranking.json", summary)
        summary_csv = [{field: row[field] for field in SUMMARY_FIELDS} for row in summary]
        write_csv(run_dir / "checkpoint_ranking.csv", summary_csv, SUMMARY_FIELDS)

    print(f"Checkpoint ranking by {split_name} CER:")
    for rank, metrics in enumerate(
        sorted(completed, key=lambda item: float(item["cer"])), start=1
    ):
        print(
            f"{rank}. {metrics['adapter']}: CER={metrics['cer']:.6f} "
            f"WER={metrics['wer']:.6f}"
        )
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
