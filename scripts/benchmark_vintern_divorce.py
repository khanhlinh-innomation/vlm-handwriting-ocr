#!/usr/bin/env python3
"""Benchmark pinned Vintern-1B-v3.5 on the frozen divorce test split."""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from pathlib import Path

os.environ["HF_HOME"] = "/workspace/vlm-handwriting/cache/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from vlm_handwriting.metrics import (
    character_error_rate,
    corpus_metrics,
    normalize_for_evaluation,
    word_error_rate,
)

ROOT = Path("/workspace/vlm-handwriting")
MODEL_ID = "5CD-AI/Vintern-1B-v3_5"
MODEL_REVISION = "fe7c963f86aa5bf017f01363cd63524765b6c41c"
MANIFEST = ROOT / "data/eval/divorce_petition/test.csv"
DATASET = ROOT / (
    "data/kagglehub/datasets/ntklinhfitus/"
    "vietnamese-handwritten-divorce-petition-ocr/versions/1/"
    "vietnamese-handwritten-divorce-petition-ocr"
)
OUTPUT = ROOT / "outputs/vintern_1b_v3_5/divorce_test_greedy_v4"
PROMPT = (
    "<image>\nHãy chép lại nguyên văn toàn bộ chữ nhìn thấy trong ảnh theo thứ tự đọc. "
    "Giữ nguyên dấu tiếng Việt, chữ in, chữ viết tay và xuống dòng. "
    "Chỉ trả về nội dung được chép, không giải thích và không thêm Markdown."
)
GENERATION = {
    "max_new_tokens": 2048,
    "do_sample": False,
    "num_beams": 1,
    "repetition_penalty": 1.1,
}
IMAGE_SIZE = 448
MAX_TILES = 6
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_transform():
    return T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )


def closest_ratio(aspect_ratio, ratios, width, height):
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for ratio in ratios:
        target = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target)
        if diff < best_diff or (
            diff == best_diff
            and area > 0.5 * IMAGE_SIZE * IMAGE_SIZE * ratio[0] * ratio[1]
        ):
            best_diff = diff
            best = ratio
    return best


def load_image(path: Path):
    started = time.perf_counter()
    image = Image.open(path).convert("RGB")
    width, height = image.size
    ratios = sorted(
        {
            (i, j)
            for n in range(1, MAX_TILES + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= MAX_TILES
        },
        key=lambda value: value[0] * value[1],
    )
    ratio = closest_ratio(width / height, ratios, width, height)
    target_width, target_height = IMAGE_SIZE * ratio[0], IMAGE_SIZE * ratio[1]
    resized = image.resize((target_width, target_height))
    tiles = []
    columns = target_width // IMAGE_SIZE
    for index in range(ratio[0] * ratio[1]):
        left = (index % columns) * IMAGE_SIZE
        top = (index // columns) * IMAGE_SIZE
        tiles.append(resized.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE)))
    if len(tiles) != 1:
        tiles.append(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
    transform = build_transform()
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    return pixel_values, time.perf_counter() - started, len(tiles)


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    text_dir = OUTPUT / "prediction_texts"
    text_dir.mkdir()
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig", newline="")))
    assert len(rows) == 12

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        use_flash_attn=False,
    ).eval().cuda()

    # One unscored warm-up uses the first test image but never its label.
    warm_pixels, _, _ = load_image(DATASET / rows[0]["relative_path"])
    with torch.inference_mode():
        model.chat(
            tokenizer,
            warm_pixels.to(torch.bfloat16).cuda(),
            PROMPT,
            GENERATION,
            history=None,
            return_history=True,
        )
    del warm_pixels
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    predictions = []
    started_all = time.perf_counter()
    for index, row in enumerate(rows, 1):
        pixels, preprocess_latency, tile_count = load_image(DATASET / row["relative_path"])
        pixels = pixels.to(torch.bfloat16).cuda()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            response, _ = model.chat(
                tokenizer,
                pixels,
                PROMPT,
                GENERATION,
                history=None,
                return_history=True,
            )
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        prediction = normalize_for_evaluation(response)
        ground_truth = normalize_for_evaluation(row["text"])
        output_tokens = len(tokenizer.encode(response, add_special_tokens=False))
        result = {
            "model": "Vintern-1B-v3.5",
            "split": "test",
            "writer_id": row["writer_id"],
            "filename": row["filename"],
            "relative_path": row["relative_path"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "sample_CER": character_error_rate(ground_truth, prediction),
            "sample_WER": word_error_rate(ground_truth, prediction),
            "exact_match": ground_truth == prediction,
            "preprocess_latency_sec": preprocess_latency,
            "inference_latency_sec": latency,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / latency if latency else None,
            "image_tiles": tile_count,
        }
        predictions.append(result)
        (text_dir / Path(row["filename"]).with_suffix(".txt")).write_text(
            prediction, encoding="utf-8"
        )
        print(
            f"[{index}/12] CER={result['sample_CER']:.4f} "
            f"latency={latency:.3f}s tokens={output_tokens} tiles={tile_count} "
            f"file={row['filename']}",
            flush=True,
        )
    runtime = time.perf_counter() - started_all
    references = [row["ground_truth"] for row in predictions]
    hypotheses = [row["prediction"] for row in predictions]
    metrics = corpus_metrics(references, hypotheses)
    latencies = [row["inference_latency_sec"] for row in predictions]
    metrics.update(
        {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "split": "test",
            "n": len(predictions),
            "prompt": PROMPT,
            "generation": GENERATION,
            "image_size": IMAGE_SIZE,
            "max_tiles": MAX_TILES,
            "warmup_runs": 1,
            "latency_scope": "model.chat only; image preprocessing excluded",
            "latency_mean_sec": statistics.fmean(latencies),
            "latency_p50_sec": statistics.median(latencies),
            "latency_p90_sec": sorted(latencies)[int(0.9 * (len(latencies) - 1))],
            "preprocess_latency_mean_sec": statistics.fmean(
                row["preprocess_latency_sec"] for row in predictions
            ),
            "tokens_per_sec_mean": statistics.fmean(
                row["tokens_per_sec"] for row in predictions
            ),
            "peak_gpu_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "total_runtime_sec": runtime,
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        }
    )
    fields = list(predictions[0])
    with (OUTPUT / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    (OUTPUT / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"COMPLETE {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
