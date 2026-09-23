#!/usr/bin/env python3
"""Run the selected Vintern divorce LoRA checkpoint once on the sealed test split.

Mirrors `benchmark_vintern_divorce.py` — same manifest, prompt, generation
config, tiling and warm-up — so the fine-tuned numbers sit directly beside the
base-model run. The only difference is how the model is loaded.

The checkpoint keeps LoRA wrapped (`language_model.base_model.model...`), and the
Hub copy of `modeling_internvl_chat.py` has no `use_llm_lora` handling, so
`AutoModel` + `trust_remote_code` would silently evaluate the base model. This
loads `InternVLChatModel` from the training repo and aborts if any key is
missing or unexpected.

This is the one authorised test run for this checkpoint. Do not tune against it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/vlm-handwriting")
os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.transforms.functional import InterpolationMode  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MANIFEST = ROOT / "data/eval/divorce_petition/test.csv"
DATASET = ROOT / (
    "data/kagglehub/datasets/ntklinhfitus/"
    "vietnamese-handwritten-divorce-petition-ocr/versions/1/"
    "vietnamese-handwritten-divorce-petition-ocr"
)

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

REFUSAL = re.compile(
    r"(tôi (không|ko) thể|không thể (đọc|chép|trích|nhận)|xin lỗi|i (can'?t|cannot))",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/vintern_1b_v3_5/divorce_lora_10epoch",
    )
    parser.add_argument("--label", default="vintern_divorce_lora_epoch10")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/vintern_1b_v3_5/divorce_test_finetuned",
    )
    return parser.parse_args()


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
    best_diff, best = float("inf"), (1, 1)
    area = width * height
    for ratio in ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff or (
            diff == best_diff and area > 0.5 * IMAGE_SIZE * IMAGE_SIZE * ratio[0] * ratio[1]
        ):
            best_diff, best = diff, ratio
    return best


def load_image(path: Path, transform):
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
    columns = target_width // IMAGE_SIZE
    tiles = []
    for index in range(ratio[0] * ratio[1]):
        left = (index % columns) * IMAGE_SIZE
        top = (index // columns) * IMAGE_SIZE
        tiles.append(resized.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE)))
    if len(tiles) != 1:
        tiles.append(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    return pixel_values, time.perf_counter() - started, len(tiles)


def repeated_ngram_ratio(text: str, n: int = 8) -> float:
    tokens = text.split()
    if len(tokens) < n * 2:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


def load_model(path: Path):
    from internvl.model.internvl_chat import InternVLChatModel

    model, info = InternVLChatModel.from_pretrained(
        str(path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        output_loading_info=True,
    )
    missing = [k for k in info.get("missing_keys", []) if "lora" in k or "language_model" in k]
    unexpected = info.get("unexpected_keys", [])
    if missing or unexpected:
        raise RuntimeError(
            f"{path} did not load cleanly: {len(missing)} missing, "
            f"{len(unexpected)} unexpected; trained weights would be ignored."
        )
    return model.eval().cuda()


def main() -> int:
    sys.path.insert(0, str(ROOT / "repo/src"))
    from vlm_handwriting.metrics import (
        character_error_rate,
        corpus_metrics,
        normalize_for_evaluation,
        word_error_rate,
    )

    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    text_dir = args.output / "prediction_texts"
    text_dir.mkdir(parents=True)

    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig", newline="")))
    if len(rows) != 12:
        raise ValueError(f"Expected 12 sealed test rows, got {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.checkpoint), trust_remote_code=True, use_fast=False
    )
    model = load_model(args.checkpoint)
    transform = build_transform()

    # One unscored warm-up on the first image; its label is never read.
    warm, _, _ = load_image(DATASET / rows[0]["relative_path"], transform)
    with torch.inference_mode():
        model.chat(tokenizer, warm.to(torch.bfloat16).cuda(), PROMPT, GENERATION)
    del warm
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    predictions = []
    started_all = time.perf_counter()
    for index, row in enumerate(rows, 1):
        pixels, preprocess_latency, tiles = load_image(
            DATASET / row["relative_path"], transform
        )
        pixels = pixels.to(torch.bfloat16).cuda()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            response = model.chat(tokenizer, pixels, PROMPT, GENERATION)
        torch.cuda.synchronize()
        latency = time.perf_counter() - started

        prediction = normalize_for_evaluation(response)
        ground_truth = normalize_for_evaluation(row["text"])
        output_tokens = len(tokenizer.encode(response, add_special_tokens=False))
        cer = character_error_rate(ground_truth, prediction)
        record = {
            "model": args.label,
            "split": "test",
            "writer_id": row.get("writer_id", ""),
            "filename": row["filename"],
            "relative_path": row["relative_path"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "sample_CER": cer,
            "sample_WER": word_error_rate(ground_truth, prediction),
            "exact_match": ground_truth == prediction,
            "preprocess_latency_sec": preprocess_latency,
            "inference_latency_sec": latency,
            "output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / latency if latency else None,
            "image_tiles": tiles,
            "hit_token_cap": output_tokens >= GENERATION["max_new_tokens"] - 8,
            "looks_like_refusal": bool(REFUSAL.search(prediction)),
            "repeated_8gram_ratio": round(repeated_ngram_ratio(prediction), 4),
            "length_ratio": round(len(prediction) / max(len(ground_truth), 1), 4),
        }
        predictions.append(record)
        (text_dir / Path(row["filename"]).with_suffix(".txt")).write_text(
            prediction, encoding="utf-8"
        )
        print(
            f"[{index}/12] CER={cer:.4f} latency={latency:.2f}s "
            f"tokens={output_tokens} tiles={tiles} {row['filename']}",
            flush=True,
        )

    runtime = time.perf_counter() - started_all
    metrics = corpus_metrics(
        [r["ground_truth"] for r in predictions], [r["prediction"] for r in predictions]
    )
    latencies = [r["inference_latency_sec"] for r in predictions]
    sample_cers = [r["sample_CER"] for r in predictions]
    metrics.update(
        {
            "model": args.label,
            "checkpoint": str(args.checkpoint),
            "split": "test",
            "n": len(predictions),
            "prompt": PROMPT,
            "generation": GENERATION,
            "image_size": IMAGE_SIZE,
            "max_tiles": MAX_TILES,
            "warmup_runs": 1,
            "latency_scope": "model.chat only; image preprocessing excluded",
            "median_sample_CER": statistics.median(sample_cers),
            "worst_sample_CER": max(sample_cers),
            "pages_CER_over_1": sum(1 for c in sample_cers if c > 1),
            "pages_hit_token_cap": sum(1 for r in predictions if r["hit_token_cap"]),
            "pages_refusal": sum(1 for r in predictions if r["looks_like_refusal"]),
            "latency_mean_sec": statistics.fmean(latencies),
            "latency_p50_sec": statistics.median(latencies),
            "latency_p90_sec": sorted(latencies)[int(0.9 * (len(latencies) - 1))],
            "preprocess_latency_mean_sec": statistics.fmean(
                r["preprocess_latency_sec"] for r in predictions
            ),
            "tokens_per_sec_mean": statistics.fmean(r["tokens_per_sec"] for r in predictions),
            "peak_gpu_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "total_runtime_sec": runtime,
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        }
    )

    with (args.output / "predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"COMPLETE {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
