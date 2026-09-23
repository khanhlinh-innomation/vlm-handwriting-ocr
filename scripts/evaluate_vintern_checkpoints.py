#!/usr/bin/env python3
"""Rank Vintern divorce LoRA checkpoints on the held-out validation split.

Checkpoints are saved with LoRA still wrapped: their tensors are named
`language_model.base_model.model...` and 336 of them are LoRA weights. The
`modeling_internvl_chat.py` published on the Hub does not implement
`use_llm_lora`, so loading a checkpoint through `AutoModel` + `trust_remote_code`
silently drops every trained weight and evaluates the base model instead. This
loads `InternVLChatModel` from the training repo, which honours the flag, and
asserts that nothing was dropped.

Selection uses validation CER only. The 12-page test split stays sealed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/vlm-handwriting")
DATA = ROOT / "data/internvl/divorce_ocr"
IMAGE_ROOT = ROOT / "data/llamafactory/glm_ocr_divorce"

IMAGE_SIZE = 448
MAX_TILES = 6  # matches --max_dynamic_patch 6 used in training
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Identical to the base-model benchmark so the comparison is like for like.
GENERATION = {
    "max_new_tokens": 2048,
    "do_sample": False,
    "num_beams": 1,
    "repetition_penalty": 1.1,
}

REFUSAL = re.compile(
    r"(tôi (không|ko) thể|không thể (đọc|chép|trích|nhận)|xin lỗi|i (can'?t|cannot))",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeatable, e.g. --checkpoint epoch3=/path/checkpoint-81",
    )
    parser.add_argument("--annotation", type=Path, default=DATA / "validation.jsonl")
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/vintern_1b_v3_5/divorce_validation_ranking",
    )
    return parser.parse_args()


def build_transform():
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

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
    """Dynamic tiling identical to the benchmark and to training preprocessing."""
    import torch
    from PIL import Image

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
    return torch.stack([transform(tile) for tile in tiles]), len(tiles)


def repeated_ngram_ratio(text: str, n: int = 8) -> float:
    """Share of n-grams that are duplicates; a looping output approaches 1."""
    tokens = text.split()
    if len(tokens) < n * 2:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


def load_model(path: Path):
    import torch
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
            f"Checkpoint {path} did not load cleanly: "
            f"{len(missing)} missing, {len(unexpected)} unexpected. "
            "The trained weights would be silently ignored."
        )
    return model.eval().cuda()


def evaluate(label: str, path: Path, rows: list[dict], image_root: Path, out_dir: Path) -> dict:
    import torch
    from transformers import AutoTokenizer

    from vlm_handwriting.metrics import (
        character_error_rate,
        corpus_metrics,
        normalize_for_evaluation,
        word_error_rate,
    )

    print(f"\n=== {label}: {path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, use_fast=False)
    model = load_model(path)
    transform = build_transform()

    # One unscored warm-up so the first timing is not a CUDA-init artefact.
    warm, _ = load_image(image_root / rows[0]["image"], transform)
    with torch.inference_mode():
        model.chat(tokenizer, warm.to(torch.bfloat16).cuda(), rows[0]["question"], GENERATION)
    del warm
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    text_dir = out_dir / label / "prediction_texts"
    text_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    for index, row in enumerate(rows, 1):
        pixels, tiles = load_image(image_root / row["image"], transform)
        pixels = pixels.to(torch.bfloat16).cuda()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            response = model.chat(tokenizer, pixels, row["question"], GENERATION)
        torch.cuda.synchronize()
        latency = time.perf_counter() - started

        prediction = normalize_for_evaluation(response)
        reference = normalize_for_evaluation(row["answer"])
        output_tokens = len(tokenizer.encode(response, add_special_tokens=False))
        cer = character_error_rate(reference, prediction)
        record = {
            "label": label,
            "image": row["image"],
            "reference": reference,
            "prediction": prediction,
            "sample_CER": cer,
            "sample_WER": word_error_rate(reference, prediction),
            "output_tokens": output_tokens,
            # The two known Vintern failure modes, which corpus CER alone hides.
            "hit_token_cap": output_tokens >= GENERATION["max_new_tokens"] - 8,
            "looks_like_refusal": bool(REFUSAL.search(prediction)),
            "repeated_8gram_ratio": round(repeated_ngram_ratio(prediction), 4),
            "length_ratio": round(len(prediction) / max(len(reference), 1), 4),
            "latency_sec": latency,
            "image_tiles": tiles,
        }
        predictions.append(record)
        (text_dir / (Path(row["image"]).stem + ".txt")).write_text(prediction, encoding="utf-8")
        print(
            f"  [{index}/{len(rows)}] CER={cer:.4f} tokens={output_tokens} "
            f"{latency:.2f}s {Path(row['image']).name}",
            flush=True,
        )

    metrics = corpus_metrics(
        [r["reference"] for r in predictions], [r["prediction"] for r in predictions]
    )
    sample_cers = [r["sample_CER"] for r in predictions]
    metrics.update(
        {
            "label": label,
            "checkpoint": str(path),
            "n": len(predictions),
            "median_sample_CER": statistics.median(sample_cers),
            "worst_sample_CER": max(sample_cers),
            "pages_CER_over_1": sum(1 for c in sample_cers if c > 1),
            "pages_hit_token_cap": sum(1 for r in predictions if r["hit_token_cap"]),
            "pages_refusal": sum(1 for r in predictions if r["looks_like_refusal"]),
            "latency_mean_sec": statistics.fmean(r["latency_sec"] for r in predictions),
            "peak_gpu_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "generation": GENERATION,
            "max_tiles": MAX_TILES,
        }
    )

    with (out_dir / label / "predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (out_dir / label / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    del model
    torch.cuda.empty_cache()
    return metrics


def main() -> int:
    sys.path.insert(0, str(ROOT / "repo/src"))

    args = parse_args()
    rows = []
    for line in args.annotation.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append(
            {
                "image": record["image"],
                # Reuse the exact training prompt so evaluation never asks
                # something the model was not trained on.
                "question": record["conversations"][0]["value"],
                "answer": record["conversations"][1]["value"],
            }
        )
    print(f"Validation rows: {len(rows)}")

    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for spec in args.checkpoint:
        if "=" not in spec:
            raise SystemExit(f"--checkpoint needs LABEL=PATH, got: {spec}")
        label, _, path = spec.partition("=")
        summary.append(evaluate(label, Path(path), rows, args.image_root, args.output))

    summary.sort(key=lambda metrics: metrics["cer"])
    (args.output / "ranking.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== RANKING (validation CER, lower is better) =====")
    for rank, m in enumerate(summary, 1):
        print(
            f"{rank}. {m['label']:10s} CER={m['cer']:.4f} WER={m['wer']:.4f} "
            f"median={m['median_sample_CER']:.4f} worst={m['worst_sample_CER']:.4f} "
            f"cap={m['pages_hit_token_cap']} refusal={m['pages_refusal']} "
            f"{m['latency_mean_sec']:.2f}s/page"
        )
    print(f"\nArtifacts: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
