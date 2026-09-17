"""PaddleOCR-VL element-recognition loading and inference helpers."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from vlm_handwriting.metrics import normalize_for_evaluation


def load_paddle_vl_base(
    config: dict[str, Any],
) -> tuple[Any, Any, Any, Any, Any]:
    """Load PaddleOCR-VL through its official Transformers element API."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("PaddleOCR-VL requires a CUDA GPU")

    seed = int(config["evaluation"]["smoke_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = torch.device("cuda:0")
    model_id = str(config["model"]["id"])
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device).eval()
    return torch, processor, model, device, dtype


def _move_inputs(inputs: Any, *, torch: Any, device: Any) -> dict[str, Any]:
    """Move processor tensors to CUDA without changing their native dtypes."""
    return {
        key: value.to(device=device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


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
    """Generate one line transcription using the official OCR prompt."""
    from PIL import Image

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": processor.image_processor.min_pixels,
                "longest_edge": 1280 * 28 * 28,
            }
        },
    )
    moved = _move_inputs(inputs, torch=torch, device=device)

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **moved,
            max_new_tokens=int(generation["max_new_tokens"]),
            do_sample=bool(generation["do_sample"]),
            repetition_penalty=float(generation["repetition_penalty"]),
            use_cache=bool(generation["use_cache"]),
        )
    torch.cuda.synchronize()
    latency = time.perf_counter() - started

    prompt_tokens = moved["input_ids"].shape[-1]
    output_ids = outputs[0][prompt_tokens:].detach().cpu()
    prediction = processor.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return normalize_for_evaluation(prediction), latency
