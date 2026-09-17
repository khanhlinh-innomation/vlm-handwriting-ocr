"""Shared TeleOCR model loading and single-image inference."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from vlm_handwriting.metrics import normalize_for_evaluation


def load_teleocr_base(config: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    """Load TeleOCR with its official remote model code and native processor."""
    import torch
    from transformers import AutoModel, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("TeleOCR requires a CUDA GPU")

    seed = int(config["evaluation"]["smoke_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_id = str(config["model"]["id"])
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto",
    ).eval()
    device = next(model.parameters()).device
    return torch, processor, model, device, dtype


def _move_inputs(inputs: Any, *, torch: Any, device: Any, dtype: Any) -> dict[str, Any]:
    """Move integer and floating tensors without casting token IDs to BF16."""
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def predict_one(
    *,
    torch: Any,
    processor: Any,
    model: Any,
    device: Any,
    dtype: Any,
    image_path: Path,
    system_prompt: str,
    prompt: str,
    generation: dict[str, Any],
) -> tuple[str, float]:
    """Generate one OCR transcription with synchronized CUDA latency."""
    from PIL import Image

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    chat_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[chat_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    moved = _move_inputs(inputs, torch=torch, device=device, dtype=dtype)

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **moved,
            use_cache=bool(generation["use_cache"]),
            max_new_tokens=int(generation["max_new_tokens"]),
            do_sample=bool(generation["do_sample"]),
        )
    torch.cuda.synchronize()
    latency = time.perf_counter() - started
    prompt_tokens = moved["input_ids"].shape[-1]
    output_ids = output[0][prompt_tokens:].detach().cpu().tolist()
    prediction = processor.batch_decode(
        [output_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return normalize_for_evaluation(prediction), latency
