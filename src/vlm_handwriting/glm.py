"""Shared GLM-OCR model loading and single-image inference."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from vlm_handwriting.metrics import normalize_for_evaluation


def load_glm_base(config: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    """Load the configured base model using BF16 where supported."""
    import torch
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("GLM-OCR requires a CUDA GPU")

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
    return torch, processor, model, device, dtype


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
    """Generate an OCR transcription and return it with synchronized latency."""
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
