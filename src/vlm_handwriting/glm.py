"""Shared GLM-OCR model loading and single-image inference."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from vlm_handwriting.metrics import normalize_for_evaluation


def validate_adapter_path(adapter_path: Path) -> Path:
    """Resolve an adapter directory and require its PEFT config and weights."""
    resolved = Path(adapter_path).expanduser().resolve()
    if not (resolved / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter_config.json under {resolved}")
    if not any(resolved.glob("adapter_model.*")):
        raise FileNotFoundError(f"Missing adapter weights under {resolved}")
    return resolved


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


def load_glm_adapter(
    config: dict[str, Any], adapter_path: Path
) -> tuple[Any, Any, Any, Any, Any]:
    """Load the configured base model and attach a saved PEFT adapter."""
    adapter_path = validate_adapter_path(adapter_path)
    torch, processor, base_model, device, dtype = load_glm_base(config)
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False).eval()
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
    """Generate an OCR transcription and return it with synchronized latency.

    When ``fallback_no_repeat_ngram_size`` is configured, a generation that reaches
    ``max_new_tokens`` without EOS is retried once with that no-repeat constraint.
    The returned latency includes both attempts.
    """
    prediction, latency, _ = predict_one_detailed(
        torch=torch,
        processor=processor,
        model=model,
        device=device,
        image_path=image_path,
        prompt=prompt,
        generation=generation,
    )
    return prediction, latency


def predict_one_detailed(
    *,
    torch: Any,
    processor: Any,
    model: Any,
    device: Any,
    image_path: Path,
    prompt: str,
    generation: dict[str, Any],
) -> tuple[str, float, dict[str, Any]]:
    """Generate OCR text and expose stopping/fallback telemetry."""
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

    prompt_tokens = inputs["input_ids"].shape[-1]
    max_new_tokens = int(generation["max_new_tokens"])
    eos_token_ids = model.generation_config.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = []
    elif isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    else:
        eos_token_ids = list(eos_token_ids)

    def generate_once(no_repeat_ngram_size: int) -> tuple[str, float, dict[str, Any]]:
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(generation["do_sample"]),
            "repetition_penalty": float(generation["repetition_penalty"]),
        }
        if no_repeat_ngram_size > 0:
            kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, **kwargs)
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        generated = output[0][prompt_tokens:]
        generated_tokens = int(generated.shape[-1])
        last_token = int(generated[-1].item()) if generated_tokens else None
        ended_with_eos = last_token in eos_token_ids
        hit_max_new_tokens = generated_tokens >= max_new_tokens and not ended_with_eos
        prediction = processor.decode(generated, skip_special_tokens=True).strip()
        metadata = {
            "generated_tokens": generated_tokens,
            "ended_with_eos": ended_with_eos,
            "hit_max_new_tokens": hit_max_new_tokens,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        }
        return normalize_for_evaluation(prediction), latency, metadata

    primary_no_repeat = int(generation.get("no_repeat_ngram_size", 0))
    prediction, latency, primary = generate_once(primary_no_repeat)
    fallback_size = int(generation.get("fallback_no_repeat_ngram_size", 0))
    metadata: dict[str, Any] = {
        "fallback_used": False,
        "primary": primary,
        "selected": primary,
    }
    if (
        primary["hit_max_new_tokens"]
        and fallback_size > 0
        and fallback_size != primary_no_repeat
    ):
        fallback_prediction, fallback_latency, fallback = generate_once(fallback_size)
        prediction = fallback_prediction
        latency += fallback_latency
        metadata.update(
            {
                "fallback_used": True,
                "fallback": fallback,
                "selected": fallback,
            }
        )
    metadata["total_latency_sec"] = latency
    return prediction, latency, metadata
