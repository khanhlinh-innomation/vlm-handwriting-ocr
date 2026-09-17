"""TeleOCR supervised batching and language-only LoRA target guards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_handwriting.data import resolve_image_path

DEFAULT_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def is_language_lora_target(name: str, suffixes: tuple[str, ...]) -> bool:
    """Allow only known projection layers under the language model."""
    lowered = name.lower()
    return (
        "language_model" in lowered
        and "vision" not in lowered
        and "projector" not in lowered
        and name.rsplit(".", 1)[-1] in suffixes
    )


def discover_language_lora_targets(
    model: Any,
    *,
    torch: Any,
    suffixes: tuple[str, ...] = DEFAULT_TARGET_SUFFIXES,
) -> list[str]:
    """Discover exact language-model linear modules and reject an empty match."""
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and is_language_lora_target(name, suffixes)
    ]
    if not targets:
        linear_names = [
            name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
        ]
        raise RuntimeError(
            "No TeleOCR language-model LoRA targets found. "
            f"First linear modules: {linear_names[:80]}"
        )
    return targets


class ManifestRowDataset:
    """Minimal Trainer-compatible dataset over frozen manifest rows."""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


class TeleOCRCollator:
    """Create one native image/chat SFT batch and mask prompt tokens."""

    def __init__(
        self,
        *,
        processor: Any,
        raw_root: Path,
        system_prompt: str,
        prompt: str,
    ) -> None:
        self.processor = processor
        self.raw_root = Path(raw_root)
        self.system_prompt = system_prompt
        self.prompt = prompt

    def __call__(self, features: list[dict[str, str]]) -> Any:
        if len(features) != 1:
            raise ValueError("TeleOCR smoke collator requires micro-batch size 1")

        from PIL import Image

        row = features[0]
        image_path = resolve_image_path(self.raw_root, row["relative_path"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        prompt_messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            },
        ]
        full_messages = [*prompt_messages, {"role": "assistant", "content": row["text"]}]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_batch = self.processor(
            text=[prompt_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        full_batch = self.processor(
            text=[full_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        prompt_ids = prompt_batch["input_ids"]
        full_ids = full_batch["input_ids"]
        prompt_length = prompt_ids.shape[1]
        if full_ids.shape[1] <= prompt_length:
            raise RuntimeError("TeleOCR supervised sequence contains no assistant target tokens")
        if not (full_ids[:, :prompt_length] == prompt_ids).all().item():
            raise RuntimeError("TeleOCR prompt tokens are not a prefix of the supervised sequence")

        labels = full_ids.clone()
        labels[:, :prompt_length] = -100
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[full_ids == pad_id] = -100
        if not (labels != -100).any().item():
            raise RuntimeError("TeleOCR supervised batch has no unmasked target tokens")
        full_batch["labels"] = labels
        return full_batch
