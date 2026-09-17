#!/usr/bin/env python3
"""Run TeleOCR finite-loss, language-only LoRA, and two-step smoke gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from vlm_handwriting.artifacts import write_json
from vlm_handwriting.data import find_manifest_root, load_manifest
from vlm_handwriting.environment import collect_environment
from vlm_handwriting.smoke import select_smoke_rows
from vlm_handwriting.teleocr_training import (
    ManifestRowDataset,
    TeleOCRCollator,
    discover_language_lora_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TeleOCR supervised-loss and two-step LoRA smoke training"
    )
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/teleocr/lora_smoke.yaml")
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    for section in ("model", "data", "lora", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing config section: {section}")
    return config


def require_new_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty TeleOCR smoke output: {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def move_batch(batch: Any, *, torch: Any, device: Any, dtype: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    output_dir = require_new_output_dir(Path(str(training["output_dir"])))
    manifest_root = find_manifest_root(args.manifest_root)
    train_rows = load_manifest(manifest_root / "train.csv")
    validation_rows = load_manifest(manifest_root / "val.csv")
    seed = int(config["data"]["seed"])
    smoke_train = select_smoke_rows(
        train_rows, size=int(config["data"]["train_size"]), seed=seed
    )
    smoke_validation = select_smoke_rows(
        validation_rows, size=int(config["data"]["validation_size"]), seed=seed
    )

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoProcessor, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError("TeleOCR smoke training requires CUDA")
    torch.manual_seed(seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_id = str(config["model"]["id"])
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )
    base_model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=dtype,
    )
    base_model.config.use_cache = False
    if bool(training["gradient_checkpointing"]):
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()

    suffixes = tuple(str(value) for value in config["lora"]["target_suffixes"])
    target_names = discover_language_lora_targets(
        base_model,
        torch=torch,
        suffixes=suffixes,
    )
    collator = TeleOCRCollator(
        processor=processor,
        raw_root=args.raw_root,
        system_prompt=str(config["model"]["system_prompt"]),
        prompt=str(config["model"]["prompt"]),
    )

    device = torch.device("cuda:0")
    base_model.to(device)
    preflight_batch = move_batch(
        collator([smoke_train[0]]),
        torch=torch,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        preflight_output = base_model(**preflight_batch)
    loss = getattr(preflight_output, "loss", None)
    if loss is None or not torch.isfinite(loss).item():
        raise RuntimeError(f"TeleOCR supervised-loss gate failed: {loss}")
    preflight_loss = float(loss.detach().cpu())
    del preflight_batch, preflight_output
    torch.cuda.empty_cache()

    lora_config = LoraConfig(
        r=int(config["lora"]["rank"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        bias="none",
        target_modules=target_names,
    )
    model = get_peft_model(base_model, lora_config)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or not all("lora_" in name.lower() for name in trainable_names):
        raise RuntimeError(f"Unexpected trainable TeleOCR parameters: {trainable_names[:50]}")
    if any("vision" in name.lower() or "projector" in name.lower() for name in trainable_names):
        raise RuntimeError("TeleOCR vision/projector parameter unexpectedly became trainable")
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    write_json(
        output_dir / "preflight.json",
        {
            "model_id": model_id,
            "supervised_loss": preflight_loss,
            "target_module_count": len(target_names),
            "target_modules": target_names,
            "trainable_parameter_count": trainable_parameters,
            "total_parameter_count": total_parameters,
            "trainable_fraction": trainable_parameters / total_parameters,
            "train_rows": len(smoke_train),
            "validation_rows": len(smoke_validation),
            "dtype": str(dtype),
            "gpu_name": torch.cuda.get_device_name(0),
        },
    )
    write_json(output_dir / "environment.json", collect_environment())
    write_json(output_dir / "resolved_config.json", config)
    print(
        json.dumps(
            {
                "supervised_loss": preflight_loss,
                "target_module_count": len(target_names),
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
            },
            indent=2,
        ),
        flush=True,
    )

    trainer_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=int(training["max_steps"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=str(training["scheduler"]),
        warmup_ratio=float(training["warmup_ratio"]),
        bf16=bool(training["bf16"]),
        fp16=not bool(training["bf16"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        logging_steps=int(training["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(training["eval_steps"]),
        save_strategy="steps",
        save_steps=int(training["save_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        save_only_model=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        seed=int(training["seed"]),
        overwrite_output_dir=bool(training["overwrite_output_dir"]),
        skip_memory_metrics=False,
        label_names=["labels"],
    )
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=ManifestRowDataset(smoke_train),
        eval_dataset=ManifestRowDataset(smoke_validation),
        data_collator=collator,
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", eval_metrics)
    trainer.save_state()
    print(json.dumps({"train": train_result.metrics, "eval": eval_metrics}, indent=2))
    print(f"TeleOCR LoRA smoke saved: {output_dir}")


if __name__ == "__main__":
    main()
