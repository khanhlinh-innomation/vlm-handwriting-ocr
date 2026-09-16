"""Collect reproducibility metadata without requiring model dependencies."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "huggingface-hub",
    "PyYAML",
    "sentencepiece",
    "peft",
    "kagglehub",
    "paddlepaddle",
    "paddlepaddle-gpu",
)


def _package_versions() -> dict[str, str | None]:
    result = {}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _nvidia_smi() -> dict[str, Any] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return {"query": command[1], "rows": completed.stdout.splitlines()}


def collect_environment() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "nvidia_smi": _nvidia_smi(),
    }
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        report["torch_cuda"] = {
            "build": torch.version.cuda,
            "available": cuda_available,
            "architectures": torch.cuda.get_arch_list() if cuda_available else [],
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "compute_capability": list(torch.cuda.get_device_capability(0))
            if cuda_available
            else None,
            "bf16_supported": torch.cuda.is_bf16_supported() if cuda_available else False,
        }
    except ImportError:
        report["torch_cuda"] = None
    return report


def format_environment(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
