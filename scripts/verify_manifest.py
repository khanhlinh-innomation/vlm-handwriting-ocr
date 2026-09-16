#!/usr/bin/env python3
"""Verify the frozen manifest contract before running a model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vlm_handwriting.data import validate_frozen_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=os.getenv("VLM_MANIFEST_ROOT"),
        required=os.getenv("VLM_MANIFEST_ROOT") is None,
    )
    parser.add_argument("--raw-root", type=Path, default=os.getenv("VLM_RAW_ROOT"))
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()
    report = validate_frozen_manifests(
        args.manifest_root,
        raw_root=args.raw_root,
        check_images=args.check_images,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
