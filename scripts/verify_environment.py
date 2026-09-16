#!/usr/bin/env python3
"""Print and optionally save reproducibility metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm_handwriting.artifacts import write_json
from vlm_handwriting.environment import collect_environment, format_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    report = collect_environment()
    print(format_environment(report))
    if args.output:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
