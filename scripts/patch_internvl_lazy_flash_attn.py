#!/usr/bin/env python3
"""Make InternVL's LLaMA FlashAttention monkey patches import lazily.

`internvl/patch/__init__.py` eagerly imports two LLaMA-specific FlashAttention
monkey patches:

    from .llama2_flash_attn_monkey_patch import replace_llama2_attn_with_flash_attn
    from .llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

Both do `from flash_attn import ...` at module level, so importing anything from
`internvl.patch` fails outright when FlashAttention is absent. The fine-tuning
entrypoint only needs `concat_pad_data_collator`,
`replace_llama_rmsnorm_with_fused_rmsnorm` and `replace_train_sampler`, and
never calls either LLaMA patch. Vintern's language tower is Qwen2, so the LLaMA
patches are inapplicable regardless.

There is no prebuilt FlashAttention wheel for torch 2.10 on x86_64 (only an
aarch64/cu13 build) and the instance has no `nvcc`, so a source build is not
available either.

This rewrites the two eager imports as a PEP 562 module `__getattr__`. The
public API is unchanged: both names still resolve on attribute access and still
raise ImportError there if FlashAttention is genuinely needed. The script is
idempotent and refuses to touch a file it does not recognise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "# PATCHED: lazy FlashAttention imports"

EAGER = (
    "from .llama2_flash_attn_monkey_patch import replace_llama2_attn_with_flash_attn\n"
    "from .llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn\n"
)

LAZY = f"""{MARKER}
# The two LLaMA FlashAttention patches below are imported on attribute access
# instead of at module import time. See scripts/patch_internvl_lazy_flash_attn.py.
_LAZY_ATTRS = {{
    'replace_llama2_attn_with_flash_attn': '.llama2_flash_attn_monkey_patch',
    'replace_llama_attn_with_flash_attn': '.llama_flash_attn_monkey_patch',
}}


def __getattr__(name):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f'module {{__name__!r}} has no attribute {{name!r}}')
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)

"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patch-init",
        type=Path,
        default=Path(
            "/workspace/vlm-handwriting/tools/Vintern/internvl_chat/internvl/patch/__init__.py"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.patch_init
    if not path.is_file():
        print(f"Missing file: {path}", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"Already patched: {path}")
        return 0
    if EAGER not in text:
        print(f"Unexpected contents, refusing to patch: {path}", file=sys.stderr)
        return 1

    backup = path.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    path.write_text(text.replace(EAGER, LAZY), encoding="utf-8")
    print(f"Patched: {path}")
    print(f"Original kept at: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
