#!/usr/bin/env python3
"""Let the InternVL trainer save only at chosen epochs.

InternVL saves the whole model at every checkpoint, not just the LoRA adapter:
roughly 2.0 GB per epoch against LLaMA-Factory's ~170 MB for the GLM runs. Ten
epoch checkpoints would need about 21 GB, and the instance has less free space
than that, so an unmodified ten-epoch run fills the disk near the end and loses
the whole thing.

`save_total_limit` cannot express this: it keeps the *last* N, and with only 220
divorce pages the best checkpoint may well be an early one. Staged runs with
`--resume_from_checkpoint` would work but rebuild the cosine schedule at each
stage, so the run would no longer match a single continuous schedule.

This adds a callback that suppresses saving at every epoch not listed in the
`SAVE_EPOCHS` environment variable (for example `SAVE_EPOCHS=3,7`). It is
registered after the default flow callback, so it can clear the save flag the
default callback just set. The final `save_model()` at the end of training is
untouched and still writes the last epoch to the output root.

Idempotent; refuses to touch a file it does not recognise; keeps a `.orig`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "# PATCHED: selective epoch saving"

# Injected between the Trainer construction and the training block, so `trainer`
# exists and `os`/`logger` are already in module scope.
TRAIN_ANCHOR = "    if training_args.do_train:\n"

TRAIN_INJECT = '''    _save_epochs = os.environ.get('SAVE_EPOCHS', '').strip()
    if _save_epochs:
        from transformers import TrainerCallback

        _wanted = {int(part) for part in _save_epochs.split(',') if part.strip()}
        logger.info(f'Selective epoch saving enabled for epochs: {sorted(_wanted)}')

        class _SelectiveEpochSave(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                if round(state.epoch) not in _wanted:
                    control.should_save = False
                return control

        trainer.add_callback(_SelectiveEpochSave())

'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finetune-script",
        type=Path,
        default=Path(
            "/workspace/vlm-handwriting/tools/Vintern/internvl_chat/"
            "internvl/train/internvl_chat_finetune.py"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.finetune_script
    if not path.is_file():
        print(f"Missing file: {path}", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"Already patched: {path}")
        return 0
    if text.count(TRAIN_ANCHOR) != 1:
        print(
            f"Expected exactly one 'if training_args.do_train:' block, "
            f"found {text.count(TRAIN_ANCHOR)}; refusing to patch {path}",
            file=sys.stderr,
        )
        return 1

    backup = path.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    patched = text.replace(TRAIN_ANCHOR, f"{MARKER}\n{TRAIN_INJECT}{TRAIN_ANCHOR}", 1)
    path.write_text(patched, encoding="utf-8")
    print(f"Patched: {path}")
    print(f"Original kept at: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
