# Repository Instructions

This repository implements reproducible Vietnamese handwritten line OCR experiments.

## Canonical rules

- Treat `docs/PROJECT_CONTEXT.md` as the current project context.
- Treat `docs/archive/` and `notebooks/legacy_colab/` as historical inputs.
- Use only the frozen UIT-HWDB-line split: train 6,346, validation 682, test 201.
- Never regenerate the split, mix word/paragraph samples into round 1, or guess labels.
- Never use the test set for prompt, checkpoint, or hyperparameter selection.
- Normalize OCR evaluation text with Unicode NFC only. Preserve case, punctuation, spacing, and Vietnamese diacritics.
- Select checkpoints by lowest validation CER.
- Require save, reload, and inference verification before declaring training successful.

## Engineering rules

- Keep raw data, credentials, caches, checkpoints, outputs, and logs outside Git.
- Resolve images from manifest `relative_path`; reject absolute paths and path traversal.
- Keep native model processors, prompts, and templates unless evidence supports a change.
- Run deterministic smoke tests before full validation or training.
- Record environment metadata, peak VRAM, latency, throughput, and versioned output paths.
- Do not silently replace the server's working `torch==2.10.0+cu128` build.
- Install server packages with `uv pip install` after checking the active environment.
- Write project documentation and experiment artifacts in English unless requested otherwise.

## Validation

```bash
python -m pytest
python scripts/verify_environment.py
python scripts/verify_manifest.py --manifest-root "$VLM_MANIFEST_ROOT"
```
