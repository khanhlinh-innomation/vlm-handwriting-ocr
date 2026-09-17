# TeleOCR workflow

This run follows the same frozen data and evaluation protocol as GLM-OCR. TeleOCR
uses its official native processor, system prompt, OCR prompt, and remote model code.
The downstream LoRA recipe remains experimental, so full training is gated behind
base inference and save/reload smoke checks.

## Fixed inputs

- Model: `StarDoc-AI/TeleOCR`
- System prompt: `You are a helpful assistant.`
- OCR prompt: `Please output the text content from the image.`
- Train / validation / test: 6,346 / 682 / 201 writer-disjoint lines
- Smoke set: 20 deterministic validation rows, seed 42
- Evaluation normalization: Unicode NFC only
- Primary metric: corpus CER

## Gate 1: install without replacing Torch

```bash
cd /workspace/vlm-handwriting/repo
git pull --ff-only

/venv/main/bin/python -c \
  'import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)'

uv pip install \
  --python /venv/main/bin/python \
  -r requirements/teleocr.txt

/venv/main/bin/python -c \
  'import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__); assert torch.__version__ == "2.10.0+cu128"'
```

Stop if the verified `torch==2.10.0+cu128` build changes.

## Gate 2: one fixed validation image

```bash
cd /workspace/vlm-handwriting/repo
export HF_HOME=/workspace/vlm-handwriting/cache/huggingface
mkdir -p /workspace/vlm-handwriting/{outputs,logs,cache}
set -o pipefail

/venv/main/bin/python scripts/benchmark_teleocr.py \
  --manifest-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/hwdb-manifest/versions/1/uit_hwdb_line_ready \
  --raw-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/uit-hwdb/versions/1/UIT_HWDB_line/UIT_HWDB_line \
  --output-root /workspace/vlm-handwriting/outputs \
  --mode one \
  2>&1 | tee /workspace/vlm-handwriting/logs/teleocr-base-one.log
```

Review the prediction artifact before proceeding. The gate passes when model and
processor loading, image input, Vietnamese output, latency reporting, and artifact
writing all succeed without OOM or repetition collapse.

## Gate 3: fixed validation smoke20

Run only after the one-image output has been reviewed:

```bash
/venv/main/bin/python scripts/benchmark_teleocr.py \
  --manifest-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/hwdb-manifest/versions/1/uit_hwdb_line_ready \
  --raw-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/uit-hwdb/versions/1/UIT_HWDB_line/UIT_HWDB_line \
  --output-root /workspace/vlm-handwriting/outputs \
  --mode smoke \
  2>&1 | tee /workspace/vlm-handwriting/logs/teleocr-base-smoke20.log
```

Do not tune from the test set. If smoke20 reveals a prompt or generation failure,
fix it and repeat validation smoke before opening test.

## Gate 4: frozen base test

After freezing the native prompt and generation config, run the base test once:

```bash
/venv/main/bin/python scripts/benchmark_teleocr.py \
  --manifest-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/hwdb-manifest/versions/1/uit_hwdb_line_ready \
  --raw-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/uit-hwdb/versions/1/UIT_HWDB_line/UIT_HWDB_line \
  --output-root /workspace/vlm-handwriting/outputs \
  --mode test \
  --allow-test \
  2>&1 | tee /workspace/vlm-handwriting/logs/teleocr-base-test.log
```

The explicit `--allow-test` flag prevents accidental test access.

## Training gates

Do not launch full TeleOCR training immediately after the base test. Implement and
verify these gates in an isolated environment first:

1. Create the supervised image/text batch with the native chat template.
2. Confirm a finite supervised loss for one batch.
3. Discover language-model LoRA targets and prove that vision/projector weights stay frozen.
4. Train a deterministic small subset.
5. Save the adapter, reload it in a fresh process, and run one validation inference.
6. Only then run the full train with epoch checkpoints.
7. Rank every checkpoint on all 682 validation rows by CER.
8. Freeze exactly one checkpoint and evaluate the 201-row test once.

Generated artifacts remain outside Git under:

```text
/workspace/vlm-handwriting/outputs/teleocr/
/workspace/vlm-handwriting/checkpoints/teleocr/
/workspace/vlm-handwriting/logs/teleocr-*.log
```
