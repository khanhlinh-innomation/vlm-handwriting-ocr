# TeleOCR workflow

This run follows the same frozen data and evaluation protocol as GLM-OCR. TeleOCR
uses its official native processor, system prompt, OCR prompt, and remote model code.
The downstream LoRA recipe remains experimental, so full training is gated behind
base inference and save/reload smoke checks.

## Fixed inputs

- Model: `StarDoc-AI/TeleOCR`
- System prompt: `You are a helpful assistant.`
- OCR prompt: `Please output the text content from the image.`
- Deterministic generation: `do_sample=false`, `max_new_tokens=256`, `repetition_penalty=1.1`
- Train / validation / test: 6,346 / 682 / 201 writer-disjoint lines
- Smoke set: 20 deterministic validation rows, seed 42
- Evaluation normalization: Unicode NFC only
- Primary metric: corpus CER

## Gate 1: isolated compatible runtime

TeleOCR's official repository pins `transformers==4.57.1`. Do not run the remote
model code with Transformers 5.x: its Qwen2.5-VL RoPE implementation expects the
4.x `ROPE_INIT_FUNCTIONS["default"]` API. Keep `/venv/main` unchanged and create a
model-specific environment that reuses the verified CUDA-enabled Torch packages.

```bash
cd /workspace/vlm-handwriting/repo
git pull --ff-only

/venv/main/bin/python -c \
  'import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)'

mkdir -p /workspace/vlm-handwriting/venvs

uv venv \
  --python /venv/main/bin/python \
  --system-site-packages \
  /workspace/vlm-handwriting/venvs/teleocr

uv pip install \
  --python /workspace/vlm-handwriting/venvs/teleocr/bin/python \
  -r requirements/teleocr.txt

/workspace/vlm-handwriting/venvs/teleocr/bin/python -c \
  'import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__); assert torch.__version__ == "2.10.0+cu128"; assert transformers.__version__ == "4.57.1"'
```

Expected version pair in this environment:

```text
torch        2.10.0+cu128
transformers 4.57.1
```

Stop if the verified Torch build changes. The GLM environment remains `/venv/main`
and is not modified by this setup.

## Gate 2: one fixed validation image

```bash
cd /workspace/vlm-handwriting/repo
export HF_HOME=/workspace/vlm-handwriting/cache/huggingface
mkdir -p /workspace/vlm-handwriting/{outputs,logs,cache}
set -o pipefail

/workspace/vlm-handwriting/venvs/teleocr/bin/python scripts/benchmark_teleocr.py \
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
/workspace/vlm-handwriting/venvs/teleocr/bin/python scripts/benchmark_teleocr.py \
  --manifest-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/hwdb-manifest/versions/1/uit_hwdb_line_ready \
  --raw-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/uit-hwdb/versions/1/UIT_HWDB_line/UIT_HWDB_line \
  --output-root /workspace/vlm-handwriting/outputs \
  --mode smoke \
  2>&1 | tee /workspace/vlm-handwriting/logs/teleocr-base-smoke20.log
```

Do not tune from the test set. If smoke20 reveals a prompt or generation failure,
fix it and repeat validation smoke before opening test.

The first smoke20 attempt used `max_new_tokens=512` and exposed a repetition
collapse on `28.jpg`: 57 reference characters versus 537 predicted characters,
CER 9.2281, and 13.702 seconds latency. The validation-only correction reduces
the cap to 256 and adds repetition penalty 1.1. The exact smoke20 set must be run
again and pass before test access.

## Gate 4: frozen base test

After freezing the native prompt and generation config, run the base test once:

```bash
/workspace/vlm-handwriting/venvs/teleocr/bin/python scripts/benchmark_teleocr.py \
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

### Compatibility and two-step LoRA smoke

Reinstall the TeleOCR requirements after pulling so the isolated environment also
contains the pinned PEFT runtime, then launch the guarded smoke script:

```bash
cd /workspace/vlm-handwriting/repo
git pull --ff-only

uv pip install \
  --python /workspace/vlm-handwriting/venvs/teleocr/bin/python \
  -r requirements/teleocr.txt

set -o pipefail
/workspace/vlm-handwriting/venvs/teleocr/bin/python \
  scripts/train_teleocr_smoke.py \
  --manifest-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/hwdb-manifest/versions/1/uit_hwdb_line_ready \
  --raw-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/uit-hwdb/versions/1/UIT_HWDB_line/UIT_HWDB_line \
  2>&1 | tee /workspace/vlm-handwriting/logs/teleocr-lora-smoke.log
```

This command stops before optimization unless the native model returns a finite
supervised loss and all trainable parameters are LoRA parameters under the language
model. It then runs exactly two optimizer steps over deterministic 32/8 train and
validation subsets and saves the adapter under:

```text
/workspace/vlm-handwriting/checkpoints/teleocr/lora_smoke
```

Reload the saved adapter in a fresh process:

```bash
/workspace/vlm-handwriting/venvs/teleocr/bin/python \
  scripts/verify_teleocr_adapter.py \
  --adapter-path /workspace/vlm-handwriting/checkpoints/teleocr/lora_smoke \
  --manifest-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/hwdb-manifest/versions/1/uit_hwdb_line_ready \
  --raw-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/uit-hwdb/versions/1/UIT_HWDB_line/UIT_HWDB_line \
  --output-dir /workspace/vlm-handwriting/outputs/teleocr/lora_smoke_verify \
  2>&1 | tee /workspace/vlm-handwriting/logs/teleocr-lora-smoke-verify.log
```

Do not launch full training until both commands pass.

Generated artifacts remain outside Git under:

```text
/workspace/vlm-handwriting/outputs/teleocr/
/workspace/vlm-handwriting/checkpoints/teleocr/
/workspace/vlm-handwriting/logs/teleocr-*.log
```
