# GLM-OCR LoRA Training

The active training path follows the official GLM-OCR LLaMA-Factory recipe: multimodal
ShareGPT data, template `glm_ocr`, prompt `<image>Text Recognition:`, LoRA rank 8,
target `all`, learning rate `1e-4`, cosine schedule, and BF16.

Upstream references: [GLM-OCR fine-tuning guide](https://github.com/zai-org/GLM-OCR/blob/main/examples/finetune/README.md)
and [LLaMA-Factory v0.9.5](https://github.com/hiyouga/LLaMA-Factory/releases/tag/v0.9.5).

Training uses a separate virtual environment. LLaMA-Factory v0.9.5 has a narrower
Transformers/Accelerate compatibility range than the verified base-inference environment;
do not install it into `/venv/main`.

## 1. Create the isolated training environment

```bash
cd /workspace/vlm-handwriting/repo
mkdir -p /workspace/vlm-handwriting/{tools,venvs}

git clone --depth 1 --branch v0.9.5 \
  https://github.com/hiyouga/LlamaFactory.git \
  /workspace/vlm-handwriting/tools/LlamaFactory

uv venv \
  --python /venv/main/bin/python \
  --system-site-packages \
  /workspace/vlm-handwriting/venvs/glm-train

uv pip install \
  --python /workspace/vlm-handwriting/venvs/glm-train/bin/python \
  --index-strategy unsafe-first-match \
  -c requirements/glm-training-constraints.txt \
  -e /workspace/vlm-handwriting/tools/LlamaFactory

uv pip install \
  --python /workspace/vlm-handwriting/venvs/glm-train/bin/python \
  --no-deps -e .
```

The explicit index strategy is required because the official CUDA wheel index contains
`torchdata`, but not the pinned `torchdata==0.11.0`. `unsafe-first-match` keeps the CUDA
index preferred, then falls back to PyPI only when that index has no compatible version.
The direct training packages remain pinned by the constraints file.

Verify the isolated environment and the untouched inference environment:

```bash
/workspace/vlm-handwriting/venvs/glm-train/bin/python -c \
  'import torch, transformers, accelerate, peft; print(torch.__version__, transformers.__version__, accelerate.__version__, peft.__version__); assert torch.__version__ == "2.10.0+cu128"'

/workspace/vlm-handwriting/venvs/glm-train/bin/llamafactory-cli version

/venv/main/bin/python -c \
  'import torch, transformers; print(torch.__version__, transformers.__version__)'
```

## 2. Prepare the frozen training data

```bash
/workspace/vlm-handwriting/venvs/glm-train/bin/python \
  scripts/prepare_glm_training.py \
  --manifest-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/hwdb-manifest/versions/1/uit_hwdb_line_ready \
  --raw-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/uit-hwdb/versions/1/UIT_HWDB_line/UIT_HWDB_line \
  --output-dir /workspace/vlm-handwriting/data/llamafactory/glm_ocr
```

This creates the full 6,346/682 training and validation files plus deterministic
32-train/8-validation smoke files. Images remain in the raw dataset and are exposed through
a symlink; they are not copied into Git.

## 3. Run the two-step training smoke

Run inside `tmux`:

```bash
export HF_HOME=/workspace/vlm-handwriting/cache/huggingface
export DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=0
set -o pipefail

/workspace/vlm-handwriting/venvs/glm-train/bin/llamafactory-cli train \
  /workspace/vlm-handwriting/repo/configs/glm/lora_smoke.yaml \
  2>&1 | tee /workspace/vlm-handwriting/logs/glm-lora-smoke.log
```

Do not start the full configuration yet. The smoke output must contain LoRA adapter files and
must pass reload inference:

```bash
/workspace/vlm-handwriting/venvs/glm-train/bin/python \
  scripts/verify_glm_adapter.py \
  --adapter-path /workspace/vlm-handwriting/checkpoints/glm_ocr/lora_smoke \
  --manifest-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/hwdb-manifest/versions/1/uit_hwdb_line_ready \
  --raw-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/uit-hwdb/versions/1/UIT_HWDB_line/UIT_HWDB_line \
  --output-dir /workspace/vlm-handwriting/outputs/glm_ocr/lora_smoke_verify
```

## 4. Full training gate

`configs/glm/lora.yaml` is the three-epoch full configuration. It remains gated until the
smoke run saves an adapter and the separate reload process completes inference. Epoch
checkpoints are later ranked by full-validation CER, not training or validation loss alone.
