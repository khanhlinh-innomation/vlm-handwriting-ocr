# Vast Server Setup

## Confirmed environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB |
| Compute capability | 12.0 (`sm_120`) |
| Driver | 570.169 |
| PyTorch | 2.10.0+cu128 |
| Torchvision | 0.25.0+cu128 |
| Python | 3.12.14, `/venv/main/bin/python` |
| CUDA test | BF16 matrix multiplication passed |
| System RAM | 251 GiB |
| Initial free disk | approximately 140 GiB |

GPU telemetry remained at 100 percent with zero allocated VRAM and no process reported by `nvidia-smi pmon`. Treat that field as unreliable and measure real job throughput.

## Rules

- Read `/workspace/AGENTS.md` before changing the server environment.
- Use the existing `main` environment and `uv pip install` unless a verified conflict requires isolation.
- Do not implicitly replace the working CUDA-enabled Torch build.
- Avoid optional CUDA extension compilation during the first GLM smoke run.
- Never commit or report Kaggle, Hugging Face, SSH, Jupyter, Syncthing, or Vast credentials.

## Bootstrap

```bash
mkdir -p /workspace/vlm-handwriting/{data,checkpoints,outputs,logs,cache}
cd /workspace/vlm-handwriting/repo
uv pip install --python /venv/main/bin/python -e ".[data,dev]"
/venv/main/bin/python scripts/verify_environment.py \
  --output /workspace/vlm-handwriting/logs/environment.json
```

## GLM-OCR base gate

Install model-specific dependencies and verify that Torch stays on the confirmed CUDA build:

```bash
uv pip install --python /venv/main/bin/python -r requirements/glm.txt
/venv/main/bin/python -c 'import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)'
export HF_HOME=/workspace/vlm-handwriting/cache/huggingface
```

Run the first fixed validation image:

```bash
/venv/main/bin/python scripts/benchmark_glm.py \
  --manifest-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/hwdb-manifest/versions/1/uit_hwdb_line_ready \
  --raw-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/uit-hwdb/versions/1/UIT_HWDB_line/UIT_HWDB_line \
  --output-root /workspace/vlm-handwriting/outputs \
  --mode one \
  2>&1 | tee /workspace/vlm-handwriting/logs/glm-base-one.log
```

Review the one-image artifact before running `--mode smoke`. Only use `--mode validation --allow-full-validation` after smoke20 review. Only use `--mode test --allow-test` after the inference configuration is frozen.
