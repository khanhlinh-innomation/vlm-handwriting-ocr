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
mkdir -p /workspace/data/{uit-hwdb,uit-hwdb-manifest}
mkdir -p /workspace/{checkpoints,outputs,logs}
cd /workspace/vlm-handwriting-ocr
uv pip install -e ".[data,dev]"
python scripts/verify_environment.py --output environment.json
```

Introduce model-specific dependencies one model at a time, beginning with GLM-OCR.
