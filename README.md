# Vietnamese Handwriting VLM OCR

Reproducible benchmarking and fine-tuning for Vietnamese handwritten line recognition with GLM-OCR, TeleOCR, and PaddleOCR-VL-1.6.

The project uses a frozen, writer-disjoint UIT-HWDB-line split and strict OCR evaluation. Unicode text is normalized to NFC only; case, punctuation, spaces, and Vietnamese diacritics are preserved.

## Frozen data contract

| Split | Samples | Writers |
|---|---:|---:|
| Train | 6,346 | 224 |
| Validation | 682 | 25 |
| Test | 201 | 6 |

- Raw Kaggle dataset: `ntklinhfitus/uit-hwdb`
- Manifest Kaggle dataset: `ntklinhfitus/hwdb-manifest`
- Fixed smoke benchmark: 20 validation rows, seed 42
- Primary metric: corpus character error rate (CER)
- Best checkpoint: lowest validation CER

Do not regenerate the split or tune against the test set.

## Repository status

- The frozen data contract has been verified against all 7,229 images on the Vast server.
- The GLM-OCR pipeline is complete: frozen base test, LoRA smoke, three-epoch full train,
  validation-CER checkpoint selection, and one frozen fine-tuned test.
- `checkpoint-1191` reduced test CER from 31.95% to 12.46% and WER from 77.41% to 29.01%.
- The final GLM result and accuracy/latency tradeoffs are documented in
  [GLM results](docs/GLM_RESULTS.md).

See [the canonical project context](docs/PROJECT_CONTEXT.md), [experiment protocol](docs/EXPERIMENT_PROTOCOL.md), and [server setup](docs/SERVER_SETUP.md).
The gated GLM LoRA workflow is documented in [GLM training](docs/GLM_TRAINING.md).

## Layout

```text
configs/                 experiment definitions
data/manifests/          manifest contract only; actual data is ignored
docs/                    current documentation and archived plans
notebooks/legacy_colab/  original Colab-oriented notebooks
scripts/                 environment, download, and validation commands
src/vlm_handwriting/     reusable experiment code
tests/                   CPU-only contract tests
```

Mutable data belongs outside the repository under `/workspace/vlm-handwriting/{data,checkpoints,outputs,logs,cache}`.

## Local development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Vast server bootstrap

The confirmed server environment is Python 3.12 with `torch 2.10.0+cu128`, RTX 5090 `sm_120`, and working BF16 CUDA. Preserve that Torch build.

```bash
cd /workspace/vlm-handwriting/repo
uv pip install --python /venv/main/bin/python -e ".[data,dev]"
/venv/main/bin/python scripts/verify_environment.py \
  --output /workspace/vlm-handwriting/logs/environment.json
/venv/main/bin/python scripts/download_data.py \
  --data-root /workspace/vlm-handwriting/data
/venv/main/bin/python scripts/verify_manifest.py \
  --manifest-root <manifest_root printed by download_data.py> \
  --raw-root <raw_root printed by download_data.py> \
  --check-images
```

KaggleHub may return nested download paths. The verifier discovers the directory containing the three split CSVs.

## GLM-OCR base benchmark

Install the official-compatible GLM dependency range without replacing the verified Torch pair, then confirm Torch is unchanged:

```bash
cd /workspace/vlm-handwriting/repo
uv pip install --python /venv/main/bin/python -r requirements/glm.txt
/venv/main/bin/python -c 'import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)'
export HF_HOME=/workspace/vlm-handwriting/cache/huggingface
```

Run exactly one fixed validation sample first:

```bash
/venv/main/bin/python scripts/benchmark_glm.py \
  --manifest-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/hwdb-manifest/versions/1/uit_hwdb_line_ready \
  --raw-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/uit-hwdb/versions/1/UIT_HWDB_line/UIT_HWDB_line \
  --output-root /workspace/vlm-handwriting/outputs \
  --mode one \
  2>&1 | tee /workspace/vlm-handwriting/logs/glm-base-one.log
```

Inspect the ground truth and prediction artifact before changing `--mode one` to `--mode smoke`. After smoke review, freeze the prompt and decoding configuration, then run the test baseline once with `--mode test --allow-test`.

The files under `notebooks/legacy_colab/` are historical Colab references. The reproducible Vast execution path uses `scripts/` and shared code under `src/`; a notebook is optional for visual inspection and is not required for benchmark or training jobs.

Run long jobs inside `tmux`. The laptop and SSH connection may disconnect after detaching, but the Vast instance itself must remain running.

## Licensing

No repository license has been selected. Dataset and model licenses remain governed by their upstream sources.
