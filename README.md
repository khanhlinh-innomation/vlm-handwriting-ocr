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

- Current documentation and nine statically reviewed Colab notebooks are imported.
- Shared data-contract, metric, smoke-selection, artifact, and environment utilities are scaffolded and tested.
- The legacy notebooks have not been executed end to end on the Vast server.
- Server-native model entry points are the next phase, starting with GLM-OCR.

See [the canonical project context](docs/PROJECT_CONTEXT.md), [experiment protocol](docs/EXPERIMENT_PROTOCOL.md), and [server setup](docs/SERVER_SETUP.md).

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

Mutable data belongs outside the repository under `/workspace/data`, `/workspace/checkpoints`, `/workspace/outputs`, and `/workspace/logs`.

## Local development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Vast server bootstrap

The confirmed server environment is Python 3.12 with `torch 2.10.0+cu128`, RTX 5090 `sm_120`, and working BF16 CUDA. Preserve that Torch build.

```bash
cd /workspace/vlm-handwriting-ocr
uv pip install -e ".[data,dev]"
python scripts/verify_environment.py --output environment.json
python scripts/download_data.py --data-root /workspace/data
python scripts/verify_manifest.py \
  --manifest-root <manifest_root printed by download_data.py> \
  --raw-root <raw_root printed by download_data.py> \
  --check-images
```

KaggleHub may return nested download paths. The verifier discovers the directory containing the three split CSVs.

## Licensing

No repository license has been selected. Dataset and model licenses remain governed by their upstream sources.
