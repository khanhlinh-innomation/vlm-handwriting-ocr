# VLM Handwriting OCR — Codex Migration Context

> Purpose: hand this file to Codex running in the remote terminal so it has the full project context, current decisions, frozen data protocol, model plans, and immediate next steps.
>
> Language note: the project documentation and experiment artifacts should be written in English unless explicitly requested otherwise.

---

# 1. Project Goal

Fine-tune and benchmark small Vision-Language Models (VLMs) for **Vietnamese handwritten line OCR**.

Core task:

```text
handwritten line image
        ↓
exact transcription
```

The target is **recognition only**:
- preserve Vietnamese diacritics,
- preserve case,
- preserve punctuation,
- preserve meaningful spacing,
- do not return explanations,
- do not perform spell correction,
- do not normalize away OCR errors during evaluation.

The three current model candidates are:

1. **GLM-OCR**
2. **TeleOCR** (formerly NaviDC-OCR)
3. **PaddleOCR-VL-1.6**

---

# 2. Broader Production Context

The original product/runtime plan targets local/in-process OCR inference.

Important production constraints from the original plan:
- PaddleOCR continues to provide detection / geometry.
- The VLM is intended to replace recognition for difficult line crops.
- Production runtime is planned around **Rust + Hugging Face Candle**.
- No Python runtime in the final production path.
- LoRA adapters may need to be merged into base weights before production export.
- Tokenizer vocabulary should remain unchanged unless explicitly required.
- Runtime integration and training are separate concerns.

Current task scope is primarily:

```text
data preparation
→ base benchmark
→ fine-tuning
→ checkpoint evaluation
```

Do **not** mix production Rust/Candle integration into the current training experiments unless explicitly requested.

---

# 3. Dataset Sources

## 3.1 Raw dataset

Source repository:

```text
https://github.com/nghiangh/UIT-HWDB-dataset
```

Kaggle dataset already uploaded:

```text
ntklinhfitus/uit-hwdb
```

UIT-HWDB official inventory:

| Subset | Samples |
|---|---:|
| UIT-HWDB-word | 110,745 |
| UIT-HWDB-line | 7,273 |
| UIT-HWDB-paragraph | 1,144 |
| Total | 119,162 |

Round 1 uses **UIT-HWDB-line only** because the production task is line-crop → transcription.

Do not mix word or paragraph samples into the first training round.

---

## 3.2 Frozen manifest dataset

A second Kaggle dataset was created:

```text
ntklinhfitus/hwdb-manifest
```

It contains the frozen experiment split and metadata.

Expected files:

```text
master_manifest.csv
master_manifest.jsonl
train.csv
val.csv
test.csv
dataset_report.csv
writer_split.csv
```

The manifest dataset does **not** need to duplicate the images.

The raw image dataset and the manifest dataset should be treated separately:

```text
uit-hwdb
    → actual images + label.json

uit-hwdb-manifest
    → frozen train / validation / test definitions
```

---

# 4. UIT-HWDB-line Label Schema

Each writer/source directory contains a `label.json`.

Observed schema:

```json
{
  "1.jpg": "KHÁI QUÁT VỀ BIỂN ĐẢO VIỆT NAM",
  "2.jpg": "Nước ta giáp với biển Đông ở hai phía Đông và Nam. ...",
  "3.jpg": "Đông."
}
```

Therefore:

```text
filename → ground-truth transcription
```

Example raw structure:

```text
UIT_HWDB_line/
└── UIT_HWDB_line/
    ├── train_data/
    │   ├── 1/
    │   │   ├── 1.jpg
    │   │   ├── 2.jpg
    │   │   └── label.json
    │   ├── 2/
    │   └── ...
    └── test_data/
        ├── 250/
        │   ├── ...
        │   └── label.json
        └── ...
```

---

# 5. Meaning of Writer / Group

A `writer_id` represents the writer/source group corresponding to one dataset folder.

Handwriting recognition should be evaluated on writers not seen during training whenever possible.

Therefore the split is **writer-disjoint**:

```text
one writer must belong to exactly one split
```

Do not randomly split individual lines across train/validation if that causes the same writer to appear in both.

---

# 6. Frozen Dataset Split

The original UIT-HWDB-line package contains 7,273 image files.

Audit result:

```text
Total image files:          7,273
Usable labeled samples:     7,229
Unlabeled extra images:        44
```

The 44 excluded images:
- exist in the training folders,
- do not have matching entries in `label.json`,
- are therefore unusable for supervised OCR training/evaluation,
- must not be assigned guessed labels.

Observed usable original split:

```text
Original train labeled rows: 7,028
Original test labeled rows:    201
Total usable labeled rows:    7,229
```

The existing test split is preserved.

A writer-disjoint validation split was created from the original training writers using:

```python
GroupShuffleSplit(
    n_splits=1,
    test_size=0.10,
    random_state=42
)
```

Final frozen split:

| Split | Samples | Writers |
|---|---:|---:|
| Train | 6,346 | 224 |
| Validation | 682 | 25 |
| Test | 201 | 6 |
| Total | 7,229 | 255 groups overall |

Validation writer IDs:

```text
[7, 10, 16, 20, 25, 61, 68, 69, 98, 105,
 113, 115, 138, 163, 176, 181, 184, 194, 197,
 200, 202, 216, 224, 240, 247]
```

Important rule:

> Never re-split the dataset unless explicitly authorized.

All models must use the exact same frozen train/validation/test definitions.

---

# 7. Dataset Audit Result

Audit status:

```text
✓ zero writer leakage
✓ zero empty labels
✓ zero corrupt labeled images
✓ zero NFC changes needed in the current labels
✓ image ↔ label visual sanity check passed
✓ 44 unlabeled images identified and excluded
```

Final text-length statistics:

| Split | Samples | Writers | Mean chars | p50 | p90 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 6,346 | 224 | 68.03 | 71 | 86 | 99.00 | 158 |
| Validation | 682 | 25 | 65.72 | 69 | 85 | 95.38 | 110 |
| Test | 201 | 6 | 65.18 | 68 | 83 | 95.00 | 99 |

The length distributions are reasonably similar across splits.

---

# 8. Manifest Usage

`master_manifest.csv` is the single source of truth for the dataset.

Typical columns include:

```text
writer_id
image_id
filename
relative_path
text_raw
text
text_length
original_split
split
```

Important path rule:

Use:

```text
relative_path
```

as the portable source of truth.

Do not depend on Kaggle-specific absolute paths such as:

```text
/kaggle/input/...
```

When running on another machine:

```python
image_path = RAW_ROOT / row["relative_path"]
```

The split CSVs:

```text
train.csv
val.csv
test.csv
```

are convenience files for training/evaluation scripts.

Model-specific dataset formats must be generated from the same frozen CSVs.

---

# 9. Evaluation Protocol

Primary metric:

```text
CER ↓
```

Secondary metrics:

```text
WER ↓
Exact-Line Accuracy ↑
Diacritic error / analysis
Latency
Peak GPU VRAM
```

Strict OCR normalization:

```python
unicodedata.normalize("NFC", text)
```

Do **not**:
- lowercase,
- remove punctuation,
- remove spaces,
- remove Vietnamese diacritics,
- spell-correct.

The same evaluator implementation must be used for all models.

Checkpoint selection:

```text
best checkpoint = lowest validation CER
```

Do not select checkpoints based on training loss alone.

---

# 10. Test-Set Rule

The test set is frozen and must not be used for hyperparameter selection.

Correct protocol:

```text
TRAIN
→ optimize model

VALIDATION
→ debug inference
→ select epoch/checkpoint
→ compare configurations

TEST
→ final evaluation only
```

For base models:
- run one validation image,
- run the fixed validation smoke20,
- freeze the inference configuration,
- then run the frozen test baseline once.

Full validation is not required for an unchanged base model. It is required during
fine-tuning to compare configurations and select the best checkpoint.

Do not tune prompts/decoding using test results.

---

# 11. Fixed Smoke Benchmark

All models should first use the same deterministic validation smoke set:

```python
SMOKE_SIZE = 20
SMOKE_SEED = 42
```

Protocol:

```text
20 fixed validation samples
        ↓
check:
- model loads
- processor loads
- image input works
- prompt/template works
- output contains OCR text only
- Vietnamese encoding is correct
- no repetition collapse
- no OOM
- record latency
- record peak GPU VRAM
```

Only after this passes should the prompt and decoding configuration be frozen. The
base test baseline may then run once. Do not tune from its result.

---

# 12. Current Model Plans

## 12.1 GLM-OCR

Approximate size:

```text
~0.9B
```

Base model:

```text
zai-org/GLM-OCR
```

Official OCR prompt used in current work:

```text
Text Recognition:
```

Training stack:

```text
LLaMA-Factory + LoRA
```

Current round-1 setup:

```text
LoRA rank:            8
LoRA target:          all (official-style recipe)
Learning rate:        1e-4
Epochs:               3
Scheduler:            cosine
Warmup ratio:         0.1
Precision:            BF16 where supported
Effective batch:      ~16
```

Planning VRAM:

```text
~8–12 GB for LoRA
```

GLM-OCR is the most straightforward first model because the official fine-tuning path is the clearest.

---

## 12.2 TeleOCR / NaviDC-OCR

Current project/model name:

```text
TeleOCR
```

Previous name:

```text
NaviDC-OCR
```

Current model reference used in planning:

```text
StarDoc-AI/TeleOCR
```

Approximate size:

```text
~1.2B
```

Training support is less mature than GLM-OCR.

Current proposed training stack:

```text
Transformers + PEFT LoRA
```

Alternative if compatible:

```text
ms-swift + PEFT
```

Round-1 hypothesis:

```text
freeze vision encoder
freeze projector initially
LoRA on language-model layers

rank:               8
alpha:              32
learning rate:      1e-4
epochs:             3
precision:          BF16
effective batch:    ~16
gradient checkpointing: ON if useful
```

Important:

> TeleOCR fine-tuning is experimental because there is no official downstream LoRA recipe as clean as GLM-OCR's.

Mandatory smoke gate:
- load model + processor,
- forward/loss,
- inject LoRA,
- train small subset,
- save adapter,
- reload adapter,
- inference.

Planning VRAM:

```text
~12–20 GB
24 GB+ preferred for comfort until measured
```

---

## 12.3 PaddleOCR-VL-1.6

Model reference:

```text
PaddlePaddle/PaddleOCR-VL-1.6
```

Approximate size:

```text
~0.9–1.0B
```

Training stack:

```text
ERNIEKit
```

Round-1 method:

```text
Full SFT
```

Current starting configuration:

```text
Learning rate:   5e-6
Epochs:          2 initially
Precision:       BF16
Acceleration:    Flash Attention / packing where supported
```

Epoch-3 rule:

```text
run epoch 3 only if validation CER is still improving after epoch 2
```

Why start with 2 epochs:
- training set is only ~6.3k lines,
- full SFT updates the whole model,
- higher overfitting/catastrophic-forgetting risk than LoRA,
- validation CER determines whether epoch 3 is justified.

Planning VRAM:

```text
24 GB+ planning floor
40 GB+ preferred headroom
```

Paddle is the most likely model to need a larger team GPU for full SFT.

---

# 13. Unsloth Decision

Unsloth is **not the main training stack** currently.

Current main stacks:

```text
GLM-OCR       → LLaMA-Factory + LoRA
TeleOCR       → Transformers + PEFT LoRA
PaddleOCR-VL  → ERNIEKit Full SFT
```

Potential Unsloth use:

```text
GLM-OCR only, as an optional optimization experiment
```

If Unsloth is tested for GLM:
- prefer 16-bit LoRA if the goal is to stay comparable,
- keep same dataset,
- same split,
- same prompt,
- same LoRA rank,
- same LR,
- same epochs,
- avoid switching to 4-bit QLoRA unless explicitly testing quantized training.

Do not migrate TeleOCR or Paddle to Unsloth by default.

---

# 14. Vast.ai Environment Context

The team lead provisioned access to a Vast.ai GPU server.

The user has an SSH key pair locally:
- public key was shared with the team lead,
- private key must never be shared or committed.

The exact SSH IP/port should be obtained from the lead/current Vast instance and should **not** be committed into the repository.

Typical connection:

```bash
ssh -p <PORT> root@<SERVER_IP>
```

Once connected, first inspect:

```bash
nvidia-smi
pwd
ls -la
```

The work runs on the remote Vast GPU machine, not on the laptop GPU.

---

# 15. Vast Template Context

The lead shared an **Unsloth Studio** Vast template.

The template includes:
- Unsloth Studio,
- PyTorch,
- CUDA image,
- Jupyter,
- SSH,
- root access,
- terminal access,
- ability to install additional packages.

Important decision:

> The template is usable as a remote GPU environment even if the main training pipeline does not use Unsloth.

For the full 3-model project, a generic CUDA 12.x + PyTorch/Jupyter environment is conceptually cleaner because:
- GLM uses LLaMA-Factory,
- TeleOCR uses Transformers/PEFT,
- Paddle uses ERNIEKit/Paddle.

However the Unsloth Studio template can still be used if the environment allows custom packages and tooling.

Template choice does not primarily determine GPU price; Vast pricing is mainly tied to the GPU/host/storage/bandwidth.

---

# 16. Recommended Remote Project Layout

Use a Git repository for code, notebooks, configs, and documentation.

Do **not** commit raw datasets or large checkpoints into normal Git.

Recommended layout:

```text
/workspace/
├── vlm-handwriting/              # git repository
│   ├── README.md
│   ├── CONTEXT.md
│   │
│   ├── notebooks/
│   │   ├── glm_ocr/
│   │   ├── teleocr/
│   │   └── paddleocr_vl/
│   │
│   ├── src/
│   │   ├── data.py
│   │   ├── metrics.py
│   │   ├── benchmark.py
│   │   └── utils.py
│   │
│   ├── scripts/
│   │   ├── benchmark_glm.py
│   │   ├── train_glm.py
│   │   ├── benchmark_teleocr.py
│   │   ├── train_teleocr.py
│   │   ├── benchmark_paddle.py
│   │   └── train_paddle.py
│   │
│   └── configs/
│       ├── glm_lora.yaml
│       ├── teleocr_lora.yaml
│       └── paddle_sft.yaml
│
├── data/
│   ├── uit-hwdb/
│   └── uit-hwdb-manifest/
│
├── checkpoints/
│   ├── glm_ocr/
│   ├── teleocr/
│   └── paddleocr_vl/
│
├── outputs/
│   ├── glm_ocr/
│   ├── teleocr/
│   └── paddleocr_vl/
│
└── logs/
```

---

# 17. Git / Notebook / Python Strategy

Recommended workflow:

```text
Notebook
→ development, inspection, smoke debugging, visual checks

Python scripts
→ full benchmarks, long training jobs, reproducible runs
```

Do not throw away the notebooks.

Refactor reusable logic into `src/` so notebooks and scripts share:
- data loading,
- split validation,
- metrics,
- prediction output format,
- logging.

For long remote jobs use:

```bash
tmux
```

Example:

```bash
tmux new -s glm_train
python scripts/train_glm.py 2>&1 | tee logs/glm_train.log
```

Detach:

```text
Ctrl+B, then D
```

Reattach:

```bash
tmux attach -t glm_train
```

This allows the laptop to disconnect while the Vast job keeps running.

---

# 18. Current 9-Notebook Experiment Design

The original plan separates each model into:

```text
base benchmark
→ fine-tune
→ fine-tuned benchmark
```

Target notebook organization:

```text
glm_ocr/
├── 01_base_benchmark.ipynb
├── 02_finetune.ipynb
└── 03_finetuned_benchmark.ipynb

teleocr/
├── 01_base_benchmark.ipynb
├── 02_finetune.ipynb
└── 03_finetuned_benchmark.ipynb

paddleocr_vl/
├── 01_base_benchmark.ipynb
├── 02_finetune.ipynb
└── 03_finetuned_benchmark.ipynb
```

A Colab-oriented version of all 9 notebooks was previously created.

When migrating to Vast:
- remove Colab Drive-specific setup,
- replace `/content/...` paths,
- use the remote `/workspace/...` layout,
- retain the experiment logic and frozen split,
- prefer scripts for full long-running jobs.

---

# 19. Expected Outputs

## Base benchmark

For each model:

```text
*_base_smoke20_predictions.csv
*_base_smoke20_metrics.json

*_base_test_predictions.csv
*_base_test_metrics.json

environment.json
```

Common prediction schema:

```text
model
split
writer_id
filename
relative_path
ground_truth
prediction
sample_CER
sample_WER
exact_match
latency_sec
```

---

## Training

Expected artifacts:

```text
training_config.json / yaml
training_log.csv
checkpoints/
best checkpoint
validation metrics
environment metadata
```

GLM / TeleOCR:
- likely LoRA adapters.

Paddle:
- full SFT checkpoint.

---

## Fine-tuned benchmark

For each model:

```text
*_finetuned_val_predictions.csv
*_finetuned_val_metrics.json

*_finetuned_test_predictions.csv
*_finetuned_test_metrics.json
```

Final comparison should include:

| Model | Version | Val CER | Test CER | WER | Exact-Line Accuracy | Latency | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|

Versions:

```text
Base
Fine-tuned
```

---

# 20. Compute / VRAM Policy

Initial planning estimates:

| Model | Method | Planning VRAM |
|---|---|---:|
| GLM-OCR | LoRA | ~8–12 GB |
| TeleOCR | LoRA | ~12–20 GB |
| PaddleOCR-VL | Full SFT | 24 GB+; 40 GB+ preferred |

These are planning estimates only.

Actual GPU sizing must be based on smoke-run measurements:
- peak VRAM,
- throughput,
- latency,
- stability.

Do not claim parameter count alone proves a fixed VRAM requirement.

---

# 21. GPU Escalation Philosophy

Original compute policy was:

```text
Kaggle / Colab first
→ measure
→ request team GPU if needed
```

Now a Vast.ai team GPU instance is available, so the immediate goal is to move the experiments there.

Still preserve the same principle:

```text
measure first
→ then decide batch size / gradient accumulation / larger GPU need
```

---

# 22. Immediate Migration Steps for Codex

When this file is handed to Codex on the Vast server, do the following in order.

## Step 1 — Inspect environment

Run:

```bash
nvidia-smi
python --version
which python
pwd
ls -la
df -h
```

Record:
- GPU model,
- VRAM,
- driver,
- CUDA compatibility,
- free disk.

Do not install large packages before checking the current image.

---

## Step 2 — Establish repository

Preferred:

```bash
cd /workspace
git clone <PRIVATE_GITHUB_REPO_URL> vlm-handwriting
cd vlm-handwriting
```

If no repo exists yet, initialize one and prepare the recommended structure.

Never commit:
- SSH private keys,
- Kaggle credentials,
- Hugging Face tokens,
- raw datasets,
- large checkpoints,
- generated model weights.

Add those to `.gitignore`.

---

## Step 3 — Create/verify data directories

Expected:

```text
/workspace/data/uit-hwdb
/workspace/data/uit-hwdb-manifest
```

Download the two Kaggle datasets using Kaggle API if not already present.

Dataset IDs:

```text
ntklinhfitus/uit-hwdb
ntklinhfitus/hwdb-manifest
```

After download, verify:

```text
train.csv = 6,346
val.csv   = 682
test.csv  = 201
```

Verify no writer leakage again before running a model.

Do not regenerate the split.

---

## Step 4 — Build shared project utilities first

Before duplicating model code, create reusable modules for:

```text
data loading
path resolution
strict OCR normalization
CER / WER / exact match
smoke-set selection
prediction CSV saving
environment metadata
GPU memory measurement
```

The fixed validation smoke sample must use:

```text
seed = 42
size = 20
```

---

## Step 5 — Run GLM-OCR base benchmark first

Order:

```text
1 sample inference
→ 20 validation smoke
→ inspect output
→ freeze inference config
→ one-time 201 test baseline
```

Do not begin GLM fine-tuning until the base inference pipeline is reliable.

---

## Step 6 — GLM fine-tuning

Use official-style LLaMA-Factory LoRA setup.

Before full run:

```text
small smoke training
→ save adapter
→ reload adapter
→ inference
```

Only then run the full 3-epoch training.

Select best checkpoint using validation CER.

---

## Step 7 — GLM fine-tuned benchmark

Load:
- same base model,
- best LoRA adapter.

Run:
- validation,
- frozen test.

Save the same metrics/prediction schema as the base benchmark.

---

## Step 8 — Repeat for TeleOCR

Be more conservative because fine-tuning support is experimental.

Require:
- model load,
- processor load,
- forward/loss,
- PEFT injection,
- small training smoke,
- save/reload,
- inference

before any full run.

---

## Step 9 — Repeat for PaddleOCR-VL-1.6

Use ERNIEKit/Paddle-specific setup.

Do not force a PyTorch-only abstraction onto Paddle if it breaks official tooling.

Start with:
- environment compatibility,
- data conversion,
- one-sample / smoke SFT,
- measured VRAM.

Full plan:
- 2 epochs first,
- epoch 3 only if validation CER still improves.

---

# 23. Important Rules for Codex

Codex should NOT:

1. Re-split the dataset.
2. Mix word/paragraph samples into round 1.
3. Use the test set for prompt/hyperparameter selection.
4. Lowercase or strip Vietnamese diacritics for CER.
5. Guess labels for the 44 unlabeled images.
6. Replace official/native model templates without evidence.
7. Switch GLM to QLoRA/4-bit silently.
8. Switch TeleOCR/Paddle to Unsloth by default.
9. Commit secrets or checkpoints to Git.
10. Delete or overwrite previous best checkpoints without versioning.
11. Claim an experiment succeeded unless save → reload → inference is verified.
12. Assume the Colab-specific paths still apply on Vast.

---

# 24. Preferred Engineering Style

Priorities:

```text
correctness
> reproducibility
> official model compatibility
> clean experiment tracking
> speed optimization
```

Use:
- explicit configs,
- deterministic seeds where possible,
- versioned outputs,
- environment metadata,
- clear logs,
- fail-fast assertions,
- small smoke tests before expensive runs.

When a framework/model behavior is uncertain:
- inspect the official source/docs/model code,
- do not guess,
- document whether a setting is official or experimental.

---

# 25. Current State Summary

Completed:

```text
✓ raw dataset uploaded to Kaggle
✓ manifest dataset uploaded to Kaggle
✓ line-only round-1 decision
✓ label schema confirmed
✓ 44 unlabeled extra images explained/excluded
✓ frozen writer-disjoint split
✓ train = 6,346
✓ val = 682
✓ test = 201
✓ dataset audit passed
✓ evaluation protocol defined
✓ 3-model shortlist defined
✓ 9-notebook experiment design defined
✓ remote Vast.ai access provisioned
```

Next:

```text
→ inspect Vast GPU/server
→ clone/init repository
→ download 2 Kaggle datasets
→ refactor common utilities
→ run GLM-OCR base smoke benchmark
→ run GLM full base validation
→ freeze GLM base inference config
→ run GLM test baseline
→ proceed to GLM fine-tuning
```

---

# 26. One-Sentence Project Summary

> Build a reproducible, writer-disjoint Vietnamese handwriting OCR benchmark and fine-tuning pipeline for GLM-OCR, TeleOCR, and PaddleOCR-VL-1.6 using the frozen UIT-HWDB-line split, strict CER evaluation, and remote GPU execution on Vast.ai.
