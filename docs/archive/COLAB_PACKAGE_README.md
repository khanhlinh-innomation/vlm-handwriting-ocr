# Vietnamese Handwriting VLM — Colab Pro Notebook Package

This package contains the complete 3-model × 3-phase experiment workflow for the frozen UIT-HWDB-line split.

## Frozen dataset contract

- Raw Kaggle dataset: `ntklinhfitus/uit-hwdb`
- Manifest Kaggle dataset: `ntklinhfitus/uit-hwdb-manifest`
- Usable labeled line samples: 7,229
- Train: 6,346 samples / 224 writers
- Validation: 682 samples / 25 writers
- Test: 201 samples / 6 writers
- Writer-disjoint; seed 42
- Strict OCR evaluation: Unicode NFC only; preserve case, punctuation, spaces and Vietnamese diacritics.

The notebooks download data with KaggleHub. In Colab Secrets you may add `KAGGLE_API_TOKEN`. All long-lived checkpoints and results are written under:

`/content/drive/MyDrive/vlm_handwriting_ocr`

## Run order

1. `01_glm_ocr_base_benchmark_colab.ipynb`
2. `02_glm_ocr_lora_finetune_colab.ipynb`
3. `03_glm_ocr_finetuned_benchmark_colab.ipynb`
4. `04_teleocr_base_benchmark_colab.ipynb`
5. `05_teleocr_lora_finetune_colab.ipynb`
6. `06_teleocr_finetuned_benchmark_colab.ipynb`
7. `07_paddleocr_vl_1_6_base_benchmark_colab.ipynb`
8. `08_paddleocr_vl_1_6_full_sft_colab.ipynb`
9. `09_paddleocr_vl_1_6_finetuned_benchmark_colab.ipynb`

## Source status / confidence

### GLM-OCR — highest confidence
Training is based directly on the official GLM-OCR LLaMA-Factory guide and official LoRA config. The notebooks preserve the official `glm_ocr` template, `<image>Text Recognition:` prompt, rank 8, target `all`, LR 1e-4, 3 epochs, cosine schedule and warmup 0.1. Micro-batch is reduced only when needed while keeping effective batch 16.

Official sources:
- https://github.com/zai-org/GLM-OCR/blob/main/examples/finetune/README.md
- https://github.com/zai-org/GLM-OCR/blob/main/examples/finetune/glm_ocr_lora_sft.yaml
- https://huggingface.co/docs/transformers/main/model_doc/glm_ocr

### TeleOCR — inference official, training experimental
NaviDC-OCR was renamed TeleOCR on 2026-09-10. The official model card documents inference, but no official downstream LoRA fine-tuning recipe was found when this package was prepared. Notebook 05 is explicitly an experimental PEFT adaptation: it targets only language-model projection layers and has mandatory forward/loss and smoke-training gates.

Official/model sources:
- https://huggingface.co/StarDoc-AI/TeleOCR
- https://huggingface.co/docs/peft/en/developer_guides/custom_models

### PaddleOCR-VL-1.6 — official ERNIEKit SFT family recipe adapted to v1.6
PaddleOCR currently recommends ERNIEKit for PaddleOCR-VL SFT. The published ERNIEKit recipe/config is for the PaddleOCR-VL 0.9B/v1-era model family. PaddleOCR-VL-1.6 is officially stated to be architecture-compatible with v1.5, so notebook 08 uses the same ERNIEKit interface with the 1.6 model checkpoint. The official documentation prefers a CUDA 12.x Paddle Docker image; Colab cannot use that Docker workflow, so the notebook uses a best-effort manual install and a mandatory smoke gate.

Official sources:
- https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6
- https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md
- https://github.com/PaddlePaddle/ERNIE/blob/release/v1.5/docs/paddleocr_vl_sft.md
- https://github.com/PaddlePaddle/ERNIE/blob/release/v1.5/examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml

## Safety gates

Benchmark notebooks default to smoke-only; full validation and test flags are OFF until manually enabled. Training notebooks run a smoke job first and keep the full job OFF by default. Test sets are never used for prompt/hyperparameter tuning. Fine-tuned checkpoint selection is by full validation CER.

## Important review note

These notebooks were statically reviewed for notebook JSON validity, Python syntax in normal Python cells, split consistency, gated test usage, and source alignment. They were not end-to-end executed on a live Colab Pro GPU in this environment. Library/model changes can still require small compatibility edits, especially the experimental TeleOCR training notebook and PaddleOCR/ERNIEKit manual Colab environment.
