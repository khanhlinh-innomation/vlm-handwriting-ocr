# Static Review Report

Generated notebooks were checked for valid nbformat JSON and Python syntax after excluding Colab magics/shell lines. Benchmark notebooks were also checked for disabled-by-default frozen-test gates.

| Notebook | Cells | JSON | Syntax |
|---|---:|---|---|
| 01_glm_ocr_base_benchmark_colab.ipynb | 27 | PASS | PASS |
| 02_glm_ocr_lora_finetune_colab.ipynb | 23 | PASS | PASS |
| 03_glm_ocr_finetuned_benchmark_colab.ipynb | 18 | PASS | PASS |
| 04_teleocr_base_benchmark_colab.ipynb | 22 | PASS | PASS |
| 05_teleocr_lora_finetune_colab.ipynb | 31 | PASS | PASS |
| 06_teleocr_finetuned_benchmark_colab.ipynb | 18 | PASS | PASS |
| 07_paddleocr_vl_1_6_base_benchmark_colab.ipynb | 21 | PASS | PASS |
| 08_paddleocr_vl_1_6_full_sft_colab.ipynb | 25 | PASS | PASS |
| 09_paddleocr_vl_1_6_finetuned_benchmark_colab.ipynb | 18 | PASS | PASS |

## Limitations

Static review is not a substitute for running the notebooks on a live Colab GPU. TeleOCR fine-tuning is an experimental PEFT adaptation because no official TeleOCR downstream fine-tuning recipe was located. PaddleOCR-VL-1.6 SFT uses the official ERNIEKit PaddleOCR-VL recipe adapted to the architecture-compatible 1.6 checkpoint; Colab uses manual installation rather than the officially preferred Docker environment.
