# Final round-one results

## Decision

Use **PaddleOCR-VL-1.6 full-SFT checkpoint-300** as the recommended model for
Vietnamese handwritten line recognition. It leads all three adapted models on
the same frozen 201-line, writer-disjoint test split.

## Final adapted-model comparison

| Rank | Model | Method | Selected checkpoint | Test CER | Test WER | Exact line | Mean latency | Peak inference VRAM |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | **PaddleOCR-VL-1.6** | Full SFT | `checkpoint-300` | **5.40%** | **12.77%** | **43.28%** | **0.374 s** | **1.925 GiB** |
| 2 | GLM-OCR | LoRA | `checkpoint-1191` | 12.46% | 29.01% | 12.94% | 0.688 s | 3.197 GiB |
| 3 | TeleOCR | LoRA | `checkpoint-1191` | 28.95% | 61.58% | 0.50% | 1.459 s | 3.435 GiB |

Paddle's test CER is 56.62% lower and its WER is 55.96% lower than the
second-place GLM checkpoint. It is also faster and uses less inference memory.

## Base-to-adapted change

| Model | Base test CER | Adapted test CER | Relative CER reduction | Base test WER | Adapted test WER | Relative WER reduction |
|---|---:|---:|---:|---:|---:|---:|
| GLM-OCR | 31.95% | 12.46% | 61.01% | 77.41% | 29.01% | 62.53% |
| TeleOCR | 89.36% | 28.95% | 67.60% | 112.61% | 61.58% | 45.32% |
| **PaddleOCR-VL-1.6** | 26.32% | **5.40%** | **79.47%** | 62.97% | **12.77%** | **79.71%** |

Paddle exact-line accuracy increased from 0.50% to 43.28%, a gain of 42.79
percentage points. Its adapted mean latency was 8.02% lower than its base
latency, and throughput increased by 7.13%.

## Validation-based checkpoint selection

| Model | Best validation checkpoint | Validation CER | Validation WER |
|---|---|---:|---:|
| GLM-OCR | `checkpoint-1191` | 10.76% | 25.77% |
| TeleOCR | `checkpoint-1191` | 27.39% | 59.77% |
| **PaddleOCR-VL-1.6** | `checkpoint-300` | **4.06%** | **9.69%** |

All epoch checkpoints were ranked using the full 682-line validation split.
Only the selected checkpoint from each model was evaluated on test.

## Training summary

| Model | Training method | Epochs / steps | Runtime | Final reported train loss |
|---|---|---:|---:|---:|
| GLM-OCR | LoRA, LLaMA-Factory | 3 / 1,191 | 31:14.78 | 0.8435 |
| TeleOCR | LoRA, Transformers/PEFT | 3 / 1,191 | 3:20:02 | 4.0339 |
| PaddleOCR-VL-1.6 | Full SFT, ERNIEKit | approximately 3 passes / 300 | 52:02.39 | 0.6181 |

These losses are framework- and objective-dependent and must not be compared
across models as quality metrics. Final OCR quality is determined by the common
CER/WER evaluation above.

## Paddle final-run evidence

- Validation artifacts:
  `/workspace/vlm-handwriting/outputs/paddleocr_vl/full_sft/validation/20260917T234405Z`
- Test artifacts:
  `/workspace/vlm-handwriting/outputs/paddleocr_vl/full_sft/test/20260918T000654Z`
- Selected checkpoint:
  `/workspace/vlm-handwriting/checkpoints/paddleocr_vl/full_sft_3ep_run1/checkpoint-300`
- Frozen test runtime: 79.015 seconds for 201 lines
- Test throughput: 2.5438 samples/s
- Test p50 / p90 latency: 0.3705 / 0.4801 seconds

## Protocol note

The data split is fixed at 6,346 train, 682 validation, and 201 test lines with
zero writer overlap. Metrics use NFC normalization only and preserve case,
punctuation, spacing, and Vietnamese diacritics. The test split is now sealed;
no subsequent prompt, decoding, checkpoint, or hyperparameter selection may use
these test results.
