# GLM-OCR Vietnamese Handwriting Results

## Executive summary

GLM-OCR was fine-tuned with LoRA on the frozen, writer-disjoint UIT-HWDB-line split.
Checkpoint selection used full-validation corpus CER, and the frozen test set was opened once
after checkpoint selection. The selected adapter reduced test CER by **61.01% relative** and
test WER by **62.53% relative** compared with the frozen base-model baseline.

The selected model is `checkpoint-1191` (epoch 3). It is the recommended GLM-OCR artifact
for the current round.

## Data and protocol

| Split | Samples | Writers | Use |
|---|---:|---:|---|
| Train | 6,346 | 224 | LoRA optimization |
| Validation | 682 | 25 | Epoch checkpoint selection by corpus CER |
| Test | 201 | 6 | One final evaluation after checkpoint freeze |

- Writer overlap between splits: zero.
- Text normalization: Unicode NFC only.
- Prompt: `Text Recognition:`.
- Decoding: deterministic, maximum 256 new tokens, repetition penalty 1.1.
- Precision: BF16.
- Test data was not exported to LLaMA-Factory training data.
- Test mode accepted exactly one validation-selected adapter.

## Training

| Item | Result |
|---|---:|
| Method | LoRA, rank 8, target `all` |
| Epochs | 3 |
| Optimizer steps | 1,191 |
| Effective batch size | 16 |
| Learning rate | 1e-4 with cosine schedule |
| Average train loss | 0.8435 |
| Validation loss, epoch 1 | 0.8846 |
| Validation loss, epoch 2 | 0.6996 |
| Validation loss, epoch 3 | 0.6659 |
| Training runtime | 31 min 14.78 sec |
| Training throughput | 10.155 samples/sec |
| Approximate peak allocated GPU memory | 4.57 GiB |

Training completed without CUDA OOM or NaN loss. The adapter save, fresh-process reload,
and validation inference gate passed before the full run.

## Checkpoint selection

All epoch checkpoints generated predictions for the same 682 validation rows.

| Rank | Checkpoint | Validation CER | Validation WER |
|---:|---|---:|---:|
| 1 | `checkpoint-1191` | **10.7568%** | **25.7728%** |
| 2 | `checkpoint-794` | 11.6850% | 28.2324% |
| 3 | `checkpoint-397` | 14.4607% | 35.2187% |

`checkpoint-1191` was frozen before test evaluation.

## Frozen test result

| Metric | Base GLM-OCR | LoRA `checkpoint-1191` | Change |
|---|---:|---:|---:|
| Corpus CER | 31.9493% | **12.4561%** | **-19.4932 pp / 61.01% relative reduction** |
| Corpus WER | 77.4118% | **29.0086%** | **-48.4032 pp / 62.53% relative reduction** |
| Exact-line accuracy | 0.00% | **12.9353%** | +12.9353 pp |
| Mean latency | 0.502 sec | 0.688 sec | +37.06% |
| P50 latency | 0.514 sec | 0.694 sec | +34.94% |
| P90 latency | 0.622 sec | 0.877 sec | +41.00% |
| Throughput | 1.930 samples/sec | 1.387 samples/sec | -28.14% |
| Peak allocated GPU memory | 3.129 GiB | 3.197 GiB | +0.068 GiB |
| Evaluation runtime | 104.168 sec | 144.964 sec | +40.797 sec |

The fine-tuned adapter materially improves recognition accuracy while adding modest GPU
memory use. Its main tradeoff is slower autoregressive inference under the same decoding
configuration.

## Artifacts

- Base frozen test runs: `/workspace/vlm-handwriting/outputs/glm_ocr/base`
- Training checkpoints: `/workspace/vlm-handwriting/checkpoints/glm_ocr/lora_full`
- Validation ranking: `/workspace/vlm-handwriting/outputs/glm_ocr/lora/validation/20260916T141000Z`
- Fine-tuned frozen test: `/workspace/vlm-handwriting/outputs/glm_ocr/lora/test/20260916T144816Z`
- Training log: `/workspace/vlm-handwriting/logs/glm-lora-full.log`
- Validation log: `/workspace/vlm-handwriting/logs/glm-lora-validation.log`
- Fine-tuned test log: `/workspace/vlm-handwriting/logs/glm-lora-test.log`

## Conclusion and next decision

The GLM-OCR LoRA pipeline is complete end to end. `checkpoint-1191` is the current
recommended model because it won validation selection and achieved a large accuracy gain on
the untouched test set. Further GLM experiments must return to train/validation only; this
test result must not be used to tune hyperparameters or select another checkpoint.

TeleOCR and PaddleOCR-VL remain deferred under the current GLM-only scope. A future
multi-model comparison should apply the same frozen split, normalization, decoding-artifact,
and one-time-test rules.
