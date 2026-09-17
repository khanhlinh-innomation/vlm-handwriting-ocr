# PaddleOCR-VL-1.6 workflow

This run uses the same frozen writer-disjoint UIT-HWDB-line split and strict
NFC-only evaluation protocol as GLM-OCR and TeleOCR.

## Runtime decision

The model is `PaddlePaddle/PaddleOCR-VL-1.6`. For base element recognition on
already-cropped text lines, use the model card's official Transformers 5 API with
the native `OCR:` prompt. This avoids page layout analysis that is irrelevant to
the line-crop task and reuses the already verified server stack:

```text
torch        2.10.0+cu128
transformers 5.17.0
GPU          NVIDIA GeForce RTX 5090 (sm_120)
```

Do not install PaddlePaddle into `/venv/main`. Full SFT will later use the official
PaddlePaddle/ERNIEKit recipe in a separate environment after the base gates pass.

Primary references:

- <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6>
- <https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html>
- <https://github.com/PaddlePaddle/ERNIE/blob/release/v1.5/docs/paddleocr_vl_sft.md>

## Gate 1: verify the existing runtime

```bash
cd /workspace/vlm-handwriting/repo
git pull --ff-only

/venv/main/bin/python -c \
  'import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__); print(torch.cuda.get_device_name(0)); assert torch.__version__ == "2.10.0+cu128"; assert int(transformers.__version__.split(".")[0]) >= 5; assert torch.cuda.is_available()'
```

Stop if the verified Torch build changes.

## Gate 2: one fixed validation image

```bash
cd /workspace/vlm-handwriting/repo
export HF_HOME=/workspace/vlm-handwriting/cache/huggingface
mkdir -p /workspace/vlm-handwriting/{outputs,logs,cache}
set -o pipefail

/venv/main/bin/python scripts/benchmark_paddle.py \
  --manifest-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/hwdb-manifest/versions/1/uit_hwdb_line_ready \
  --raw-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/uit-hwdb/versions/1/UIT_HWDB_line/UIT_HWDB_line \
  --output-root /workspace/vlm-handwriting/outputs \
  --mode one \
  2>&1 | tee /workspace/vlm-handwriting/logs/paddle-base-one.log
```

Review the raw prediction before smoke20. This command may download roughly 2 GB
of model files on first use. Test access remains blocked unless both `--mode test`
and `--allow-test` are supplied.

The fixed one-image run passed in BF16 on the RTX 5090: CER 0.061728, WER
0.176471, latency 1.186 seconds, and peak allocated VRAM 1.822 GiB. Inspect the
saved raw reference and prediction before continuing; a single example is a
pipeline gate, not a quality estimate.

The first server attempt loaded all weights but exposed a Transformers 5 processor
API difference: the live `PaddleOCRVLImageProcessor` did not publish a direct
`min_pixels` attribute used by the model-card snippet. The runner now delegates
image sizing to the checkpoint's own `preprocessor_config.json` defaults
(`min_pixels=112896`, `max_pixels=1003520`) instead of reading that unstable
attribute. Cached upstream code is not patched.

## Gate 3: fixed validation smoke20

After confirming that the one-image prediction contains transcription only, run:

```bash
/venv/main/bin/python scripts/benchmark_paddle.py \
  --manifest-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/hwdb-manifest/versions/1/uit_hwdb_line_ready \
  --raw-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/uit-hwdb/versions/1/UIT_HWDB_line/UIT_HWDB_line \
  --output-root /workspace/vlm-handwriting/outputs \
  --mode smoke \
  2>&1 | tee /workspace/vlm-handwriting/logs/paddle-base-smoke20.log
```

Review per-line predictions and any repetition outlier before freezing the base
generation configuration. The test split remains sealed during this gate.

Smoke20 completed successfully: corpus CER 0.235016, WER 0.631034, mean latency
0.412 seconds, p90 latency 0.508 seconds, throughput 2.355 samples/s, peak
allocated VRAM 1.869 GiB, and total runtime 8.493 seconds. Before freezing the
configuration, inspect the three rows with highest sample CER for unexpected
markup, special tokens, or repetition. Artifacts are under
`outputs/paddleocr_vl/base/20260917T171155Z`.

The three highest-CER rows contained short transcription-only outputs with 17,
59, and 19 predicted characters. Their errors were ordinary recognition errors;
there was no markup, special-token leakage, or repetition. Freeze the native
`OCR:` prompt with deterministic decoding, a 256-token cap, repetition penalty
1.0, and native checkpoint image sizing.

## Gate 4: frozen base test

Run the frozen base configuration on all 201 test lines exactly once:

```bash
/venv/main/bin/python scripts/benchmark_paddle.py \
  --manifest-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/hwdb-manifest/versions/1/uit_hwdb_line_ready \
  --raw-root /workspace/vlm-handwriting/data/kagglehub/datasets/ntklinhfitus/uit-hwdb/versions/1/UIT_HWDB_line/UIT_HWDB_line \
  --output-root /workspace/vlm-handwriting/outputs \
  --mode test \
  --allow-test \
  2>&1 | tee /workspace/vlm-handwriting/logs/paddle-base-test.log
```

Do not alter the base prompt or generation settings from the test result.

The one authorized base test run completed on all 201 frozen test lines:

| Metric | Result |
|---|---:|
| CER | 0.263166 (26.32%) |
| WER | 0.629741 (62.97%) |
| Exact-line accuracy | 0.004975 (1/201, 0.50%) |
| Mean / p50 / p90 latency | 0.407 / 0.400 / 0.518 s |
| Throughput | 2.374 samples/s |
| Peak allocated VRAM | 1.925 GiB |
| Total runtime | 84.650 s |

Artifacts are under `outputs/paddleocr_vl/base/20260917T173018Z`. The base test
is now sealed: do not tune prompts or generation settings from these test rows.
PaddleOCR-VL-1.6 is the strongest frozen base by test CER (26.32% versus GLM
31.95% and TeleOCR 89.36%), while fine-tuned GLM remains the strongest result
overall at 12.46% test CER.

## Gate 5: isolated Paddle GPU runtime

The first compatibility probe used PaddlePaddle 3.2.1 from the CUDA 12.6 index.
Import and GPU discovery succeeded, but the first BF16 operation aborted because
that wheel was compiled only for architectures 61, 70, 75, 80, 86, 89, and 90;
the RTX 5090 reports compute capability 12.0 (`sm_120`). This is a wheel
architecture mismatch, not a data, model-code, or VRAM failure.

Paddle's current official compatibility table recommends CUDA 12.9 for
consumer Blackwell `sm_120`, and its current Linux install guide publishes
PaddlePaddle 3.3.0 on the CUDA 12.9 index. Replace only the package inside the
isolated `paddle-train` environment, then repeat the BF16 operation before
installing ERNIEKit. Do not install Paddle into `/venv/main` and do not compile
an unofficial wheel.

The replacement gate passed: PaddlePaddle 3.3.0 reported CUDA runtime 12.9,
found the RTX 5090 at compute capability 12.0, and completed a synchronized
1024-by-1024 BF16 matrix multiplication. The host Driver API reported 12.8; this
did not block the tested operation. The missing-`ccache` and deprecated
`paddle.device.cuda.synchronize` messages were non-blocking warnings.

## Gate 6: install ERNIEKit without changing the verified wheel

Clone the pinned upstream branch outside the experiment repository and install
its dependencies into the same isolated environment. Use
`requirements/paddle-training-constraints.txt`, then recheck the Paddle version
and BF16 operation before any training smoke. The repository includes
`scripts/prepare_paddle_training.py`, which converts only frozen train and
validation rows to the official ERNIEKit JSONL structure; it intentionally
exports zero test rows.

The conversion gate passed with 6,346 training rows, 682 validation rows,
deterministic 128/32 smoke subsets, valid absolute image paths, the masked
native `OCR:` prompt, and zero exported test rows. The next command must use
`configs/paddle/erniekit_smoke.yaml`: exactly two optimizer steps of full SFT,
not the planned two-epoch run. Its purpose is to prove finite loss, checkpoint
save, evaluation, and measured RTX 5090 memory usage.

ERNIEKit's distributed launcher invokes `python` by name for its child process.
Calling the `erniekit` executable by absolute path is therefore insufficient
when the shell still has `/venv/main/bin` first on `PATH`: the child process
cannot import Paddle. Activate `/workspace/vlm-handwriting/venvs/paddle-train`
and verify `command -v python` before every ERNIEKit launch. Do not install
PaddlePaddle into `/venv/main` as a workaround.

ERNIEKit also treats `model_name_or_path` as a local directory during image
processor initialization; it does not materialize a Hugging Face repository ID
at that point. Download `PaddlePaddle/PaddleOCR-VL-1.6` with
`huggingface_hub.snapshot_download` into
`/workspace/vlm-handwriting/models/PaddleOCR-VL-1.6` before launching. The
smoke config points to that local directory so `preprocessor_config.json` and
the remaining model assets are resolved deterministically.

With the local snapshot in place, the trainer reached `on_train_begin` but the
default VisualDL callback failed while serializing a boolean hyperparameter into
an integer protobuf field. The smoke config sets the officially supported
`report_to: none` value to disable external reporting. Console logs, evaluation,
trainer result files, and checkpoints remain enabled; VisualDL/TensorBoard
visualization is intentionally omitted from this compatibility smoke.

The failed VisualDL initialization left the original `full_sft_smoke` output
directory non-empty. Preserve it as setup-failure evidence rather than enabling
overwrite or deleting it. The versioned smoke config writes the first real
training attempt to the clean `full_sft_smoke_run1` directory.

That run completed its two optimizer steps, then failed before checkpoint save
when the trainer attempted scheduled evaluation. In ERNIEKit release/v1.5's
OCR-VL workflow, `eval_dataset` is hard-coded to `None`; the configured
validation JSONL is therefore not attached to the trainer. Evaluation is called
before checkpoint saving, so the in-memory updates from this failed process are
not a reusable artifact. The corrected smoke disables inline evaluation, writes
to the clean `full_sft_smoke_run2` directory, and validates the saved model in a
separate generation pass. Full-run checkpoint selection likewise uses separate
full-validation generation CER, matching the project protocol.

The corrected run2 completed both optimizer steps and exited with code 0. It
reported finite train loss 1.945588, runtime 48.539 seconds, throughput 1.3185
samples/s and 0.0412 optimizer steps/s, then saved the full model to
`checkpoints/paddleocr_vl/full_sft_smoke_run2`. The single-card asynchronous-save
warning is non-blocking because synchronous saving completed. Before declaring
the smoke fully passed, record peak memory from the step logs and verify a fresh
process can reload the saved artifact and generate one frozen validation line.

The first fresh-process attempt loaded all 608 fine-tuned weight entries, then
stopped before generation because ERNIEKit's HF export did not include
`chat_template.jinja`. This does not invalidate the weights. The verification
runner now loads the unchanged official base processor/chat template and the
fine-tuned model weights separately, avoiding mutation of the saved artifact.

The next inference attempt again loaded all weights, but the ERNIEKit-exported
config caused the vision encoder to return a tuple while Transformers expected a
model-output object. The loader now uses the official base runtime config with
`return_dict=True` for both the composite and vision configs, while continuing
to source every trained tensor from the saved artifact. It also preserves the
official `tie_word_embeddings=False`; no checkpoint file is edited.

Fresh-process reload and validation inference subsequently passed. The full run
therefore uses `configs/paddle/erniekit_full_3ep.yaml`: 300 optimizer steps,
approximately 100 steps per epoch, and checkpoints at steps 100, 200, and 300.
Running all three epochs in one job is safe for selection because every epoch
artifact is retained; the winner is chosen later by separate full-validation
generation CER. The test split remains sealed until that choice is frozen.

## Remaining order

1. Install ERNIEKit release/v1.5 under the verified PaddlePaddle 3.3.0/cu129
   constraint, then rerun an import/GPU check.
2. Convert only train/validation manifests to ERNIEKit multimodal JSONL.
3. Run finite-loss and tiny full-SFT smoke gates; measure VRAM on the RTX 5090.
4. Launch two epochs only if the smoke passes within 32 GB.
5. Select by full-validation generation CER; run epoch 3 only if justified.
6. Evaluate exactly one frozen checkpoint on test.
