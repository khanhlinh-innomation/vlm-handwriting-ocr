# Vintern-1B-v3.5 Divorce-Domain LoRA

Fine-tunes `5CD-AI/Vintern-1B-v3_5` on the private divorce-petition pages with
UIT-HWDB and MeddiesOCR replay, using the official Vintern recipe.

## Run it

```bash
# 1. compatibility gate: 2 optimizer steps on 8 rows
bash /workspace/vlm-handwriting/repo/scripts/train_vintern_divorce_lora.sh --smoke

# 2. the real run: 10 epochs
bash /workspace/vlm-handwriting/repo/scripts/train_vintern_divorce_lora.sh
```

Nothing else needs preparing. The environment, the training repo, the converted
dataset and the meta files are already on the server.

Checkpoints land in `checkpoints/vintern_1b_v3_5/divorce_lora_10epoch`. The
script refuses to start if that directory already exists, so a rerun cannot
silently overwrite a finished run.

### Checkpoints are saved at epochs 3, 7 and 10 only

InternVL writes the **whole model** at every checkpoint, not just the LoRA
adapter: about 2.0 GB each, against roughly 170 MB for the GLM runs under
LLaMA-Factory. Ten epoch checkpoints would need about 21 GB and the instance had
20 GB free, so an unmodified ten-epoch run fills the disk near the end and loses
everything. A two-step smoke does not expose this, because it saves once.

`save_total_limit` cannot express the fix — it keeps the *last* N, and with only
220 divorce pages the best checkpoint may well be an early one. Staged runs with
`--resume_from_checkpoint` would work but rebuild the cosine schedule at each
stage, so the run would no longer match one continuous schedule.

`scripts/patch_internvl_selective_epoch_save.py` adds a callback that suppresses
saving at any epoch not listed in `SAVE_EPOCHS`. It is registered after the
default flow callback, so it clears the save flag the default callback just set;
a callback-order test confirms epochs 3 and 7 save and the rest do not. The
training script sets `SAVE_EPOCHS=3,7`, and epoch 10 is the final `save_model()`
at the output root. Three artifacts, about 5.9 GB:

| Artifact | Epoch | Resume state |
|---|---|---|
| `checkpoint-84/` | 3 | yes |
| `checkpoint-196/` | 7 | yes |
| output root | 10 | model only |

Re-apply after any `git pull` of the training repo, alongside the FlashAttention
patch.

## Why not LLaMA-Factory

Vintern-1B-v3.5 ships as `InternVLChatModel` (`model_type: internvl_chat`), the
remote-code InternVL layout: InternViT-300M + Qwen2-0.5B, `template: Hermes-2`,
no `preprocessor_config.json`. LLaMA-Factory 0.9.5 refuses that layout outright:

```python
# tools/LlamaFactory/src/llamafactory/model/patcher.py:349
if isinstance(architectures, list) and "InternVLChatModel" in architectures:
    raise ValueError(
        "Please download the internvl models in a Hugging Face–compatible format "
        "(for example, https://huggingface.co/OpenGVLab/InternVL3-8B-hf)."
    )
```

Converting the weights to the `-hf` layout was rejected for round one: Vintern
carries a custom 151,674-token vocabulary, and a silent conversion mismatch
would be hard to detect from metrics alone. The model trains normally through
its own upstream recipe, which is what this phase uses.

## Frozen decisions

| Item | Value |
|---|---|
| Base model | `5CD-AI/Vintern-1B-v3_5` |
| Pinned revision | `fe7c963f86aa5bf017f01363cd63524765b6c41c` |
| Training repo | `tools/Vintern` — `5CD-AI/Vintern`, fork of `OpenGVLab/InternVL` |
| Entrypoint | `internvl_chat/internvl/train/internvl_chat_finetune.py` |
| Conversation style | `Hermes-2` |
| Method | LLM LoRA rank 16; vision tower, MLP projector and LLM base frozen |
| Learning rate | `2e-5`, cosine, warmup ratio 0.05 |
| Effective batch | 16 (1 per device x 16 accumulation, single GPU) |
| Precision | BF16 + TF32, gradient checkpointing on |
| Tiling | `max_dynamic_patch 6`, thumbnail on, `ps_version v2` |
| Sequence cutoff | `max_seq_length 4096` |
| Epochs | 10 (~28 optimizer steps per epoch) |

### conv_style must be Hermes-2

`config.json` declares `template: Hermes-2`, and `modeling_internvl_chat.chat()`
resolves the inference template from that field. Training with any other style
would teach the model a prompt format it is never served. `Hermes-2` is also
what the official Vintern fine-tuning notebook passes.

The OpenGVLab default `internvl2_5` is not merely different here — it is not
registered in this fork's `internvl/conversation.py` at all.

### Tiling note

`config.json` declares `max_dynamic_patch: 4`, while the official script, the
Vintern notebook and the existing benchmark all use 6. Training uses 6 so the
fine-tuned model matches the inference configuration already benchmarked. If the
tile count changes, change it in training and inference together.

## Environment

`/workspace/vlm-handwriting/venvs/vintern-train`. The `glm-train`,
`vintern-bench` and `vintern-isolated` environments are untouched;
`vintern-isolated` is broken (`ImportError: cannot import name
'EncoderDecoderCache'`) and unused by this phase.

| Package | Version |
|---|---|
| torch | 2.10.0+cu128 |
| transformers | 4.37.2 |
| peft | 0.10.0 |
| timm | 0.9.12 |
| deepspeed | 0.19.7 |

### FlashAttention is absent by necessity

There is no prebuilt FlashAttention wheel for torch 2.10 on x86_64 — the only
torch 2.10 asset is an aarch64/cu13 build — and the instance has no `nvcc`, so a
source build is unavailable. This is not a problem in itself: the checkpoint
already sets `vision_config.use_flash_attn: false`, and `modeling_intern_vit.py`
falls back to naive attention when the package is missing.

The blocker was an unrelated eager import. `internvl/patch/__init__.py` imports
two LLaMA-specific FlashAttention monkey patches at module level, so importing
anything from `internvl.patch` failed:

```
File "internvl/patch/llama2_flash_attn_monkey_patch.py", line 8
    from flash_attn import __version__ as flash_attn_version
ModuleNotFoundError: No module named 'flash_attn'
```

The fine-tuning entrypoint imports only `concat_pad_data_collator`,
`replace_llama_rmsnorm_with_fused_rmsnorm` and `replace_train_sampler` from that
package, and never calls either LLaMA patch. Vintern's language tower is Qwen2,
so the LLaMA patches are inapplicable regardless.

`scripts/patch_internvl_lazy_flash_attn.py` rewrites those two eager imports as
a PEP 562 module `__getattr__`. Both names still resolve on attribute access and
still raise `ImportError` there if FlashAttention is genuinely needed, so the
public API is unchanged. The script is idempotent, refuses to touch a file it
does not recognise, and keeps the original as `__init__.py.orig`.

Re-apply after any `git pull` of the training repo:

```bash
/workspace/vlm-handwriting/venvs/vintern-train/bin/python \
  scripts/patch_internvl_lazy_flash_attn.py
```

## Data

`scripts/prepare_vintern_training.py` converts the frozen GLM ShareGPT export to
InternVL chat JSONL plus meta files, writing to `data/internvl/divorce_ocr`.

| Source | Rows |
|---|---:|
| Divorce pages | 220 |
| MeddiesOCR replay | 126 |
| UIT-HWDB replay | 88 |
| **Train total** | **434** |
| Divorce validation | 11 |
| Divorce test (untouched) | 12 |

This reproduces the mixture the GLM divorce run used, which keeps the two models
comparable. `divorce_domain_train.json` is a pre-mixed file — 440 rows of which
only 220 are divorce pages, the rest replay baked in — so the converter asserts
that exact composition rather than assuming the file is single-domain.

### Mojibake filter

Six of the 132 Meddies replay labels carry broken VNI-era encoding (`COÂNG`,
`ÑAÀU`, `Ö`). They are dropped, leaving 126. Training on them would teach Vintern
to emit corrupted Vietnamese diacritics — the one thing it currently does well.
The removed image paths are listed in `preparation_summary.json`. Divorce and
UIT labels are clean.

### Prompt

Training uses the same Vietnamese instruction as the benchmark, so the model is
never asked a question it was not trained on:

```text
Hãy chép lại nguyên văn toàn bộ chữ nhìn thấy trong ảnh theo thứ tự đọc.
Giữ nguyên dấu tiếng Việt, chữ in, chữ viết tay và xuống dòng.
Chỉ trả về nội dung được chép, không giải thích và không thêm Markdown.
```

This differs from the GLM prompt `<image>Text Recognition:`. Each model family
keeps its own native prompt; the semantic task is identical.

### Sealed test split

The converter cross-checks divorce train page names against `val.csv` and
`test.csv` and raises on any overlap. `preparation_summary.json` records
`test_used_for_training: false` and `test_rows_untouched: 12`.

## Verification done so far

CPU-only, no GPU and no writes:

- The fine-tuning entrypoint imports cleanly after the lazy-import patch.
- `get_conv_template("Hermes-2")` resolves in this fork; the tokenizer loads
  from the pinned snapshot.
- Sequence lengths against the 4,096 cutoff, counting 7 tiles x 256 visual
  tokens plus the templated text:

  | Split | n | p50 | p90 | p99 | max | over 4,096 |
  |---|---:|---:|---:|---:|---:|---:|
  | train | 434 | 2,263 | 2,687 | 3,499 | 4,725 | 1 |
  | validation | 11 | 2,283 | 2,604 | 2,658 | 2,658 | 0 |

  The single over-length row is `ocr-000082`
  (`meddies_images/images/img_0022625.jpg`), a Meddies replay page rather than a
  divorce page, so the truncation does not touch the target domain. Raising the
  cutoff to accommodate one replay row is not worth the extra activation memory.

- A real training batch builds end to end from the smoke meta. For 8 rows the
  collator returns `input_ids (8, 2719)` and `pixel_values (56, 3, 448, 448)` —
  56 tiles is exactly 8 x (6 + thumbnail). `<IMG_CONTEXT>` placeholders number
  14,336, exactly 256 x 56, so the vision and text streams line up. Labels are
  83.9% masked and the prompt region is fully masked, so supervision lands only
  on the transcription target.
- LoRA wiring works under peft 0.10 / transformers 4.37.2: `wrap_llm_lora(r=16,
  lora_alpha=32)` yields 336 trainable tensors, all of them LoRA, 8,798,208 of
  946,991,232 parameters (0.93%). The vision tower and `mlp1` are frozen.

- One forward and backward pass completes on a real batch. Loss is 0.2522 and
  finite; all 336 LoRA tensors receive a gradient.

  Exactly 168 of those 336 gradients are nonzero, and the sampled `lora_A`
  gradient norm is 0. This is LoRA at initialisation, not a wiring fault: peft
  initialises `lora_B` to zeros, so the adapter output `B @ A @ x` is zero and
  `dL/dA` is proportional to `B`, hence zero on the first step. The nonzero half
  is the `lora_B` matrices, which is what moves first. From step 2 onward both
  halves receive gradient.

The smoke gate then passed on the GPU: 2/2 optimizer steps, train loss 0.1392,
42.3 s runtime, adapter and tokenizer written. Two startup faults surfaced there
and are fixed in the script — `LAUNCHER` defaults to `slurm` in
`internvl_chat_finetune.py:607` and must be set to `pytorch`, and
`--use_fast_tokenizer` is not an accepted argument in this fork, which hardcodes
`use_fast=False` when building the tokenizer.

**Not measured:** peak VRAM. `skip_memory_metrics` is on, so the trainer does not
report it; watch `nvidia-smi` during the run if the number is wanted.

## After training

Rank the ten epoch checkpoints on the 11-row divorce validation split by
generated CER — never on training loss, never on the sealed 12-page test split.
Record repetition-collapse rate, generated-token count and EOS status alongside
CER: base Vintern's failures on these pages were three refusals to transcribe
and one unterminated repetition at the 2,048-token cap, and corpus CER alone
does not distinguish those from ordinary recognition error.

`tools/merge_lora.py <input_path> <output_path>` folds the adapter into the base
weights once a checkpoint is selected.

## What this phase is trying to fix

The base Vintern audit over the 12 test pages found 4 severe failures out of 12:
three refusals ("Tôi không thể đọc…") and one unterminated repetition. The five
clean pages scored CER 0.116 to 0.290. These are instruction-following failures,
not recognition failures. Training on exact-transcription targets is aimed
squarely at the refusals; the repetition mode still needs inference-side
handling, as it did for GLM.

## Alternative mixture

`data/internvl/vintern_divorce_mixture/` holds a larger three-way export —
220 divorce pages repeated 16x, plus 3,500 UIT lines and 3,500 Meddies pages,
for 10,520 exposures per epoch. It is not wired into the training script. It
trades comparability with the GLM run for stronger replay, and its Meddies
sample has not been mojibake-filtered.
