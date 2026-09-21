# GLM-OCR Printed-Page Adaptation with MeddiesOCR

This phase continues the selected Vietnamese-handwriting LoRA adapter on full-page
printed documents while replaying the frozen UIT-HWDB training set. MeddiesOCR is
weak supervision derived from PDF text; no manual MeddiesOCR audit is required.

## Frozen decisions

- Base model: `zai-org/GLM-OCR`
- Starting adapter: `checkpoint-1191` from the completed UIT-HWDB GLM run
- MeddiesOCR source: `Leo1903/meddiesOCR`
- Pinned revision: `584aa75cac164df05fa999453432d18ad3783f5c`
- Split: document-disjoint 80/10/10 with seed 42
- Prompt: `<image>Text Recognition:`
- LoRA: rank 8, target `all`, continue the existing adapter
- Learning rate: `3e-5`
- Epochs: 3
- Initial mixture: 80% MeddiesOCR / 20% UIT-HWDB interleave probability. With
  33,548 printed train pages and 6,346 handwriting lines, this exhausts MeddiesOCR
  while replaying UIT-HWDB about 1.3 times per epoch; 50/50 would over-repeat every
  handwriting line more than five times per epoch.
- Initial cutoff: 4096; the preparation report records target-only truncation risk
- The UIT-HWDB and MeddiesOCR test splits remain unavailable to training

## Verified local preparation result

The pinned source contains 41,986 annotation/image rows from 5,498 documents.
Deterministic preparation accepted 41,984 rows from 5,496 documents and rejected
two rows: one 151 MP image above the 100 MP safety limit and one label containing
the Unicode replacement character. No manual audit was performed.

| Split | Documents | Pages |
|---|---:|---:|
| Train | 4,396 | 33,548 |
| Validation | 550 | 4,110 |
| Test | 550 | 4,326 |

Document overlap is zero. Exact GLM tokenizer profiling of target text produced
p50 682, p90 1,242, p99 2,460, and maximum 10,340 tokens. There are 780 targets
above 2,048 tokens, 25 above 4,096, and one above 8,192. These counts exclude
visual and prompt tokens, so the smoke run must still measure actual preprocessing
truncation and memory at cutoff 4,096.

## Local/repository phase (steps 1-7)

Install the data utilities without replacing the verified server Torch build:

```bash
uv pip install --python /workspace/vlm-handwriting/venvs/glm-train/bin/python \
  --no-deps -e .
uv pip install --python /workspace/vlm-handwriting/venvs/glm-train/bin/python \
  "huggingface_hub>=1.0" "Pillow>=10.0" "tokenizers>=0.20"
```

Download the pinned public dataset outside Git:

```bash
/workspace/vlm-handwriting/venvs/glm-train/bin/python \
  scripts/download_meddies.py \
  --output-dir /workspace/vlm-handwriting/data/raw/meddiesOCR
```

Download the exact tokenizer artifact used for target-length profiling:

```bash
hf download zai-org/GLM-OCR tokenizer.json \
  --local-dir /workspace/vlm-handwriting/cache/glm-ocr-tokenizer
```

Prepare images, deterministic rejection records, the document-disjoint split,
ShareGPT JSON, smoke subsets, checksums, and length statistics:

```bash
/workspace/vlm-handwriting/venvs/glm-train/bin/python \
  scripts/prepare_meddies_glm_training.py \
  --dataset-root /workspace/vlm-handwriting/data/raw/meddiesOCR \
  --output-dir /workspace/vlm-handwriting/data/llamafactory/glm_ocr_meddies \
  --tokenizer-json /workspace/vlm-handwriting/cache/glm-ocr-tokenizer/tokenizer.json
```

Add the already prepared frozen UIT-HWDB replay data. This copies only JSON files
and creates an image symlink; it never exports the handwriting test set:

```bash
/workspace/vlm-handwriting/venvs/glm-train/bin/python \
  scripts/assemble_glm_meddies_mixture.py \
  --meddies-dir /workspace/vlm-handwriting/data/llamafactory/glm_ocr_meddies \
  --handwriting-dir /workspace/vlm-handwriting/data/llamafactory/glm_ocr
```

Required gates before inference or training:

1. `preparation_summary.json` records the pinned source revision.
2. Train, validation, and test `doc_id` overlap is zero.
3. Every accepted image passes Pillow verification.
4. `dataset_info.json` contains both MeddiesOCR and UIT-HWDB entries.
5. No dataset entry containing `test` is present in either training config.
6. The selected cutoff and any target truncation count are recorded.

## Server/GPU phase (step 8 onward)

Step 8 first evaluates the unchanged `checkpoint-1191` on MeddiesOCR validation.
Do not train until this B0 artifact exists. Then run the two-step continuation smoke:

```bash
/workspace/vlm-handwriting/venvs/glm-train/bin/llamafactory-cli train \
  configs/glm/meddies_continued_lora_smoke.yaml
```

The smoke must produce finite loss and pass adapter save, fresh-process reload,
and one full-page inference. After the smoke gate, launch the three-epoch run:

```bash
/workspace/vlm-handwriting/venvs/glm-train/bin/llamafactory-cli train \
  configs/glm/meddies_continued_lora.yaml
```

Rank every epoch checkpoint separately on:

- MeddiesOCR printed-page validation: micro/macro/median CER, WER, omissions,
  repetition collapse, generated-token count, and EOS status.
- Frozen UIT-HWDB handwriting validation: strict CER, WER, and exact-line rate.

Freeze one checkpoint before opening the MeddiesOCR test split or the four-page
divorce-form pilot. Do not use the previously opened handwriting test set for new
checkpoint or hyperparameter selection.

## Parallel divorce-form labeling lane

While public-data training runs, label private forms independently. Keep label status
explicit as `ai_draft`, `needs_review`, or `human_verified`; group all pages from one
PDF/case into one future split. Fully verify validation and test before using the
remaining drafts as training data for the later in-domain phase.
