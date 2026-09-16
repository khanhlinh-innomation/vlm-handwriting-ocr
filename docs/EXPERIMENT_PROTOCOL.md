# Experiment Protocol

## Task and data

Input one handwritten Vietnamese line image and return its exact transcription without explanation or spelling correction.

Use UIT-HWDB-line only in round 1. The frozen split contains 7,229 labeled samples: 6,346 train from 224 writers, 682 validation from 25 writers, and 201 test from 6 writers. Writer sets must remain disjoint. Forty-four raw images without matching labels are excluded.

## Evaluation

Normalize reference and prediction with `unicodedata.normalize("NFC", text)` only. Report corpus CER, corpus WER, exact-line accuracy, latency, peak GPU VRAM, runtime, and throughput.

Do not lowercase, strip punctuation, collapse meaningful spaces, remove diacritics, or spell-correct.

## Execution gates

For every base model:

```text
one validation sample -> fixed validation smoke20
-> freeze inference configuration -> one frozen test baseline
```

The one-sample and smoke20 runs are debugging gates, not final baseline scores. A full
validation benchmark is not required for an unchanged base model. Validation becomes
mandatory during fine-tuning for checkpoint and configuration selection.

For every fine-tuning method:

```text
forward/loss smoke -> short training smoke -> save -> reload
-> inference -> full training
```

Select the best checkpoint using validation CER only. Run the frozen test after the configuration and checkpoint are fixed.

## Model order

1. GLM-OCR: LLaMA-Factory LoRA.
2. TeleOCR: experimental Transformers/PEFT LoRA.
3. PaddleOCR-VL-1.6: ERNIEKit full SFT.
