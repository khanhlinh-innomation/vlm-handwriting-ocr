#!/usr/bin/env bash
# Vintern-1B-v3.5 LoRA fine-tuning on the divorce mixture.
#
# One command. Run the smoke gate first:
#   bash scripts/train_vintern_divorce_lora.sh --smoke
# then the real run:
#   bash scripts/train_vintern_divorce_lora.sh
#
# Recipe source: 5CD-AI/Vintern (fork of OpenGVLab/InternVL), the training repo
# named by the official Vintern fine-tuning notebook.
#
#   conv_style Hermes-2   matches the checkpoint's own config.template, which is
#                         what modeling_internvl_chat.chat() uses at inference.
#                         Training on any other template would teach the model a
#                         prompt format it is never served.
#   max_seq_length 4096   full A4 transcriptions: 7 tiles x 256 visual tokens
#                         plus targets reaching roughly 2.5k tokens.
#   save_strategy epoch   every epoch is retained so the checkpoint is chosen by
#                         validation CER, never by training loss.

set -euo pipefail

ROOT=/workspace/vlm-handwriting
INTERNVL_ROOT="$ROOT/tools/Vintern/internvl_chat"
PYTHON="$ROOT/venvs/vintern-train/bin/python"
MODEL="$ROOT/cache/huggingface/hub/models--5CD-AI--Vintern-1B-v3_5/snapshots/fe7c963f86aa5bf017f01363cd63524765b6c41c"
DATA="$ROOT/data/internvl/divorce_ocr"

SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
fi

if [[ "$SMOKE" == "1" ]]; then
  META="$DATA/meta_smoke.json"
  OUTPUT="$ROOT/checkpoints/vintern_1b_v3_5/divorce_lora_smoke"
  LOG="$ROOT/logs/vintern-divorce-lora-smoke.log"
  EPOCH_ARGS=(--max_steps 2 --save_steps 2 --save_strategy steps --save_total_limit 1)
  rm -rf "$OUTPUT"
else
  META="$DATA/meta_train.json"
  OUTPUT="$ROOT/checkpoints/vintern_1b_v3_5/divorce_lora_10epoch"
  LOG="$ROOT/logs/vintern-divorce-lora-10epoch.log"
  EPOCH_ARGS=(--num_train_epochs 10 --save_strategy epoch --save_total_limit 10)
  if [[ -e "$OUTPUT" ]]; then
    echo "Refusing to overwrite existing output: $OUTPUT" >&2
    exit 2
  fi
fi

test -x "$PYTHON"
test -d "$MODEL"
test -f "$META"
test -f "$INTERNVL_ROOT/internvl/train/internvl_chat_finetune.py"

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG")"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME="$ROOT/cache/huggingface"
# internvl_chat_finetune.py reads os.environ.get('LAUNCHER', 'slurm'); without
# this it takes the SLURM path and dies on KeyError: 'SLURM_PROCID'.
export LAUNCHER=pytorch
export PYTHONPATH="$INTERNVL_ROOT"
export TOKENIZERS_PARALLELISM=true
export TF_CPP_MIN_LOG_LEVEL=3
cd "$INTERNVL_ROOT"

"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "$MODEL" \
  --conv_style Hermes-2 \
  --output_dir "$OUTPUT" \
  --meta_path "$META" \
  --overwrite_output_dir False \
  --force_image_size 448 \
  --min_dynamic_patch 1 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 2 \
  --bf16 True \
  --tf32 True \
  "${EPOCH_ARGS[@]}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --evaluation_strategy no \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version v2 \
  --seed 42 \
  --data_seed 42 \
  --remove_unused_columns False \
  --report_to none \
  2>&1 | tee "$LOG"
