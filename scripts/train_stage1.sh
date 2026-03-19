#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
TRAIN_RUN="${REPO_ROOT}/src/tabicl/train/run.py"
PRIOR_GENLOAD="${REPO_ROOT}/src/tabicl/prior/genload.py"

require_env_dir() {
    var_name="$1"
    description="$2"
    eval "value=\${$var_name:-}"

    if [ -z "$value" ]; then
        printf '%s\n' "$description" >&2
        exit 1
    fi
}

WANDB_DIR="${WANDB_DIR:-}"
STAGE1_CHECKPOINT_DIR="${STAGE1_CHECKPOINT_DIR:-}"
STAGE1_PRIOR_DIR="${STAGE1_PRIOR_DIR:-}"

require_env_dir WANDB_DIR "Please set WANDB_DIR to a writable directory."
require_env_dir STAGE1_CHECKPOINT_DIR "Please set STAGE1_CHECKPOINT_DIR to a writable directory."
require_env_dir STAGE1_PRIOR_DIR "Please set STAGE1_PRIOR_DIR to a writable directory."

mkdir -p "$WANDB_DIR" "$STAGE1_CHECKPOINT_DIR" "$STAGE1_PRIOR_DIR"

# This script is used to train TabICL for the first stage of the curriculum learning

# ----------------------------------
# Generate prior datasets on the fly
# ----------------------------------

torchrun --standalone --nproc_per_node=8 "${TRAIN_RUN}" \
    --wandb_log True \
    --wandb_project TabICL \
    --wandb_name Stage1 \
    --wandb_dir "$WANDB_DIR" \
    --wandb_mode online \
    --device cuda \
    --dtype float32 \
    --np_seed 43 \
    --torch_seed 42 \
    --max_steps 160000 \
    --batch_size 512 \
    --micro_batch_size 4 \
    --lr 5e-5 \
    --scheduler cosine_warmup \
    --warmup_proportion 0.05 \
    --gradient_clipping 1.0 \
    --prior_type mix_scm \
    --prior_device cpu \
    --batch_size_per_gp 4 \
    --min_features 2 \
    --max_features 100 \
    --max_classes 10 \
    --max_seq_len 2048 \
    --min_train_size 0.1 \
    --max_train_size 0.9 \
    --embed_dim 128 \
    --col_num_blocks 3 \
    --col_nhead 4 \
    --col_num_inds 128 \
    --row_num_blocks 3 \
    --row_nhead 8 \
    --row_num_cls 4 \
    --row_rope_base 100000 \
    --icl_num_blocks 12 \
    --icl_nhead 4 \
    --ff_factor 2 \
    --norm_first True \
    --checkpoint_dir "$STAGE1_CHECKPOINT_DIR" \
    --save_temp_every 50 \
    --save_perm_every 5000
    --only_load_model True \

# # ------------------------------------------------------
# # Save prior datasets to disk and load them for training
# # ------------------------------------------------------

# # Saving to disk
# python "${PRIOR_GENLOAD}" \
#     --save_dir "$STAGE1_PRIOR_DIR" \
#     --np_seed 43 \
#     --torch_seed 42 \
#     --num_batches 100000 \
#     --resume_from 0 \
#     --batch_size 512 \
#     --batch_size_per_gp 4 \
#     --prior_type mix_scm \
#     --min_features 2 \
#     --max_features 100 \
#     --max_classes 10 \
#     --max_seq_len 2048 \
#     --min_train_size 0.1 \
#     --max_train_size 0.9 \
#     --n_jobs -1 \
#     --num_threads_per_generate 1 \
#     --device cpu

# # Loading from disk and training
# torchrun --standalone --nproc_per_node=8 "${TRAIN_RUN}" \
#     --wandb_log True \
#     --wandb_project TabICL \
#     --wandb_name Stage1 \
#     --wandb_dir "$WANDB_DIR" \
#     --wandb_mode online \
#     --device cuda \
#     --dtype float32 \
#     --np_seed 42 \
#     --torch_seed 42 \
#     --max_steps 100000 \
#     --batch_size 512 \
#     --micro_batch_size 4 \
#     --lr 1e-4 \
#     --scheduler cosine_warmup \
#     --warmup_proportion 0.02 \
#     --gradient_clipping 1.0 \
#     --prior_dir "$STAGE1_PRIOR_DIR" \
#     --load_prior_start 0 \
#     --delete_after_load False \
#     --prior_device cpu \
#     --embed_dim 128 \
#     --col_num_blocks 3 \
#     --col_nhead 4 \
#     --col_num_inds 128 \
#     --row_num_blocks 3 \
#     --row_nhead 8 \
#     --row_num_cls 4 \
#     --row_rope_base 100000 \
#     --icl_num_blocks 12 \
#     --icl_nhead 4 \
#     --ff_factor 2 \
#     --norm_first True \
#     --checkpoint_dir "$STAGE1_CHECKPOINT_DIR" \
#     --save_temp_every 50 \
#     --save_perm_every 5000
