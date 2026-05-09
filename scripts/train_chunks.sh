#!/bin/bash
GPU_NUM=$1
RESUME_FROM=$2

export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

if [ -z "$GPU_NUM" ]; then
    GPU_NUM=1
fi

if [ -z "$RESUME_FROM" ]; then
    RESUME_FROM=None
fi

python -m torch.distributed.launch --master_port=$((29500 + RANDOM % 100)) \
    --nproc_per_node=$GPU_NUM train.py \
    --config configs/pgocc_chunks.py \
    --override debug=True \
    resume_from=$RESUME_FROM
