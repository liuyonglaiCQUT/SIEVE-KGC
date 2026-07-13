#!/usr/bin/env bash
set -e
CKPT=${1:-checkpoint/wn18rr_sieve/model_best.mdl}
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python evaluate.py \
  --task WN18RR \
  --data_dir data/WN18RR \
  --pretrained_model ./bert-base-uncased \
  --checkpoint "$CKPT" \
  --pooling mean \
  --max_num_tokens 50 \
  --eval_batch_size 32 \
  --entity_batch_size 256 \
  --sieve_num_anchors 4 \
  --sieve_anchor_pool 12 \
  --sieve_eval_weight 1.0 \
  --sieve_path_weight 1.0 \
  --sieve_nei_weight 0.3 \
  --num_workers 0 \
  --filter_known true
