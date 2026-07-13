#!/usr/bin/env bash
set -e
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python main.py \
  --task WN18RR \
  --data_dir data/WN18RR \
  --pretrained_model ./bert-base-uncased \
  --pooling mean \
  --max_num_tokens 50 \
  --batch_size 32 \
  --eval_batch_size 32 \
  --entity_batch_size 256 \
  --learning_rate 2e-5 \
  --weight_decay 1e-4 \
  --epochs 120 \
  --warmup_ratio 0.015 \
  --use_amp true \
  --pre_batch 0 \
  --use_self_negative true \
  --bidirectional_inbatch_loss true \
  --sieve_num_anchors 4 \
  --sieve_anchor_pool 12 \
  --sieve_random_train_anchors false \
  --sieve_aux_weight 0.2 \
  --sieve_eval_weight 1.0 \
  --sieve_path_weight 1.0 \
  --sieve_nei_weight 0.3 \
  --num_workers 0 \
  --eval_every 1 \
  --early_stop_patience 25 \
  --output_dir checkpoint/wn18rr_sieve
