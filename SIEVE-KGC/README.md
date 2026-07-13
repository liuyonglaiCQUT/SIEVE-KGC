# SIEVE-KGC

A clean implementation of **SIEVE-KGC: Structural Evidence-Enhanced Query Encoding for Knowledge Graph Completion**.

This release keeps only the code required to train and evaluate the main SIEVE-KGC model.
Visualization scripts, ablation-only utilities, efficiency scripts, and external baseline wrappers are not included.
SIEVE is always enabled inside the model; there is no command-line switch for disabling the structural evidence branch.

## Expected data format

Each dataset directory should contain:

```text
train.json
valid.json
test.json
entities.json
```

Triples use JSON objects with fields such as `head_id`, `head`, `relation`, `tail_id`, and `tail`.
Entities use JSON objects with `entity_id`, `entity`, and optionally `entity_desc`.

## Train on WN18RR

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python main.py \
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
```

## Evaluate

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --task WN18RR \
  --data_dir data/WN18RR \
  --pretrained_model ./bert-base-uncased \
  --checkpoint checkpoint/wn18rr_sieve/model_best.mdl \
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
```

### Acknowledgement
Many thanks to previous works:
- [SimKGC](https://github.com/intfloat/SimKGC)
- [RAA-KGC](https://github.com/DuanyangYuan/RAA-KGC)

## Files

- `main.py`: training entry point.
- `evaluate.py`: filtered full-entity evaluation.
- `models.py`: SIEVE-KGC bi-encoder.
- `sieve.py`: structural evidence retrieval.
- `trainer.py`, `triplet.py`, `metric.py`, `dict_hub.py`, `doc.py`, `utils.py`, `triplet_mask.py`: core training and data utilities.
- `preprocess.py`: optional converter for raw WN18RR/FB15k-237 style files.
