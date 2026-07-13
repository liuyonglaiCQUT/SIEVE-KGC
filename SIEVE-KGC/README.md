# SIEVE-KGC

Official implementation of **SIEVE-KGC: Structural Evidence Enhanced Query Encoding for Knowledge Graph Completion**.

SIEVE-KGC augments a text-based bi-encoder with query-conditioned structural evidence retrieved from the observed graph. It combines relation-compatible path closure and relation-conditioned neighborhood consistency, encodes the selected evidence on the query side, and preserves full-entity ranking with cached candidate representations.

> **Reproducibility note.** The commands below record the settings used for the reported experiments whenever they have been verified. Paths are relative to the repository root. The large-scale Wikidata5M-Trans experiment used a separate streaming pipeline on an NVIDIA A800; see [Wikidata5M-Trans](#wikidata5m-trans) before attempting to reproduce it.

## Requirements

The code was run on Linux with Python, PyTorch, and Hugging Face Transformers.

Hardware used in the paper:

- **NVIDIA GeForce RTX 4090:** WN18RR, FB15k-237, the inductive splits, and the compatibility/efficiency experiments.
- **NVIDIA A800:** Wikidata5M-Trans, including large-scale full-entity evaluation.

We recommend creating an isolated environment and installing the locked dependencies supplied with this repository:

```bash
conda create -n sieve-kgc python=3.8 -y
conda activate sieve-kgc
pip install -r requirements.txt
```

Download `bert-base-uncased` in advance and place it at:

```text
./bert-base-uncased/
```

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
