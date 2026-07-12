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

All commands below enable offline Hugging Face loading:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

If you use a different CUDA/PyTorch build, verify that it is compatible with your GPU driver. Exact numerical results may also vary slightly across hardware and library versions.

## Repository layout

```text
SIEVE-KGC/
├── main.py                  # transductive training
├── evaluate.py              # filtered full-entity evaluation
├── config.py                # command-line arguments
├── models.py                # text encoders and scoring model
├── sieve.py                 # structural evidence retrieval
├── trainer.py               # optimization and validation
├── data/
│   ├── WN18RR/
│   ├── FB15k237/
│   ├── Wikidata5M-Trans/
│   ├── WN18RR_ind_v1/ ... WN18RR_ind_v4/
│   └── FB15k237_ind_v1/ ... FB15k237_ind_v4/
├── checkpoint/
└── bert-base-uncased/
```

## Data

We evaluate on the following transductive benchmarks:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| WN18RR | 86,835 | 3,034 | 3,134 |
| FB15k-237 | 272,115 | 17,535 | 20,466 |
| Wikidata5M-Trans | 20,614,279 | 5,163 | 5,163 |

We additionally use the standard `v1`--`v4` inductive splits of WN18RR and FB15k-237. Please obtain the datasets from their original distributions and comply with their respective licenses. This repository need not redistribute third-party raw data.

### Transductive JSON format

Each transductive directory should contain:

```text
data/<dataset>/
├── train.json
├── valid.json
├── test.json
├── entities.json
└── relations.json
```

Triple files contain JSON records with head, relation, tail, and the corresponding textual fields. `entities.json` contains entity identifiers, names, and descriptions. The loaders in `triplet.py` and `dict_hub.py` define the exact accepted keys.

### Inductive protocol

For an inductive split, the source graph and target support graph must remain separate. Model parameters are learned on the source graph and then frozen. At test time, SIEVE may construct a **non-parametric evidence index from the observed target support triples only**. Target validation/test query labels must never be inserted into that index.

A prepared inductive directory may contain:

```text
entities_all.json  entities_tv.json  entities_ind.json
train.json         valid.json        test.json
ind_train.json     ind_valid.json
test_trans.json    test_ind.json
```

The exact file mapping must match the command used for a particular split. Do not treat `test_ind.json` or any answer-bearing test file as an observed support graph.

## Training

Run all commands from the repository root. Checkpoints are selected by validation MRR and written to the specified `output_dir`.

### WN18RR

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
  --learning_rate 1e-5 \
  --weight_decay 1e-4 \
  --epochs 160 \
  --warmup_ratio 0.02 \
  --use_amp true \
  --pre_batch 0 \
  --use_self_negative true \
  --bidirectional_inbatch_loss true \
  --use_sieve true \
  --sieve_train true \
  --sieve_eval true \
  --sieve_num_anchors 4 \
  --sieve_anchor_pool 12 \
  --sieve_random_train_anchors false \
  --sieve_aux_weight 0.2 \
  --sieve_eval_weight 1.0 \
  --sieve_path_weight 1.0 \
  --sieve_nei_weight 0.3 \
  --sieve_max_degree 24 \
  --sieve_max_paths_per_pair 6 \
  --sieve_max_path_vocab 8192 \
  --sieve_max_pair_index 2000000 \
  --sieve_skip_backtrack true \
  --num_workers 0 \
  --eval_every 1 \
  --early_stop_patience 30 \
  --seed 42 \
  --output_dir checkpoint/wn18rr_sieve_e160_lr1e5_a4
```

### FB15k-237

The command-line task name and directory name used by this code are `custom` and `FB15k237`, respectively; the benchmark is written as **FB15k-237** in the paper.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python main.py \
  --task custom \
  --data_dir data/FB15k237 \
  --pretrained_model ./bert-base-uncased \
  --pooling mean \
  --max_num_tokens 32 \
  --batch_size 64 \
  --gradient_accumulation_steps 8 \
  --eval_batch_size 64 \
  --entity_batch_size 512 \
  --learning_rate 1e-5 \
  --weight_decay 1e-4 \
  --epochs 20 \
  --warmup_ratio 0.02 \
  --grad_clip 10.0 \
  --use_amp true \
  --amp_init_scale 1024 \
  --t 0.05 \
  --finetune_t true \
  --additive_margin 0.02 \
  --use_self_negative true \
  --bidirectional_inbatch_loss true \
  --pre_batch 16 \
  --pre_batch_weight 0.5 \
  --use_sieve true \
  --sieve_train true \
  --sieve_eval true \
  --sieve_aux_weight 0.3 \
  --sieve_eval_weight 1.0 \
  --sieve_num_anchors 4 \
  --sieve_anchor_pool 12 \
  --sieve_random_train_anchors true \
  --sieve_path_weight 1.0 \
  --sieve_nei_weight 0.30 \
  --sieve_max_degree 24 \
  --sieve_max_paths_per_pair 6 \
  --sieve_max_path_vocab 8192 \
  --sieve_max_pair_index 2000000 \
  --sieve_skip_backtrack true \
  --eval_every 1 \
  --early_stop_patience 5 \
  --num_workers 0 \
  --seed 42 \
  --print_freq 100 \
  --output_dir checkpoint/fb15k237_sieve
```

If the above batch configuration exceeds the available memory, reduce `batch_size` and increase `gradient_accumulation_steps` so that the effective batch size remains comparable. Such a change is a memory workaround, not a strictly identical run.

## Evaluation

Evaluation performs filtered full-entity ranking and reports MRR and Hits@1/3/10. Use the best checkpoint selected on the validation set. Do not tune fusion weights on the test set.

### WN18RR

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --checkpoint checkpoint/wn18rr_sieve_e160_lr1e5_a4/model_best.mdl \
  --task WN18RR \
  --data_dir data/WN18RR \
  --test_path data/WN18RR/test.json \
  --entities_path data/WN18RR/entities.json \
  --pretrained_model ./bert-base-uncased \
  --eval_batch_size 32 \
  --entity_batch_size 256 \
  --use_sieve true \
  --sieve_eval true \
  --sieve_eval_weight 1.0 \
  --sieve_num_anchors 4 \
  --sieve_anchor_pool 12 \
  --sieve_path_weight 1.0 \
  --sieve_nei_weight 0.3 \
  --sieve_max_degree 24 \
  --sieve_max_paths_per_pair 6 \
  --sieve_max_path_vocab 8192 \
  --sieve_max_pair_index 2000000 \
  --sieve_skip_backtrack true \
  --filter_known true \
  --num_workers 0
```

### FB15k-237

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --checkpoint checkpoint/fb15k237_sieve/model_best.mdl \
  --task custom \
  --data_dir data/FB15k237 \
  --test_path data/FB15k237/test.json \
  --entities_path data/FB15k237/entities.json \
  --pretrained_model ./bert-base-uncased \
  --eval_batch_size 64 \
  --entity_batch_size 512 \
  --use_sieve true \
  --sieve_eval true \
  --sieve_eval_weight 1.0 \
  --sieve_num_anchors 4 \
  --sieve_anchor_pool 12 \
  --sieve_path_weight 1.0 \
  --sieve_nei_weight 0.30 \
  --sieve_max_degree 24 \
  --sieve_max_paths_per_pair 6 \
  --sieve_max_path_vocab 8192 \
  --sieve_max_pair_index 2000000 \
  --sieve_skip_backtrack true \
  --filter_known true \
  --num_workers 0
```

`evaluate.py` restores model arguments from the checkpoint and then applies explicitly supplied evaluation/path overrides. Nevertheless, keeping the structural settings in the command makes the run auditable. If a validation-selected `sieve_eval_weight` differs from `1.0`, use that recorded value consistently for the corresponding test run.

## Wikidata5M-Trans

Wikidata5M-Trans contains more than 20 million training triples. The paper experiment was run on an **NVIDIA A800** with a separate memory-aware training pipeline and streaming full-entity evaluator. It should not be replaced by the compact `eval_wiki5m_trans.py` placeholder: that file exits deliberately and does not reproduce the paper experiment.

The following **confirmed evaluation settings** were used by the large-scale streaming run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python evaluate_stream.py \
  --checkpoint checkpoint/wikidata5m_lazy_sieve_final/model_last.mdl \
  --task custom \
  --data_dir data/Wikidata5M-Trans \
  --pretrained_model ./bert-base-uncased \
  --pooling mean \
  --max_num_tokens 32 \
  --eval_batch_size 16 \
  --entity_batch_size 32768 \
  --use_sieve true \
  --sieve_eval true \
  --sieve_eval_weight 0.3 \
  --sieve_num_anchors 1 \
  --sieve_anchor_pool 3 \
  --sieve_path_weight 0.5 \
  --sieve_nei_weight 0.1 \
  --sieve_max_degree 4 \
  --sieve_max_paths_per_pair 1 \
  --sieve_max_pair_index 200000 \
  --sieve_verbose true \
  --filter_known true \
  --num_workers 0
```

Before making a reproducibility release, copy the exact historical streaming implementation into the repository as `evaluate_stream.py` and verify this command end to end. The exact historical **Wikidata5M-Trans training launch command has not yet been recovered from the available files**, so it is intentionally not reconstructed here. Publish it only after checking the original shell history, training log, or checkpoint-saved arguments. This prevents an unverified command from being presented as the one that produced the paper result.

## Expected main results

The paper reports filtered link-prediction results in percentage points:

| Dataset | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|
| WN18RR | 62.36 | 53.86 | 67.57 | 77.30 |
| FB15k-237 | 31.78 | 22.99 | 34.37 | 49.62 |
| Wikidata5M-Trans | 32.35 | 28.10 | 32.92 | 45.67 |

Small deviations can arise from stochastic optimization and software/hardware differences. Large discrepancies should first be checked against data preprocessing, checkpoint selection, filtered evaluation, and the validation-selected fusion weight.

## Reproducibility checklist

- Use the exact dataset split and textual descriptions stated in the paper.
- Keep `bert-base-uncased` fixed across compared text-based methods.
- Select checkpoints and hyperparameters using validation data only.
- Use filtered full-entity ranking for final evaluation.
- Set and report the random seed.
- Record the complete command, Git commit, package versions, GPU model, and output log.
- For inductive evaluation, keep target query labels outside the support/evidence graph.
- For Wikidata5M-Trans, use the verified streaming implementation rather than the compact placeholder.

## Citation

The citation will be added after the paper receives verifiable public bibliographic metadata. Please do not create or circulate a fictitious journal citation for a manuscript that has not yet been published.

## Acknowledgements

This work was supported by the Major Project of the Science and Technology Research Program of Chongqing Municipal Education Commission (Grant No. KJZD-M202400901).

## License

Add a `LICENSE` file before public release and state the selected license here. The license for this code does not override the licenses of BERT, the benchmark datasets, or other third-party resources.
