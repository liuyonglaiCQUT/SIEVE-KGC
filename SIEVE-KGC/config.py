import argparse
import os
from typing import Optional


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Invalid boolean value: {}".format(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("SIEVE-KGC")

    # Data
    parser.add_argument("--task", default="WN18RR", choices=["WN18RR", "FB15k237", "custom"])
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--train_path", default="")
    parser.add_argument("--valid_path", default="")
    parser.add_argument("--test_path", default="")
    parser.add_argument("--entities_path", default="")
    parser.add_argument("--max_num_tokens", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)

    # Encoder and contrastive learning.
    parser.add_argument("--pretrained_model", default="bert-base-uncased")
    parser.add_argument("--pooling", default="mean", choices=["cls", "mean", "max"])
    parser.add_argument("--t", type=float, default=0.05)
    parser.add_argument("--finetune_t", type=str2bool, default=True)
    parser.add_argument("--additive_margin", type=float, default=0.02)
    parser.add_argument("--pre_batch", type=int, default=0)
    parser.add_argument("--pre_batch_weight", type=float, default=0.5)
    parser.add_argument("--use_self_negative", type=str2bool, default=True)
    parser.add_argument(
        "--bidirectional_inbatch_loss",
        type=str2bool,
        default=True,
        help="Use both query-to-tail and tail-to-query in-batch contrastive loss.",
    )

    # Optimization
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--entity_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_amp", type=str2bool, default=True)
    parser.add_argument("--amp_init_scale", type=float, default=1024.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_freq", type=int, default=100)   # 50
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=4)
    parser.add_argument("--output_dir", default="checkpoint/sieve_kgc")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda")

    # Evaluation
    parser.add_argument("--filter_known", type=str2bool, default=True)
    parser.add_argument("--save_predictions", type=str2bool, default=False)
    parser.add_argument("--prediction_path", default="predictions.json")
    # SIEVE structural evidence branch.
    # The released model always uses SIEVE; no command-line switch is exposed to disable it.
    parser.add_argument(
        "--sieve_aux_weight",
        type=float,
        default=0.20,
        help="Weight of the SIEVE-enhanced query contrastive loss.",
    )
    parser.add_argument(
        "--sieve_eval_weight",
        type=float,
        default=1.0,
        help="Score fusion: base_score + weight * sieve_query_score.",
    )
    parser.add_argument("--sieve_num_anchors", type=int, default=4)
    parser.add_argument(
        "--sieve_anchor_pool",
        type=int,
        default=12,
        help="Precomputed top structural evidence pool; training samples anchors from it.",
    )
    parser.add_argument("--sieve_random_train_anchors", type=str2bool, default=True)
    parser.add_argument("--sieve_path_weight", type=float, default=1.0)
    parser.add_argument("--sieve_nei_weight", type=float, default=0.30)
    parser.add_argument("--sieve_max_degree", type=int, default=24)
    parser.add_argument("--sieve_max_paths_per_pair", type=int, default=6)
    parser.add_argument("--sieve_max_path_vocab", type=int, default=8192)
    parser.add_argument("--sieve_max_pair_index", type=int, default=2_000_000)
    parser.add_argument("--sieve_skip_backtrack", type=str2bool, default=True)
    parser.add_argument("--sieve_verbose", type=str2bool, default=True)

    return parser


def _resolve_paths(args):
    if not args.data_dir:
        args.data_dir = os.path.join("data", args.task)
    if not args.train_path:
        args.train_path = os.path.join(args.data_dir, "train.json")
    if not args.valid_path:
        args.valid_path = os.path.join(args.data_dir, "valid.json")
    if not args.test_path:
        args.test_path = os.path.join(args.data_dir, "test.json")
    if not args.entities_path:
        args.entities_path = os.path.join(args.data_dir, "entities.json")
    return args


def get_args(argv: Optional[list] = None):
    args = build_parser().parse_args(argv)
    return _resolve_paths(args)
