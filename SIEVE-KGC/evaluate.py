import json
import os
from argparse import Namespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import get_args
from dict_hub import EntityDict, PositiveTripleIndex
from doc import Example, load_examples
from logger_config import logger
from metric import compute_metrics
from models import build_model
from sieve import SIEVEContextBuilder
from triplet import EntityCollator, EntityDataset, TripletCollator, TripletDataset
from triplet_mask import set_positive_index
from utils import load_checkpoint, move_to_device, resolve_device


@torch.no_grad()
def encode_all_entities(model, entity_dict, tokenizer, args, device):
    dataset = EntityDataset(entity_dict.entities)
    loader = DataLoader(
        dataset,
        batch_size=int(args.entity_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=EntityCollator(tokenizer, args.max_num_tokens),
    )
    vectors = []
    model.eval()
    for batch in loader:
        batch = move_to_device(batch, device)
        vectors.append(
            model.encode_entities(
                batch["tail_token_ids"],
                batch["tail_mask"],
                batch["tail_token_type_ids"],
            ).cpu()
        )
    return torch.cat(vectors, dim=0)


def _filter_known_scores(scores, examples, entity_dict, positive_index):
    for row, example in enumerate(examples):
        gold_idx = entity_dict.entity_to_idx(example.tail_id)
        for tail_id in positive_index.get_tails(example.head_id, example.relation):
            idx = entity_dict.id_to_idx.get(tail_id)
            if idx is not None and idx != gold_idx:
                scores[row, idx] = -torch.inf


@torch.no_grad()
def evaluate_model(
    model,
    examples: Sequence[Example],
    entity_dict: EntityDict,
    tokenizer,
    args,
    positive_index: PositiveTripleIndex,
    evidence_map: Mapping[Tuple[str, str], Tuple[str, ...]],
    device: torch.device,
    save_predictions: Optional[str] = None,
):
    entity_vectors = encode_all_entities(model, entity_dict, tokenizer, args, device)
    entity_vectors_device = entity_vectors.to(device)

    dataset = TripletDataset(examples)
    loader = DataLoader(
        dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=TripletCollator(
            tokenizer=tokenizer,
            max_length=args.max_num_tokens,
            use_self_negative=args.use_self_negative,
            entity_dict=entity_dict,
            evidence_map=evidence_map,
            use_sieve=bool(args.use_sieve and args.sieve_eval),
            num_anchors=args.sieve_num_anchors,
            random_train_anchors=False,
            training=False,
        ),
    )

    ranks: List[int] = []
    predictions = []
    model.eval()
    for batch in loader:
        batch_examples = batch["batch_data"]
        batch = move_to_device(batch, device)
        base_vector, sieve_vector = model.encode_query_batch(batch)
        scores = base_vector.mm(entity_vectors_device.t())
        if (
            bool(args.use_sieve)
            and bool(args.sieve_eval)
            and sieve_vector is not None
        ):
            sieve_scores = sieve_vector.mm(entity_vectors_device.t())
            scores = scores + float(args.sieve_eval_weight) * sieve_scores

        if bool(args.filter_known):
            _filter_known_scores(scores, batch_examples, entity_dict, positive_index)

        gold_indices = torch.tensor(
            [entity_dict.entity_to_idx(ex.tail_id) for ex in batch_examples],
            dtype=torch.long,
            device=device,
        )
        gold_scores = scores.gather(1, gold_indices.unsqueeze(1)).squeeze(1)
        batch_ranks = 1 + (scores > gold_scores.unsqueeze(1)).sum(dim=1)
        ranks.extend(batch_ranks.cpu().tolist())

        if save_predictions:
            topk = min(10, scores.size(1))
            top_indices = scores.topk(topk, dim=1).indices.cpu().tolist()
            for ex, rank, indices in zip(
                batch_examples, batch_ranks.cpu().tolist(), top_indices
            ):
                predictions.append(
                    {
                        "head_id": ex.head_id,
                        "relation": ex.relation,
                        "tail_id": ex.tail_id,
                        "rank": int(rank),
                        "top10": [
                            entity_dict.get_entity_by_idx(idx).entity_id
                            for idx in indices
                        ],
                    }
                )

    metrics = compute_metrics(ranks).to_dict()
    if save_predictions:
        os.makedirs(os.path.dirname(save_predictions) or ".", exist_ok=True)
        with open(save_predictions, "w", encoding="utf-8") as writer:
            json.dump(predictions, writer, ensure_ascii=False, indent=2)
    return metrics


def _load_cli_model(cli_args):
    if not cli_args.checkpoint:
        raise ValueError("--checkpoint is required for standalone evaluation")
    checkpoint = load_checkpoint(cli_args.checkpoint, map_location="cpu")
    saved = dict(checkpoint.get("args", {})) or vars(cli_args).copy()

    override_keys = {
        "data_dir",
        "train_path",
        "valid_path",
        "test_path",
        "entities_path",
        "eval_batch_size",
        "entity_batch_size",
        "num_workers",
        "device",
        "filter_known",
        "save_predictions",
        "prediction_path",
        "use_sieve",
        "sieve_eval",
        "sieve_train",
        "sieve_eval_weight",
        "sieve_num_anchors",
        "sieve_anchor_pool",
        "sieve_path_weight",
        "sieve_nei_weight",
        "sieve_max_degree",
        "sieve_max_paths_per_pair",
        "sieve_max_path_vocab",
        "sieve_max_pair_index",
        "sieve_skip_backtrack",
        "sieve_verbose",
    }
    for key in override_keys:
        saved[key] = getattr(cli_args, key)
    saved["checkpoint"] = cli_args.checkpoint
    args = Namespace(**saved)

    entity_dict = EntityDict(args.entities_path)
    model = build_model(args)
    model.load_state_dict(checkpoint["model"], strict=True)
    return args, entity_dict, model


def main():
    cli_args = get_args()
    args, entity_dict, model = _load_cli_model(cli_args)
    device = resolve_device(args.device)
    model.to(device)
    local_only = os.path.isdir(str(args.pretrained_model))
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        use_fast=True,
        local_files_only=local_only,
    )

    train_examples = load_examples(args.train_path, add_inverse=True)
    valid_examples = load_examples(args.valid_path, add_inverse=True)
    test_examples = load_examples(args.test_path, add_inverse=True)
    positive_index = PositiveTripleIndex(
        train_examples + valid_examples + test_examples
    )
    set_positive_index(PositiveTripleIndex(train_examples))

    evidence_map = {}
    if bool(args.use_sieve):
        builder = SIEVEContextBuilder(args, entity_dict, train_examples)
        evidence_map = builder.precompute(
            test_examples,
            pool_size=max(args.sieve_anchor_pool, args.sieve_num_anchors),
        )

    metrics = evaluate_model(
        model=model,
        examples=test_examples,
        entity_dict=entity_dict,
        tokenizer=tokenizer,
        args=args,
        positive_index=positive_index,
        evidence_map=evidence_map,
        device=device,
        save_predictions=args.prediction_path if args.save_predictions else None,
    )
    logger.info("Test metrics: %s", metrics)


if __name__ == "__main__":
    main()
