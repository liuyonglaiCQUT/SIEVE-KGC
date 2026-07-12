import os
from transformers import AutoTokenizer

from config import get_args
from dict_hub import EntityDict, PositiveTripleIndex
from doc import load_examples
from logger_config import logger
from models import build_model
from sieve import SIEVEContextBuilder
from trainer import Trainer
from triplet_mask import set_positive_index
from utils import load_checkpoint, resolve_device, set_seed


def main():
    args = get_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    logger.info("Loading data from %s", args.data_dir)
    entity_dict = EntityDict(args.entities_path)
    train_examples = load_examples(args.train_path, add_inverse=True)
    valid_examples = load_examples(args.valid_path, add_inverse=True)
    test_examples = load_examples(args.test_path, add_inverse=True)

    train_positive_index = PositiveTripleIndex(train_examples)
    eval_positive_index = PositiveTripleIndex(
        train_examples + valid_examples + test_examples
    )
    set_positive_index(train_positive_index)

    logger.info(
        "entities=%d train=%d valid=%d test=%d",
        len(entity_dict),
        len(train_examples),
        len(valid_examples),
        len(test_examples),
    )

    local_only = os.path.isdir(str(args.pretrained_model))
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        use_fast=True,
        local_files_only=local_only,
    )

    evidence_map = {}
    if bool(args.use_sieve):
        builder = SIEVEContextBuilder(args, entity_dict, train_examples)
        evidence_map = builder.precompute(
            train_examples + valid_examples,
            pool_size=max(args.sieve_anchor_pool, args.sieve_num_anchors),
        )

    model = build_model(args).to(device)
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        logger.info("Loaded checkpoint %s", args.checkpoint)

    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_examples=train_examples,
        valid_examples=valid_examples,
        entity_dict=entity_dict,
        eval_positive_index=eval_positive_index,
        evidence_map=evidence_map,
        device=device,
    )
    best_path = trainer.train()
    logger.info("Training completed; best checkpoint: %s", best_path)


if __name__ == "__main__":
    main()
