import json
import os
import random
from argparse import Namespace
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    output = {}
    for key, value in batch.items():
        output[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return output


def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int, best_mrr: float, args):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_mrr": best_mrr,
        "args": vars(args) if hasattr(args, "__dict__") else dict(args),
    }
    torch.save(state, path)


def load_checkpoint(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def namespace_from_dict(values: Dict[str, Any]) -> Namespace:
    return Namespace(**values)


def dump_args(args, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as writer:
        json.dump(vars(args), writer, ensure_ascii=False, indent=2)
