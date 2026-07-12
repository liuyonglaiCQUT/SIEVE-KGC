import random
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from dict_hub import EntityDict
from doc import EntityExample, Example
from triplet_mask import construct_mask, construct_self_negative_mask


def _join_entity_text(name: str, desc: str) -> str:
    name = str(name or "").strip()
    desc = str(desc or "").strip()
    if desc and desc.lower() not in name.lower():
        return "{}: {}".format(name, desc)
    return name or desc


class TripletDataset(Dataset):
    def __init__(self, examples: Sequence[Example]):
        self.examples = list(examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]


class EntityDataset(Dataset):
    def __init__(self, entities: Sequence[EntityExample]):
        self.entities = list(entities)

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, index: int) -> EntityExample:
        return self.entities[index]


def _ensure_token_type_ids(encoded: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "token_type_ids" in encoded:
        return encoded["token_type_ids"]
    return torch.zeros_like(encoded["input_ids"])


class TripletCollator:
    """Create classic SimKGC queries and RAA-style SIEVE-enhanced queries.

    RAA-KGC encodes one query per anchor and averages those anchor-query vectors.
    This collator does the same efficiently: all evidence-enhanced queries are
    flattened into one token batch, and `sieve_owner` maps them back to the
    original examples for scatter-mean aggregation in the model.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int,
        use_self_negative: bool = True,
        entity_dict: EntityDict = None,
        evidence_map: Mapping[Tuple[str, str], Tuple[str, ...]] = None,
        use_sieve: bool = False,
        num_anchors: int = 4,
        random_train_anchors: bool = True,
        training: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.use_self_negative = bool(use_self_negative)
        self.entity_dict = entity_dict
        self.evidence_map = evidence_map or {}
        self.use_sieve = bool(use_sieve)
        self.num_anchors = max(1, int(num_anchors))
        self.random_train_anchors = bool(random_train_anchors)
        self.training = bool(training)

    def _encode_hr(self, examples: Sequence[Example]):
        head_texts = [_join_entity_text(ex.head, ex.head_desc) for ex in examples]
        relation_texts = [ex.relation for ex in examples]
        return self.tokenizer(
            head_texts,
            relation_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def _encode_entity_texts(self, names: List[str], descs: List[str]):
        texts = [_join_entity_text(name, desc) for name, desc in zip(names, descs)]
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def _select_evidence_ids(self, example: Example) -> List[str]:
        pool = list(self.evidence_map.get((example.head_id, example.relation), tuple()))
        if self.training:
            # The current positive tail is removed during training to prevent a
            # trivial target-text shortcut.  Evaluation selection never reads
            # the gold target and therefore remains a real inference procedure.
            pool = [entity_id for entity_id in pool if entity_id != example.tail_id]
        if not pool:
            return []
        keep = min(self.num_anchors, len(pool))
        if self.training and self.random_train_anchors and len(pool) > keep:
            return random.sample(pool, keep)
        return pool[:keep]

    def _encode_sieve_queries(self, examples: Sequence[Example]):
        if not self.use_sieve or self.entity_dict is None:
            return None

        flat_heads: List[str] = []
        flat_pairs: List[str] = []
        owners: List[int] = []
        counts = torch.zeros(len(examples), dtype=torch.long)

        for owner, example in enumerate(examples):
            head_text = _join_entity_text(example.head, example.head_desc)
            evidence_ids = self._select_evidence_ids(example)
            if not evidence_ids:
                # RAA code falls back to the original example when no anchor is
                # available.  This keeps the auxiliary branch well-defined.
                flat_heads.append(head_text)
                flat_pairs.append(example.relation)
                owners.append(owner)
                counts[owner] += 1
                continue

            for entity_id in evidence_ids:
                entity = self.entity_dict.get(entity_id)
                evidence_text = _join_entity_text(entity.entity, entity.entity_desc)
                # Same insertion pattern as RAA-KGC: relation and one anchor are
                # placed in the second BERT sequence, separated by [SEP].
                pair_text = "{} [SEP] shadow evidence: {}".format(
                    example.relation, evidence_text
                )
                flat_heads.append(head_text)
                flat_pairs.append(pair_text)
                owners.append(owner)
                counts[owner] += 1

        encoded = self.tokenizer(
            flat_heads,
            flat_pairs,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "sieve_hr_token_ids": encoded["input_ids"],
            "sieve_hr_mask": encoded["attention_mask"],
            "sieve_hr_token_type_ids": _ensure_token_type_ids(encoded),
            "sieve_owner": torch.tensor(owners, dtype=torch.long),
            "sieve_count": counts.clamp_min(1),
        }

    def __call__(self, examples: Sequence[Example]):
        examples = list(examples)
        hr = self._encode_hr(examples)
        tail = self._encode_entity_texts(
            [ex.tail for ex in examples], [ex.tail_desc for ex in examples]
        )
        head = self._encode_entity_texts(
            [ex.head for ex in examples], [ex.head_desc for ex in examples]
        )

        batch = {
            "hr_token_ids": hr["input_ids"],
            "hr_mask": hr["attention_mask"],
            "hr_token_type_ids": _ensure_token_type_ids(hr),
            "tail_token_ids": tail["input_ids"],
            "tail_mask": tail["attention_mask"],
            "tail_token_type_ids": _ensure_token_type_ids(tail),
            "head_token_ids": head["input_ids"],
            "head_mask": head["attention_mask"],
            "head_token_type_ids": _ensure_token_type_ids(head),
            "triplet_mask": construct_mask(examples, examples),
            "self_negative_mask": construct_self_negative_mask(examples),
            "batch_data": examples,
        }
        sieve_batch = self._encode_sieve_queries(examples)
        if sieve_batch is not None:
            batch.update(sieve_batch)
        return batch


class EntityCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, entities: Sequence[EntityExample]):
        entities = list(entities)
        texts = [_join_entity_text(ex.entity, ex.entity_desc) for ex in entities]
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "tail_token_ids": encoded["input_ids"],
            "tail_mask": encoded["attention_mask"],
            "tail_token_type_ids": _ensure_token_type_ids(encoded),
            "entity_ids": [ex.entity_id for ex in entities],
        }
