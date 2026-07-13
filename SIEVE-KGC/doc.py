from dataclasses import dataclass
from typing import Dict, Iterable, List
import json


@dataclass(frozen=True)
class EntityExample:
    entity_id: str
    entity: str
    entity_desc: str = ""

    @classmethod
    def from_dict(cls, obj: Dict):
        return cls(
            entity_id=str(obj["entity_id"]),
            entity=str(obj.get("entity", obj.get("name", obj["entity_id"]))),
            entity_desc=str(obj.get("entity_desc", obj.get("description", "")) or ""),
        )


@dataclass(frozen=True)
class Example:
    head_id: str
    head: str
    relation: str
    tail_id: str
    tail: str
    head_desc: str = ""
    tail_desc: str = ""

    @classmethod
    def from_dict(cls, obj: Dict):
        return cls(
            head_id=str(obj["head_id"]),
            head=str(obj.get("head", obj["head_id"])),
            relation=str(obj["relation"]),
            tail_id=str(obj["tail_id"]),
            tail=str(obj.get("tail", obj["tail_id"])),
            head_desc=str(obj.get("head_desc", "") or ""),
            tail_desc=str(obj.get("tail_desc", "") or ""),
        )

    def inverse(self):
        relation = self.relation
        if relation.startswith("inverse "):
            relation = relation[len("inverse ") :]
        else:
            relation = f"inverse {relation}"
        return Example(
            head_id=self.tail_id,
            head=self.tail,
            head_desc=self.tail_desc,
            relation=relation,
            tail_id=self.head_id,
            tail=self.head,
            tail_desc=self.head_desc,
        )


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as reader:
        return json.load(reader)


def load_examples(path: str, add_inverse: bool = True) -> List[Example]:
    raw = load_json(path)
    examples = [Example.from_dict(obj) for obj in raw]
    if add_inverse:
        examples = examples + [example.inverse() for example in examples]
    return examples


def load_entities(path: str) -> List[EntityExample]:
    return [EntityExample.from_dict(obj) for obj in load_json(path)]


def dump_json(path: str, objects: Iterable[Dict]):
    with open(path, "w", encoding="utf-8") as writer:
        json.dump(list(objects), writer, ensure_ascii=False, indent=2)
