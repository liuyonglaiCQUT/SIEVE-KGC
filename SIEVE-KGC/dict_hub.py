from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

from doc import EntityExample, Example, load_entities


class EntityDict:
    def __init__(self, path: str):
        self.entities: List[EntityExample] = load_entities(path)
        self.id_to_idx: Dict[str, int] = {
            entity.entity_id: idx for idx, entity in enumerate(self.entities)
        }

    def __len__(self):
        return len(self.entities)

    def get_entity_by_idx(self, idx: int) -> EntityExample:
        return self.entities[idx]

    def entity_to_idx(self, entity_id: str) -> int:
        return self.id_to_idx[str(entity_id)]

    def get(self, entity_id: str) -> EntityExample:
        return self.entities[self.entity_to_idx(entity_id)]


class PositiveTripleIndex:
    def __init__(self, examples: Iterable[Example]):
        self.tails: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for example in examples:
            self.tails[(example.head_id, example.relation)].add(example.tail_id)

    def is_positive(self, head_id: str, relation: str, tail_id: str) -> bool:
        return str(tail_id) in self.tails.get((str(head_id), str(relation)), set())

    def get_tails(self, head_id: str, relation: str) -> Set[str]:
        return self.tails.get((str(head_id), str(relation)), set())
