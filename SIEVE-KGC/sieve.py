"""SIEVE structural evidence retrieval.

This module retrieves compact evidence entities for each (head, relation)
query from the observed graph. The retrieved entities are used as query-side
context and are never treated as candidate answers.
"""

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from dict_hub import EntityDict
from doc import Example
from logger_config import logger


QueryKey = Tuple[str, str]


class SIEVEContextBuilder:
    def __init__(
        self,
        args,
        entity_dict: EntityDict,
        train_examples: Sequence[Example],
    ):
        self.args = args
        self.entity_dict = entity_dict
        self.path_weight = float(getattr(args, "sieve_path_weight", 1.0))
        self.nei_weight = float(getattr(args, "sieve_nei_weight", 0.30))
        self.max_degree = max(1, int(getattr(args, "sieve_max_degree", 24)))
        self.max_paths_per_pair = max(
            1, int(getattr(args, "sieve_max_paths_per_pair", 6))
        )
        self.max_path_vocab = max(
            1, int(getattr(args, "sieve_max_path_vocab", 8192))
        )
        self.max_pair_index = max(
            1, int(getattr(args, "sieve_max_pair_index", 2_000_000))
        )
        self.skip_backtrack = bool(
            getattr(args, "sieve_skip_backtrack", True)
        )
        self.verbose = bool(getattr(args, "sieve_verbose", True))

        self.ent2idx = dict(entity_dict.id_to_idx)
        self.idx2ent = [entity.entity_id for entity in entity_dict.entities]
        self.rel2idx: Dict[str, int] = {}
        self.query_cache: Dict[QueryKey, Tuple[str, ...]] = {}
        self.head_to_candidates: Dict[int, Tuple[int, ...]] = {}
        self.pair_paths: Dict[Tuple[int, int], Tuple[int, ...]] = {}

        self.ent_sig = torch.zeros(1, 1)
        self.tail_proto = torch.zeros(1, 1)
        self.comp_table = torch.zeros(1, 1)
        self._build_statistics(list(train_examples))

    def _build_statistics(self, train_examples: List[Example]):
        relation_names = sorted({ex.relation for ex in train_examples})
        self.rel2idx = {relation: idx for idx, relation in enumerate(relation_names)}
        entity_count = len(self.entity_dict)
        relation_count = len(relation_names)
        if entity_count == 0 or relation_count == 0:
            return

        triples: List[Tuple[int, int, int]] = []
        adjacency: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        ent_sig = torch.zeros(entity_count, relation_count, dtype=torch.float32)

        for ex in train_examples:
            h = self.ent2idx.get(ex.head_id)
            t = self.ent2idx.get(ex.tail_id)
            r = self.rel2idx.get(ex.relation)
            if h is None or t is None or r is None:
                continue
            triples.append((h, r, t))
            adjacency[h].append((r, t))
            ent_sig[h, r] += 1.0

        for head in list(adjacency.keys()):
            # Deterministic capped adjacency keeps the evidence index reproducible.
            adjacency[head] = sorted(
                adjacency[head], key=lambda item: (item[0], item[1])
            )[: self.max_degree]

        ent_sig = F.normalize(torch.log1p(ent_sig), p=2, dim=-1)

        tail_proto = torch.zeros(
            relation_count, relation_count, dtype=torch.float32
        )
        proto_count = torch.zeros(relation_count, 1, dtype=torch.float32)
        for _, relation, tail in triples:
            tail_proto[relation] += ent_sig[tail]
            proto_count[relation] += 1.0
        tail_proto = F.normalize(
            tail_proto / proto_count.clamp_min(1.0), p=2, dim=-1
        )

        raw_pair_paths: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        head_candidates: Dict[int, set] = defaultdict(set)
        pair_counter = Counter()
        raw_cap = max(1, self.max_paths_per_pair * 4)
        indexed_pair_count = 0
        stop = False

        # Direct neighbors are always retained, even if the optional 2-hop index
        # later reaches its global memory cap.
        for head, neighbors in adjacency.items():
            for _, tail in neighbors:
                if tail != head:
                    head_candidates[head].add(tail)

        for head in range(entity_count):
            for r1, middle in adjacency.get(head, []):
                for r2, tail in adjacency.get(middle, []):
                    if self.skip_backtrack and tail == head:
                        continue
                    pair = (r1, r2)
                    pair_counter[pair] += 1
                    key = (head, tail)
                    if len(raw_pair_paths[key]) < raw_cap:
                        raw_pair_paths[key].append(pair)
                    head_candidates[head].add(tail)
                    indexed_pair_count += 1
                    if indexed_pair_count >= self.max_pair_index:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break

        if pair_counter:
            top_pairs = [
                pair
                for pair, _ in pair_counter.most_common(self.max_path_vocab)
            ]
            pair2id = {pair: idx for idx, pair in enumerate(top_pairs)}
            comp_table = torch.zeros(
                relation_count, len(top_pairs), dtype=torch.float32
            )
            comp_counter = Counter()
            for head, relation, tail in triples:
                for pair in raw_pair_paths.get((head, tail), []):
                    pid = pair2id.get(pair)
                    if pid is not None:
                        comp_counter[(relation, pid)] += 1
            for (relation, pid), count in comp_counter.items():
                comp_table[relation, pid] = float(count)
            comp_table = torch.log1p(comp_table)
            comp_table = comp_table / comp_table.max(
                dim=1, keepdim=True
            ).values.clamp_min(1.0)

            pair_paths: Dict[Tuple[int, int], Tuple[int, ...]] = {}
            for key, pairs in raw_pair_paths.items():
                ids: List[int] = []
                seen = set()
                for pair in pairs:
                    pid = pair2id.get(pair)
                    if pid is None or pid in seen:
                        continue
                    ids.append(pid)
                    seen.add(pid)
                    if len(ids) >= self.max_paths_per_pair:
                        break
                if ids:
                    pair_paths[key] = tuple(ids)
            self.pair_paths = pair_paths
            self.comp_table = comp_table
            path_vocab_size = len(top_pairs)
        else:
            self.pair_paths = {}
            self.comp_table = torch.zeros(relation_count, 1, dtype=torch.float32)
            path_vocab_size = 0

        self.ent_sig = ent_sig
        self.tail_proto = tail_proto
        self.head_to_candidates = {
            head: tuple(sorted(candidates))
            for head, candidates in head_candidates.items()
        }

        if self.verbose:
            logger.info(
                "SIEVE statistics | entities=%d relations=%d triples=%d "
                "path_vocab=%d pair_paths=%d query_heads=%d comp_nnz=%d",
                entity_count,
                relation_count,
                len(triples),
                path_vocab_size,
                len(self.pair_paths),
                len(self.head_to_candidates),
                int((self.comp_table > 0).sum().item()),
            )

    def _score_candidates(self, head: int, relation: int, candidates: List[int]):
        if not candidates:
            return torch.empty(0, dtype=torch.float32)

        candidate_tensor = torch.tensor(candidates, dtype=torch.long)
        nei = torch.mv(
            self.ent_sig.index_select(0, candidate_tensor),
            self.tail_proto[relation],
        )
        path = torch.zeros(len(candidates), dtype=torch.float32)
        if self.comp_table.numel() > 1 and self.pair_paths:
            relation_row = self.comp_table[relation]
            for idx, tail in enumerate(candidates):
                pids = self.pair_paths.get((head, tail))
                if pids:
                    pid_tensor = torch.tensor(pids, dtype=torch.long)
                    path[idx] = relation_row.index_select(0, pid_tensor).max()
        return self.path_weight * path + self.nei_weight * nei

    def get_evidence_ids(
        self,
        head_id: str,
        relation_name: str,
        max_items: int,
    ) -> Tuple[str, ...]:
        """Return a deterministic top structural evidence pool for one query.

        The target entity is never used to construct this pool.  All statistics
come from the observed graph used to build the evidence index.
        """
        if max_items <= 0:
            return tuple()
        key = (str(head_id), str(relation_name))
        cached = self.query_cache.get(key)
        if cached is not None:
            return cached[:max_items]

        head = self.ent2idx.get(str(head_id))
        relation = self.rel2idx.get(str(relation_name))
        if head is None or relation is None:
            self.query_cache[key] = tuple()
            return tuple()

        candidates = [
            tail
            for tail in self.head_to_candidates.get(head, tuple())
            if tail != head
        ]
        if not candidates:
            self.query_cache[key] = tuple()
            return tuple()

        scores = self._score_candidates(head, relation, candidates)
        keep = min(max_items, len(candidates))
        # Stable tie breaking: first rank by score, then by entity index.
        ranked = sorted(
            zip(candidates, scores.tolist()),
            key=lambda item: (-item[1], item[0]),
        )[:keep]
        result = tuple(self.idx2ent[entity_idx] for entity_idx, _ in ranked)
        self.query_cache[key] = result
        return result

    def precompute(
        self,
        examples: Iterable[Example],
        pool_size: int,
    ) -> Dict[QueryKey, Tuple[str, ...]]:
        keys = sorted({(ex.head_id, ex.relation) for ex in examples})
        evidence: Dict[QueryKey, Tuple[str, ...]] = {}
        for head_id, relation in keys:
            evidence[(head_id, relation)] = self.get_evidence_ids(
                head_id, relation, pool_size
            )
        if self.verbose:
            nonempty = sum(1 for values in evidence.values() if values)
            avg_size = (
                sum(len(values) for values in evidence.values()) / max(1, len(evidence))
            )
            logger.info(
                "SIEVE evidence cache | queries=%d nonempty=%d avg_pool=%.2f",
                len(evidence),
                nonempty,
                avg_size,
            )
        return evidence
