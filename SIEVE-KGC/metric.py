from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List
import numpy as np


@dataclass
class RankingMetrics:
    mr: float
    mrr: float
    hits1: float
    hits3: float
    hits10: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mr": self.mr,
            "mrr": self.mrr,
            "hits@1": self.hits1,
            "hits@3": self.hits3,
            "hits@10": self.hits10,
        }


def compute_metrics(ranks: Iterable[int]) -> RankingMetrics:
    ranks = np.asarray(list(ranks), dtype=np.float64)
    if ranks.size == 0:
        return RankingMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
    return RankingMetrics(
        mr=float(ranks.mean()),
        mrr=float((1.0 / ranks).mean()),
        hits1=float((ranks <= 1).mean()),
        hits3=float((ranks <= 3).mean()),
        hits10=float((ranks <= 10).mean()),
    )
