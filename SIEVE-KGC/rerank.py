"""Compatibility layer for code expecting the original SimKGC rerank module."""


def rerank_by_graph(batch_score, examples, entity_dict=None):
    # SimKGC-SIEVE performs graph reranking explicitly through
    # model.rerank_full_scores() in evaluate.py.
    return batch_score
