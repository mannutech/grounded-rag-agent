"""Rank/score fusion for hybrid retrieval.

Two strategies:

* **Reciprocal Rank Fusion (RRF)** — rank-based and scale-free, so it needs no
  calibration between cosine similarity (bounded ~[-1, 1]) and BM25 (unbounded).
  This is the default.
* **Weighted score fusion** — min-max normalises each list to [0, 1] then takes a
  weighted sum. Offered mainly so the evaluation can compare it against RRF.

Both return ``(index, fused_score)`` pairs sorted by score descending, ties
broken by ascending index.
"""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[int]], *, k: int = 60) -> list[tuple[int, float]]:
    """Fuse several ranked index lists via RRF.

    Each item's score is ``sum over lists of 1 / (k + rank + 1)`` where ``rank`` is
    its 0-based position in that list (absent => no contribution).
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _min_max_normalise(pairs: list[tuple[int, float]]) -> dict[int, float]:
    if not pairs:
        return {}
    values = [score for _, score in pairs]
    lo, hi = min(values), max(values)
    if hi == lo:  # all equal — order is a tie, map to full membership
        return {idx: 1.0 for idx, _ in pairs}
    span = hi - lo
    return {idx: (score - lo) / span for idx, score in pairs}


def weighted_score_fusion(
    dense: list[tuple[int, float]],
    sparse: list[tuple[int, float]],
    *,
    dense_weight: float,
) -> list[tuple[int, float]]:
    """Min-max normalise each list then combine with ``dense_weight`` in [0, 1].

    ``dense_weight == 1.0`` reproduces the pure dense ordering (sparse-only items
    contribute 0 and sort to the bottom).
    """
    dense_norm = _min_max_normalise(dense)
    sparse_norm = _min_max_normalise(sparse)
    sparse_weight = 1.0 - dense_weight
    combined: dict[int, float] = {}
    for idx in set(dense_norm) | set(sparse_norm):
        combined[idx] = (
            dense_weight * dense_norm.get(idx, 0.0) + sparse_weight * sparse_norm.get(idx, 0.0)
        )
    return sorted(combined.items(), key=lambda item: (-item[1], item[0]))
