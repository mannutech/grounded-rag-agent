"""Fusion: RRF formula + dominance, and weighted normalisation behaviour."""

from __future__ import annotations

import pytest

from grounded_rag.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion


def test_rrf_doc_in_both_lists_beats_doc_in_one() -> None:
    # doc 1 is rank 0 in both lists; doc 5 is rank 0 in one and absent in the other.
    fused = reciprocal_rank_fusion([[1, 5], [1, 9]], k=60)
    ranked = [idx for idx, _ in fused]
    assert ranked[0] == 1
    assert ranked.index(1) < ranked.index(5)


def test_rrf_score_matches_formula() -> None:
    fused = dict(reciprocal_rank_fusion([[7, 3], [3, 7]], k=60))
    # both 7 and 3 appear once at rank 0 and once at rank 1.
    expected = 1.0 / (60 + 0 + 1) + 1.0 / (60 + 1 + 1)
    assert fused[7] == pytest.approx(expected)
    assert fused[3] == pytest.approx(expected)


def test_rrf_ties_break_by_ascending_index() -> None:
    fused = reciprocal_rank_fusion([[2, 1]], k=60)
    # 2 is rank 0 (higher score) so leads; equal-score case would order by index.
    assert [idx for idx, _ in fused] == [2, 1]
    tie = reciprocal_rank_fusion([[1], [2]], k=60)  # both rank 0 in separate lists
    assert [idx for idx, _ in tie] == [1, 2]


def test_weighted_dense_weight_one_reproduces_dense_order() -> None:
    dense = [(0, 0.9), (1, 0.5), (2, 0.1)]
    sparse = [(2, 9.0), (1, 1.0)]  # different scale, would reorder if it counted
    fused = weighted_score_fusion(dense, sparse, dense_weight=1.0)
    assert [idx for idx, _ in fused] == [0, 1, 2]


def test_weighted_normalises_each_list() -> None:
    # dense_weight 0.5: both lists min-max'd to [0,1] then averaged.
    dense = [(0, 100.0), (1, 0.0)]  # -> {0:1.0, 1:0.0}
    sparse = [(1, 10.0), (0, 0.0)]  # -> {1:1.0, 0:0.0}
    fused = dict(weighted_score_fusion(dense, sparse, dense_weight=0.5))
    assert fused[0] == pytest.approx(0.5)
    assert fused[1] == pytest.approx(0.5)
