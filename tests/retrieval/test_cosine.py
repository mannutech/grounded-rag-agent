"""Cosine math: exact values, tie-breaking, and total handling of degenerate input."""

from __future__ import annotations

import numpy as np
import pytest

from grounded_rag.retrieval.store import cosine_similarity, cosine_top_k


def test_cosine_similarity_known_values() -> None:
    a = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0)
    assert cosine_similarity(a, np.array([-1.0, 0.0, 0.0])) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_is_zero_not_nan() -> None:
    a = np.array([0.0, 0.0, 0.0])
    score = cosine_similarity(a, np.array([1.0, 2.0, 3.0]))
    assert score == 0.0
    assert not np.isnan(score)


def test_cosine_top_k_orders_and_breaks_ties_by_index() -> None:
    query = np.array([1.0, 0.0])
    # rows 0 and 2 are identical to the query (tie); row 1 is orthogonal.
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    result = cosine_top_k(query, matrix, k=3)
    assert [idx for idx, _ in result] == [0, 2, 1]  # tie (0,2) broken by ascending index
    assert result[0][1] == pytest.approx(1.0)
    assert result[2][1] == pytest.approx(0.0)


def test_cosine_top_k_k_larger_than_corpus() -> None:
    query = np.array([1.0, 0.0])
    matrix = np.array([[1.0, 0.0], [0.5, 0.5]])
    assert len(cosine_top_k(query, matrix, k=10)) == 2


def test_cosine_top_k_degenerate_inputs() -> None:
    query = np.array([1.0, 0.0])
    assert cosine_top_k(query, np.zeros((0, 2)), k=5) == []          # empty corpus
    assert cosine_top_k(query, np.array([[1.0, 0.0]]), k=0) == []     # k == 0
    assert cosine_top_k(np.array([0.0, 0.0]), np.array([[1.0, 0.0]]), k=1) == []  # zero query
