"""Diversity re-ranking: Maximal Marginal Relevance and a greedy DPP.

Both re-rankers take the *same* inputs — a relevance score per candidate and
a content-based item-item similarity matrix — and trade relevance for
diversity differently:

``mmr_rerank``
    Greedy, one item at a time: at each step, pick the candidate maximizing
    ``lambda * relevance - (1 - lambda) * max_similarity_to_selected``. A
    single scalar, ``lambda``, controls the relevance/diversity trade-off.

``dpp_greedy_map``
    A Determinantal Point Process models the whole selected *set* jointly:
    the probability of a subset is proportional to the determinant of its
    kernel submatrix, which is large exactly when the subset is both
    individually relevant (large diagonal) and mutually dissimilar (small
    off-diagonal). Exact MAP inference is NP-hard, so this implements the
    standard fast greedy approximation (Chen, Zhang and Zhou, 2018), which
    incrementally updates a Cholesky-like factor instead of recomputing a
    determinant at every step.
"""

from __future__ import annotations

import numpy as np


def mmr_rerank(
    relevance: np.ndarray,
    similarity: np.ndarray,
    k: int,
    lambda_param: float = 0.5,
) -> np.ndarray:
    """Re-rank candidates by Maximal Marginal Relevance.

    Args:
        relevance: Relevance score per candidate, shape ``(n,)``.
        similarity: Candidate-candidate similarity, shape ``(n, n)``.
        k: Number of items to select.
        lambda_param: Weight on relevance versus diversity, in ``[0, 1]``.
            ``1.0`` reduces to plain relevance ranking; ``0.0`` maximizes
            diversity only.

    Returns:
        Indices into the candidate arrays, selection order (best first).

    Raises:
        ValueError: If ``lambda_param`` is outside ``[0, 1]`` or shapes
            disagree.
    """
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError(f"lambda_param must be in [0, 1], got {lambda_param}.")
    n = relevance.shape[0]
    if similarity.shape != (n, n):
        raise ValueError(f"similarity must be ({n}, {n}), got {similarity.shape}.")

    k = min(k, n)
    selected: list[int] = []
    remaining = set(range(n))
    max_similarity_to_selected = np.zeros(n)

    for _ in range(k):
        best_item, best_score = None, -np.inf
        for item in remaining:
            score = (
                lambda_param * relevance[item]
                - (1 - lambda_param) * max_similarity_to_selected[item]
            )
            if score > best_score:
                best_item, best_score = item, score
        selected.append(best_item)
        remaining.discard(best_item)
        max_similarity_to_selected = np.maximum(max_similarity_to_selected, similarity[best_item])

    return np.array(selected, dtype=np.int64)


def dpp_greedy_map(kernel: np.ndarray, k: int) -> np.ndarray:
    """Fast greedy MAP inference for a Determinantal Point Process.

    Implements the incremental Cholesky-update algorithm of Chen, Zhang and
    Zhou (2018), which selects the subset of size ``k`` approximately
    maximizing ``log det(kernel[S, S])`` in ``O(k^2 n)`` instead of the
    naive ``O(k n^4)`` of recomputing each candidate determinant from
    scratch.

    Args:
        kernel: Positive semi-definite kernel ``L``, shape ``(n, n)``. In
            this project, ``L = diag(relevance) @ similarity @ diag(relevance)``.
        k: Number of items to select.

    Returns:
        Indices into ``kernel``, selection order (best first).
    """
    n = kernel.shape[0]
    k = min(k, n)
    d2 = np.diag(kernel).copy().astype(np.float64)
    c = np.zeros((n, k), dtype=np.float64)
    selected: list[int] = []

    for step in range(k):
        j = int(np.argmax(d2))
        if d2[j] <= 1e-12:
            break
        selected.append(j)
        if step == k - 1:
            break
        ej = (kernel[j] - c[:, :step] @ c[j, :step]) / np.sqrt(d2[j])
        c[:, step] = ej
        d2 -= ej**2
        d2[j] = -np.inf

    return np.array(selected, dtype=np.int64)
