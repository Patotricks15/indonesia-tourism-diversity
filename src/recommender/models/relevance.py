"""Collaborative relevance scorer: shrunk cosine item-item co-visitation.

This is the *base* ranker. The point of this project is not this scorer —
it is what happens to its output next: MMR and DPP re-ranking for
diversity (``recommender.models.rerank``). The relevance model itself is a
standard, non-neural co-occurrence kNN, identical in spirit to the
``ItemKNN`` baseline used elsewhere in this portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommender.data.splitting import Split


@dataclass
class ItemKNNRelevance:
    """Shrunk cosine item-item co-occurrence over visited destinations."""

    top_k_neighbours: int = 50
    shrinkage: float = 10.0
    similarity: np.ndarray | None = None
    popularity: np.ndarray | None = None
    n_items: int = 0

    def fit(self, split: Split) -> ItemKNNRelevance:
        users = split.train["user_index"].to_numpy(dtype=np.int64)
        items = split.train["item_index"].to_numpy(dtype=np.int64)
        matrix = np.zeros((split.n_items, split.n_users), dtype=np.float32)
        matrix[items, users] = 1.0
        cooccurrence = matrix @ matrix.T
        norms = np.sqrt(np.diag(cooccurrence))
        denominator = np.outer(norms, norms) + self.shrinkage
        self.similarity = np.divide(
            cooccurrence,
            denominator,
            out=np.zeros_like(cooccurrence),
            where=denominator > 0,
        )
        np.fill_diagonal(self.similarity, 0.0)
        self.n_items = split.n_items
        if self.top_k_neighbours < self.n_items:
            remove = self.n_items - self.top_k_neighbours
            threshold = np.partition(self.similarity, remove, axis=1)[:, remove]
            self.similarity[self.similarity < threshold[:, None]] = 0.0
        self.popularity = matrix.sum(axis=1)
        return self

    def score(self, history: set[int]) -> np.ndarray:
        """Relevance score for every destination, given a visit history."""
        if self.similarity is None or self.popularity is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        if not history:
            return self.popularity.copy()
        return self.similarity[list(history)].sum(axis=0)

    def candidates(self, history: set[int], pool_size: int) -> np.ndarray:
        """Top-``pool_size`` unseen destinations by relevance, best first."""
        scores = self.score(history).copy()
        if history:
            scores[list(history)] = -np.inf
        pool_size = min(pool_size, int(np.isfinite(scores).sum()))
        if pool_size == 0:
            return np.array([], dtype=np.int64)
        top = np.argpartition(-scores, pool_size - 1)[:pool_size]
        return top[np.argsort(-scores[top], kind="stable")]
