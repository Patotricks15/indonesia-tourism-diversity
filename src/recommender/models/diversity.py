"""Diversity-aware recommenders combining relevance with MMR or DPP.

Each class exposes the portfolio-standard ``recommend(user_index, k,
exclude)`` interface so :func:`recommender.evaluation.metrics.evaluate` can
score it exactly like the non-diversified baselines, letting the three
strategies be compared head-to-head on the same candidate pools.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommender.models.relevance import ItemKNNRelevance
from recommender.models.rerank import dpp_greedy_map, mmr_rerank


@dataclass
class RelevanceOnlyRecommender:
    """Pure relevance ranking: the baseline MMR and DPP are compared against."""

    relevance: ItemKNNRelevance
    candidate_pool: int = 50

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None  # noqa: ARG002
    ) -> np.ndarray:
        history = exclude or set()
        return self.relevance.candidates(history, max(k, self.candidate_pool))[:k]


@dataclass
class MMRRecommender:
    """Relevance ranking re-ordered by Maximal Marginal Relevance."""

    relevance: ItemKNNRelevance
    similarity: np.ndarray
    lambda_param: float = 0.5
    candidate_pool: int = 50

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None  # noqa: ARG002
    ) -> np.ndarray:
        history = exclude or set()
        candidates = self.relevance.candidates(history, self.candidate_pool)
        if candidates.size == 0:
            return candidates
        rel = self.relevance.score(history)[candidates]
        sim = self.similarity[np.ix_(candidates, candidates)]
        order = mmr_rerank(rel, sim, k=k, lambda_param=self.lambda_param)
        return candidates[order]


@dataclass
class DPPRecommender:
    """Relevance ranking re-ordered by greedy DPP MAP inference."""

    relevance: ItemKNNRelevance
    similarity: np.ndarray
    candidate_pool: int = 50

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None  # noqa: ARG002
    ) -> np.ndarray:
        history = exclude or set()
        candidates = self.relevance.candidates(history, self.candidate_pool)
        if candidates.size == 0:
            return candidates
        rel = np.clip(self.relevance.score(history)[candidates], 1e-6, None)
        sim = self.similarity[np.ix_(candidates, candidates)]
        # Quality-diversity decomposition: L_ij = rel_i * rel_j * sim_ij,
        # with sim_ii = 1 on the diagonal so L_ii = rel_i^2 (pure quality).
        np.fill_diagonal(sim, 1.0)
        kernel = np.outer(rel, rel) * sim
        order = dpp_greedy_map(kernel, k=k)
        return candidates[order]
