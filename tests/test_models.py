"""Tests for the relevance scorer and the MMR/DPP diversity re-rankers."""

from __future__ import annotations

import numpy as np

from recommender.models import (
    DPPRecommender,
    ItemKNNRelevance,
    MMRRecommender,
    RelevanceOnlyRecommender,
    build_item_similarity,
    dpp_greedy_map,
    mmr_rerank,
)


def test_mmr_with_lambda_one_matches_pure_relevance_order():
    relevance = np.array([0.9, 0.5, 0.8, 0.1])
    similarity = np.ones((4, 4))
    np.fill_diagonal(similarity, 0.0)
    order = mmr_rerank(relevance, similarity, k=4, lambda_param=1.0)
    assert list(order) == [0, 2, 1, 3]


def test_mmr_prefers_diverse_item_when_lambda_is_small():
    # Item 1 is almost as relevant as item 0 but near-identical to it;
    # item 2 is less relevant but dissimilar. Diversity-heavy MMR should
    # pick item 2 second, not the near-duplicate item 1.
    relevance = np.array([1.0, 0.95, 0.5])
    similarity = np.array(
        [[0.0, 0.99, 0.0], [0.99, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    order = mmr_rerank(relevance, similarity, k=2, lambda_param=0.1)
    assert list(order) == [0, 2]


def test_dpp_greedy_map_avoids_duplicate_items():
    # Two near-identical high-relevance items plus one distinct, lower
    # relevance item: the DPP kernel should favor the distinct item over
    # the duplicate once one near-identical item is already selected.
    rel = np.array([1.0, 1.0, 0.6])
    sim = np.array([[1.0, 0.95, 0.1], [0.95, 1.0, 0.1], [0.1, 0.1, 1.0]])
    kernel = np.outer(rel, rel) * sim
    selected = dpp_greedy_map(kernel, k=2)
    assert 2 in selected


def test_relevance_only_recommender_excludes_history(split):
    relevance = ItemKNNRelevance().fit(split)
    model = RelevanceOnlyRecommender(relevance, candidate_pool=20)
    history = split.user_positives()[0]
    ranked = model.recommend(0, k=5, exclude=history)
    assert not set(map(int, ranked)) & history


def test_mmr_recommender_returns_valid_candidates(split, catalogue, interactions):
    relevance = ItemKNNRelevance().fit(split)
    similarity = build_item_similarity(catalogue, interactions.item_ids)
    model = MMRRecommender(relevance, similarity, candidate_pool=20)
    history = split.user_positives()[0]
    ranked = model.recommend(0, k=5, exclude=history)
    assert len(ranked) <= 5
    assert not set(map(int, ranked)) & history


def test_dpp_recommender_returns_valid_candidates(split, catalogue, interactions):
    relevance = ItemKNNRelevance().fit(split)
    similarity = build_item_similarity(catalogue, interactions.item_ids)
    model = DPPRecommender(relevance, similarity, candidate_pool=20)
    history = split.user_positives()[0]
    ranked = model.recommend(0, k=5, exclude=history)
    assert len(ranked) <= 5
    assert not set(map(int, ranked)) & history
