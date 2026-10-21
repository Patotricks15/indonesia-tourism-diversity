"""Tests that MMR/DPP actually increase diversity relative to pure relevance."""

from __future__ import annotations

from recommender.evaluation import evaluate_diversity
from recommender.models import (
    ItemKNNRelevance,
    MMRRecommender,
    RelevanceOnlyRecommender,
    build_item_similarity,
)


def test_mmr_increases_intra_list_diversity_over_relevance_only(split, catalogue, interactions):
    relevance = ItemKNNRelevance().fit(split)
    similarity = build_item_similarity(catalogue, interactions.item_ids)
    categories = (
        catalogue.set_index("place_id")["category"].reindex(interactions.item_ids).to_numpy()
    )

    relevance_only = RelevanceOnlyRecommender(relevance, candidate_pool=20)
    mmr = MMRRecommender(relevance, similarity, lambda_param=0.2, candidate_pool=20)

    base_report = evaluate_diversity(
        relevance_only, split, similarity, categories, k=5, partition="test", model_name="base"
    )
    mmr_report = evaluate_diversity(
        mmr, split, similarity, categories, k=5, partition="test", model_name="mmr"
    )

    assert mmr_report.intra_list_diversity >= base_report.intra_list_diversity
