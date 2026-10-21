"""Command-line entry points for training, evaluation, and recommendation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from recommender.data import build_implicit_feedback, load_ratings, user_holdout_split
from recommender.evaluation import evaluate, evaluate_diversity
from recommender.models import (
    DPPRecommender,
    ItemKNNRelevance,
    MMRRecommender,
    PopularityRecommender,
    RelevanceOnlyRecommender,
    build_item_similarity,
)

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_POOL = 50
MMR_LAMBDA = 0.5


def _load():
    ratings = load_ratings(ROOT / "data" / "ratings.csv")
    places = pd.read_csv(ROOT / "data" / "places.csv")
    data = build_implicit_feedback(ratings)
    split = user_holdout_split(data)
    return data, places, split


def _build_models(split, similarity):
    relevance = ItemKNNRelevance().fit(split)
    return {
        "RelevanceOnly": RelevanceOnlyRecommender(relevance, candidate_pool=CANDIDATE_POOL),
        "MMR": MMRRecommender(relevance, similarity, lambda_param=MMR_LAMBDA,
                               candidate_pool=CANDIDATE_POOL),
        "DPP": DPPRecommender(relevance, similarity, candidate_pool=CANDIDATE_POOL),
        "Popularity": PopularityRecommender().fit(split),
    }


def train_command(_: argparse.Namespace) -> None:
    _, _, split = _load()
    relevance = ItemKNNRelevance().fit(split)
    print(
        json.dumps(
            {"n_items": relevance.n_items, "top_k_neighbours": relevance.top_k_neighbours}
        )
    )


def evaluate_command(args: argparse.Namespace) -> None:
    data, places, split = _load()
    similarity = build_item_similarity(places, data.item_ids)
    categories = places.set_index("place_id")["category"].reindex(data.item_ids).to_numpy()
    models = _build_models(split, similarity)

    rows = []
    for name, model in models.items():
        ranking = evaluate(model, split, k=args.k, partition=args.partition, model_name=name)
        diversity = evaluate_diversity(
            model, split, similarity, categories, k=args.k, partition=args.partition,
            model_name=name,
        )
        rows.append({**ranking.as_row(), **diversity.as_row()})
    print(json.dumps(rows, indent=2))


def recommend_command(args: argparse.Namespace) -> None:
    data, places, split = _load()
    matches = (data.user_ids == args.user_id).nonzero()[0]
    if matches.size == 0:
        raise SystemExit(f"Unknown user id {args.user_id}.")
    user = int(matches[0])
    similarity = build_item_similarity(places, data.item_ids)
    relevance = ItemKNNRelevance().fit(split)
    history = split.user_positives().get(user, set())

    strategy = {
        "relevance": RelevanceOnlyRecommender(relevance, candidate_pool=CANDIDATE_POOL),
        "mmr": MMRRecommender(relevance, similarity, lambda_param=MMR_LAMBDA,
                               candidate_pool=CANDIDATE_POOL),
        "dpp": DPPRecommender(relevance, similarity, candidate_pool=CANDIDATE_POOL),
    }[args.strategy]
    ranked = strategy.recommend(user, k=args.k, exclude=history)

    lookup = places.set_index("place_id")
    payload = [
        {
            "rank": rank,
            "place_id": int(data.item_ids[item]),
            "category": str(lookup.loc[int(data.item_ids[item]), "category"]),
        }
        for rank, item in enumerate(ranked, start=1)
    ]
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.set_defaults(func=train_command)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--partition", choices=["validation", "test"], default="test")
    evaluate_parser.add_argument("--k", type=int, default=10)
    evaluate_parser.set_defaults(func=evaluate_command)

    recommend = commands.add_parser("recommend")
    recommend.add_argument("--user-id", type=int, required=True)
    recommend.add_argument("--k", type=int, default=10)
    recommend.add_argument("--strategy", choices=["relevance", "mmr", "dpp"], default="mmr")
    recommend.set_defaults(func=recommend_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
