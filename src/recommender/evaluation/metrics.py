"""Top-N ranking and diversity metrics for the tourism recommender."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from recommender.data.splitting import Split


class Recommender(Protocol):
    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray: ...


def _ndcg_at_k(ranked: np.ndarray, targets: set[int], k: int) -> float:
    gains = sum(
        1.0 / np.log2(position + 2)
        for position, item in enumerate(ranked)
        if int(item) in targets
    )
    ideal = sum(1.0 / np.log2(position + 2) for position in range(min(len(targets), k)))
    return gains / ideal if ideal else 0.0


def _average_precision_at_k(ranked: np.ndarray, targets: set[int], k: int) -> float:
    hits = 0
    precision_sum = 0.0
    for position, item in enumerate(ranked):
        if int(item) in targets:
            hits += 1
            precision_sum += hits / (position + 1)
    denominator = min(len(targets), k)
    return precision_sum / denominator if denominator else 0.0


@dataclass(frozen=True)
class EvaluationReport:
    model: str
    protocol: str
    k: int
    ndcg: float
    recall: float
    precision: float
    average_precision: float
    hit_rate: float
    coverage: float
    n_users: int

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "model": self.model,
            "protocol": self.protocol,
            "k": self.k,
            "ndcg": round(self.ndcg, 4),
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "map": round(self.average_precision, 4),
            "hit_rate": round(self.hit_rate, 4),
            "coverage": round(self.coverage, 4),
            "n_users": self.n_users,
        }


def evaluate(
    model: Recommender,
    split: Split,
    k: int = 10,
    partition: str = "test",
    model_name: str = "model",
) -> EvaluationReport:
    """Evaluate ranking quality against each user's held-out positives."""
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be 'validation' or 'test'.")
    frame = getattr(split, partition)
    if frame.empty:
        raise ValueError(f"Partition '{partition}' is empty.")

    histories = split.user_positives()
    relevant = {
        int(user): set(group["item_index"].astype(int))
        for user, group in frame.groupby("user_index")
    }
    ndcg_total = recall_total = precision_total = ap_total = 0.0
    hits = 0
    recommended: set[int] = set()

    for user, targets in relevant.items():
        ranked = model.recommend(user, k=k, exclude=histories.get(user, set()))
        recommended.update(map(int, ranked))
        positions = [position for position, item in enumerate(ranked) if int(item) in targets]
        if positions:
            hits += 1
        ndcg_total += _ndcg_at_k(ranked, targets, k)
        ap_total += _average_precision_at_k(ranked, targets, k)
        recall_total += len(positions) / len(targets)
        precision_total += len(positions) / k

    n_users = len(relevant)
    return EvaluationReport(
        model=model_name,
        protocol=split.protocol,
        k=k,
        ndcg=ndcg_total / n_users,
        recall=recall_total / n_users,
        precision=precision_total / n_users,
        average_precision=ap_total / n_users,
        hit_rate=hits / n_users,
        coverage=len(recommended) / split.n_items,
        n_users=n_users,
    )


@dataclass(frozen=True)
class DiversityReport:
    """List-quality metrics measuring the effect of a diversity re-ranker.

    Attributes:
        model: Name of the evaluated recommender.
        k: List length.
        intra_list_diversity: Mean pairwise dissimilarity (``1 - similarity``)
            across all pairs in each user's top-``k`` list, averaged over
            users. Higher is more diverse.
        category_coverage: Mean number of distinct categories per top-``k``
            list, normalized by ``k``. Higher means less redundant lists.
        n_users: Number of users the lists were built for.
    """

    model: str
    k: int
    intra_list_diversity: float
    category_coverage: float
    n_users: int

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "model": self.model,
            "k": self.k,
            "intra_list_diversity": round(self.intra_list_diversity, 4),
            "category_coverage": round(self.category_coverage, 4),
            "n_users": self.n_users,
        }


def evaluate_diversity(
    model: Recommender,
    split: Split,
    similarity: np.ndarray,
    categories: np.ndarray,
    k: int = 10,
    partition: str = "test",
    model_name: str = "model",
) -> DiversityReport:
    """Measure intra-list diversity and category coverage of a ranking.

    Args:
        model: A fitted recommender exposing ``recommend``.
        split: Partition supplying training history and the evaluated users.
        similarity: Content-based item-item similarity, ``(n_items, n_items)``.
        categories: Category label per internal item index, ``(n_items,)``.
        k: List length.
        partition: ``"validation"`` or ``"test"``.
        model_name: Label attached to the report.

    Returns:
        A :class:`DiversityReport`.

    Raises:
        ValueError: If ``partition`` is invalid or the evaluated partition
            is empty.
    """
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be 'validation' or 'test'.")
    frame = getattr(split, partition)
    if frame.empty:
        raise ValueError(f"Partition '{partition}' is empty.")

    histories = split.user_positives()
    users = sorted(frame["user_index"].unique())

    ild_total = coverage_total = 0.0
    n_lists = 0
    for user in users:
        ranked = model.recommend(int(user), k=k, exclude=histories.get(int(user), set()))
        if len(ranked) < 2:
            continue
        pairs = [
            1.0 - similarity[int(a), int(b)]
            for i, a in enumerate(ranked)
            for b in ranked[i + 1 :]
        ]
        ild_total += float(np.mean(pairs))
        coverage_total += len(set(categories[list(map(int, ranked))])) / k
        n_lists += 1

    if n_lists == 0:
        return DiversityReport(model=model_name, k=k, intra_list_diversity=0.0,
                                category_coverage=0.0, n_users=0)
    return DiversityReport(
        model=model_name,
        k=k,
        intra_list_diversity=ild_total / n_lists,
        category_coverage=coverage_total / n_lists,
        n_users=n_lists,
    )
