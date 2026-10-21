"""Individual feature transforms.

Every ``train_only`` transform is fitted strictly on the training partition
of a :class:`~recommender.data.splitting.Split`; a serving-time lookup for a
user/item absent from training falls back to the declared ``fill`` value in
``conf/features.yaml`` rather than raising.
"""

from __future__ import annotations

import pandas as pd

from recommender.data.splitting import Split


def user_activity_count(split: Split) -> pd.Series:
    """Training-time positive-visit count per user."""
    return split.train.groupby("user_index").size().astype("float64")


def user_mean_rating(split: Split) -> pd.Series:
    """Training-time mean overall rating per user."""
    return split.train.groupby("user_index")["rating"].mean()


def item_popularity_rank(split: Split) -> pd.Series:
    """Fraction-rank (0, 1] of each destination's training-time popularity."""
    counts = split.train.groupby("item_index").size()
    return counts.rank(pct=True)


def item_category(catalogue: pd.DataFrame) -> pd.Series:
    """Declared category, read directly from the destination catalogue."""
    return catalogue.set_index("place_id")["category"].astype("category")


def item_city(catalogue: pd.DataFrame) -> pd.Series:
    """City, read directly from the destination catalogue."""
    return catalogue.set_index("place_id")["city"].astype("category")
