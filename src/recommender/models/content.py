"""Content-based item-item similarity for diversity re-ranking.

Built from catalogue metadata only (category and city), never from
interaction data: the whole point of MMR/DPP re-ranking is to diversify
*what kind* of destination appears in the list, which requires a notion of
similarity that is independent of collaborative co-visitation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_item_similarity(catalogue: pd.DataFrame, item_ids: np.ndarray) -> np.ndarray:
    """Cosine similarity over one-hot (category, city) descriptors.

    Args:
        catalogue: Destination metadata with ``place_id``, ``category``,
            ``city`` columns.
        item_ids: Original place ids, positionally aligned to internal item
            index; defines the row/column order of the returned matrix.

    Returns:
        A symmetric ``(n_items, n_items)`` similarity matrix in ``[0, 1]``,
        with a zero diagonal (no destination is "similar to itself" for
        re-ranking purposes).
    """
    indexed = catalogue.set_index("place_id").reindex(item_ids)
    category = pd.get_dummies(indexed["category"].fillna("unknown"), prefix="cat")
    city = pd.get_dummies(indexed["city"].fillna("unknown"), prefix="city")
    # Down-weight city relative to category: two destinations in the same
    # city but different categories should not be treated as near-duplicates.
    descriptors = pd.concat([category, 0.5 * city], axis=1).to_numpy(dtype=np.float64)

    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = descriptors / norms
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, 0.0)
    return np.clip(similarity, 0.0, 1.0)
