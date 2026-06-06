import logging
from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from .base import BaseRecommender


logger = logging.getLogger(__name__)


class ItemItemRecommender(BaseRecommender):
    def __init__(self, max_similar_items: int = 100):
        super().__init__()
        self.max_similar_items = max_similar_items
        self.similar_item_indices = None
        self.similar_item_scores = None
        self.train_matrix = None

    def fit(self, user_item_matrix: csr_matrix, **kwargs):
        logger.info("Fitting ItemItemRecommender with top-%s neighbors per item...", self.max_similar_items)
        self.train_matrix = user_item_matrix.tocsr()
        item_user_matrix = self.train_matrix.T.tocsr()
        n_items = item_user_matrix.shape[0]
        n_neighbors = min(self.max_similar_items + 1, n_items)

        nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors, n_jobs=-1)
        nn.fit(item_user_matrix)
        distances, indices = nn.kneighbors(item_user_matrix, return_distance=True)

        self.similar_item_indices = indices[:, 1:]
        self.similar_item_scores = 1.0 - distances[:, 1:]
        self.is_fitted = True

    def recommend(self, user_idx: int, k: int = 10, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        if user_idx >= self.train_matrix.shape[0]:
            return []

        seen_items = self._get_user_seen_items(user_idx, self.train_matrix)
        if not seen_items:
            return []

        scores = {}
        user_row = self.train_matrix[user_idx]
        user_weights = {
            int(item_idx): float(weight)
            for item_idx, weight in zip(user_row.indices, user_row.data)
        }

        for item_idx in seen_items:
            neighbor_items = self.similar_item_indices[item_idx]
            neighbor_scores = self.similar_item_scores[item_idx]
            history_weight = user_weights.get(int(item_idx), 1.0)
            for neighbor, sim in zip(neighbor_items, neighbor_scores):
                neighbor = int(neighbor)
                if exclude_seen and neighbor in seen_items:
                    continue
                scores[neighbor] = scores.get(neighbor, 0.0) + float(sim) * history_weight

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(int(item_idx), float(score)) for item_idx, score in ranked[:k]]
