import numpy as np
from typing import List, Tuple
from scipy.sparse import csr_matrix
import logging
from .base import BaseRecommender

logger = logging.getLogger(__name__)

class PopularityRecommender(BaseRecommender):
    def __init__(self):
        super().__init__()
        self.item_scores = None
        self.popular_items_sorted = None
        self.train_matrix = None
        
    def fit(self, user_item_matrix: csr_matrix, **kwargs):
        logger.info("Fitting PopularityRecommender...")
        self.train_matrix = user_item_matrix
        
        # Calculate popularity as sum of interactions (or just counts) across all users
        item_sums = np.array(user_item_matrix.sum(axis=0)).flatten()
        self.item_scores = item_sums
        
        # Sort items by popularity descending
        self.popular_items_sorted = np.argsort(self.item_scores)[::-1]
        self.is_fitted = True
        
    def recommend(self, user_idx: int, k: int = 10, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        seen_items = set()
        if exclude_seen and self.train_matrix is not None:
            seen_items = self._get_user_seen_items(user_idx, self.train_matrix)
            
        recommendations = []
        for item_idx in self.popular_items_sorted:
            if item_idx not in seen_items:
                recommendations.append((int(item_idx), float(self.item_scores[item_idx])))
            if len(recommendations) == k:
                break
                
        return recommendations
