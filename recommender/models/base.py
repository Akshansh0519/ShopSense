from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np
from scipy.sparse import csr_matrix

class BaseRecommender(ABC):
    def __init__(self):
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, user_item_matrix: csr_matrix, **kwargs):
        """Train the model on the provided interaction matrix."""
        pass
        
    @abstractmethod
    def recommend(self, user_idx: int, k: int = 10, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        Recommend top-k items for a given user.
        Returns a list of tuples (item_idx, score).
        """
        pass
        
    def _get_user_seen_items(self, user_idx: int, user_item_matrix: csr_matrix) -> set:
        """Helper to get items already seen by the user."""
        if user_idx >= user_item_matrix.shape[0]:
            return set()
        
        row_start = user_item_matrix.indptr[user_idx]
        row_end = user_item_matrix.indptr[user_idx + 1]
        seen_items = set(user_item_matrix.indices[row_start:row_end])
        return seen_items
