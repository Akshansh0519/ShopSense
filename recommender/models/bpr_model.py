import numpy as np
from typing import List, Tuple
from scipy.sparse import csr_matrix
import logging
import implicit
from .base import BaseRecommender

logger = logging.getLogger(__name__)

class BPRRecommender(BaseRecommender):
    def __init__(self, factors: int = 64, regularization: float = 0.01, iterations: int = 100):
        super().__init__()
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        
        self.model = implicit.bpr.BayesianPersonalizedRanking(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=False
        )
        self.train_matrix = None
        
    def fit(self, user_item_matrix: csr_matrix, **kwargs):
        logger.info(f"Fitting BPRRecommender (factors={self.factors}, iter={self.iterations})...")
        self.train_matrix = user_item_matrix
        
        # BPR works well with binary interactions or positive weights
        self.model.fit(user_item_matrix)
        self.is_fitted = True
        
    def recommend(self, user_idx: int, k: int = 10, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        if user_idx >= self.train_matrix.shape[0]:
            return []
            
        ids, scores = self.model.recommend(
            userid=user_idx,
            user_items=self.train_matrix[user_idx],
            N=k,
            filter_already_liked_items=exclude_seen
        )
        
        recommendations = [(int(item_id), float(score)) for item_id, score in zip(ids, scores)]
        return recommendations
