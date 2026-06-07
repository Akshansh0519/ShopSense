import numpy as np
from typing import List, Tuple
from scipy.sparse import csr_matrix
import logging
import implicit
from .base import BaseRecommender

logger = logging.getLogger(__name__)

class ALSRecommender(BaseRecommender):
    def __init__(self, factors: int = 64, regularization: float = 0.1, iterations: int = 30, alpha: float = 20.0):
        super().__init__()
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        
        # Note: implicit ALS expects item-user matrix for training in older versions, 
        # but in newer versions (>=0.7.0) it takes user-item matrix directly.
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=False # Explicitly disable GPU for local stability unless requested
        )
        self.train_matrix = None
        
    def fit(self, user_item_matrix: csr_matrix, **kwargs):
        logger.info(f"Fitting ALSRecommender (factors={self.factors}, iter={self.iterations})...")
        self.train_matrix = user_item_matrix
        
        # Scale matrix by alpha for confidence
        confidence_matrix = (user_item_matrix * self.alpha).astype(np.float32)
        
        # Fit model
        self.model.fit(confidence_matrix)
        self.is_fitted = True
        
    def recommend(self, user_idx: int, k: int = 10, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        if user_idx >= self.train_matrix.shape[0]:
            return []
            
        # implicit recommend returns (item_ids, scores)
        ids, scores = self.model.recommend(
            userid=user_idx,
            user_items=self.train_matrix[user_idx],
            N=k,
            filter_already_liked_items=exclude_seen
        )
        
        recommendations = [(int(item_id), float(score)) for item_id, score in zip(ids, scores)]
        return recommendations
