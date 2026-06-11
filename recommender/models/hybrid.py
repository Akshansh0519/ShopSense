import numpy as np
from typing import List, Tuple, Dict
from scipy.sparse import csr_matrix
import logging
from .base import BaseRecommender

logger = logging.getLogger(__name__)

class HybridRecommender(BaseRecommender):
    def __init__(self, models: Dict[str, BaseRecommender], weights: Dict[str, float]):
        super().__init__()
        self.models = models
        self.weights = weights
        self.is_fitted = True # We assume sub-models are already fitted
        
    def fit(self, user_item_matrix: csr_matrix, **kwargs):
        self.is_fitted = True
        
    def _normalize_scores(self, scores: List[Tuple[int, float]]) -> Dict[int, float]:
        if not scores:
            return {}
            
        items = [x[0] for x in scores]
        vals = np.array([x[1] for x in scores])
        
        if len(vals) == 0:
            return {}
            
        val_min = vals.min()
        val_max = vals.max()
        
        if val_max == val_min:
            normalized = np.ones_like(vals)
        else:
            normalized = (vals - val_min) / (val_max - val_min)
            
        return {item: float(norm) for item, norm in zip(items, normalized)}
        
    def recommend_with_signals(self, user_idx: int, k: int = 10, exclude_seen: bool = True) -> List[Dict]:
        # Get candidates and scores from all models (fetch more than k for merging)
        fetch_k = max(k * 10, 100) 
        
        aggregated_scores = {}
        signal_scores = {}
        
        for name, model in self.models.items():
            if name not in self.weights or self.weights[name] == 0:
                continue
                
            model_recs = model.recommend(user_idx, k=fetch_k, exclude_seen=exclude_seen)
            norm_scores = self._normalize_scores(model_recs)
            
            weight = self.weights[name]
            for item, score in norm_scores.items():
                contribution = score * weight
                aggregated_scores[item] = aggregated_scores.get(item, 0.0) + contribution
                signal_scores.setdefault(item, {})
                signal_scores[item][name] = float(contribution)
                
        # Sort and select top-k
        sorted_recs = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "item_idx": int(item_idx),
                "score": float(score),
                "signals": signal_scores.get(item_idx, {}),
            }
            for item_idx, score in sorted_recs[:k]
        ]

    def recommend(self, user_idx: int, k: int = 10, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        return [
            (rec["item_idx"], rec["score"])
            for rec in self.recommend_with_signals(user_idx, k=k, exclude_seen=exclude_seen)
        ]
