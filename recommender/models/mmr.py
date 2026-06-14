import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .base import BaseRecommender

logger = logging.getLogger(__name__)

class MMRReranker:
    def __init__(self, lambda_param: float = 0.75):
        self.lambda_param = lambda_param
        
    def rerank(self, candidates: List[Tuple[int, float]], item_embeddings: np.ndarray, final_k: int = 10) -> List[Tuple[int, float]]:
        if not candidates:
            return []
            
        if len(candidates) <= final_k:
            return candidates
            
        candidate_items = [c[0] for c in candidates]
        candidate_scores = np.array([c[1] for c in candidates])
        
        # Normalize scores to [0,1] for MMR formula
        score_min = candidate_scores.min()
        score_max = candidate_scores.max()
        if score_max > score_min:
            candidate_scores = (candidate_scores - score_min) / (score_max - score_min)
        else:
            candidate_scores = np.ones_like(candidate_scores)
            
        candidate_embeddings = item_embeddings[candidate_items]
        
        selected_indices = []
        selected_scores = []
        
        # Start with the highest scored item
        first_idx = int(np.argmax(candidate_scores))
        selected_indices.append(first_idx)
        selected_scores.append(candidates[first_idx][1]) # Keep original score
        
        remaining_indices = set(range(len(candidates))) - {first_idx}
        
        while len(selected_indices) < final_k and remaining_indices:
            best_mmr_score = -float('inf')
            best_idx = -1
            
            selected_embeds = candidate_embeddings[selected_indices]
            
            for idx in remaining_indices:
                relevance = candidate_scores[idx]
                
                # Max similarity to already selected items
                sims = cosine_similarity(candidate_embeddings[idx].reshape(1, -1), selected_embeds)[0]
                max_sim = np.max(sims)
                
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx
                    
            selected_indices.append(best_idx)
            selected_scores.append(candidates[best_idx][1]) # Keep original score
            remaining_indices.remove(best_idx)
            
        # Return final selected items with original scores
        reranked = [(candidate_items[idx], selected_scores[i]) for i, idx in enumerate(selected_indices)]
        return reranked


class MMRRecommender(BaseRecommender):
    def __init__(self, base_model: BaseRecommender, item_embeddings: np.ndarray, lambda_param: float = 0.75, candidate_pool_size: int = 100):
        super().__init__()
        self.base_model = base_model
        self.item_embeddings = item_embeddings
        self.lambda_param = lambda_param
        self.candidate_pool_size = candidate_pool_size
        self.reranker = MMRReranker(lambda_param=lambda_param)
        self.is_fitted = True

    def fit(self, user_item_matrix, **kwargs):
        self.is_fitted = True

    def recommend_with_signals(self, user_idx: int, k: int = 10, exclude_seen: bool = True) -> List[Dict]:
        fetch_k = max(self.candidate_pool_size, k)
        if hasattr(self.base_model, "recommend_with_signals"):
            base_recs = self.base_model.recommend_with_signals(user_idx, k=fetch_k, exclude_seen=exclude_seen)
            candidate_pairs = [(rec["item_idx"], rec["score"]) for rec in base_recs]
            signal_lookup = {rec["item_idx"]: rec.get("signals", {}) for rec in base_recs}
        else:
            candidate_pairs = self.base_model.recommend(user_idx, k=fetch_k, exclude_seen=exclude_seen)
            signal_lookup = {item_idx: {} for item_idx, _ in candidate_pairs}

        reranked = self.reranker.rerank(candidate_pairs, self.item_embeddings, final_k=k)
        return [
            {
                "item_idx": int(item_idx),
                "score": float(score),
                "signals": signal_lookup.get(item_idx, {}),
                "mmr_adjusted": rank > 0,
            }
            for rank, (item_idx, score) in enumerate(reranked)
        ]

    def recommend(self, user_idx: int, k: int = 10, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        return [
            (rec["item_idx"], rec["score"])
            for rec in self.recommend_with_signals(user_idx, k=k, exclude_seen=exclude_seen)
        ]
