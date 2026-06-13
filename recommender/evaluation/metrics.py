import numpy as np
from typing import List, Set, Dict
from sklearn.metrics.pairwise import cosine_similarity

def recall_at_k(recommended: List[int], actual: Set[int], k: int = 10) -> float:
    if not actual:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & actual)
    return hits / min(len(actual), k)

def precision_at_k(recommended: List[int], actual: Set[int], k: int = 10) -> float:
    if not actual:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & actual)
    return hits / k

def ndcg_at_k(recommended: List[int], actual: Set[int], k: int = 10) -> float:
    if not actual:
        return 0.0
    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2)
            
    idcg = 0.0
    for i in range(min(len(actual), k)):
        idcg += 1.0 / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def map_at_k(recommended: List[int], actual: Set[int], k: int = 10) -> float:
    if not actual:
        return 0.0
    rec_k = recommended[:k]
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(rec_k):
        if item in actual:
            hits += 1
            sum_precisions += hits / (i + 1)
            
    return sum_precisions / min(len(actual), k)

def coverage_at_k(all_recommended_items: Set[int], total_items: int) -> float:
    if total_items == 0:
        return 0.0
    return len(all_recommended_items) / total_items

def diversity_at_k(recommended: List[int], item_embeddings: np.ndarray, k: int = 10) -> float:
    rec_k = [item for item in recommended[:k] if 0 <= item < len(item_embeddings)]
    if len(rec_k) < 2:
        return 0.0
    embeddings = item_embeddings[rec_k]
    sims = cosine_similarity(embeddings)
    upper = sims[np.triu_indices_from(sims, k=1)]
    return float(1.0 - np.mean(upper)) if len(upper) else 0.0

def novelty_at_k(recommended: List[int], item_popularity_prob: Dict[int, float], k: int = 10) -> float:
    """
    item_popularity_prob: dict mapping item_idx to its probability of being interacted with
    """
    rec_k = recommended[:k]
    novelty = 0.0
    for item in rec_k:
        prob = item_popularity_prob.get(item, 1e-9)
        novelty += -np.log2(prob)
    return novelty / k if k > 0 else 0.0

def serendipity_at_k(recommended: List[int], actual: Set[int], user_dominant_category: str, item_categories: Dict[int, str], k: int = 10) -> float:
    """
    Serendipity: fraction of successful recommendations (hits) that are outside the user's dominant category.
    """
    if not actual:
        return 0.0
    rec_k = recommended[:k]
    serendipitous_hits = 0
    total_hits = 0
    for item in rec_k:
        if item in actual:
            total_hits += 1
            if item_categories.get(item, "") != user_dominant_category:
                serendipitous_hits += 1
                
    return serendipitous_hits / total_hits if total_hits > 0 else 0.0
