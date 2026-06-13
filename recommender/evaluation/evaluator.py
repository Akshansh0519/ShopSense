import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging
from . import metrics

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, test_df: pd.DataFrame, num_items: int):
        self.test_df = test_df
        self.num_items = num_items
        
        # Group actuals by user for fast lookup
        self.actuals = self.test_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
        
    def evaluate(self, model, k: int = 10, item_popularity_prob: Dict[int, float] = None,
                 item_embeddings=None,
                 user_dominant_categories: Dict[int, str] = None, item_categories: Dict[int, str] = None) -> Dict[str, float]:
        logger.info(f"Evaluating model at k={k}...")
        
        metric_sums = {
            'recall': 0.0, 'precision': 0.0, 'ndcg': 0.0, 'map': 0.0, 
            'novelty': 0.0, 'serendipity': 0.0
        }
        latencies = []
        all_recommended_items = set()
        
        valid_users = 0
        
        for user_idx, actual_items in self.actuals.items():
            start_time = time.time()
            recs = model.recommend(user_idx, k=k)
            latency = (time.time() - start_time) * 1000 # ms
            latencies.append(latency)
            
            if not recs:
                continue
                
            rec_items = [r[0] for r in recs]
            all_recommended_items.update(rec_items)
            
            metric_sums['recall'] += metrics.recall_at_k(rec_items, actual_items, k)
            metric_sums['precision'] += metrics.precision_at_k(rec_items, actual_items, k)
            metric_sums['ndcg'] += metrics.ndcg_at_k(rec_items, actual_items, k)
            metric_sums['map'] += metrics.map_at_k(rec_items, actual_items, k)
            
            if item_popularity_prob:
                metric_sums['novelty'] += metrics.novelty_at_k(rec_items, item_popularity_prob, k)
            if item_embeddings is not None:
                metric_sums.setdefault('diversity', 0.0)
                metric_sums['diversity'] += metrics.diversity_at_k(rec_items, item_embeddings, k)
                
            if user_dominant_categories and item_categories:
                user_cat = user_dominant_categories.get(user_idx, "")
                metric_sums['serendipity'] += metrics.serendipity_at_k(rec_items, actual_items, user_cat, item_categories, k)
                
            valid_users += 1
            
        if valid_users == 0:
            return {k: 0.0 for k in metric_sums}
            
        results = {
            f'recall@{k}': metric_sums['recall'] / valid_users,
            f'precision@{k}': metric_sums['precision'] / valid_users,
            f'ndcg@{k}': metric_sums['ndcg'] / valid_users,
            f'map@{k}': metric_sums['map'] / valid_users,
            f'coverage@{k}': metrics.coverage_at_k(all_recommended_items, self.num_items),
            'p50_latency_ms': np.percentile(latencies, 50) if latencies else 0.0,
            'p95_latency_ms': np.percentile(latencies, 95) if latencies else 0.0,
            'p99_latency_ms': np.percentile(latencies, 99) if latencies else 0.0
        }
        
        if item_popularity_prob:
            results[f'novelty@{k}'] = metric_sums['novelty'] / valid_users
        if item_embeddings is not None:
            results[f'diversity@{k}'] = metric_sums.get('diversity', 0.0) / valid_users
        if user_dominant_categories and item_categories:
            results[f'serendipity@{k}'] = metric_sums['serendipity'] / valid_users
            
        return results
