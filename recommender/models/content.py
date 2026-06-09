import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from scipy.sparse import csr_matrix
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender

logger = logging.getLogger(__name__)

class ContentRecommender(BaseRecommender):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.item_embeddings = None
        self.user_profiles = None
        self.train_matrix = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["encoder"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.encoder = None
        
    def prepare_item_text(self, articles_df: pd.DataFrame, item_mapping: Dict[str, int]) -> pd.DataFrame:
        """Create a unified text field for each item."""
        df = articles_df.copy()
        
        # Keep only items in the mapping
        df = df[df['article_id'].isin(item_mapping.keys())].copy()
        df['item_idx'] = df['article_id'].map(item_mapping)
        
        # Fill NA
        text_cols = ['product_type_name', 'product_group_name', 'colour_group_name', 'department_name', 'detail_desc']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')
                
        # Combine text
        df['combined_text'] = df.apply(
            lambda x: f"{x.get('product_group_name', '')} {x.get('product_type_name', '')} {x.get('department_name', '')} {x.get('colour_group_name', '')} {x.get('detail_desc', '')}",
            axis=1
        )
        
        return df.sort_values('item_idx').reset_index(drop=True)
        
    def fit(self, user_item_matrix: csr_matrix, **kwargs):
        """
        Expects 'articles_df' and 'item_mapping' in kwargs.
        Optional 'artifacts_dir' in kwargs to load precomputed embeddings.
        """
        logger.info("Fitting ContentRecommender...")
        articles_df = kwargs.get('articles_df')
        item_mapping = kwargs.get('item_mapping')
        artifacts_dir = kwargs.get('artifacts_dir', 'artifacts')
        
        if articles_df is None or item_mapping is None:
            raise ValueError("articles_df and item_mapping must be provided in kwargs.")
            
        self.train_matrix = user_item_matrix
        num_items = user_item_matrix.shape[1]
        
        embeddings_path = Path(artifacts_dir) / 'item_embeddings.npy'
        
        if embeddings_path.exists():
            logger.info(f"Loading PRE-COMPUTED embeddings from {embeddings_path}... (Skipping Neural Network training!)")
            self.item_embeddings = np.load(embeddings_path)
        else:
            if self.encoder is None:
                self.encoder = SentenceTransformer(self.model_name)
            logger.info("Preparing text and encoding items...")
            df_text = self.prepare_item_text(articles_df, item_mapping)
            
            self.item_embeddings = np.zeros((num_items, self.encoder.get_sentence_embedding_dimension()), dtype=np.float32)
            
            # Encode valid items
            valid_indices = df_text['item_idx'].values
            valid_texts = df_text['combined_text'].tolist()
            logger.info("Running Sentence-Transformers (This will take a long time on CPU)...")
            embeddings = self.encoder.encode(valid_texts, show_progress_bar=True)
            
            self.item_embeddings[valid_indices] = np.asarray(embeddings, dtype=np.float32)
            np.save(embeddings_path, self.item_embeddings)
        
        # Build user profiles (mean of their interacted items)
        logger.info("Building user profiles...")
        # To compute quickly: sparse matrix (num_users x num_items) @ dense matrix (num_items x embed_dim)
        # Normalize rows to get mean
        row_sums = np.array(user_item_matrix.sum(axis=1)).astype(np.float32).flatten()
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        
        summed_embeddings = user_item_matrix.dot(self.item_embeddings).astype(np.float32)
        self.user_profiles = (summed_embeddings / row_sums[:, np.newaxis]).astype(np.float32)
        
        self.is_fitted = True
        
    def recommend(self, user_idx: int, k: int = 10, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        if user_idx >= self.train_matrix.shape[0]:
            return []
            
        user_vector = self.user_profiles[user_idx].reshape(1, -1)
        
        # If user has no interactions (zero vector), return empty
        if not np.any(user_vector):
            return []
            
        # Compute cosine similarity with all items
        scores = cosine_similarity(user_vector, self.item_embeddings)[0]
        
        seen_items = set()
        if exclude_seen:
            seen_items = self._get_user_seen_items(user_idx, self.train_matrix)
            
        # Get indices of non-zero scores
        candidate_indices = np.where(scores > 0)[0]
        
        # Sort candidates by score descending
        sorted_candidates = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        
        recommendations = []
        for item_idx in sorted_candidates:
            if item_idx not in seen_items:
                recommendations.append((int(item_idx), float(scores[item_idx])))
            if len(recommendations) == k:
                break
                
        return recommendations
