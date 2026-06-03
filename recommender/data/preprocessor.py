import pandas as pd
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
    def filter_interactions(self, df: pd.DataFrame, min_user_interactions: int = 3, min_item_interactions: int = 5) -> pd.DataFrame:
        logger.info(f"Initial shape: {df.shape}")
        
        # Filter items
        item_counts = df['article_id'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df['article_id'].isin(valid_items)]
        logger.info(f"After item filtering: {df.shape}")
        
        # Filter users
        user_counts = df['customer_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df['customer_id'].isin(valid_users)]
        logger.info(f"After user filtering: {df.shape}")
        
        return df.copy()
        
    def create_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating user and item integer mappings...")
        
        unique_users = df['customer_id'].unique()
        unique_items = df['article_id'].unique()
        
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
        
        df['user_idx'] = df['customer_id'].map(self.user_mapping)
        df['item_idx'] = df['article_id'].map(self.item_mapping)
        
        return df
        
    def save_mappings(self):
        logger.info("Saving mappings to artifacts...")
        with open(self.artifacts_dir / "user_mapping.pkl", "wb") as f:
            pickle.dump(self.user_mapping, f)
        with open(self.artifacts_dir / "item_mapping.pkl", "wb") as f:
            pickle.dump(self.item_mapping, f)
        with open(self.artifacts_dir / "reverse_user_mapping.pkl", "wb") as f:
            pickle.dump(self.reverse_user_mapping, f)
        with open(self.artifacts_dir / "reverse_item_mapping.pkl", "wb") as f:
            pickle.dump(self.reverse_item_mapping, f)
            
    def load_mappings(self):
        logger.info("Loading mappings from artifacts...")
        with open(self.artifacts_dir / "user_mapping.pkl", "rb") as f:
            self.user_mapping = pickle.load(f)
        with open(self.artifacts_dir / "item_mapping.pkl", "rb") as f:
            self.item_mapping = pickle.load(f)
        with open(self.artifacts_dir / "reverse_user_mapping.pkl", "rb") as f:
            self.reverse_user_mapping = pickle.load(f)
        with open(self.artifacts_dir / "reverse_item_mapping.pkl", "rb") as f:
            self.reverse_item_mapping = pickle.load(f)
