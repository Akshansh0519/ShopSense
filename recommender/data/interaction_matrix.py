import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger(__name__)

class InteractionMatrixBuilder:
    def __init__(self, num_users: int, num_items: int):
        self.num_users = num_users
        self.num_items = num_items
        
    def build_raw_matrix(self, df: pd.DataFrame) -> csr_matrix:
        logger.info("Building raw interaction matrix (binary 1.0 weights)...")
        # For implicit feedback, multiple purchases of the same item can either sum or be 1. 
        # Here we group by user and item and use 1.0.
        grouped = df.groupby(['user_idx', 'item_idx']).size().reset_index(name='count')
        
        weights = np.ones(len(grouped))
        
        matrix = csr_matrix(
            (weights, (grouped['user_idx'], grouped['item_idx'])),
            shape=(self.num_users, self.num_items)
        )
        return matrix
        
    def build_time_decay_matrix(self, df: pd.DataFrame, decay_rate: float = 0.01) -> csr_matrix:
        logger.info(f"Building time-decayed interaction matrix (rate={decay_rate})...")
        max_date = df['t_dat'].max()
        
        # Calculate days ago
        df_copy = df.copy()
        df_copy['days_ago'] = (max_date - df_copy['t_dat']).dt.days
        df_copy['weight'] = np.exp(-decay_rate * df_copy['days_ago'])
        
        # Aggregate weights if user bought item multiple times (e.g. sum weights)
        grouped = df_copy.groupby(['user_idx', 'item_idx'])['weight'].sum().reset_index()
        
        matrix = csr_matrix(
            (grouped['weight'], (grouped['user_idx'], grouped['item_idx'])),
            shape=(self.num_users, self.num_items)
        )
        return matrix
