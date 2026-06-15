import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class SegmentEvaluator:
    def __init__(self):
        self.segments = {
            'cold': (3, 5),
            'warm': (6, 20),
            'hot': (21, float('inf'))
        }
        
    def assign_segments(self, train_df: pd.DataFrame) -> Dict[str, set]:
        logger.info("Assigning users to segments...")
        user_counts = train_df['user_idx'].value_counts()
        
        segment_users = {}
        for name, (min_count, max_count) in self.segments.items():
            users = user_counts[(user_counts >= min_count) & (user_counts <= max_count)].index
            segment_users[name] = set(users)
            logger.info(f"Segment '{name}': {len(users)} users")
            
        return segment_users
        
    def get_segment_test_data(self, test_df: pd.DataFrame, segment_users: set) -> pd.DataFrame:
        return test_df[test_df['user_idx'].isin(segment_users)].copy()
