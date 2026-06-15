import hashlib
import pandas as pd
import numpy as np
import logging
from scipy.stats import mannwhitneyu

logger = logging.getLogger(__name__)

class ABSimulator:
    def __init__(self, control_model, treatment_model):
        self.control_model = control_model
        self.treatment_model = treatment_model
        
    def _assign_variant(self, user_id: str) -> str:
        # Deterministic hash-based split
        hash_val = int(hashlib.md5(user_id.encode('utf-8')).hexdigest(), 16)
        return 'control' if hash_val % 2 == 0 else 'treatment'
        
    def simulate(self, test_df: pd.DataFrame, reverse_user_mapping: dict, k: int = 10):
        logger.info("Running offline A/B simulator...")
        
        # Group actuals by user for fast lookup
        actuals = test_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
        
        control_hits = []
        treatment_hits = []
        
        for user_idx, actual_items in actuals.items():
            user_id = str(reverse_user_mapping.get(user_idx, str(user_idx)))
            variant = self._assign_variant(user_id)
            
            if variant == 'control':
                recs = self.control_model.recommend(user_idx, k=k)
                rec_items = [r[0] for r in recs]
                hit = 1 if set(rec_items) & actual_items else 0
                control_hits.append(hit)
            else:
                recs = self.treatment_model.recommend(user_idx, k=k)
                rec_items = [r[0] for r in recs]
                hit = 1 if set(rec_items) & actual_items else 0
                treatment_hits.append(hit)
                
        control_ctr = np.mean(control_hits) if control_hits else 0.0
        treatment_ctr = np.mean(treatment_hits) if treatment_hits else 0.0
        
        lift = (treatment_ctr - control_ctr) / control_ctr if control_ctr > 0 else 0.0
        
        # Statistical test
        p_value = 1.0
        if control_hits and treatment_hits:
            try:
                _, p_value = mannwhitneyu(treatment_hits, control_hits, alternative='greater')
            except ValueError:
                # Happens if all values are identical
                p_value = 1.0
                
        decision = "Promote" if (lift > 0 and p_value < 0.05) else "Rollback"
        
        result = {
            'control_users': len(control_hits),
            'treatment_users': len(treatment_hits),
            'control_simulated_ctr': control_ctr,
            'treatment_simulated_ctr': treatment_ctr,
            'lift': lift,
            'p_value': p_value,
            'decision': decision
        }
        
        logger.info(f"A/B Result: {decision}. Lift: {lift:.2%}, p-value: {p_value:.4f}")
        return result
        
if __name__ == "__main__":
    print("A/B Simulator module.")
