from typing import Dict
import logging

logger = logging.getLogger(__name__)

class ReasonGenerator:
    def __init__(self):
        pass
        
    def generate_reason(self, signals: Dict[str, float], is_mmr_adjusted: bool = False) -> str:
        if is_mmr_adjusted:
            return "Selected to improve variety in your recommendation list"
            
        if not signals:
            return "Trending item"
            
        # Find dominant signal
        dominant_signal = max(signals.items(), key=lambda x: x[1])
        signal_name = dominant_signal[0]
        
        if signal_name in ('als', 'bpr'):
            return "Users with similar purchase patterns liked this item"
        elif signal_name == 'content':
            return "Similar to products in your history"
        elif signal_name == 'popularity':
            return "Trending item among similar shoppers"
        elif signal_name == 'freshness':
            return "Recently added item you might like"
            
        return "Recommended for you"
