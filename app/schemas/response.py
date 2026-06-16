from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class SignalDecomposition(BaseModel):
    als: Optional[float] = 0.0
    bpr: Optional[float] = 0.0
    content: Optional[float] = 0.0
    popularity: Optional[float] = 0.0
    freshness: Optional[float] = 0.0

class RecommendationItem(BaseModel):
    item_id: str
    rank: int
    score: float
    reason: str
    signals: Optional[SignalDecomposition] = None

class RecommendationResponse(BaseModel):
    user_id: str
    model_version: str
    cached: bool
    recommendations: List[RecommendationItem]
    latency_ms: float
