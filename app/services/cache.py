import redis
import json
import logging
from typing import List, Dict, Optional
import os

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self, ttl_seconds: int = 86400):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            self.client = redis.from_url(redis_url, decode_responses=True)
            self.ttl = ttl_seconds
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None
            
    def get_recommendations(self, user_id: str, model_version: str) -> Optional[List[Dict]]:
        if not self.client:
            return None
            
        key = f"recs:{model_version}:{user_id}"
        
        try:
            # We store the full JSON payload as the value in the sorted set
            # Score is the rank (or -score) to keep it sorted
            cached_data = self.client.zrange(key, 0, -1, desc=True)
            if cached_data:
                logger.info(f"Cache hit for {key}")
                return [json.loads(item) for item in cached_data]
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
            
    def set_recommendations(self, user_id: str, model_version: str, recommendations: List[Dict]):
        if not self.client or not recommendations:
            return
            
        key = f"recs:{model_version}:{user_id}"
        
        try:
            pipeline = self.client.pipeline()
            # Clear existing
            pipeline.delete(key)
            
            # Add new (ZADD expects mapping {value: score})
            mapping = {json.dumps(rec): rec['score'] for rec in recommendations}
            pipeline.zadd(key, mapping)
            pipeline.expire(key, self.ttl)
            pipeline.execute()
            logger.info(f"Cached {len(recommendations)} items for {key}")
        except Exception as e:
            logger.error(f"Redis set error: {e}")
