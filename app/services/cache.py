import redis
import json
import logging
import time
from typing import List, Dict, Optional
import os

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self, ttl_seconds: int = 86400):
        self.ttl = ttl_seconds
        self.client = None
        self._memory_cache = {}  # In-memory fallback
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        # Try to connect to Redis
        try:
            kwargs = {
                "decode_responses": True,
                "socket_timeout": 2.5,         # Fail fast if Redis hangs, but allow serverless wakeups
                "socket_connect_timeout": 2.5  # Fail fast on connection
            }
            # Many managed Redis providers (like Render or Upstash) use rediss:// 
            # and may fail strict SSL certificate checks on free tiers.
            if ".upstash.io" in redis_url and redis_url.startswith("redis://"):
                redis_url = redis_url.replace("redis://", "rediss://", 1)
                
            if redis_url.startswith("rediss://"):
                kwargs["ssl_cert_reqs"] = None
                
            client = redis.from_url(redis_url, **kwargs)
            client.ping()  # Force connection test
            self.client = client
            logger.info(f"Successfully connected to Redis at {redis_url}")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to connect to Redis at {redis_url}. Error: {e}")
            logger.warning("Falling back to IN-MEMORY cache since Redis is unreachable.")
            self.client = None
            
    def get_recommendations(self, user_id: str, model_version: str) -> Optional[List[Dict]]:
        key = f"recs:{model_version}:{user_id}"
        
        if self.client:
            try:
                cached_data = self.client.zrange(key, 0, -1, desc=True)
                if cached_data:
                    logger.info(f"Redis cache hit for {key}")
                    return [json.loads(item) for item in cached_data]
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                
        # Fallback memory cache (used if client is None, or if Redis get failed/returned nothing)
        if key in self._memory_cache:
            data, expiry = self._memory_cache[key]
            if time.time() < expiry:
                logger.info(f"Memory cache hit for {key}")
                return data
            else:
                del self._memory_cache[key]
        
        return None
            
    def set_recommendations(self, user_id: str, model_version: str, recommendations: List[Dict]):
        if not recommendations:
            return
            
        key = f"recs:{model_version}:{user_id}"
        
        redis_success = False
        if self.client:
            try:
                pipeline = self.client.pipeline()
                pipeline.delete(key)
                mapping = {json.dumps(rec): rec['score'] for rec in recommendations}
                pipeline.zadd(key, mapping)
                pipeline.expire(key, self.ttl)
                pipeline.execute()
                logger.info(f"Cached {len(recommendations)} items in Redis for {key}")
                redis_success = True
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                
        if not redis_success:
            # Fallback memory cache
            self._memory_cache[key] = (recommendations, time.time() + self.ttl)
            logger.info(f"Cached {len(recommendations)} items in Memory for {key}")
