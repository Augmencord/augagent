"""Rate Limiting and Caching Subsystem for AugAgent."""

import time
import abc
import os
import asyncio
from typing import Optional

try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

class BaseRateLimiter(abc.ABC):
    """Abstract interface for rate limiting."""
    
    @abc.abstractmethod
    async def is_rate_limited(self, tenant_id: str, endpoint: str, limit: int, window: int) -> bool:
        """Return True if the tenant has exceeded the limit within the window."""
        pass


class InMemoryRateLimiter(BaseRateLimiter):
    """In-memory rate limiter using a sliding window. Ideal for dev/testing."""
    
    def __init__(self):
        # Format: { "{tenant_id}:{endpoint}": [timestamp1, timestamp2, ...] }
        self._store = {}
        self._lock = asyncio.Lock()

    async def is_rate_limited(self, tenant_id: str, endpoint: str, limit: int, window: int) -> bool:
        key = f"{tenant_id}:{endpoint}"
        now = time.time()
        
        async with self._lock:
            timestamps = self._store.get(key, [])
            # Filter timestamps outside the window
            timestamps = [t for t in timestamps if now - t < window]
            
            if len(timestamps) >= limit:
                self._store[key] = timestamps
                return True
                
            timestamps.append(now)
            self._store[key] = timestamps
            return False


class RedisRateLimiter(BaseRateLimiter):
    """Distributed rate limiter using Redis sorted sets (sliding window)."""
    
    def __init__(self, redis_url: str):
        if not HAS_REDIS:
            raise ImportError("redis is required for RedisRateLimiter. Install with `pip install augagent[redis]`")
        self.redis = redis.from_url(redis_url, decode_responses=True)
        # Lua script for atomic sliding window rate limiting
        self.lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local clear_before = now - window
        
        redis.call('ZREMRANGEBYSCORE', key, 0, clear_before)
        local count = redis.call('ZCARD', key)
        
        if count >= limit then
            return 1
        else
            redis.call('ZADD', key, now, now)
            redis.call('EXPIRE', key, window)
            return 0
        end
        """

    async def is_rate_limited(self, tenant_id: str, endpoint: str, limit: int, window: int) -> bool:
        key = f"augagent:ratelimit:{tenant_id}:{endpoint}"
        now = time.time()
        
        result = await self.redis.eval(self.lua_script, 1, key, now, window, limit)
        return result == 1


# Auto-detect rate limiter based on environment
def get_rate_limiter() -> BaseRateLimiter:
    redis_url = os.getenv("REDIS_URL")
    if redis_url and HAS_REDIS:
        return RedisRateLimiter(redis_url)
    return InMemoryRateLimiter()

# Global rate limiter instance
rate_limiter = get_rate_limiter()
