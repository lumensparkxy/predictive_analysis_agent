"""
Cache management system using diskcache for high-performance local caching.

Provides intelligent caching for query results, processed data,
and computed metrics with automatic cleanup and size management.
"""

import logging
from typing import Any, Dict, Optional

from diskcache import Cache

logger = logging.getLogger(__name__)


class CacheStore:
    """High-performance disk-based cache manager."""
    
    def __init__(self, cache_dir: str = "cache", size_limit: int = 1_000_000_000):  # 1GB default
        """Initialize cache store with size limits."""
        try:
            self.cache = Cache(cache_dir, size_limit=size_limit)
            logger.info(f"Cache initialized: {cache_dir}, limit: {size_limit / (1024**3):.1f}GB")
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with hit/miss tracking."""
        try:
            value = self.cache.get(key, default)
            if value != default:
                logger.debug(f"Cache HIT: {key}")
            else:
                logger.debug(f"Cache MISS: {key}")
            return value
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL in seconds."""
        try:
            if ttl:
                success = self.cache.set(key, value, expire=ttl)
            else:
                success = self.cache.set(key, value)
            
            if success:
                logger.debug(f"Cache SET: {key}" + (f" (TTL: {ttl}s)" if ttl else ""))
            else:
                logger.warning(f"Cache SET failed: {key}")
            
            return success
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            success = self.cache.delete(key)
            if success:
                logger.debug(f"Cache DELETE: {key}")
            return success
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            self.cache.clear()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return key in self.cache
        except Exception as e:
            logger.error(f"Cache exists check error for key {key}: {e}")
            return False
    
    def get_many(self, keys: list) -> Dict[str, Any]:
        """Get multiple values from cache."""
        results = {}
        for key in keys:
            results[key] = self.get(key)
        return results
    
    def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> Dict[str, bool]:
        """Set multiple values in cache."""
        results = {}
        for key, value in mapping.items():
            results[key] = self.set(key, value, ttl)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            stats = {
                "size_bytes": self.cache.volume(),
                "size_mb": self.cache.volume() / (1024 * 1024),
                "count": len(self.cache),
                "hit_rate": 0.0,
                "directory": str(self.cache.directory)
            }
            
            # Calculate hit rate if stats are available
            if hasattr(self.cache, 'hits') and hasattr(self.cache, 'misses'):
                total_requests = self.cache.hits + self.cache.misses
                if total_requests > 0:
                    stats["hit_rate"] = self.cache.hits / total_requests
                stats["hits"] = self.cache.hits
                stats["misses"] = self.cache.misses
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
    
    def cleanup_expired(self) -> int:
        """Clean up expired cache entries and return count."""
        try:
            expired_count = self.cache.expire()
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired cache entries")
            return expired_count
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache entries: {e}")
            return 0
    
    def optimize(self) -> Dict[str, int]:
        """Optimize cache by removing expired entries and defragmenting."""
        try:
            stats_before = self.get_stats()
            
            # Clean expired entries
            expired_count = self.cleanup_expired()
            
            # Get stats after cleanup
            stats_after = self.get_stats()
            
            optimization_stats = {
                "expired_removed": expired_count,
                "size_before_mb": stats_before.get("size_mb", 0),
                "size_after_mb": stats_after.get("size_mb", 0),
                "space_freed_mb": stats_before.get("size_mb", 0) - stats_after.get("size_mb", 0)
            }
            
            logger.info(f"Cache optimization completed: {optimization_stats}")
            return optimization_stats
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            return {"error": str(e)}
    
    def get_keys_by_pattern(self, pattern: str) -> list:
        """Get all keys matching a pattern (simple string matching)."""
        try:
            matching_keys = []
            for key in self.cache:
                if pattern in key:
                    matching_keys.append(key)
            return matching_keys
        except Exception as e:
            logger.error(f"Failed to get keys by pattern {pattern}: {e}")
            return []
    
    def delete_by_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        matching_keys = self.get_keys_by_pattern(pattern)
        deleted_count = 0
        
        for key in matching_keys:
            if self.delete(key):
                deleted_count += 1
        
        logger.info(f"Deleted {deleted_count} keys matching pattern: {pattern}")
        return deleted_count
    
    def is_healthy(self) -> bool:
        """Check if cache is healthy and responsive."""
        try:
            # Test basic operations
            test_key = "__health_check__"
            test_value = "healthy"
            
            # Test set
            if not self.set(test_key, test_value):
                return False
            
            # Test get
            if self.get(test_key) != test_value:
                return False
            
            # Test delete
            if not self.delete(test_key):
                return False
            
            return True
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return False


# Global cache instance
_global_cache = None


def get_cache() -> CacheStore:
    """Get global cache instance (singleton pattern)."""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheStore()
    return _global_cache


def cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments."""
    import hashlib
    
    # Create a string representation of all arguments
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        key_parts.append(str(arg))
    
    # Add keyword arguments (sorted for consistency)
    for key, value in sorted(kwargs.items()):
        key_parts.append(f"{key}={value}")
    
    # Create hash of the combined key
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached_function(ttl: int = 3600):
    """Decorator to cache function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            key = f"func:{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for function {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            logger.debug(f"Cached result for function {func.__name__}")
            
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test the cache store
    import time
    
    cache = CacheStore("test_cache", size_limit=10_000_000)  # 10MB for testing
    
    # Test basic operations
    cache.set("test_key", {"data": "test_value", "timestamp": time.time()})
    value = cache.get("test_key")
    print("Cached value:", value)
    
    # Test TTL
    cache.set("ttl_key", "expires_soon", ttl=2)
    print("TTL value (should exist):", cache.get("ttl_key"))
    time.sleep(3)
    print("TTL value (should be None):", cache.get("ttl_key"))
    
    # Test function caching
    @cached_function(ttl=60)
    def expensive_function(x, y):
        time.sleep(0.1)  # Simulate expensive operation
        return x * y + time.time()
    
    start_time = time.time()
    result1 = expensive_function(5, 10)
    first_call_time = time.time() - start_time
    
    start_time = time.time()
    result2 = expensive_function(5, 10)  # Should be cached
    second_call_time = time.time() - start_time
    
    print(f"First call: {first_call_time:.3f}s, Second call: {second_call_time:.3f}s")
    print(f"Results match: {result1 == result2}")
    
    # Test stats
    stats = cache.get_stats()
    print("Cache stats:", stats)
    
    # Test health check
    health = cache.is_healthy()
    print(f"Cache healthy: {health}")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_cache", ignore_errors=True)
    
    print("✅ Cache store test completed successfully")
