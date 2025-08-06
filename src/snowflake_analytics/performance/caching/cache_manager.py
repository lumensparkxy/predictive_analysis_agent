"""
Central cache manager for multi-layer caching with intelligent cache policies.
"""

import time
import threading
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import OrderedDict
from enum import Enum
import hashlib
import weakref
import gc
from functools import wraps


class CacheLevel(Enum):
    """Cache levels for multi-layer caching."""
    MEMORY_L1 = "memory_l1"      # Fastest, smallest
    MEMORY_L2 = "memory_l2"      # Fast, medium size
    DISK = "disk"                # Slower, large capacity
    DISTRIBUTED = "distributed"  # Shared across instances


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    FIFO = "fifo"        # First In, First Out
    TTL = "ttl"          # Time To Live based
    ADAPTIVE = "adaptive" # Adaptive based on access patterns


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


class MemoryCache:
    """In-memory cache implementation with configurable eviction policies."""
    
    def __init__(self, 
                 max_size_mb: int = 100,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
                 default_ttl_seconds: Optional[int] = None):
        """
        Initialize memory cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            eviction_policy: Eviction policy to use
            default_ttl_seconds: Default TTL for entries
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.default_ttl_seconds = default_ttl_seconds
        
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = OrderedDict()  # For LRU
        self._frequency_counter: Dict[str, int] = {}  # For LFU
        self._current_size_bytes = 0
        self._lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired_entries = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self.misses += 1
                return None
            
            # Check if expired
            if entry.is_expired:
                self._remove_entry(key)
                self.misses += 1
                self.expired_entries += 1
                return None
            
            # Update access metadata
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
            # Update eviction policy data
            self._update_access_tracking(key)
            
            self.hits += 1
            return entry.value
    
    def put(self, 
            key: str, 
            value: Any, 
            ttl_seconds: Optional[int] = None,
            tags: List[str] = None) -> bool:
        """Put value in cache."""
        with self._lock:
            # Calculate size
            try:
                size_bytes = self._calculate_size(value)
            except Exception:
                size_bytes = 1024  # Default size if calculation fails
            
            # Check if we need to evict entries
            while (self._current_size_bytes + size_bytes > self.max_size_bytes and 
                   len(self._cache) > 0):
                evicted = self._evict_entry()
                if not evicted:
                    break
            
            # If still too large, don't cache
            if size_bytes > self.max_size_bytes:
                return False
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl_seconds=ttl_seconds or self.default_ttl_seconds,
                tags=tags or []
            )
            
            # Add to cache
            self._cache[key] = entry
            self._current_size_bytes += size_bytes
            self._update_access_tracking(key)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            return self._remove_entry(key)
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._frequency_counter.clear()
            self._current_size_bytes = 0
    
    def evict_by_tags(self, tags: List[str]):
        """Evict all entries with specified tags."""
        with self._lock:
            keys_to_remove = []
            for key, entry in self._cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
    
    def cleanup_expired(self):
        """Remove all expired entries."""
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
                self.expired_entries += 1
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback size estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in value.items())
            else:
                return 1024  # Default size
    
    def _update_access_tracking(self, key: str):
        """Update access tracking for eviction policies."""
        if self.eviction_policy == EvictionPolicy.LRU:
            # Move to end of OrderedDict
            self._access_order.pop(key, None)
            self._access_order[key] = True
        
        elif self.eviction_policy == EvictionPolicy.LFU:
            self._frequency_counter[key] = self._frequency_counter.get(key, 0) + 1
    
    def _evict_entry(self) -> bool:
        """Evict an entry based on the eviction policy."""
        if not self._cache:
            return False
        
        key_to_evict = None
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used (first in OrderedDict)
            key_to_evict = next(iter(self._access_order))
        
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            if self._frequency_counter:
                key_to_evict = min(self._frequency_counter.items(), key=lambda x: x[1])[0]
        
        elif self.eviction_policy == EvictionPolicy.FIFO:
            # Remove oldest entry
            oldest_key = None
            oldest_time = None
            for key, entry in self._cache.items():
                if oldest_time is None or entry.created_at < oldest_time:
                    oldest_time = entry.created_at
                    oldest_key = key
            key_to_evict = oldest_key
        
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Remove entry with shortest remaining TTL
            shortest_ttl_key = None
            shortest_remaining = float('inf')
            
            for key, entry in self._cache.items():
                if entry.ttl_seconds:
                    remaining = entry.ttl_seconds - entry.age_seconds
                    if remaining < shortest_remaining:
                        shortest_remaining = remaining
                        shortest_ttl_key = key
            
            key_to_evict = shortest_ttl_key or next(iter(self._cache))
        
        elif self.eviction_policy == EvictionPolicy.ADAPTIVE:
            # Adaptive eviction based on access patterns
            key_to_evict = self._adaptive_evict()
        
        if key_to_evict:
            self._remove_entry(key_to_evict)
            self.evictions += 1
            return True
        
        return False
    
    def _adaptive_evict(self) -> Optional[str]:
        """Adaptive eviction considering multiple factors."""
        if not self._cache:
            return None
        
        best_candidate = None
        best_score = float('inf')
        
        for key, entry in self._cache.items():
            # Score based on multiple factors (lower is better)
            age_factor = entry.age_seconds / 3600  # Hours
            frequency_factor = 1.0 / max(entry.access_count, 1)
            size_factor = entry.size_bytes / (1024 * 1024)  # MB
            recency_factor = (datetime.now() - entry.last_accessed).total_seconds() / 3600
            
            score = age_factor + frequency_factor + size_factor * 0.1 + recency_factor
            
            if score < best_score:
                best_score = score
                best_candidate = key
        
        return best_candidate
    
    def _remove_entry(self, key: str) -> bool:
        """Remove entry and update tracking structures."""
        entry = self._cache.pop(key, None)
        if entry:
            self._current_size_bytes -= entry.size_bytes
            self._access_order.pop(key, None)
            self._frequency_counter.pop(key, None)
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'entries': len(self._cache),
                'size_mb': self._current_size_bytes / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization_percent': (self._current_size_bytes / self.max_size_bytes) * 100,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate_percent': hit_rate,
                'evictions': self.evictions,
                'expired_entries': self.expired_entries
            }


class DiskCache:
    """Disk-based cache for larger, less frequently accessed data."""
    
    def __init__(self, cache_dir: str = '/tmp/cache', max_size_mb: int = 1000):
        """Initialize disk cache."""
        import os
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        self._metadata: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        try:
            with self._lock:
                if key not in self._metadata:
                    return None
                
                metadata = self._metadata[key]
                
                # Check if expired
                if self._is_expired(metadata):
                    self._remove_entry(key)
                    return None
                
                # Load from disk
                filepath = self._get_filepath(key)
                with open(filepath, 'rb') as f:
                    value = pickle.load(f)
                
                # Update access time
                metadata['last_accessed'] = datetime.now().isoformat()
                metadata['access_count'] += 1
                
                return value
        except Exception as e:
            print(f"Error reading from disk cache: {e}")
            return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Put value in disk cache."""
        try:
            with self._lock:
                # Serialize value
                serialized = pickle.dumps(value)
                size_bytes = len(serialized)
                
                # Check space and evict if needed
                self._ensure_space(size_bytes)
                
                # Save to disk
                filepath = self._get_filepath(key)
                with open(filepath, 'wb') as f:
                    f.write(serialized)
                
                # Update metadata
                self._metadata[key] = {
                    'size_bytes': size_bytes,
                    'created_at': datetime.now().isoformat(),
                    'last_accessed': datetime.now().isoformat(),
                    'access_count': 1,
                    'ttl_seconds': ttl_seconds,
                    'filepath': filepath
                }
                
                return True
        except Exception as e:
            print(f"Error writing to disk cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete entry from disk cache."""
        with self._lock:
            return self._remove_entry(key)
    
    def _get_filepath(self, key: str) -> str:
        """Get filepath for cache key."""
        import os
        # Use hash to avoid filesystem issues with key names
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def _is_expired(self, metadata: Dict) -> bool:
        """Check if entry is expired."""
        if not metadata.get('ttl_seconds'):
            return False
        
        created_at = datetime.fromisoformat(metadata['created_at'])
        return (datetime.now() - created_at).total_seconds() > metadata['ttl_seconds']
    
    def _ensure_space(self, needed_bytes: int):
        """Ensure sufficient space by evicting entries if needed."""
        current_size = sum(meta['size_bytes'] for meta in self._metadata.values())
        
        while current_size + needed_bytes > self.max_size_bytes and self._metadata:
            # Find oldest entry to evict (simple FIFO)
            oldest_key = min(self._metadata.items(), 
                           key=lambda x: x[1]['created_at'])[0]
            self._remove_entry(oldest_key)
            current_size = sum(meta['size_bytes'] for meta in self._metadata.values())
    
    def _remove_entry(self, key: str) -> bool:
        """Remove entry from disk and metadata."""
        metadata = self._metadata.pop(key, None)
        if metadata:
            try:
                import os
                if os.path.exists(metadata['filepath']):
                    os.remove(metadata['filepath'])
                return True
            except Exception as e:
                print(f"Error removing disk cache file: {e}")
        return False


class CacheManager:
    """
    Central cache manager that coordinates multi-layer caching with
    intelligent cache policies and automatic optimization.
    """
    
    def __init__(self):
        """Initialize cache manager."""
        self.caches: Dict[CacheLevel, Any] = {}
        self._cache_policies: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        
        # Initialize default caches
        self.caches[CacheLevel.MEMORY_L1] = MemoryCache(
            max_size_mb=50,
            eviction_policy=EvictionPolicy.LRU,
            default_ttl_seconds=300  # 5 minutes
        )
        
        self.caches[CacheLevel.MEMORY_L2] = MemoryCache(
            max_size_mb=200,
            eviction_policy=EvictionPolicy.ADAPTIVE,
            default_ttl_seconds=1800  # 30 minutes
        )
        
        self.caches[CacheLevel.DISK] = DiskCache(
            cache_dir='/tmp/cache',
            max_size_mb=1000
        )
        
        # Global statistics
        self.total_gets = 0
        self.total_puts = 0
        self.cache_promotions = 0
        self.cache_demotions = 0
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache, checking layers in order.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        self.total_gets += 1
        
        # Check L1 cache first (fastest)
        value = self.caches[CacheLevel.MEMORY_L1].get(key)
        if value is not None:
            return value
        
        # Check L2 cache
        value = self.caches[CacheLevel.MEMORY_L2].get(key)
        if value is not None:
            # Promote to L1 cache
            self.caches[CacheLevel.MEMORY_L1].put(key, value, ttl_seconds=300)
            self.cache_promotions += 1
            return value
        
        # Check disk cache
        value = self.caches[CacheLevel.DISK].get(key)
        if value is not None:
            # Promote to L2 cache
            self.caches[CacheLevel.MEMORY_L2].put(key, value, ttl_seconds=1800)
            self.cache_promotions += 1
            return value
        
        return default
    
    def put(self, 
            key: str, 
            value: Any, 
            ttl_seconds: Optional[int] = None,
            cache_level: CacheLevel = CacheLevel.MEMORY_L1,
            tags: List[str] = None) -> bool:
        """
        Put value in specified cache level.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            cache_level: Target cache level
            tags: Tags for cache invalidation
            
        Returns:
            True if successfully cached
        """
        self.total_puts += 1
        
        if cache_level == CacheLevel.MEMORY_L1 or cache_level == CacheLevel.MEMORY_L2:
            return self.caches[cache_level].put(key, value, ttl_seconds, tags)
        elif cache_level == CacheLevel.DISK:
            return self.caches[cache_level].put(key, value, ttl_seconds)
        
        return False
    
    def put_multilayer(self, 
                      key: str, 
                      value: Any, 
                      ttl_seconds: Optional[int] = None,
                      tags: List[str] = None) -> bool:
        """
        Put value in multiple cache layers based on intelligent policies.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            tags: Tags for cache invalidation
            
        Returns:
            True if cached in at least one layer
        """
        success = False
        
        # Always try L1 cache for fast access
        if self.caches[CacheLevel.MEMORY_L1].put(key, value, ttl_seconds or 300, tags):
            success = True
        
        # Determine if value should be cached in L2 based on size and expected access
        try:
            value_size = len(pickle.dumps(value))
            
            # Cache in L2 if reasonably sized and longer TTL
            if value_size < 1024 * 1024:  # Less than 1MB
                if self.caches[CacheLevel.MEMORY_L2].put(key, value, ttl_seconds or 1800, tags):
                    success = True
            
            # Cache on disk for large values or long-term storage
            if value_size > 1024 * 10 or (ttl_seconds and ttl_seconds > 3600):  # >10KB or >1hour TTL
                if self.caches[CacheLevel.DISK].put(key, value, ttl_seconds):
                    success = True
        except Exception:
            pass
        
        return success
    
    def delete(self, key: str):
        """Delete key from all cache layers."""
        for cache in self.caches.values():
            cache.delete(key)
    
    def invalidate_by_tags(self, tags: List[str]):
        """Invalidate cache entries by tags."""
        for cache_level, cache in self.caches.items():
            if hasattr(cache, 'evict_by_tags'):
                cache.evict_by_tags(tags)
    
    def clear_all(self):
        """Clear all cache layers."""
        for cache in self.caches.values():
            cache.clear()
    
    def cleanup_expired(self):
        """Remove expired entries from all caches."""
        for cache in self.caches.values():
            if hasattr(cache, 'cleanup_expired'):
                cache.cleanup_expired()
    
    def cache_decorator(self, 
                       ttl_seconds: int = 300,
                       cache_level: CacheLevel = CacheLevel.MEMORY_L1,
                       key_func: Optional[Callable] = None,
                       tags: List[str] = None):
        """
        Decorator for caching function results.
        
        Args:
            ttl_seconds: Cache TTL
            cache_level: Target cache level
            key_func: Custom key generation function
            tags: Tags for cache invalidation
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__module__}.{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
                
                # Try to get from cache
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.put(cache_key, result, ttl_seconds, cache_level, tags)
                
                return result
            
            return wrapper
        return decorator
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'total_gets': self.total_gets,
            'total_puts': self.total_puts,
            'cache_promotions': self.cache_promotions,
            'cache_demotions': self.cache_demotions,
            'layers': {}
        }
        
        for level, cache in self.caches.items():
            if hasattr(cache, 'get_stats'):
                stats['layers'][level.value] = cache.get_stats()
        
        # Calculate overall hit rate
        total_hits = sum(layer.get('hits', 0) for layer in stats['layers'].values())
        total_misses = sum(layer.get('misses', 0) for layer in stats['layers'].values())
        total_requests = total_hits + total_misses
        
        if total_requests > 0:
            stats['overall_hit_rate_percent'] = (total_hits / total_requests) * 100
        else:
            stats['overall_hit_rate_percent'] = 0
        
        return stats
    
    def optimize_cache_configuration(self) -> Dict[str, Any]:
        """Analyze usage patterns and suggest cache optimizations."""
        stats = self.get_statistics()
        recommendations = []
        
        # Analyze L1 cache performance
        l1_stats = stats['layers'].get('memory_l1', {})
        l1_hit_rate = l1_stats.get('hit_rate_percent', 0)
        l1_utilization = l1_stats.get('utilization_percent', 0)
        
        if l1_hit_rate < 60:
            recommendations.append({
                'cache': 'L1',
                'issue': 'Low hit rate',
                'suggestion': 'Consider increasing L1 cache size or adjusting TTL',
                'current_hit_rate': l1_hit_rate
            })
        
        if l1_utilization > 90:
            recommendations.append({
                'cache': 'L1',
                'issue': 'High utilization',
                'suggestion': 'Increase L1 cache size to reduce evictions',
                'current_utilization': l1_utilization
            })
        
        # Analyze promotion/demotion patterns
        promotion_rate = (self.cache_promotions / max(self.total_gets, 1)) * 100
        if promotion_rate > 20:
            recommendations.append({
                'cache': 'Multi-layer',
                'issue': 'High promotion rate',
                'suggestion': 'Consider adjusting cache layer sizes or policies',
                'promotion_rate': promotion_rate
            })
        
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'overall_performance': {
                'hit_rate': stats.get('overall_hit_rate_percent', 0),
                'total_requests': self.total_gets,
                'promotion_rate': promotion_rate
            },
            'recommendations': recommendations,
            'layer_performance': stats['layers']
        }
    
    def export_cache_report(self, filepath: str) -> bool:
        """Export comprehensive cache performance report."""
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'statistics': self.get_statistics(),
                'optimization_analysis': self.optimize_cache_configuration()
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting cache report: {e}")
            return False