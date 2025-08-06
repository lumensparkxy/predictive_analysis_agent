"""
Memory optimizer for garbage collection and memory usage optimization.
"""

import gc
import threading
import weakref
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    memory_percent: float
    gc_collections: Dict[int, int]
    large_objects: int
    weak_references: int


class MemoryOptimizer:
    """
    Memory optimizer for monitoring memory usage, managing garbage collection,
    and optimizing memory allocation patterns.
    """
    
    def __init__(self):
        """Initialize memory optimizer."""
        self._weak_refs: Dict[str, weakref.ref] = {}
        self._object_pools: Dict[str, List[Any]] = {}
        self._memory_history: List[MemoryStats] = []
        self._lock = threading.Lock()
        
        # Memory thresholds
        self.high_memory_threshold = 85.0  # Percentage
        self.gc_trigger_threshold = 80.0
        self.large_object_threshold_mb = 10
        
        # Statistics
        self.forced_gc_count = 0
        self.memory_warnings = 0
        self.pool_hits = 0
        self.pool_misses = 0
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        try:
            # Try to get system memory info
            try:
                import psutil
                memory = psutil.virtual_memory()
                total_mb = memory.total / (1024 * 1024)
                available_mb = memory.available / (1024 * 1024)
                used_mb = memory.used / (1024 * 1024)
                percent = memory.percent
            except ImportError:
                # Fallback to basic info
                total_mb = 8192  # Assume 8GB
                available_mb = 2048  # Assume 2GB available
                used_mb = total_mb - available_mb
                percent = (used_mb / total_mb) * 100
            
            # Garbage collection stats
            gc_stats = {}
            for generation in range(3):
                gc_stats[generation] = gc.get_count()[generation]
            
            # Count large objects and weak references
            large_objects = len([obj for obj in gc.get_objects() 
                               if hasattr(obj, '__sizeof__') and 
                               obj.__sizeof__() > self.large_object_threshold_mb * 1024 * 1024])
            
            stats = MemoryStats(
                total_memory_mb=total_mb,
                available_memory_mb=available_mb,
                used_memory_mb=used_mb,
                memory_percent=percent,
                gc_collections=gc_stats,
                large_objects=large_objects,
                weak_references=len(self._weak_refs)
            )
            
            # Store in history
            with self._lock:
                self._memory_history.append(stats)
                if len(self._memory_history) > 1000:
                    self._memory_history = self._memory_history[-1000:]
            
            return stats
            
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return MemoryStats(0, 0, 0, 0, {}, 0, 0)
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization actions."""
        stats = self.get_memory_stats()
        actions_taken = []
        
        # Check if memory usage is high
        if stats.memory_percent > self.gc_trigger_threshold:
            # Force garbage collection
            collected = gc.collect()
            self.forced_gc_count += 1
            actions_taken.append(f"Forced GC collected {collected} objects")
            
            # Clean up weak references
            cleaned_refs = self._cleanup_weak_references()
            if cleaned_refs > 0:
                actions_taken.append(f"Cleaned {cleaned_refs} dead weak references")
        
        # Check for memory warning
        if stats.memory_percent > self.high_memory_threshold:
            self.memory_warnings += 1
            actions_taken.append("Memory usage above warning threshold")
        
        # Clean object pools
        cleaned_pools = self._cleanup_object_pools()
        if cleaned_pools > 0:
            actions_taken.append(f"Cleaned {cleaned_pools} object pools")
        
        return {
            'memory_stats': {
                'memory_percent': stats.memory_percent,
                'available_mb': stats.available_memory_mb,
                'large_objects': stats.large_objects
            },
            'actions_taken': actions_taken,
            'optimization_timestamp': datetime.now().isoformat()
        }
    
    def register_weak_reference(self, key: str, obj: Any, cleanup_callback: Optional[Callable] = None):
        """Register object with weak reference for memory tracking."""
        def cleanup_ref():
            self._weak_refs.pop(key, None)
            if cleanup_callback:
                cleanup_callback()
        
        self._weak_refs[key] = weakref.ref(obj, cleanup_ref)
    
    def get_object_pool(self, pool_name: str, factory: Callable, max_size: int = 100) -> Any:
        """Get object from pool or create new one."""
        with self._lock:
            pool = self._object_pools.setdefault(pool_name, [])
            
            if pool:
                obj = pool.pop()
                self.pool_hits += 1
                return obj
            else:
                self.pool_misses += 1
                return factory()
    
    def return_to_pool(self, pool_name: str, obj: Any, reset_func: Optional[Callable] = None):
        """Return object to pool for reuse."""
        with self._lock:
            pool = self._object_pools.setdefault(pool_name, [])
            
            if len(pool) < 100:  # Max pool size
                if reset_func:
                    reset_func(obj)
                pool.append(obj)
    
    def _cleanup_weak_references(self) -> int:
        """Clean up dead weak references."""
        with self._lock:
            dead_refs = []
            for key, ref in self._weak_refs.items():
                if ref() is None:
                    dead_refs.append(key)
            
            for key in dead_refs:
                self._weak_refs.pop(key, None)
            
            return len(dead_refs)
    
    def _cleanup_object_pools(self) -> int:
        """Clean up unused object pools."""
        with self._lock:
            cleaned = 0
            for pool_name, pool in list(self._object_pools.items()):
                if len(pool) > 50:  # Keep only 50 objects max per pool
                    removed = len(pool) - 50
                    self._object_pools[pool_name] = pool[:50]
                    cleaned += removed
            
            return cleaned
    
    def analyze_memory_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Analyze memory usage trends."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_stats = [
                stats for stats in self._memory_history
                if len(self._memory_history) > 0  # Basic check since we don't store timestamps
            ][-100:]  # Get last 100 readings
        
        if len(recent_stats) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate trends
        memory_usage = [s.memory_percent for s in recent_stats]
        start_usage = memory_usage[0]
        end_usage = memory_usage[-1]
        max_usage = max(memory_usage)
        min_usage = min(memory_usage)
        avg_usage = sum(memory_usage) / len(memory_usage)
        
        trend_direction = 'stable'
        if end_usage > start_usage + 5:
            trend_direction = 'increasing'
        elif end_usage < start_usage - 5:
            trend_direction = 'decreasing'
        
        return {
            'trend_direction': trend_direction,
            'start_usage_percent': start_usage,
            'end_usage_percent': end_usage,
            'max_usage_percent': max_usage,
            'min_usage_percent': min_usage,
            'avg_usage_percent': avg_usage,
            'volatility': max_usage - min_usage,
            'sample_count': len(recent_stats)
        }
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report."""
        current_stats = self.get_memory_stats()
        trends = self.analyze_memory_trends()
        
        return {
            'generated_at': datetime.now().isoformat(),
            'current_memory': {
                'memory_percent': current_stats.memory_percent,
                'available_mb': current_stats.available_memory_mb,
                'used_mb': current_stats.used_memory_mb,
                'large_objects': current_stats.large_objects,
                'weak_references': current_stats.weak_references
            },
            'trends': trends,
            'optimizer_stats': {
                'forced_gc_count': self.forced_gc_count,
                'memory_warnings': self.memory_warnings,
                'pool_hits': self.pool_hits,
                'pool_misses': self.pool_misses,
                'pool_hit_rate': (self.pool_hits / max(self.pool_hits + self.pool_misses, 1)) * 100
            },
            'recommendations': self._generate_memory_recommendations(current_stats, trends)
        }
    
    def _generate_memory_recommendations(self, stats: MemoryStats, trends: Dict) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        if stats.memory_percent > 80:
            recommendations.append("High memory usage detected - consider increasing available memory")
        
        if stats.large_objects > 100:
            recommendations.append("Many large objects detected - review data structures for optimization")
        
        if trends.get('trend_direction') == 'increasing':
            recommendations.append("Memory usage trending upward - investigate potential memory leaks")
        
        if trends.get('volatility', 0) > 20:
            recommendations.append("High memory usage volatility - review allocation patterns")
        
        pool_hit_rate = (self.pool_hits / max(self.pool_hits + self.pool_misses, 1)) * 100
        if pool_hit_rate < 50:
            recommendations.append("Low object pool hit rate - review pool sizing and usage patterns")
        
        return recommendations