"""
Cache strategies for different use cases and access patterns.
"""

from enum import Enum
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass


class AccessPattern(Enum):
    """Different access patterns for cache optimization."""
    TEMPORAL = "temporal"          # Time-based access (recent items accessed more)
    FREQUENCY = "frequency"        # Frequency-based (popular items accessed more)
    SEQUENTIAL = "sequential"      # Sequential access pattern
    RANDOM = "random"             # Random access pattern
    SEASONAL = "seasonal"         # Seasonal/cyclical access pattern


class CacheStrategy(Enum):
    """Different caching strategies."""
    WRITE_THROUGH = "write_through"      # Write to cache and storage simultaneously
    WRITE_BACK = "write_back"           # Write to cache, sync to storage later
    WRITE_AROUND = "write_around"       # Write directly to storage, bypass cache
    READ_THROUGH = "read_through"       # Read from cache, load from storage if miss
    LAZY_LOADING = "lazy_loading"       # Load into cache only when requested


@dataclass
class CacheConfiguration:
    """Cache configuration for specific use case."""
    strategy: CacheStrategy
    access_pattern: AccessPattern
    ttl_seconds: int
    max_size_mb: int
    eviction_policy: str
    prefetch_enabled: bool = False
    compression_enabled: bool = False


class CacheStrategies:
    """
    Intelligent cache strategies that adapt to different access patterns
    and use cases for optimal performance.
    """
    
    def __init__(self):
        """Initialize cache strategies."""
        self.access_patterns: Dict[str, AccessPattern] = {}
        self.strategy_configs: Dict[str, CacheConfiguration] = {}
        self._access_history: Dict[str, List[datetime]] = {}
        
        # Predefined strategy configurations
        self._init_predefined_strategies()
    
    def _init_predefined_strategies(self):
        """Initialize predefined cache strategies."""
        
        # Strategy for frequently accessed data (user sessions, etc.)
        self.strategy_configs['hot_data'] = CacheConfiguration(
            strategy=CacheStrategy.WRITE_THROUGH,
            access_pattern=AccessPattern.FREQUENCY,
            ttl_seconds=300,  # 5 minutes
            max_size_mb=100,
            eviction_policy='lfu',
            prefetch_enabled=True
        )
        
        # Strategy for time-series data
        self.strategy_configs['time_series'] = CacheConfiguration(
            strategy=CacheStrategy.READ_THROUGH,
            access_pattern=AccessPattern.TEMPORAL,
            ttl_seconds=3600,  # 1 hour
            max_size_mb=500,
            eviction_policy='lru',
            prefetch_enabled=True
        )
        
        # Strategy for reference data (relatively static)
        self.strategy_configs['reference_data'] = CacheConfiguration(
            strategy=CacheStrategy.LAZY_LOADING,
            access_pattern=AccessPattern.RANDOM,
            ttl_seconds=86400,  # 24 hours
            max_size_mb=200,
            eviction_policy='lru',
            prefetch_enabled=False
        )
        
        # Strategy for temporary/computed data
        self.strategy_configs['computed_data'] = CacheConfiguration(
            strategy=CacheStrategy.WRITE_BACK,
            access_pattern=AccessPattern.FREQUENCY,
            ttl_seconds=1800,  # 30 minutes
            max_size_mb=300,
            eviction_policy='adaptive',
            prefetch_enabled=False,
            compression_enabled=True
        )
        
        # Strategy for large datasets with sequential access
        self.strategy_configs['large_sequential'] = CacheConfiguration(
            strategy=CacheStrategy.READ_THROUGH,
            access_pattern=AccessPattern.SEQUENTIAL,
            ttl_seconds=7200,  # 2 hours
            max_size_mb=1000,
            eviction_policy='fifo',
            prefetch_enabled=True
        )
    
    def analyze_access_pattern(self, key: str, access_times: List[datetime] = None) -> AccessPattern:
        """
        Analyze access pattern for a cache key.
        
        Args:
            key: Cache key to analyze
            access_times: List of access timestamps (optional)
            
        Returns:
            Detected access pattern
        """
        if access_times:
            self._access_history[key] = access_times
        elif key not in self._access_history:
            return AccessPattern.RANDOM
        
        access_times = self._access_history[key]
        
        if len(access_times) < 3:
            return AccessPattern.RANDOM
        
        # Sort access times
        sorted_times = sorted(access_times)
        
        # Calculate time intervals between accesses
        intervals = []
        for i in range(1, len(sorted_times)):
            interval = (sorted_times[i] - sorted_times[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return AccessPattern.RANDOM
        
        # Analyze patterns
        avg_interval = sum(intervals) / len(intervals)
        interval_variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        coefficient_of_variation = (interval_variance ** 0.5) / avg_interval if avg_interval > 0 else float('inf')
        
        # Recent access bias (temporal pattern)
        recent_threshold = datetime.now() - timedelta(hours=1)
        recent_accesses = sum(1 for t in access_times if t >= recent_threshold)
        recent_bias = recent_accesses / len(access_times)
        
        # Determine pattern
        if recent_bias > 0.7:
            return AccessPattern.TEMPORAL
        elif coefficient_of_variation < 0.3:  # Low variance = regular pattern
            if avg_interval < 60:  # Less than 1 minute average
                return AccessPattern.FREQUENCY
            else:
                return AccessPattern.SEASONAL
        elif len(access_times) > 10 and self._is_sequential_pattern(access_times):
            return AccessPattern.SEQUENTIAL
        else:
            return AccessPattern.RANDOM
    
    def _is_sequential_pattern(self, access_times: List[datetime]) -> bool:
        """Check if access pattern is sequential."""
        # Simplified sequential detection - in practice would be more sophisticated
        sorted_times = sorted(access_times)
        gaps = []
        
        for i in range(1, len(sorted_times)):
            gap = (sorted_times[i] - sorted_times[i-1]).total_seconds()
            gaps.append(gap)
        
        # If most gaps are similar, likely sequential
        if len(gaps) < 2:
            return False
        
        avg_gap = sum(gaps) / len(gaps)
        similar_gaps = sum(1 for gap in gaps if abs(gap - avg_gap) < avg_gap * 0.5)
        
        return similar_gaps / len(gaps) > 0.6
    
    def recommend_strategy(self, 
                          key: str,
                          data_size_mb: float,
                          expected_access_frequency: int,
                          data_volatility: str = 'medium') -> str:
        """
        Recommend cache strategy based on data characteristics.
        
        Args:
            key: Cache key
            data_size_mb: Size of data in MB
            expected_access_frequency: Expected accesses per hour
            data_volatility: 'low', 'medium', 'high'
            
        Returns:
            Recommended strategy name
        """
        # Analyze access pattern if available
        pattern = self.access_patterns.get(key, AccessPattern.RANDOM)
        
        # Large data with high frequency -> hot_data strategy
        if expected_access_frequency > 100 and data_size_mb < 50:
            return 'hot_data'
        
        # Time-based access pattern
        elif pattern == AccessPattern.TEMPORAL:
            return 'time_series'
        
        # Large sequential data
        elif data_size_mb > 100 and pattern == AccessPattern.SEQUENTIAL:
            return 'large_sequential'
        
        # Computed/volatile data
        elif data_volatility == 'high':
            return 'computed_data'
        
        # Default to reference data strategy
        else:
            return 'reference_data'
    
    def get_cache_configuration(self, strategy_name: str) -> Optional[CacheConfiguration]:
        """Get cache configuration for strategy."""
        return self.strategy_configs.get(strategy_name)
    
    def optimize_strategy(self, 
                         key: str,
                         current_hit_rate: float,
                         current_strategy: str) -> Dict[str, Any]:
        """
        Optimize cache strategy based on performance metrics.
        
        Args:
            key: Cache key
            current_hit_rate: Current cache hit rate (0-1)
            current_strategy: Current strategy name
            
        Returns:
            Optimization recommendations
        """
        recommendations = {
            'current_strategy': current_strategy,
            'current_hit_rate': current_hit_rate,
            'recommendations': [],
            'suggested_strategy': current_strategy
        }
        
        # Low hit rate analysis
        if current_hit_rate < 0.5:
            recommendations['recommendations'].append("Low hit rate detected")
            
            # Analyze access pattern
            pattern = self.access_patterns.get(key, AccessPattern.RANDOM)
            
            if pattern == AccessPattern.TEMPORAL:
                if current_strategy != 'time_series':
                    recommendations['suggested_strategy'] = 'time_series'
                    recommendations['recommendations'].append("Switch to time_series strategy for temporal access pattern")
                else:
                    recommendations['recommendations'].append("Consider increasing TTL for time_series strategy")
            
            elif pattern == AccessPattern.FREQUENCY:
                if current_strategy != 'hot_data':
                    recommendations['suggested_strategy'] = 'hot_data'
                    recommendations['recommendations'].append("Switch to hot_data strategy for frequent access")
                else:
                    recommendations['recommendations'].append("Consider increasing cache size for hot_data strategy")
        
        # High hit rate but could optimize for other factors
        elif current_hit_rate > 0.8:
            recommendations['recommendations'].append("Good hit rate - consider optimizing for memory usage")
            
            current_config = self.strategy_configs.get(current_strategy)
            if current_config and not current_config.compression_enabled:
                recommendations['recommendations'].append("Enable compression to reduce memory usage")
        
        # Medium hit rate optimization
        else:
            recommendations['recommendations'].append("Moderate hit rate - fine-tune current strategy")
            
            current_config = self.strategy_configs.get(current_strategy)
            if current_config:
                if current_config.prefetch_enabled:
                    recommendations['recommendations'].append("Prefetching enabled - monitor for over-caching")
                else:
                    recommendations['recommendations'].append("Consider enabling prefetching")
        
        return recommendations
    
    def create_custom_strategy(self,
                             name: str,
                             strategy: CacheStrategy,
                             access_pattern: AccessPattern,
                             ttl_seconds: int,
                             max_size_mb: int,
                             eviction_policy: str = 'lru',
                             prefetch_enabled: bool = False,
                             compression_enabled: bool = False) -> bool:
        """Create custom cache strategy."""
        try:
            config = CacheConfiguration(
                strategy=strategy,
                access_pattern=access_pattern,
                ttl_seconds=ttl_seconds,
                max_size_mb=max_size_mb,
                eviction_policy=eviction_policy,
                prefetch_enabled=prefetch_enabled,
                compression_enabled=compression_enabled
            )
            
            self.strategy_configs[name] = config
            return True
        except Exception as e:
            print(f"Error creating custom strategy: {e}")
            return False
    
    def analyze_cache_effectiveness(self, metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Analyze effectiveness of current cache strategies.
        
        Args:
            metrics: Cache metrics by strategy name
            
        Returns:
            Effectiveness analysis
        """
        analysis = {
            'overall_performance': 'good',
            'strategy_performance': {},
            'recommendations': []
        }
        
        total_hit_rate = 0
        strategy_count = 0
        
        for strategy_name, strategy_metrics in metrics.items():
            hit_rate = strategy_metrics.get('hit_rate_percent', 0) / 100
            utilization = strategy_metrics.get('utilization_percent', 0)
            evictions = strategy_metrics.get('evictions', 0)
            
            # Score strategy performance
            performance_score = hit_rate * 0.6 + (1 - utilization/100) * 0.2 + (1 - min(evictions/1000, 1)) * 0.2
            
            strategy_analysis = {
                'hit_rate': hit_rate,
                'utilization': utilization,
                'evictions': evictions,
                'performance_score': performance_score,
                'grade': self._grade_performance(performance_score)
            }
            
            analysis['strategy_performance'][strategy_name] = strategy_analysis
            
            total_hit_rate += hit_rate
            strategy_count += 1
            
            # Generate strategy-specific recommendations
            if performance_score < 0.6:
                analysis['recommendations'].append(f"Strategy '{strategy_name}' underperforming - review configuration")
        
        # Overall performance assessment
        if strategy_count > 0:
            avg_hit_rate = total_hit_rate / strategy_count
            if avg_hit_rate > 0.8:
                analysis['overall_performance'] = 'excellent'
            elif avg_hit_rate > 0.6:
                analysis['overall_performance'] = 'good'
            elif avg_hit_rate > 0.4:
                analysis['overall_performance'] = 'fair'
            else:
                analysis['overall_performance'] = 'poor'
        
        return analysis
    
    def _grade_performance(self, score: float) -> str:
        """Grade performance score."""
        if score >= 0.8:
            return 'A'
        elif score >= 0.6:
            return 'B'
        elif score >= 0.4:
            return 'C'
        elif score >= 0.2:
            return 'D'
        else:
            return 'F'
    
    def export_strategy_analysis(self, filepath: str, metrics: Dict[str, Dict]) -> bool:
        """Export cache strategy analysis to file."""
        try:
            analysis = {
                'generated_at': datetime.now().isoformat(),
                'available_strategies': {
                    name: {
                        'strategy': config.strategy.value,
                        'access_pattern': config.access_pattern.value,
                        'ttl_seconds': config.ttl_seconds,
                        'max_size_mb': config.max_size_mb,
                        'eviction_policy': config.eviction_policy,
                        'prefetch_enabled': config.prefetch_enabled,
                        'compression_enabled': config.compression_enabled
                    }
                    for name, config in self.strategy_configs.items()
                },
                'effectiveness_analysis': self.analyze_cache_effectiveness(metrics),
                'access_patterns': {
                    key: pattern.value 
                    for key, pattern in self.access_patterns.items()
                }
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting strategy analysis: {e}")
            return False