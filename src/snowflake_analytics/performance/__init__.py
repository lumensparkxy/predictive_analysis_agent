"""
Performance optimization module for Snowflake Analytics.

Provides comprehensive performance monitoring, profiling, and optimization capabilities
for enterprise-grade scalability and efficiency.
"""

from .profiling import (
    SystemProfiler, ApplicationProfiler, DatabaseProfiler, 
    APIProfiler, BottleneckAnalyzer
)
from .database import (
    SchemaOptimizer, QueryOptimizer, ConnectionOptimizer,
    IndexManager, PartitionManager
)
from .caching import (
    CacheManager, MemoryOptimizer, CacheStrategies,
    InvalidationManager, CacheMonitor
)
from .api import (
    ResponseOptimizer, AsyncProcessor, CompressionManager,
    RateLimiter, LoadBalancer
)
from .processing import (
    PipelineOptimizer, ParallelProcessor, MLOptimizer,
    StreamingProcessor, TransformationOptimizer
)
from .monitoring import (
    PerformanceMonitor, AutoScaler, MetricsCollector,
    AlertingEngine, BenchmarkRunner
)

__all__ = [
    # Profiling components
    'SystemProfiler', 'ApplicationProfiler', 'DatabaseProfiler',
    'APIProfiler', 'BottleneckAnalyzer',
    
    # Database optimization
    'SchemaOptimizer', 'QueryOptimizer', 'ConnectionOptimizer',
    'IndexManager', 'PartitionManager',
    
    # Caching and memory
    'CacheManager', 'MemoryOptimizer', 'CacheStrategies',
    'InvalidationManager', 'CacheMonitor',
    
    # API optimization
    'ResponseOptimizer', 'AsyncProcessor', 'CompressionManager',
    'RateLimiter', 'LoadBalancer',
    
    # Data processing
    'PipelineOptimizer', 'ParallelProcessor', 'MLOptimizer',
    'StreamingProcessor', 'TransformationOptimizer',
    
    # Monitoring and scaling
    'PerformanceMonitor', 'AutoScaler', 'MetricsCollector',
    'AlertingEngine', 'BenchmarkRunner',
]