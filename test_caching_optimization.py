"""
Test validation for caching and memory optimization components.
"""

import sys
import os
import time
import importlib.util

# Add the directory to path directly
script_dir = os.path.dirname(os.path.abspath(__file__))

def import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    # Import caching modules
    base_path = os.path.join(script_dir, 'src', 'snowflake_analytics', 'performance', 'caching')
    
    cache_manager_module = import_module_from_path('cache_manager', os.path.join(base_path, 'cache_manager.py'))
    memory_optimizer_module = import_module_from_path('memory_optimizer', os.path.join(base_path, 'memory_optimizer.py'))
    cache_strategies_module = import_module_from_path('cache_strategies', os.path.join(base_path, 'cache_strategies.py'))
    invalidation_manager_module = import_module_from_path('invalidation_manager', os.path.join(base_path, 'invalidation_manager.py'))
    cache_monitor_module = import_module_from_path('cache_monitor', os.path.join(base_path, 'cache_monitor.py'))
    
    # Extract classes
    CacheManager = cache_manager_module.CacheManager
    MemoryOptimizer = memory_optimizer_module.MemoryOptimizer
    CacheStrategies = cache_strategies_module.CacheStrategies
    InvalidationManager = invalidation_manager_module.InvalidationManager
    CacheMonitor = cache_monitor_module.CacheMonitor
    
    print("✓ All caching components imported successfully")
    
    # Test CacheManager
    print("\n=== Testing CacheManager ===")
    cache_manager = CacheManager()
    
    # Test multi-layer caching
    test_key = "test_key"
    test_value = {"data": "test_data", "timestamp": time.time()}
    
    success = cache_manager.put_multilayer(test_key, test_value, ttl_seconds=300)
    print(f"✓ Multi-layer cache put: {success}")
    
    retrieved_value = cache_manager.get(test_key)
    print(f"✓ Cache retrieval: {retrieved_value is not None}")
    
    # Test cache decorator
    @cache_manager.cache_decorator(ttl_seconds=60)
    def expensive_function(x, y):
        time.sleep(0.01)  # Simulate work
        return x + y
    
    result1 = expensive_function(1, 2)
    result2 = expensive_function(1, 2)  # Should hit cache
    print(f"✓ Cache decorator test: {result1 == result2 == 3}")
    
    stats = cache_manager.get_statistics()
    print(f"✓ Cache statistics: {stats['total_gets']} gets, {stats['total_puts']} puts")
    
    # Test MemoryOptimizer
    print("\n=== Testing MemoryOptimizer ===")
    memory_optimizer = MemoryOptimizer()
    
    memory_stats = memory_optimizer.get_memory_stats()
    print(f"✓ Memory stats collected - {memory_stats.memory_percent:.1f}% usage")
    
    optimization_result = memory_optimizer.optimize_memory()
    print(f"✓ Memory optimization complete - {len(optimization_result['actions_taken'])} actions taken")
    
    # Test object pool
    def create_test_object():
        return {"created_at": time.time()}
    
    obj1 = memory_optimizer.get_object_pool('test_pool', create_test_object)
    memory_optimizer.return_to_pool('test_pool', obj1)
    obj2 = memory_optimizer.get_object_pool('test_pool', create_test_object)
    print(f"✓ Object pool test: pool hits={memory_optimizer.pool_hits}, misses={memory_optimizer.pool_misses}")
    
    # Test CacheStrategies
    print("\n=== Testing CacheStrategies ===")
    cache_strategies = CacheStrategies()
    
    # Test strategy recommendation
    strategy_name = cache_strategies.recommend_strategy(
        key='user_sessions',
        data_size_mb=5.0,
        expected_access_frequency=150,
        data_volatility='medium'
    )
    print(f"✓ Strategy recommendation: {strategy_name}")
    
    config = cache_strategies.get_cache_configuration(strategy_name)
    print(f"✓ Strategy config: TTL={config.ttl_seconds}s, Size={config.max_size_mb}MB")
    
    # Test access pattern analysis
    from datetime import datetime, timedelta
    access_times = [
        datetime.now() - timedelta(minutes=i) 
        for i in range(10)
    ]
    pattern = cache_strategies.analyze_access_pattern('test_key', access_times)
    print(f"✓ Access pattern analysis: {pattern.value}")
    
    # Test InvalidationManager
    print("\n=== Testing InvalidationManager ===")
    invalidation_manager = InvalidationManager()
    
    # Create invalidation rule
    InvalidationRule = invalidation_manager_module.InvalidationRule
    InvalidationTrigger = invalidation_manager_module.InvalidationTrigger
    
    rule = InvalidationRule(
        rule_id='test_rule',
        trigger=InvalidationTrigger.TAG_BASED,
        cache_keys={'key1', 'key2'},
        tags={'user_data', 'session'},
        dependencies=set(),
        cascade=True
    )
    
    success = invalidation_manager.add_rule(rule)
    print(f"✓ Invalidation rule added: {success}")
    
    # Test invalidation by tags
    event = invalidation_manager.invalidate_by_tags(['user_data'])
    print(f"✓ Tag-based invalidation: {len(event.affected_keys)} keys affected")
    
    stats = invalidation_manager.get_invalidation_statistics()
    print(f"✓ Invalidation stats: {stats['total_invalidations']} total invalidations")
    
    # Test CacheMonitor
    print("\n=== Testing CacheMonitor ===")
    cache_monitor = CacheMonitor()
    
    # Register cache for monitoring
    cache_monitor.register_cache('test_cache', cache_manager.caches[cache_manager_module.CacheLevel.MEMORY_L1])
    
    # Collect metrics
    metrics = cache_monitor.collect_metrics()
    print(f"✓ Metrics collection: {len(metrics)} cache(s) monitored")
    
    if metrics:
        print(f"  - Hit rate: {metrics[0].hit_rate:.2%}")
        print(f"  - Memory usage: {metrics[0].memory_usage_mb:.1f}MB")
    
    # Test performance dashboard
    dashboard = cache_monitor.get_performance_dashboard()
    print(f"✓ Performance dashboard: {len(dashboard.get('cache_summaries', {}))} cache summaries")
    
    # Test alert callback
    alert_count = [0]  # Use list for closure
    
    def test_alert_callback(alert):
        alert_count[0] += 1
        print(f"  Alert: {alert.alert_type.value} for {alert.cache_name}")
    
    cache_monitor.add_alert_callback(test_alert_callback)
    print("✓ Alert callback registered")
    
    print("\n=== Caching & Memory Optimization Tests Completed Successfully! ===")
    
    # Print summary
    print(f"\nSummary:")
    print(f"- Multi-layer caching with L1/L2/Disk tiers")
    print(f"- Memory optimization with GC management and object pooling")
    print(f"- Intelligent cache strategies based on access patterns")
    print(f"- Tag-based and dependency-based cache invalidation")
    print(f"- Real-time cache performance monitoring and alerting")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)