"""
Basic tests for performance profiling components to validate functionality.
"""

import unittest
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Direct imports to avoid dependency issues
from snowflake_analytics.performance.profiling.system_profiler import SystemProfiler
from snowflake_analytics.performance.profiling.application_profiler import ApplicationProfiler
from snowflake_analytics.performance.profiling.database_profiler import DatabaseProfiler
from snowflake_analytics.performance.profiling.api_profiler import APIProfiler
from snowflake_analytics.performance.profiling.bottleneck_analyzer import BottleneckAnalyzer


class TestPerformanceProfiling(unittest.TestCase):
    """Test performance profiling components."""
    
    def test_system_profiler_basic(self):
        """Test basic system profiler functionality."""
        profiler = SystemProfiler(collection_interval=0.1, history_size=10)
        
        # Test current metrics collection
        metrics = profiler.get_current_metrics()
        self.assertIsNotNone(metrics.timestamp)
        self.assertIsInstance(metrics.cpu_percent, float)
        self.assertIsInstance(metrics.memory_percent, float)
        
        # Test monitoring start/stop
        self.assertTrue(profiler.start_monitoring())
        self.assertTrue(profiler.is_monitoring)
        time.sleep(0.3)  # Let it collect some data
        self.assertTrue(profiler.stop_monitoring())
        self.assertFalse(profiler.is_monitoring)
        
        # Should have some metrics history
        self.assertGreater(len(profiler.metrics_history), 0)
    
    def test_application_profiler_decorator(self):
        """Test application profiler decorator functionality."""
        profiler = ApplicationProfiler(max_history=100)
        
        @profiler.profile_function
        def test_function(x, y):
            time.sleep(0.01)  # Simulate work
            return x + y
        
        # Call function multiple times
        for i in range(5):
            result = test_function(i, i + 1)
            self.assertEqual(result, i + (i + 1))
        
        # Check profiling data
        summary = profiler.get_performance_summary()
        self.assertEqual(summary['total_functions'], 1)
        self.assertEqual(summary['total_calls'], 5)
        self.assertGreater(summary['avg_call_time_ms'], 5)  # Should be > 5ms due to sleep
        
        # Check function statistics
        slow_functions = profiler.get_top_slow_functions(1)
        self.assertEqual(len(slow_functions), 1)
        self.assertEqual(slow_functions[0].total_calls, 5)
    
    def test_database_profiler_manual_tracking(self):
        """Test database profiler manual tracking."""
        profiler = DatabaseProfiler(max_history=100)
        
        # Simulate query execution
        query = "SELECT * FROM users WHERE age > 25"
        execution_id = profiler.start_query_execution(query, connection_id="conn1")
        self.assertIsNotNone(execution_id)
        
        time.sleep(0.01)  # Simulate query execution time
        
        profiler.end_query_execution(
            execution_id, 
            rows_returned=10,
            exception=None
        )
        
        # Check profiling data
        summary = profiler.get_query_performance_summary()
        self.assertEqual(summary['total_queries'], 1)
        self.assertEqual(summary['total_executions'], 1)
        self.assertGreater(summary['avg_execution_time_ms'], 5)
    
    def test_api_profiler_manual_tracking(self):
        """Test API profiler manual tracking."""
        profiler = APIProfiler(max_history=100)
        
        # Simulate API call
        call_id = profiler.start_api_call_tracking(
            endpoint="/api/users",
            method="GET"
        )
        self.assertIsNotNone(call_id)
        
        time.sleep(0.01)  # Simulate API processing time
        
        profiler.end_api_call_tracking(
            call_id,
            status_code=200,
            response_size=1024
        )
        
        # Check profiling data
        summary = profiler.get_endpoint_performance_summary()
        self.assertEqual(summary['total_endpoints'], 1)
        self.assertEqual(summary['total_calls'], 1)
        self.assertGreater(summary['avg_response_time_ms'], 5)
    
    def test_bottleneck_analyzer_integration(self):
        """Test bottleneck analyzer with integrated profilers."""
        system_profiler = SystemProfiler()
        app_profiler = ApplicationProfiler()
        db_profiler = DatabaseProfiler()
        api_profiler = APIProfiler()
        
        analyzer = BottleneckAnalyzer(
            system_profiler=system_profiler,
            app_profiler=app_profiler,
            db_profiler=db_profiler,
            api_profiler=api_profiler
        )
        
        # Analyze current performance (should work even with no data)
        alerts = analyzer.analyze_current_performance()
        self.assertIsInstance(alerts, list)
        
        # Generate performance report
        report = analyzer.generate_performance_report()
        self.assertIn('generated_at', report)
        self.assertIn('current_alerts', report)
        self.assertIn('component_summaries', report)
        self.assertIsInstance(report['overall_health_score'], float)
    
    def test_profiler_enable_disable(self):
        """Test enabling/disabling profilers."""
        profilers = [
            SystemProfiler(),
            ApplicationProfiler(), 
            DatabaseProfiler(),
            APIProfiler()
        ]
        
        for profiler in profilers:
            # Should be enabled by default
            self.assertTrue(profiler.is_enabled() if hasattr(profiler, 'is_enabled') else profiler.enabled)
            
            # Test disable
            profiler.disable()
            self.assertFalse(profiler.is_enabled() if hasattr(profiler, 'is_enabled') else profiler.enabled)
            
            # Test enable
            profiler.enable()
            self.assertTrue(profiler.is_enabled() if hasattr(profiler, 'is_enabled') else profiler.enabled)


if __name__ == '__main__':
    unittest.main()