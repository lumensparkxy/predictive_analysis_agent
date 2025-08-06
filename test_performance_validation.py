"""
Simple validation script for performance profiling components.
"""

import sys
import os
import time

# Add mock psutil to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Replace psutil import
import mock_psutil
sys.modules['psutil'] = mock_psutil

# Direct import without going through the package structure
import importlib.util

def import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    # Import modules directly - need to handle dependencies
    base_path = os.path.join(script_dir, 'src', 'snowflake_analytics', 'performance', 'profiling')
    
    # Load modules with dependencies first
    system_module = import_module_from_path('system_profiler', os.path.join(base_path, 'system_profiler.py'))
    app_module = import_module_from_path('application_profiler', os.path.join(base_path, 'application_profiler.py'))
    db_module = import_module_from_path('database_profiler', os.path.join(base_path, 'database_profiler.py'))
    api_module = import_module_from_path('api_profiler', os.path.join(base_path, 'api_profiler.py'))
    
    # Add modules to sys.modules so bottleneck analyzer can import them
    sys.modules['system_profiler'] = system_module
    sys.modules['application_profiler'] = app_module
    sys.modules['database_profiler'] = db_module
    sys.modules['api_profiler'] = api_module
    
    # Now load bottleneck analyzer
    bottleneck_module = import_module_from_path('bottleneck_analyzer', os.path.join(base_path, 'bottleneck_analyzer.py'))
    
    # Extract classes
    SystemProfiler = system_module.SystemProfiler
    ApplicationProfiler = app_module.ApplicationProfiler
    DatabaseProfiler = db_module.DatabaseProfiler
    APIProfiler = api_module.APIProfiler
    BottleneckAnalyzer = bottleneck_module.BottleneckAnalyzer
    
    print("✓ All profiling components imported successfully")
    
    # Test SystemProfiler
    print("\n=== Testing SystemProfiler ===")
    system_profiler = SystemProfiler(collection_interval=0.1, history_size=5)
    metrics = system_profiler.get_current_metrics()
    print(f"✓ Current CPU: {metrics.cpu_percent}%")
    print(f"✓ Current Memory: {metrics.memory_percent}%")
    
    # Start/stop monitoring test
    system_profiler.start_monitoring()
    print("✓ Monitoring started")
    time.sleep(0.3)
    system_profiler.stop_monitoring()
    print("✓ Monitoring stopped")
    print(f"✓ Collected {len(system_profiler.metrics_history)} data points")
    
    # Test ApplicationProfiler
    print("\n=== Testing ApplicationProfiler ===")
    app_profiler = ApplicationProfiler(max_history=10)
    
    @app_profiler.profile_function
    def test_function(x):
        time.sleep(0.01)
        return x * 2
    
    # Call function multiple times
    results = [test_function(i) for i in range(3)]
    print(f"✓ Function results: {results}")
    
    summary = app_profiler.get_performance_summary()
    print(f"✓ Total functions profiled: {summary['total_functions']}")
    print(f"✓ Total calls: {summary['total_calls']}")
    print(f"✓ Average call time: {summary['avg_call_time_ms']:.2f}ms")
    
    # Test DatabaseProfiler
    print("\n=== Testing DatabaseProfiler ===")
    db_profiler = DatabaseProfiler(max_history=10)
    
    # Simulate query execution
    query = "SELECT * FROM users WHERE active = true"
    exec_id = db_profiler.start_query_execution(query)
    time.sleep(0.01)
    db_profiler.end_query_execution(exec_id, rows_returned=5)
    
    db_summary = db_profiler.get_query_performance_summary()
    print(f"✓ Total queries: {db_summary['total_queries']}")
    print(f"✓ Total executions: {db_summary['total_executions']}")
    print(f"✓ Average execution time: {db_summary['avg_execution_time_ms']:.2f}ms")
    
    # Test APIProfiler
    print("\n=== Testing APIProfiler ===")
    api_profiler = APIProfiler(max_history=10)
    
    # Simulate API call
    call_id = api_profiler.start_api_call_tracking("/api/test", "GET")
    time.sleep(0.01)
    api_profiler.end_api_call_tracking(call_id, status_code=200, response_size=1024)
    
    api_summary = api_profiler.get_endpoint_performance_summary()
    print(f"✓ Total endpoints: {api_summary['total_endpoints']}")
    print(f"✓ Total calls: {api_summary['total_calls']}")
    print(f"✓ Average response time: {api_summary['avg_response_time_ms']:.2f}ms")
    
    # Test BottleneckAnalyzer
    print("\n=== Testing BottleneckAnalyzer ===")
    analyzer = BottleneckAnalyzer(
        system_profiler=system_profiler,
        app_profiler=app_profiler,
        db_profiler=db_profiler,
        api_profiler=api_profiler
    )
    
    alerts = analyzer.analyze_current_performance()
    print(f"✓ Performance alerts generated: {len(alerts)}")
    
    report = analyzer.generate_performance_report()
    print(f"✓ Health score: {report['overall_health_score']}")
    print(f"✓ Top recommendations: {len(report['top_recommendations'])}")
    
    print("\n=== All Tests Passed Successfully! ===")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)