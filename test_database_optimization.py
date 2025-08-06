"""
Test validation for database optimization components.
"""

import sys
import os
import importlib.util

# Add the directory to path directly
script_dir = os.path.dirname(os.path.abspath(__file__))

def import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    # Import database optimization modules
    base_path = os.path.join(script_dir, 'src', 'snowflake_analytics', 'performance', 'database')
    
    schema_module = import_module_from_path('schema_optimizer', os.path.join(base_path, 'schema_optimizer.py'))
    query_module = import_module_from_path('query_optimizer', os.path.join(base_path, 'query_optimizer.py'))
    connection_module = import_module_from_path('connection_optimizer', os.path.join(base_path, 'connection_optimizer.py'))
    index_module = import_module_from_path('index_manager', os.path.join(base_path, 'index_manager.py'))
    partition_module = import_module_from_path('partition_manager', os.path.join(base_path, 'partition_manager.py'))
    
    # Extract classes
    SchemaOptimizer = schema_module.SchemaOptimizer
    QueryOptimizer = query_module.QueryOptimizer
    ConnectionOptimizer = connection_module.ConnectionOptimizer
    IndexManager = index_module.IndexManager
    PartitionManager = partition_module.PartitionManager
    
    print("✓ All database optimization components imported successfully")
    
    # Test SchemaOptimizer
    print("\n=== Testing SchemaOptimizer ===")
    schema_optimizer = SchemaOptimizer()
    analysis = schema_optimizer.analyze_table_structure('users')
    print(f"✓ Table analysis complete - {len(analysis.suggested_indexes)} index suggestions")
    
    recommendations = schema_optimizer.generate_schema_recommendations(['users', 'orders'])
    print(f"✓ Generated {len(recommendations)} schema recommendations")
    
    # Test QueryOptimizer
    print("\n=== Testing QueryOptimizer ===")
    query_optimizer = QueryOptimizer()
    
    test_query = "SELECT * FROM users WHERE age > 25 ORDER BY created_at"
    pattern = query_optimizer.analyze_query(test_query)
    print(f"✓ Query pattern analysis: {pattern.query_type} query on {len(pattern.tables_involved)} tables")
    
    optimizations = query_optimizer.optimize_query(test_query)
    print(f"✓ Generated {len(optimizations)} query optimizations")
    
    # Test ConnectionOptimizer
    print("\n=== Testing ConnectionOptimizer ===")
    connection_optimizer = ConnectionOptimizer()
    
    # Mock connection factory for testing
    def mock_connection():
        class MockConnection:
            def cursor(self):
                return MockCursor()
            def close(self):
                pass
        return MockConnection()
    
    class MockCursor:
        def execute(self, query):
            pass
        def fetchone(self):
            return [1]
        def close(self):
            pass
    
    # Get classes from modules
    PoolConfiguration = connection_module.PoolConfiguration
    ConnectionPool = connection_module.ConnectionPool
    config = PoolConfiguration(min_connections=2, max_connections=5)
    pool = ConnectionPool(mock_connection, config)
    
    connection_optimizer.register_connection_pool('test_pool', pool)
    analysis = connection_optimizer.analyze_connection_performance('test_pool')
    print(f"✓ Connection pool analysis complete - health score available")
    
    # Test IndexManager
    print("\n=== Testing IndexManager ===")
    index_manager = IndexManager()
    indexes = index_manager.discover_existing_indexes('users')
    print(f"✓ Discovered {len(indexes)} existing indexes")
    
    query_patterns = [
        "SELECT * FROM users WHERE email = 'test@example.com'",
        "SELECT * FROM users WHERE created_at > '2024-01-01' ORDER BY created_at"
    ]
    index_recommendations = index_manager.recommend_indexes(query_patterns, 'users')
    print(f"✓ Generated {len(index_recommendations)} index recommendations")
    
    # Test PartitionManager
    print("\n=== Testing PartitionManager ===")
    partition_manager = PartitionManager()
    partition_analysis = partition_manager.analyze_table_for_partitioning('orders', query_patterns)
    print(f"✓ Partition analysis complete - {len(partition_analysis.get('partitioning_opportunities', []))} opportunities found")
    
    partition_recs = partition_manager.generate_partitioning_recommendations(['orders', 'logs'])
    print(f"✓ Generated {len(partition_recs)} partitioning recommendations")
    
    archive_recs = partition_manager.generate_archiving_recommendations('logs')
    print(f"✓ Generated {len(archive_recs)} archiving recommendations")
    
    print("\n=== Database Optimization Tests Completed Successfully! ===")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)