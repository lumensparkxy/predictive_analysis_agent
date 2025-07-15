"""
Feature Store Examples

Comprehensive examples demonstrating feature store capabilities including:
- Feature registration and management
- Online and offline feature serving
- Feature versioning and lineage tracking
- Quality monitoring and validation
- Integration with ML pipelines
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.snowflake_analytics.feature_store import (
    FeatureStore, FeatureStoreConfig, FeatureMetadata, FeatureSchema,
    FeatureType, FeatureStatus
)


def example_1_basic_feature_registration():
    """Example 1: Basic feature registration and management"""
    
    print("=== Example 1: Basic Feature Registration ===")
    
    # Initialize feature store
    config = FeatureStoreConfig(
        storage_path="data/examples/feature_store",
        enable_lineage=True,
        enable_monitoring=True
    )
    
    with FeatureStore(config=config) as store:
        print("✓ Feature store initialized")
        
        # Register individual features
        user_age_metadata = store.register_feature(
            name="user_age",
            description="User age in years",
            feature_type=FeatureType.NUMERICAL,
            data_type="integer",
            feature_group="user_demographics",
            created_by="data_engineer"
        )
        print(f"✓ Registered feature: {user_age_metadata.name}")
        
        user_country_metadata = store.register_feature(
            name="user_country",
            description="User country code",
            feature_type=FeatureType.CATEGORICAL,
            data_type="string",
            feature_group="user_demographics",
            created_by="data_engineer",
            tags=["geography", "demographics"]
        )
        print(f"✓ Registered feature: {user_country_metadata.name}")
        
        # Register feature group
        group_id = store.register_feature_group(
            name="user_demographics",
            description="Core user demographic features",
            features=["user_age", "user_country"],
            created_by="data_engineer"
        )
        print(f"✓ Registered feature group: {group_id}")
        
        # List features
        all_features = store.list_features()
        print(f"✓ Total features registered: {len(all_features)}")
        
        # Search features
        demographics_features = store.search_features("demographics")
        print(f"✓ Found {len(demographics_features)} demographics features")


def example_2_feature_data_operations():
    """Example 2: Feature data write/read operations"""
    
    print("\n=== Example 2: Feature Data Operations ===")
    
    config = FeatureStoreConfig(
        storage_path="data/examples/feature_store",
        enable_lineage=True
    )
    
    with FeatureStore(config=config) as store:
        # Generate sample feature data
        sample_data = pd.DataFrame({
            'entity_id': [f'user_{i}' for i in range(1000)],
            'user_age': np.random.randint(18, 80, 1000),
            'user_country': np.random.choice(['US', 'UK', 'CA', 'DE', 'FR'], 1000),
            'signup_date': pd.date_range('2023-01-01', periods=1000, freq='1H'),
            'last_active': pd.date_range('2024-01-01', periods=1000, freq='6H')
        })
        
        print(f"✓ Generated sample data: {len(sample_data)} records")
        
        # Write features to store
        write_result = store.write_features(
            feature_data=sample_data,
            feature_group="user_demographics",
            timestamp_column="signup_date",
            mode="overwrite"
        )
        
        print(f"✓ Written {write_result['records_written']} records to feature store")
        
        # Read features back
        retrieved_data = store.read_features(
            feature_group="user_demographics",
            features=["user_age", "user_country"],
            limit=100
        )
        
        print(f"✓ Retrieved {len(retrieved_data)} records from feature store")
        print(f"✓ Feature columns: {list(retrieved_data.columns)}")
        
        # Read with time filters
        recent_data = store.read_features(
            feature_group="user_demographics",
            start_time=datetime.now() - timedelta(days=30),
            limit=50
        )
        
        print(f"✓ Retrieved {len(recent_data)} recent records")


def example_3_online_feature_serving():
    """Example 3: Online feature serving for real-time inference"""
    
    print("\n=== Example 3: Online Feature Serving ===")
    
    config = FeatureStoreConfig(
        storage_path="data/examples/feature_store",
        online_store_uri="redis://localhost:6379",
        cache_ttl_seconds=300
    )
    
    with FeatureStore(config=config) as store:
        # Prepare feature data for online serving
        online_data = pd.DataFrame({
            'entity_id': [f'user_{i}' for i in range(100)],
            'user_age': np.random.randint(18, 80, 100),
            'user_country': np.random.choice(['US', 'UK', 'CA'], 100),
            'credit_score': np.random.randint(300, 850, 100),
            'account_balance': np.random.uniform(0, 10000, 100)
        })
        
        # Update online store
        store.update_online_store(
            feature_group="user_demographics",
            feature_data=online_data,
            entity_column="entity_id"
        )
        print("✓ Updated online store with feature data")
        
        # Serve features for real-time inference
        entity_ids = ['user_1', 'user_2', 'user_3']
        online_features = store.get_online_features(
            feature_group="user_demographics",
            entity_ids=entity_ids,
            features=["user_age", "credit_score"]
        )
        
        print(f"✓ Served online features for {len(entity_ids)} entities")
        for entity_id, features in online_features.items():
            print(f"  {entity_id}: {features}")
        
        # Get serving statistics
        stats = store.serving_engine.get_serving_statistics()
        print(f"✓ Serving stats: {stats['requests_served']} requests, "
              f"{stats['cache_hit_rate']:.1f}% cache hit rate")


def example_4_feature_versioning_and_snapshots():
    """Example 4: Feature versioning and snapshot management"""
    
    print("\n=== Example 4: Feature Versioning and Snapshots ===")
    
    config = FeatureStoreConfig(
        storage_path="data/examples/feature_store",
        enable_versioning=True
    )
    
    with FeatureStore(config=config) as store:
        # Create feature snapshot
        snapshot_id = store.create_feature_snapshot(
            feature_group="user_demographics",
            snapshot_name="v1.0_baseline",
            description="Initial baseline features for user demographics",
            created_by="ml_engineer"
        )
        
        print(f"✓ Created feature snapshot: {snapshot_id}")
        
        # Update features (simulate schema evolution)
        updated_data = pd.DataFrame({
            'entity_id': [f'user_{i}' for i in range(500, 600)],
            'user_age': np.random.randint(18, 80, 100),
            'user_country': np.random.choice(['US', 'UK', 'CA', 'AU', 'NZ'], 100),
            'premium_user': np.random.choice([True, False], 100),
            'signup_channel': np.random.choice(['web', 'mobile', 'referral'], 100)
        })
        
        store.write_features(
            feature_data=updated_data,
            feature_group="user_demographics",
            mode="append"
        )
        
        print("✓ Updated features with new data and schema")
        
        # Create new snapshot
        snapshot_id_v2 = store.create_feature_snapshot(
            feature_group="user_demographics",
            snapshot_name="v1.1_enhanced",
            description="Enhanced features with premium status and signup channel",
            created_by="ml_engineer"
        )
        
        print(f"✓ Created enhanced snapshot: {snapshot_id_v2}")
        
        # Read from specific version
        v1_data = store.read_features(
            feature_group="user_demographics",
            version=snapshot_id,
            limit=10
        )
        
        print(f"✓ Read {len(v1_data)} records from v1.0 snapshot")
        print(f"✓ V1.0 columns: {list(v1_data.columns)}")


def example_5_feature_quality_monitoring():
    """Example 5: Feature quality monitoring and validation"""
    
    print("\n=== Example 5: Feature Quality Monitoring ===")
    
    config = FeatureStoreConfig(
        storage_path="data/examples/feature_store",
        enable_monitoring=True
    )
    
    with FeatureStore(config=config) as store:
        # Compute feature statistics
        statistics = store.compute_feature_statistics(
            feature_group="user_demographics",
            features=["user_age", "user_country"]
        )
        
        print("✓ Computed feature statistics:")
        print(f"  Records analyzed: {statistics['record_count']}")
        print(f"  Features analyzed: {statistics['feature_count']}")
        
        for feature_name, stats in statistics['features'].items():
            print(f"  {feature_name}:")
            print(f"    - Null percentage: {stats['null_percentage']:.2f}%")
            print(f"    - Unique values: {stats['unique_count']}")
            if 'mean' in stats:
                print(f"    - Mean: {stats['mean']:.2f}")
        
        # Validate feature quality
        quality_rules = {
            'max_missing_percentage': 5.0,
            'min_unique_values': 3,
            'max_outlier_percentage': 2.0
        }
        
        validation_results = store.validate_feature_quality(
            feature_group="user_demographics",
            quality_rules=quality_rules
        )
        
        print(f"\n✓ Feature quality validation:")
        print(f"  Overall passed: {validation_results['overall_passed']}")
        print(f"  Passed features: {validation_results['summary']['passed_features']}")
        print(f"  Failed features: {validation_results['summary']['failed_features']}")
        
        if validation_results['summary']['warnings']:
            print("  Warnings:")
            for warning in validation_results['summary']['warnings']:
                print(f"    - {warning}")


def example_6_feature_lineage_tracking():
    """Example 6: Feature lineage tracking and impact analysis"""
    
    print("\n=== Example 6: Feature Lineage Tracking ===")
    
    config = FeatureStoreConfig(
        storage_path="data/examples/feature_store",
        enable_lineage=True
    )
    
    with FeatureStore(config=config) as store:
        # Get feature lineage
        lineage_info = store.get_feature_lineage(
            feature_id="user_demographics_user_age",
            direction="both",
            depth=3
        )
        
        if 'error' not in lineage_info:
            print("✓ Feature lineage retrieved:")
            print(f"  Nodes in lineage: {lineage_info['statistics']['total_nodes']}")
            print(f"  Edges in lineage: {lineage_info['statistics']['total_edges']}")
            print(f"  Node types: {lineage_info['statistics']['node_types']}")
        else:
            print(f"  Lineage info: {lineage_info['error']}")
        
        # Analyze impact of feature changes
        impact_analysis = store.lineage_tracker.analyze_impact(
            feature_id="user_demographics_user_age",
            change_type="schema_change"
        ) if store.lineage_tracker else None
        
        if impact_analysis:
            print(f"\n✓ Impact analysis:")
            print(f"  Risk assessment: {impact_analysis['risk_assessment']}")
            print(f"  Directly affected: {len(impact_analysis['directly_affected'])}")
            print(f"  Indirectly affected: {len(impact_analysis['indirectly_affected'])}")
        
        # Get feature evolution timeline
        if store.lineage_tracker:
            evolution = store.lineage_tracker.get_feature_evolution(
                feature_id="user_demographics_user_age",
                start_time=datetime.now() - timedelta(days=7)
            )
            
            print(f"\n✓ Feature evolution:")
            print(f"  Total events: {evolution['summary']['total_events']}")
            print(f"  Event types: {evolution['summary']['event_types']}")


def example_7_feature_views_and_joins():
    """Example 7: Feature views and complex feature combinations"""
    
    print("\n=== Example 7: Feature Views and Joins ===")
    
    config = FeatureStoreConfig(
        storage_path="data/examples/feature_store"
    )
    
    with FeatureStore(config=config) as store:
        # Create additional feature groups for joining
        transaction_data = pd.DataFrame({
            'entity_id': [f'user_{i}' for i in range(1, 101)],
            'total_transactions': np.random.randint(0, 100, 100),
            'avg_transaction_amount': np.random.uniform(10, 1000, 100),
            'last_transaction_date': pd.date_range('2024-01-01', periods=100, freq='1D')
        })
        
        store.write_features(
            feature_data=transaction_data,
            feature_group="user_transactions",
            mode="overwrite"
        )
        
        print("✓ Created user_transactions feature group")
        
        # Create feature view combining multiple groups
        view_id = store.create_feature_view(
            name="user_profile_complete",
            description="Complete user profile combining demographics and transactions",
            feature_groups=["user_demographics", "user_transactions"],
            join_keys=["entity_id"],
            filters={
                "user_age": {"gte": 18, "lte": 65},
                "total_transactions": {"gte": 1}
            },
            created_by="ml_engineer"
        )
        
        print(f"✓ Created feature view: {view_id}")
        
        # Get data from feature view
        view_data = store.get_feature_view_data(
            view_id=view_id,
            limit=20
        )
        
        print(f"✓ Retrieved {len(view_data)} records from feature view")
        print(f"✓ Combined feature columns: {list(view_data.columns)}")


def example_8_ml_pipeline_integration():
    """Example 8: Integration with ML pipeline workflow"""
    
    print("\n=== Example 8: ML Pipeline Integration ===")
    
    config = FeatureStoreConfig(
        storage_path="data/examples/feature_store",
        enable_lineage=True,
        enable_monitoring=True
    )
    
    with FeatureStore(config=config) as store:
        # Simulate ML training pipeline
        print("✓ Simulating ML training pipeline...")
        
        # 1. Feature selection for training
        training_features = store.read_features(
            feature_group="user_demographics",
            features=["user_age", "user_country"],
            start_time=datetime.now() - timedelta(days=90)
        )
        
        print(f"  - Selected {len(training_features)} training samples")
        
        # 2. Feature validation before training
        validation_results = store.validate_feature_quality(
            feature_group="user_demographics",
            features=["user_age", "user_country"]
        )
        
        if validation_results['overall_passed']:
            print("  - Feature quality validation passed")
        else:
            print("  - Feature quality validation failed")
        
        # 3. Create training snapshot
        training_snapshot = store.create_feature_snapshot(
            feature_group="user_demographics",
            snapshot_name="ml_training_v1",
            description="Feature snapshot for ML model training",
            created_by="ml_pipeline"
        )
        
        print(f"  - Created training snapshot: {training_snapshot}")
        
        # 4. Simulate model serving
        print("\n✓ Simulating model serving...")
        
        # Get features for prediction
        prediction_entities = ['user_1', 'user_5', 'user_10']
        serving_features = store.get_online_features(
            feature_group="user_demographics",
            entity_ids=prediction_entities,
            features=["user_age", "user_country"]
        )
        
        print(f"  - Served features for {len(prediction_entities)} predictions")
        
        # 5. Monitor feature drift
        current_stats = store.compute_feature_statistics(
            feature_group="user_demographics",
            start_time=datetime.now() - timedelta(days=7)
        )
        
        print(f"  - Computed current feature statistics for drift monitoring")


def example_9_performance_optimization():
    """Example 9: Performance optimization and caching"""
    
    print("\n=== Example 9: Performance Optimization ===")
    
    config = FeatureStoreConfig(
        storage_path="data/examples/feature_store",
        enable_caching=True,
        cache_ttl_seconds=300,
        max_parallel_operations=5
    )
    
    with FeatureStore(config=config) as store:
        # Precompute features for faster serving
        high_value_users = [f'user_{i}' for i in range(1, 51)]
        
        precompute_success = store.serving_engine.precompute_features(
            feature_group="user_demographics",
            entity_ids=high_value_users,
            features=["user_age", "user_country"]
        )
        
        if precompute_success:
            print("✓ Precomputed features for high-value users")
        
        # Test serving performance with cache
        import time
        
        # First request (cache miss)
        start_time = time.time()
        features_1 = store.get_online_features(
            feature_group="user_demographics",
            entity_ids=['user_1', 'user_2'],
            features=["user_age"]
        )
        cache_miss_time = time.time() - start_time
        
        # Second request (cache hit)
        start_time = time.time()
        features_2 = store.get_online_features(
            feature_group="user_demographics",
            entity_ids=['user_1', 'user_2'],
            features=["user_age"]
        )
        cache_hit_time = time.time() - start_time
        
        print(f"✓ Performance comparison:")
        print(f"  Cache miss: {cache_miss_time*1000:.2f}ms")
        print(f"  Cache hit: {cache_hit_time*1000:.2f}ms")
        print(f"  Speedup: {cache_miss_time/cache_hit_time:.1f}x")
        
        # Get serving statistics
        serving_stats = store.serving_engine.get_serving_statistics()
        print(f"✓ Overall serving statistics:")
        print(f"  Requests served: {serving_stats['requests_served']}")
        print(f"  Cache hit rate: {serving_stats['cache_hit_rate']:.1f}%")
        print(f"  Average latency: {serving_stats['average_latency_ms']:.2f}ms")


def example_10_data_export_and_backup():
    """Example 10: Data export and backup operations"""
    
    print("\n=== Example 10: Data Export and Backup ===")
    
    config = FeatureStoreConfig(
        storage_path="data/examples/feature_store"
    )
    
    with FeatureStore(config=config) as store:
        # Export feature store metadata
        export_path = store.export_feature_store(
            output_path="data/examples/exports/feature_store_export.json",
            include_data=False,
            include_snapshots=True,
            format="json"
        )
        
        print(f"✓ Exported feature store metadata to: {export_path}")
        
        # Export feature registry
        registry_export = store.feature_registry.export_registry(
            output_path="data/examples/exports/feature_registry.json",
            include_dependencies=True
        )
        
        print(f"✓ Exported feature registry to: {registry_export}")
        
        # Export lineage graph
        if store.lineage_tracker:
            lineage_export = store.lineage_tracker.export_lineage_graph(
                output_path="data/examples/exports/feature_lineage.json",
                include_metadata=True
            )
            print(f"✓ Exported lineage graph to: {lineage_export}")
        
        # Cleanup old data (dry run)
        cleanup_results = store.cleanup_old_data(
            retention_days=30,
            dry_run=True
        )
        
        print(f"✓ Cleanup analysis (dry run):")
        print(f"  Files to delete: {len(cleanup_results['deleted_files'])}")
        print(f"  Snapshots to delete: {len(cleanup_results['deleted_snapshots'])}")
        print(f"  Space to free: {cleanup_results['space_freed_mb']:.2f}MB")


def run_all_examples():
    """Run all feature store examples"""
    
    print("🚀 Running Feature Store Examples")
    print("=" * 50)
    
    try:
        example_1_basic_feature_registration()
        example_2_feature_data_operations()
        example_3_online_feature_serving()
        example_4_feature_versioning_and_snapshots()
        example_5_feature_quality_monitoring()
        example_6_feature_lineage_tracking()
        example_7_feature_views_and_joins()
        example_8_ml_pipeline_integration()
        example_9_performance_optimization()
        example_10_data_export_and_backup()
        
        print("\n" + "=" * 50)
        print("✅ All Feature Store Examples Completed Successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs("data/examples/feature_store", exist_ok=True)
    os.makedirs("data/examples/exports", exist_ok=True)
    
    run_all_examples()
