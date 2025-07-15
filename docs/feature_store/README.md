# Feature Store Documentation

## Overview

The Feature Store is a comprehensive enterprise-grade solution for managing ML features throughout their entire lifecycle. It provides centralized feature management, versioning, serving, monitoring, and governance capabilities for production ML systems.

## Architecture

### Core Components

1. **Feature Store**: Main orchestrator providing unified API
2. **Feature Metadata Manager**: Manages feature definitions and schemas
3. **Feature Registry**: Handles feature discovery and dependency tracking
4. **Feature Lineage Tracker**: Tracks data provenance and relationships
5. **Feature Serving Engine**: Provides online/offline feature delivery

### Storage Layers

- **Offline Storage**: Parquet-based storage for batch processing
- **Online Storage**: Redis-based storage for real-time serving
- **Metadata Storage**: SQLAlchemy-based metadata persistence
- **Cache Layer**: In-memory caching for performance optimization

## Quick Start

### Basic Setup

```python
from src.snowflake_analytics.feature_store import (
    FeatureStore, FeatureStoreConfig, FeatureType
)

# Configure feature store
config = FeatureStoreConfig(
    storage_path="data/feature_store",
    online_store_uri="redis://localhost:6379",
    enable_lineage=True,
    enable_monitoring=True
)

# Initialize feature store
with FeatureStore(config=config) as store:
    # Your feature operations here
    pass
```

### Feature Registration

```python
# Register individual feature
metadata = store.register_feature(
    name="user_age",
    description="User age in years",
    feature_type=FeatureType.NUMERICAL,
    data_type="integer",
    feature_group="demographics",
    created_by="data_engineer"
)

# Register feature group
group_id = store.register_feature_group(
    name="demographics",
    description="User demographic features",
    features=["user_age", "user_country"],
    created_by="data_engineer"
)
```

### Writing Features

```python
import pandas as pd

# Prepare feature data
feature_data = pd.DataFrame({
    'entity_id': ['user_1', 'user_2', 'user_3'],
    'user_age': [25, 34, 42],
    'user_country': ['US', 'UK', 'CA']
})

# Write to feature store
result = store.write_features(
    feature_data=feature_data,
    feature_group="demographics",
    mode="append"
)
```

### Reading Features

```python
# Read offline features
offline_data = store.read_features(
    feature_group="demographics",
    features=["user_age", "user_country"],
    start_time=datetime.now() - timedelta(days=30),
    limit=1000
)

# Read online features for serving
online_features = store.get_online_features(
    feature_group="demographics",
    entity_ids=["user_1", "user_2"],
    features=["user_age", "user_country"]
)
```

## Feature Management

### Feature Types

The feature store supports various feature types:

- **NUMERICAL**: Continuous numeric values (age, income, scores)
- **CATEGORICAL**: Discrete categories (country, product_type)
- **BINARY**: Boolean values (is_premium, has_churned)
- **TEXT**: Text data (descriptions, comments)
- **TIMESTAMP**: Date/time values (signup_date, last_login)
- **EMBEDDING**: Vector embeddings (user_embedding, product_embedding)

### Feature Status Lifecycle

Features progress through different states:

1. **DRAFT**: Under development, not ready for production
2. **ACTIVE**: Production-ready and available for serving
3. **DEPRECATED**: Marked for retirement, still available
4. **RETIRED**: No longer available for new usage

### Schema Management

```python
# Define feature schema
from src.snowflake_analytics.feature_store import FeatureSchema

schema = FeatureSchema(
    name="user_age",
    feature_type=FeatureType.NUMERICAL,
    data_type="integer",
    constraints={
        "min_value": 0,
        "max_value": 120,
        "nullable": False
    },
    transformation_logic="EXTRACT(YEAR FROM CURRENT_DATE) - birth_year"
)
```

## Feature Serving

### Online Serving

Optimized for low-latency real-time inference:

```python
# Single entity
features = store.get_online_features(
    feature_group="demographics",
    entity_ids=["user_123"],
    features=["age", "country"]
)

# Batch serving
batch_features = store.serve_feature_batch([
    FeatureRequest(
        request_id="req_1",
        entity_ids=["user_1", "user_2"],
        feature_names=["age", "income"],
        feature_group="demographics"
    )
])
```

### Offline Serving

Designed for batch processing and model training:

```python
# Training data retrieval
training_data = store.get_offline_features(
    feature_group="demographics",
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 6, 30),
    features=["age", "country", "income"]
)

# Point-in-time correctness
historical_features = store.get_point_in_time_features(
    feature_group="demographics",
    entity_ids=["user_1", "user_2"],
    timestamp=datetime(2024, 3, 15),
    features=["age", "country"]
)
```

### Performance Optimization

```python
# Precompute features for hot entities
store.serving_engine.precompute_features(
    feature_group="demographics",
    entity_ids=high_value_users,
    features=["age", "income", "country"]
)

# Configure caching
config = FeatureStoreConfig(
    cache_ttl_seconds=300,
    max_parallel_operations=10,
    enable_caching=True
)
```

## Feature Versioning

### Snapshots

Create point-in-time snapshots for reproducibility:

```python
# Create snapshot
snapshot_id = store.create_feature_snapshot(
    feature_group="demographics",
    snapshot_name="v1.0_baseline",
    description="Initial production features",
    created_by="ml_engineer"
)

# Read from specific snapshot
snapshot_data = store.read_features(
    feature_group="demographics",
    version=snapshot_id,
    limit=1000
)
```

### Schema Evolution

```python
# Validate schema changes
validation_result = store.feature_registry.validate_feature_schema(
    feature_id="demographics_age",
    new_schema={
        "data_type": "integer",
        "constraints": {"min_value": 13, "max_value": 120}
    }
)

# Update schema if validation passes
if validation_result['valid']:
    store.feature_registry.update_feature_schema(
        feature_id="demographics_age",
        new_schema=new_schema
    )
```

## Feature Quality

### Statistics and Monitoring

```python
# Compute feature statistics
stats = store.compute_feature_statistics(
    feature_group="demographics",
    features=["age", "country"],
    start_time=datetime.now() - timedelta(days=7)
)

print(f"Records analyzed: {stats['record_count']}")
for feature, feature_stats in stats['features'].items():
    print(f"{feature}: {feature_stats['null_percentage']:.2f}% missing")
```

### Quality Validation

```python
# Define quality rules
quality_rules = {
    'max_missing_percentage': 5.0,
    'min_unique_values': 2,
    'max_outlier_percentage': 3.0
}

# Validate feature quality
validation_results = store.validate_feature_quality(
    feature_group="demographics",
    quality_rules=quality_rules
)

if not validation_results['overall_passed']:
    print("Quality issues detected:")
    for warning in validation_results['summary']['warnings']:
        print(f"- {warning}")
```

### Freshness Monitoring

```python
# Check feature freshness
freshness_report = store.serving_engine.validate_feature_freshness(
    feature_group="demographics",
    max_age_minutes=60
)

print(f"Fresh features: {freshness_report['fresh_features']}")
print(f"Stale features: {freshness_report['stale_features']}")
```

## Feature Lineage

### Tracking Relationships

```python
# Track feature transformation
store.lineage_tracker.track_feature_transformation(
    source_features=["raw_birth_date"],
    target_feature="user_age",
    transformation_id="age_calculation",
    transformation_metadata={
        "type": "date_extraction",
        "logic": "EXTRACT(YEAR FROM CURRENT_DATE) - EXTRACT(YEAR FROM birth_date)"
    },
    actor="etl_pipeline"
)
```

### Impact Analysis

```python
# Analyze impact of feature changes
impact_analysis = store.analyze_impact(
    feature_id="user_age",
    change_type="schema_change"
)

print(f"Risk level: {impact_analysis['risk_assessment']}")
print(f"Affected features: {len(impact_analysis['directly_affected'])}")
print(f"Affected pipelines: {len(impact_analysis['affected_pipelines'])}")
```

### Lineage Visualization

```python
# Get feature lineage
lineage = store.get_feature_lineage(
    feature_id="user_age",
    direction="both",
    depth=3
)

print(f"Lineage nodes: {lineage['statistics']['total_nodes']}")
print(f"Lineage edges: {lineage['statistics']['total_edges']}")
```

## Feature Views

### Creating Complex Feature Combinations

```python
# Create feature view joining multiple groups
view_id = store.create_feature_view(
    name="user_profile",
    description="Complete user profile features",
    feature_groups=["demographics", "transactions", "behavior"],
    join_keys=["user_id"],
    filters={
        "age": {"gte": 18, "lte": 65},
        "country": {"in": ["US", "UK", "CA"]}
    },
    aggregations={
        "total_transactions": "sum",
        "avg_transaction_amount": "mean"
    }
)

# Query feature view
view_data = store.get_feature_view_data(
    view_id=view_id,
    start_time=datetime.now() - timedelta(days=30),
    limit=1000
)
```

## Integration Patterns

### ML Pipeline Integration

```python
# Training pipeline
def train_model():
    # 1. Feature selection and validation
    features = store.read_features(
        feature_group="user_features",
        start_time=datetime.now() - timedelta(days=90)
    )
    
    # 2. Quality validation
    quality_check = store.validate_feature_quality("user_features")
    if not quality_check['overall_passed']:
        raise ValueError("Feature quality validation failed")
    
    # 3. Create training snapshot
    snapshot_id = store.create_feature_snapshot(
        feature_group="user_features",
        snapshot_name=f"training_{datetime.now().strftime('%Y%m%d')}",
        description="Training dataset snapshot",
        created_by="ml_pipeline"
    )
    
    # 4. Train model with snapshot data
    # ... model training code ...

# Inference pipeline
def predict(user_ids):
    # Get real-time features
    features = store.get_online_features(
        feature_group="user_features",
        entity_ids=user_ids,
        features=["age", "country", "income"]
    )
    
    # Make predictions
    # ... prediction code ...
```

### Streaming Integration

```python
# Real-time feature updates
def process_user_event(event):
    # Extract features from event
    feature_data = pd.DataFrame([{
        'entity_id': event['user_id'],
        'last_activity': event['timestamp'],
        'activity_type': event['type']
    }])
    
    # Update feature store
    store.write_features(
        feature_data=feature_data,
        feature_group="user_activity",
        mode="append"
    )
    
    # Update online store for serving
    store.update_online_store(
        feature_group="user_activity",
        feature_data=feature_data
    )
```

## Configuration

### Feature Store Configuration

```python
from src.snowflake_analytics.feature_store import FeatureStoreConfig

config = FeatureStoreConfig(
    # Storage settings
    storage_backend="parquet",
    storage_path="data/feature_store",
    compression="snappy",
    partitioning_columns=["year", "month"],
    
    # Registry settings
    registry_uri="sqlite:///data/feature_registry.db",
    
    # Online store settings
    online_store_uri="redis://localhost:6379",
    
    # Caching settings
    enable_caching=True,
    cache_ttl_seconds=3600,
    
    # Performance settings
    max_parallel_operations=10,
    
    # Feature settings
    enable_lineage=True,
    enable_monitoring=True,
    retention_days=365
)
```

### Environment Variables

```bash
# Feature store configuration
FEATURE_STORE_STORAGE_PATH=data/feature_store
FEATURE_STORE_REGISTRY_URI=sqlite:///data/feature_registry.db
FEATURE_STORE_ONLINE_STORE_URI=redis://localhost:6379

# Performance tuning
FEATURE_STORE_CACHE_TTL=3600
FEATURE_STORE_MAX_PARALLEL_OPS=10
FEATURE_STORE_COMPRESSION=snappy

# Feature settings
FEATURE_STORE_ENABLE_LINEAGE=true
FEATURE_STORE_ENABLE_MONITORING=true
FEATURE_STORE_RETENTION_DAYS=365
```

## Monitoring and Observability

### Serving Metrics

```python
# Get serving statistics
stats = store.serving_engine.get_serving_statistics()
print(f"Requests served: {stats['requests_served']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1f}%")
print(f"Average latency: {stats['average_latency_ms']:.2f}ms")
print(f"Error rate: {stats['error_rate']:.2f}%")
```

### Feature Drift Detection

```python
# Compare feature distributions over time
current_stats = store.compute_feature_statistics(
    feature_group="demographics",
    start_time=datetime.now() - timedelta(days=7)
)

baseline_stats = store.compute_feature_statistics(
    feature_group="demographics", 
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now() - timedelta(days=23)
)

# Detect drift by comparing statistics
for feature in current_stats['features']:
    current_mean = current_stats['features'][feature].get('mean', 0)
    baseline_mean = baseline_stats['features'][feature].get('mean', 0)
    
    if abs(current_mean - baseline_mean) / baseline_mean > 0.1:  # 10% threshold
        print(f"Drift detected in {feature}: {baseline_mean:.2f} -> {current_mean:.2f}")
```

### Alerts and Notifications

```python
# Quality-based alerting
def check_feature_quality_and_alert():
    validation_results = store.validate_feature_quality("demographics")
    
    if not validation_results['overall_passed']:
        # Send alert (integrate with notification system)
        alert_message = f"Feature quality issues detected in demographics group"
        # send_slack_alert(alert_message)
        # send_email_alert(alert_message)
        
        # Log issues
        for feature, result in validation_results['results'].items():
            if not result['passed']:
                print(f"Quality issue in {feature}: {result['issues']}")
```

## Data Governance

### Access Control

```python
# Feature-level permissions (when enabled)
config = FeatureStoreConfig(
    enable_access_control=True
)

# Example of role-based access
# store.grant_feature_access(
#     feature_id="sensitive_feature",
#     role="ml_engineer",
#     permissions=["read", "write"]
# )
```

### Audit Trails

```python
# Feature usage tracking
usage_stats = store.feature_registry.get_feature_usage_stats("user_age")
print(f"Feature: {usage_stats['feature_name']}")
print(f"Created: {usage_stats['created_at']}")
print(f"Dependencies: {usage_stats['dependencies_count']}")
print(f"Dependents: {usage_stats['dependents_count']}")
```

### Compliance

```python
# Data retention and cleanup
cleanup_results = store.cleanup_old_data(
    retention_days=365,
    dry_run=False  # Set to True for testing
)

print(f"Cleaned up {len(cleanup_results['deleted_files'])} files")
print(f"Freed {cleanup_results['space_freed_mb']:.2f}MB of space")
```

## Backup and Recovery

### Export Operations

```python
# Export feature store metadata
export_path = store.export_feature_store(
    output_path="backups/feature_store_backup.json",
    include_data=False,
    include_snapshots=True
)

# Export lineage graph
lineage_path = store.lineage_tracker.export_lineage_graph(
    output_path="backups/lineage_backup.json",
    include_metadata=True
)

# Export feature registry
registry_path = store.feature_registry.export_registry(
    output_path="backups/registry_backup.json",
    include_dependencies=True
)
```

### Import Operations

```python
# Import feature registry
import_results = store.feature_registry.import_registry(
    import_path="backups/registry_backup.json",
    merge_strategy="skip_existing"
)

print(f"Imported {import_results['features_imported']} features")
print(f"Skipped {import_results['features_skipped']} existing features")
```

## Best Practices

### Feature Design

1. **Naming Conventions**: Use descriptive, consistent names
   - `user_age_years` instead of `age`
   - `transaction_amount_usd` instead of `amount`

2. **Feature Groups**: Organize related features together
   - `user_demographics`: age, gender, country
   - `user_behavior`: login_frequency, page_views
   - `transaction_features`: amount, frequency, merchant

3. **Documentation**: Provide clear descriptions and business context

### Performance

1. **Partitioning**: Use appropriate partitioning strategies
   ```python
   config = FeatureStoreConfig(
       partitioning_columns=["year", "month", "day"]
   )
   ```

2. **Caching**: Enable caching for frequently accessed features
3. **Batch Operations**: Use batch APIs for bulk operations
4. **Precomputation**: Precompute features for critical entities

### Quality Assurance

1. **Validation Rules**: Define comprehensive quality rules
2. **Monitoring**: Set up alerts for quality degradation
3. **Testing**: Test feature pipelines thoroughly
4. **Gradual Rollouts**: Use feature flags for safe deployments

### Security

1. **Access Control**: Implement proper permissions
2. **Data Masking**: Mask sensitive features in non-production
3. **Encryption**: Encrypt sensitive data at rest and in transit
4. **Audit Logging**: Log all feature access and modifications

## Troubleshooting

### Common Issues

1. **Performance Issues**
   ```python
   # Check serving statistics
   stats = store.serving_engine.get_serving_statistics()
   if stats['average_latency_ms'] > 100:
       # Enable caching, precompute features, or optimize queries
       pass
   ```

2. **Quality Issues**
   ```python
   # Validate and diagnose quality problems
   validation = store.validate_feature_quality("feature_group")
   for feature, result in validation['results'].items():
       if not result['passed']:
           print(f"Issues in {feature}: {result['issues']}")
   ```

3. **Lineage Issues**
   ```python
   # Check lineage tracking
   if store.lineage_tracker:
       lineage = store.get_feature_lineage("feature_id")
       if 'error' in lineage:
           print(f"Lineage error: {lineage['error']}")
   ```

### Debugging

1. **Enable Detailed Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check Configuration**
   ```python
   # Validate configuration
   print(f"Storage path exists: {Path(config.storage_path).exists()}")
   print(f"Online store enabled: {store.online_store_enabled}")
   ```

3. **Verify Data Integrity**
   ```python
   # Check data consistency
   stats = store.compute_feature_statistics("feature_group")
   print(f"Record count: {stats['record_count']}")
   print(f"Missing data: {stats['overall']['missing_percentage']:.2f}%")
   ```

## API Reference

For detailed API documentation, see:
- [Feature Store API](api/feature_store.md)
- [Feature Metadata API](api/feature_metadata.md)
- [Feature Registry API](api/feature_registry.md)
- [Feature Serving API](api/feature_serving.md)
- [Feature Lineage API](api/feature_lineage.md)

## Examples

Complete examples are available in:
- [Basic Usage Examples](../examples/feature_store_examples/)
- [Integration Examples](../examples/integration_examples/)
- [Performance Examples](../examples/performance_examples/)

## Contributing

To contribute to the feature store:

1. Follow the coding standards
2. Add comprehensive tests
3. Update documentation
4. Submit pull requests for review

For more information, see [Contributing Guidelines](../CONTRIBUTING.md).
