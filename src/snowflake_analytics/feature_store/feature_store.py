"""
Feature Store

Comprehensive feature store implementation providing enterprise-grade feature management,
versioning, lineage tracking, and serving capabilities for ML pipelines.

Key capabilities:
- Feature registration and metadata management
- Version control and change tracking
- Online and offline feature serving
- Data lineage and audit trails
- Feature quality monitoring
- Time-travel capabilities
- Integration with ML pipelines
- Compliance and governance
"""

import os
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import hashlib

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from redis import Redis

from ...config.settings import SnowflakeSettings
from ...utils.logger import SnowflakeLogger
from .feature_metadata import (
    FeatureMetadataManager, FeatureMetadata, FeatureSchema, FeatureType,
    FeatureStatus, QualityMetrics, UsageStatistics
)
from .feature_registry import FeatureRegistry
from .feature_lineage import FeatureLineageTracker
from .feature_serving import FeatureServingEngine


@dataclass
class FeatureStoreConfig:
    """Feature store configuration"""
    storage_backend: str = "parquet"
    storage_path: str = "data/feature_store"
    registry_uri: str = "sqlite:///data/feature_registry.db"
    online_store_uri: str = "redis://localhost:6379"
    enable_lineage: bool = True
    enable_monitoring: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_parallel_operations: int = 10
    compression: str = "snappy"
    partitioning_columns: List[str] = None
    retention_days: int = 365


class FeatureStore:
    """
    Comprehensive feature store for ML-ready feature management.
    
    Provides complete feature lifecycle management including registration,
    versioning, serving, monitoring, and governance capabilities.
    """
    
    def __init__(
        self,
        config: Optional[FeatureStoreConfig] = None,
        config_path: Optional[str] = None
    ):
        """Initialize feature store"""
        
        # Configuration
        self.config = config or FeatureStoreConfig()
        self.settings = SnowflakeSettings(config_path)
        self.logger = SnowflakeLogger("FeatureStore").get_logger()
        
        # Setup storage directories
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metadata_manager = FeatureMetadataManager(
            registry_uri=self.config.registry_uri,
            enable_caching=self.config.enable_caching
        )
        
        self.feature_registry = FeatureRegistry(
            metadata_manager=self.metadata_manager,
            storage_path=self.storage_path
        )
        
        if self.config.enable_lineage:
            self.lineage_tracker = FeatureLineageTracker(
                storage_path=self.storage_path / "lineage"
            )
        else:
            self.lineage_tracker = None
        
        self.serving_engine = FeatureServingEngine(
            metadata_manager=self.metadata_manager,
            storage_path=self.storage_path,
            online_store_uri=self.config.online_store_uri,
            cache_ttl_seconds=self.config.cache_ttl_seconds
        )
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_parallel_operations)
        
        self.logger.info("FeatureStore initialized successfully")
    
    def register_feature(
        self,
        name: str,
        description: str,
        feature_type: FeatureType,
        data_type: str,
        feature_group: str,
        created_by: str,
        **kwargs
    ) -> FeatureMetadata:
        """Register new feature in the store"""
        
        # Create feature schema
        schema = FeatureSchema(
            name=name,
            feature_type=feature_type,
            data_type=data_type,
            description=description,
            **kwargs.get('schema_kwargs', {})
        )
        
        # Create feature metadata
        metadata = self.metadata_manager.create_feature_metadata(
            name=name,
            description=description,
            schema=schema,
            feature_group=feature_group,
            created_by=created_by,
            **kwargs
        )
        
        # Register with feature registry
        self.feature_registry.register_feature(metadata)
        
        # Track lineage if enabled
        if self.lineage_tracker:
            self.lineage_tracker.track_feature_creation(
                feature_id=metadata.feature_id,
                created_by=created_by,
                metadata=asdict(metadata)
            )
        
        self.logger.info(f"Feature registered: {metadata.feature_id}")
        return metadata
    
    def register_feature_group(
        self,
        name: str,
        description: str,
        features: List[str],
        created_by: str,
        **kwargs
    ) -> str:
        """Register feature group"""
        
        group = self.metadata_manager.create_feature_group(
            name=name,
            description=description,
            features=features,
            created_by=created_by,
            **kwargs
        )
        
        self.logger.info(f"Feature group registered: {group.group_id}")
        return group.group_id
    
    def write_features(
        self,
        feature_data: pd.DataFrame,
        feature_group: str,
        timestamp_column: Optional[str] = None,
        partition_columns: Optional[List[str]] = None,
        mode: str = "append"
    ) -> Dict[str, Any]:
        """Write feature data to the store"""
        
        try:
            # Validate feature data
            validation_results = self._validate_feature_data(feature_data, feature_group)
            if not validation_results['valid']:
                raise ValueError(f"Feature data validation failed: {validation_results['errors']}")
            
            # Add metadata columns
            enriched_data = self._enrich_feature_data(feature_data, timestamp_column)
            
            # Determine storage path
            storage_path = self._get_feature_storage_path(feature_group)
            
            # Write to storage
            write_results = self._write_to_storage(
                data=enriched_data,
                storage_path=storage_path,
                partition_columns=partition_columns or self.config.partitioning_columns,
                mode=mode
            )
            
            # Update metadata and statistics
            self._update_feature_statistics(feature_group, enriched_data)
            
            # Track lineage if enabled
            if self.lineage_tracker:
                self.lineage_tracker.track_feature_write(
                    feature_group=feature_group,
                    record_count=len(enriched_data),
                    timestamp=datetime.now(),
                    storage_path=str(storage_path)
                )
            
            # Update online store if configured
            if self.serving_engine.online_store_enabled:
                self.serving_engine.update_online_store(feature_group, enriched_data)
            
            self.logger.info(f"Features written: {feature_group} ({len(enriched_data)} records)")
            
            return {
                'feature_group': feature_group,
                'records_written': len(enriched_data),
                'storage_path': str(storage_path),
                'validation_results': validation_results,
                'write_results': write_results
            }
            
        except Exception as e:
            self.logger.error(f"Failed to write features for {feature_group}: {str(e)}")
            raise
    
    def read_features(
        self,
        feature_group: str,
        features: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        version: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Read features from the store"""
        
        try:
            # Get feature metadata
            group_features = self.feature_registry.get_feature_group_features(feature_group)
            if not group_features:
                raise ValueError(f"Feature group not found: {feature_group}")
            
            # Filter features if specified
            if features:
                missing_features = set(features) - set(group_features)
                if missing_features:
                    raise ValueError(f"Features not found in group: {missing_features}")
                group_features = features
            
            # Determine storage path
            storage_path = self._get_feature_storage_path(feature_group, version)
            
            # Read from storage
            feature_data = self._read_from_storage(
                storage_path=storage_path,
                features=group_features,
                start_time=start_time,
                end_time=end_time,
                filters=filters,
                limit=limit
            )
            
            # Update usage statistics
            self._update_usage_statistics(feature_group, len(feature_data))
            
            # Track lineage if enabled
            if self.lineage_tracker:
                self.lineage_tracker.track_feature_read(
                    feature_group=feature_group,
                    features=group_features,
                    record_count=len(feature_data),
                    timestamp=datetime.now()
                )
            
            self.logger.info(f"Features read: {feature_group} ({len(feature_data)} records)")
            return feature_data
            
        except Exception as e:
            self.logger.error(f"Failed to read features for {feature_group}: {str(e)}")
            raise
    
    def get_online_features(
        self,
        feature_group: str,
        entity_ids: List[str],
        features: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get features for online serving"""
        
        return self.serving_engine.get_online_features(
            feature_group=feature_group,
            entity_ids=entity_ids,
            features=features
        )
    
    def create_feature_view(
        self,
        name: str,
        description: str,
        feature_groups: List[str],
        join_keys: List[str],
        filters: Optional[Dict[str, Any]] = None,
        aggregations: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> str:
        """Create feature view (logical combination of feature groups)"""
        
        view_id = f"view_{name}_{int(time.time())}"
        
        # Validate feature groups exist
        for group in feature_groups:
            if not self.feature_registry.get_feature_group_features(group):
                raise ValueError(f"Feature group not found: {group}")
        
        # Create view metadata
        view_metadata = {
            'view_id': view_id,
            'name': name,
            'description': description,
            'feature_groups': feature_groups,
            'join_keys': join_keys,
            'filters': filters or {},
            'aggregations': aggregations or {},
            'created_by': created_by,
            'created_at': datetime.now().isoformat()
        }
        
        # Store view definition
        view_path = self.storage_path / "views" / f"{view_id}.json"
        view_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(view_path, 'w') as f:
            json.dump(view_metadata, f, indent=2, default=str)
        
        self.logger.info(f"Feature view created: {view_id}")
        return view_id
    
    def get_feature_view_data(
        self,
        view_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get data from feature view"""
        
        # Load view definition
        view_path = self.storage_path / "views" / f"{view_id}.json"
        if not view_path.exists():
            raise ValueError(f"Feature view not found: {view_id}")
        
        with open(view_path, 'r') as f:
            view_metadata = json.load(f)
        
        # Read data from each feature group
        group_dataframes = []
        for group in view_metadata['feature_groups']:
            group_data = self.read_features(
                feature_group=group,
                start_time=start_time,
                end_time=end_time
            )
            group_dataframes.append(group_data)
        
        # Join feature groups
        if len(group_dataframes) == 1:
            result_data = group_dataframes[0]
        else:
            result_data = group_dataframes[0]
            for df in group_dataframes[1:]:
                result_data = result_data.merge(
                    df,
                    on=view_metadata['join_keys'],
                    how='inner'
                )
        
        # Apply filters
        if view_metadata['filters']:
            for column, condition in view_metadata['filters'].items():
                if column in result_data.columns:
                    if isinstance(condition, dict):
                        if 'gte' in condition:
                            result_data = result_data[result_data[column] >= condition['gte']]
                        if 'lte' in condition:
                            result_data = result_data[result_data[column] <= condition['lte']]
                        if 'eq' in condition:
                            result_data = result_data[result_data[column] == condition['eq']]
                    else:
                        result_data = result_data[result_data[column] == condition]
        
        # Apply aggregations
        if view_metadata['aggregations']:
            agg_dict = {}
            for column, agg_func in view_metadata['aggregations'].items():
                if column in result_data.columns:
                    agg_dict[column] = agg_func
            
            if agg_dict:
                result_data = result_data.groupby(view_metadata['join_keys']).agg(agg_dict).reset_index()
        
        # Apply limit
        if limit:
            result_data = result_data.head(limit)
        
        self.logger.info(f"Feature view data retrieved: {view_id} ({len(result_data)} records)")
        return result_data
    
    def compute_feature_statistics(
        self,
        feature_group: str,
        features: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Compute comprehensive feature statistics"""
        
        # Read feature data
        feature_data = self.read_features(
            feature_group=feature_group,
            features=features,
            start_time=start_time,
            end_time=end_time
        )
        
        if feature_data.empty:
            return {'error': 'No data found for specified criteria'}
        
        statistics = {
            'feature_group': feature_group,
            'computation_time': datetime.now().isoformat(),
            'record_count': len(feature_data),
            'feature_count': len(feature_data.columns),
            'time_range': {
                'start': start_time.isoformat() if start_time else None,
                'end': end_time.isoformat() if end_time else None
            },
            'features': {}
        }
        
        # Compute statistics for each feature
        for column in feature_data.columns:
            if column.startswith('_'):  # Skip metadata columns
                continue
            
            feature_stats = self._compute_feature_column_statistics(feature_data[column])
            statistics['features'][column] = feature_stats
        
        # Compute overall statistics
        statistics['overall'] = {
            'total_missing_values': feature_data.isnull().sum().sum(),
            'missing_percentage': (feature_data.isnull().sum().sum() / feature_data.size) * 100,
            'duplicate_rows': feature_data.duplicated().sum(),
            'duplicate_percentage': (feature_data.duplicated().sum() / len(feature_data)) * 100
        }
        
        self.logger.info(f"Feature statistics computed: {feature_group}")
        return statistics
    
    def validate_feature_quality(
        self,
        feature_group: str,
        features: Optional[List[str]] = None,
        quality_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate feature quality against rules"""
        
        # Compute statistics
        statistics = self.compute_feature_statistics(feature_group, features)
        
        # Default quality rules
        default_rules = {
            'max_missing_percentage': 10.0,
            'max_duplicate_percentage': 5.0,
            'min_unique_values': 2,
            'max_outlier_percentage': 5.0
        }
        
        # Merge with custom rules
        rules = {**default_rules, **(quality_rules or {})}
        
        validation_results = {
            'feature_group': feature_group,
            'validation_time': datetime.now().isoformat(),
            'overall_passed': True,
            'rules_applied': rules,
            'results': {},
            'summary': {
                'total_features': len(statistics.get('features', {})),
                'passed_features': 0,
                'failed_features': 0,
                'warnings': []
            }
        }
        
        # Validate each feature
        for feature_name, feature_stats in statistics.get('features', {}).items():
            feature_validation = self._validate_feature_against_rules(
                feature_name, feature_stats, rules
            )
            validation_results['results'][feature_name] = feature_validation
            
            if feature_validation['passed']:
                validation_results['summary']['passed_features'] += 1
            else:
                validation_results['summary']['failed_features'] += 1
                validation_results['overall_passed'] = False
        
        # Validate overall statistics
        overall_stats = statistics.get('overall', {})
        if overall_stats.get('missing_percentage', 0) > rules['max_missing_percentage']:
            validation_results['overall_passed'] = False
            validation_results['summary']['warnings'].append(
                f"Overall missing percentage ({overall_stats['missing_percentage']:.2f}%) exceeds threshold"
            )
        
        if overall_stats.get('duplicate_percentage', 0) > rules['max_duplicate_percentage']:
            validation_results['overall_passed'] = False
            validation_results['summary']['warnings'].append(
                f"Overall duplicate percentage ({overall_stats['duplicate_percentage']:.2f}%) exceeds threshold"
            )
        
        self.logger.info(f"Feature quality validated: {feature_group}")
        return validation_results
    
    def get_feature_lineage(
        self,
        feature_id: str,
        direction: str = 'both',
        depth: int = 5
    ) -> Dict[str, Any]:
        """Get feature lineage information"""
        
        if not self.lineage_tracker:
            return {'error': 'Lineage tracking not enabled'}
        
        return self.lineage_tracker.get_feature_lineage(
            feature_id=feature_id,
            direction=direction,
            depth=depth
        )
    
    def get_feature_metadata(self, feature_id: str) -> Optional[FeatureMetadata]:
        """Get feature metadata"""
        
        return self.metadata_manager.get_feature_metadata(feature_id)
    
    def list_features(
        self,
        feature_group: Optional[str] = None,
        status: Optional[FeatureStatus] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[FeatureMetadata]:
        """List features with optional filtering"""
        
        return self.metadata_manager.list_features(
            feature_group=feature_group,
            status=status,
            tags=tags,
            limit=limit
        )
    
    def search_features(
        self,
        query: str,
        search_fields: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[FeatureMetadata]:
        """Search features by text query"""
        
        return self.metadata_manager.search_features(
            query=query,
            search_fields=search_fields,
            limit=limit
        )
    
    def create_feature_snapshot(
        self,
        feature_group: str,
        snapshot_name: str,
        description: str,
        created_by: str
    ) -> str:
        """Create feature snapshot for versioning"""
        
        # Read current feature data
        current_data = self.read_features(feature_group)
        
        # Create snapshot metadata
        snapshot_id = f"snapshot_{feature_group}_{int(time.time())}"
        snapshot_metadata = {
            'snapshot_id': snapshot_id,
            'snapshot_name': snapshot_name,
            'description': description,
            'feature_group': feature_group,
            'created_by': created_by,
            'created_at': datetime.now().isoformat(),
            'record_count': len(current_data),
            'data_hash': self._calculate_data_hash(current_data)
        }
        
        # Store snapshot data
        snapshot_path = self.storage_path / "snapshots" / feature_group / snapshot_id
        snapshot_path.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data_path = snapshot_path / "data.parquet"
        current_data.to_parquet(data_path, compression=self.config.compression)
        
        # Save metadata
        metadata_path = snapshot_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(snapshot_metadata, f, indent=2, default=str)
        
        # Track lineage if enabled
        if self.lineage_tracker:
            self.lineage_tracker.track_snapshot_creation(
                feature_group=feature_group,
                snapshot_id=snapshot_id,
                created_by=created_by
            )
        
        self.logger.info(f"Feature snapshot created: {snapshot_id}")
        return snapshot_id
    
    def cleanup_old_data(
        self,
        retention_days: Optional[int] = None,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Clean up old feature data based on retention policy"""
        
        retention_days = retention_days or self.config.retention_days
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        cleanup_results = {
            'cutoff_date': cutoff_date.isoformat(),
            'retention_days': retention_days,
            'dry_run': dry_run,
            'deleted_files': [],
            'deleted_snapshots': [],
            'space_freed_mb': 0.0
        }
        
        # Clean up old data files
        for feature_group_path in self.storage_path.glob("*/"):
            if feature_group_path.is_dir() and feature_group_path.name not in ['views', 'lineage', 'snapshots']:
                for data_file in feature_group_path.rglob("*.parquet"):
                    if data_file.stat().st_mtime < cutoff_date.timestamp():
                        file_size_mb = data_file.stat().st_size / (1024 * 1024)
                        cleanup_results['deleted_files'].append(str(data_file))
                        cleanup_results['space_freed_mb'] += file_size_mb
                        
                        if not dry_run:
                            data_file.unlink()
        
        # Clean up old snapshots
        snapshots_path = self.storage_path / "snapshots"
        if snapshots_path.exists():
            for snapshot_dir in snapshots_path.rglob("snapshot_*"):
                if snapshot_dir.is_dir():
                    metadata_file = snapshot_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        created_at = datetime.fromisoformat(metadata['created_at'])
                        if created_at < cutoff_date:
                            cleanup_results['deleted_snapshots'].append(str(snapshot_dir))
                            
                            if not dry_run:
                                import shutil
                                shutil.rmtree(snapshot_dir)
        
        self.logger.info(f"Cleanup completed: {len(cleanup_results['deleted_files'])} files, "
                        f"{len(cleanup_results['deleted_snapshots'])} snapshots, "
                        f"{cleanup_results['space_freed_mb']:.2f}MB freed")
        
        return cleanup_results
    
    def export_feature_store(
        self,
        output_path: str,
        include_data: bool = False,
        include_snapshots: bool = False,
        format: str = 'json'
    ) -> str:
        """Export feature store metadata and optionally data"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'feature_store_config': asdict(self.config),
            'features': [],
            'feature_groups': [],
            'views': []
        }
        
        # Export feature metadata
        all_features = self.list_features()
        for feature in all_features:
            feature_data = asdict(feature)
            # Convert datetime objects
            for key, value in feature_data.items():
                if isinstance(value, datetime):
                    feature_data[key] = value.isoformat()
            export_data['features'].append(feature_data)
        
        # Export feature groups
        # This would require implementing group listing in metadata manager
        
        # Export feature views
        views_path = self.storage_path / "views"
        if views_path.exists():
            for view_file in views_path.glob("*.json"):
                with open(view_file, 'r') as f:
                    view_data = json.load(f)
                export_data['views'].append(view_data)
        
        # Export to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Feature store exported to: {output_path}")
        return str(output_path)
    
    def _validate_feature_data(
        self,
        data: pd.DataFrame,
        feature_group: str
    ) -> Dict[str, Any]:
        """Validate feature data against schema"""
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check if data is empty
        if data.empty:
            validation_results['valid'] = False
            validation_results['errors'].append("Data is empty")
            return validation_results
        
        # Get feature group metadata
        group_features = self.feature_registry.get_feature_group_features(feature_group)
        if group_features:
            # Check if all expected features are present
            missing_features = set(group_features) - set(data.columns)
            if missing_features:
                validation_results['warnings'].append(f"Missing features: {missing_features}")
            
            # Check for unexpected features
            extra_features = set(data.columns) - set(group_features)
            if extra_features:
                validation_results['warnings'].append(f"Extra features: {extra_features}")
        
        # Check for null values in key columns
        null_counts = data.isnull().sum()
        high_null_features = null_counts[null_counts > len(data) * 0.5].index.tolist()
        if high_null_features:
            validation_results['warnings'].append(f"High null percentage: {high_null_features}")
        
        return validation_results
    
    def _enrich_feature_data(
        self,
        data: pd.DataFrame,
        timestamp_column: Optional[str] = None
    ) -> pd.DataFrame:
        """Add metadata columns to feature data"""
        
        enriched_data = data.copy()
        
        # Add ingestion timestamp
        enriched_data['_ingestion_timestamp'] = datetime.now()
        
        # Add event timestamp if not provided
        if timestamp_column and timestamp_column in data.columns:
            enriched_data['_event_timestamp'] = enriched_data[timestamp_column]
        else:
            enriched_data['_event_timestamp'] = datetime.now()
        
        # Add data version hash
        enriched_data['_data_hash'] = self._calculate_data_hash(data)
        
        return enriched_data
    
    def _get_feature_storage_path(
        self,
        feature_group: str,
        version: Optional[str] = None
    ) -> Path:
        """Get storage path for feature group"""
        
        if version:
            return self.storage_path / "snapshots" / feature_group / version / "data.parquet"
        else:
            return self.storage_path / feature_group
    
    def _write_to_storage(
        self,
        data: pd.DataFrame,
        storage_path: Path,
        partition_columns: Optional[List[str]] = None,
        mode: str = "append"
    ) -> Dict[str, Any]:
        """Write data to storage backend"""
        
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.storage_backend == "parquet":
            if partition_columns:
                # Partitioned write
                table = pa.Table.from_pandas(data)
                pq.write_to_dataset(
                    table,
                    root_path=str(storage_path),
                    partition_cols=partition_columns,
                    compression=self.config.compression,
                    existing_data_behavior='overwrite_or_ignore' if mode == 'append' else 'delete_matching'
                )
            else:
                # Single file write
                if mode == "append" and storage_path.exists():
                    existing_data = pd.read_parquet(storage_path)
                    combined_data = pd.concat([existing_data, data], ignore_index=True)
                    combined_data.to_parquet(storage_path, compression=self.config.compression)
                else:
                    data.to_parquet(storage_path, compression=self.config.compression)
            
            return {
                'storage_backend': 'parquet',
                'storage_path': str(storage_path),
                'compression': self.config.compression,
                'partitioned': bool(partition_columns)
            }
        
        else:
            raise ValueError(f"Unsupported storage backend: {self.config.storage_backend}")
    
    def _read_from_storage(
        self,
        storage_path: Path,
        features: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Read data from storage backend"""
        
        if not storage_path.exists():
            return pd.DataFrame()
        
        if self.config.storage_backend == "parquet":
            if storage_path.is_dir():
                # Read partitioned dataset
                dataset = pq.ParquetDataset(str(storage_path))
                table = dataset.read(columns=features)
                data = table.to_pandas()
            else:
                # Read single file
                data = pd.read_parquet(storage_path, columns=features)
            
            # Apply time filters
            if start_time or end_time:
                if '_event_timestamp' in data.columns:
                    if start_time:
                        data = data[data['_event_timestamp'] >= start_time]
                    if end_time:
                        data = data[data['_event_timestamp'] <= end_time]
            
            # Apply additional filters
            if filters:
                for column, condition in filters.items():
                    if column in data.columns:
                        if isinstance(condition, dict):
                            if 'gte' in condition:
                                data = data[data[column] >= condition['gte']]
                            if 'lte' in condition:
                                data = data[data[column] <= condition['lte']]
                            if 'eq' in condition:
                                data = data[data[column] == condition['eq']]
                            if 'in' in condition:
                                data = data[data[column].isin(condition['in'])]
                        else:
                            data = data[data[column] == condition]
            
            # Apply limit
            if limit:
                data = data.head(limit)
            
            return data
        
        else:
            raise ValueError(f"Unsupported storage backend: {self.config.storage_backend}")
    
    def _update_feature_statistics(
        self,
        feature_group: str,
        data: pd.DataFrame
    ):
        """Update feature statistics after write operation"""
        
        # This would update metadata with latest statistics
        # Implementation depends on metadata manager capabilities
        pass
    
    def _update_usage_statistics(
        self,
        feature_group: str,
        records_read: int
    ):
        """Update usage statistics after read operation"""
        
        # This would update metadata with usage statistics
        # Implementation depends on metadata manager capabilities
        pass
    
    def _compute_feature_column_statistics(
        self,
        series: pd.Series
    ) -> Dict[str, Any]:
        """Compute statistics for a single feature column"""
        
        stats = {
            'count': len(series),
            'null_count': series.isnull().sum(),
            'null_percentage': (series.isnull().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'unique_percentage': (series.nunique() / len(series)) * 100
        }
        
        # Numeric statistics
        if pd.api.types.is_numeric_dtype(series):
            stats.update({
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'median': series.median(),
                'q25': series.quantile(0.25),
                'q75': series.quantile(0.75)
            })
            
            # Detect outliers using IQR method
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            stats['outlier_count'] = len(outliers)
            stats['outlier_percentage'] = (len(outliers) / len(series)) * 100
        
        # String statistics
        elif pd.api.types.is_string_dtype(series):
            stats.update({
                'avg_length': series.astype(str).str.len().mean(),
                'min_length': series.astype(str).str.len().min(),
                'max_length': series.astype(str).str.len().max(),
                'most_frequent': series.mode().iloc[0] if len(series.mode()) > 0 else None
            })
        
        return stats
    
    def _validate_feature_against_rules(
        self,
        feature_name: str,
        feature_stats: Dict[str, Any],
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate feature against quality rules"""
        
        validation = {
            'feature': feature_name,
            'passed': True,
            'issues': []
        }
        
        # Check missing percentage
        if feature_stats['null_percentage'] > rules['max_missing_percentage']:
            validation['passed'] = False
            validation['issues'].append(
                f"Missing percentage ({feature_stats['null_percentage']:.2f}%) exceeds threshold"
            )
        
        # Check unique values
        if feature_stats['unique_count'] < rules['min_unique_values']:
            validation['passed'] = False
            validation['issues'].append(
                f"Unique values ({feature_stats['unique_count']}) below minimum"
            )
        
        # Check outliers for numeric features
        if 'outlier_percentage' in feature_stats:
            if feature_stats['outlier_percentage'] > rules['max_outlier_percentage']:
                validation['passed'] = False
                validation['issues'].append(
                    f"Outlier percentage ({feature_stats['outlier_percentage']:.2f}%) exceeds threshold"
                )
        
        return validation
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of data for versioning"""
        
        # Convert to string representation and hash
        data_str = data.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
