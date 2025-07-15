"""
Feature Serving Engine

Comprehensive feature serving system for online and offline feature delivery,
supporting real-time inference and batch processing use cases.

Key capabilities:
- Online feature serving with low latency
- Offline feature serving for batch processing
- Feature caching and optimization
- Point-in-time correctness
- Feature versioning and consistency
- Monitoring and metrics collection
"""

import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import hashlib

import pandas as pd
import numpy as np

from .feature_metadata import FeatureMetadataManager, FeatureMetadata


@dataclass
class FeatureRequest:
    """Feature serving request"""
    request_id: str
    entity_ids: List[str]
    feature_names: List[str]
    feature_group: str
    timestamp: Optional[datetime] = None
    version: Optional[str] = None
    serving_type: str = "online"  # online, offline, batch


@dataclass
class FeatureResponse:
    """Feature serving response"""
    request_id: str
    features: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]
    served_at: datetime
    latency_ms: float
    cache_hit: bool = False


@dataclass
class ServingConfig:
    """Feature serving configuration"""
    online_store_uri: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600
    max_batch_size: int = 1000
    connection_pool_size: int = 10
    timeout_seconds: int = 30
    enable_monitoring: bool = True
    enable_point_in_time: bool = True
    default_version: str = "latest"


class FeatureServingEngine:
    """
    Comprehensive feature serving engine for online and offline delivery.
    
    Provides high-performance feature serving with caching, versioning,
    and point-in-time correctness for ML applications.
    """
    
    def __init__(
        self,
        metadata_manager: FeatureMetadataManager,
        storage_path: Path,
        online_store_uri: Optional[str] = None,
        cache_ttl_seconds: int = 3600,
        config: Optional[ServingConfig] = None
    ):
        """Initialize feature serving engine"""
        
        self.metadata_manager = metadata_manager
        self.storage_path = storage_path
        self.config = config or ServingConfig()
        
        if online_store_uri:
            self.config.online_store_uri = online_store_uri
        if cache_ttl_seconds:
            self.config.cache_ttl_seconds = cache_ttl_seconds
        
        # Initialize online store connection
        self.online_store_enabled = self._setup_online_store()
        
        # Feature cache
        self._feature_cache = {}
        self._cache_timestamps = {}
        
        # Serving statistics
        self._serving_stats = {
            'requests_served': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'average_latency_ms': 0.0
        }
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=self.config.connection_pool_size)
    
    def get_online_features(
        self,
        feature_group: str,
        entity_ids: List[str],
        features: Optional[List[str]] = None,
        version: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get features for online serving (low latency)"""
        
        request_id = self._generate_request_id()
        start_time = time.time()
        
        try:
            # Create feature request
            request = FeatureRequest(
                request_id=request_id,
                entity_ids=entity_ids,
                feature_names=features or [],
                feature_group=feature_group,
                version=version,
                serving_type="online"
            )
            
            # Serve features
            response = self._serve_features(request)
            
            # Update statistics
            latency_ms = (time.time() - start_time) * 1000
            self._update_serving_stats(latency_ms, response.cache_hit)
            
            return response.features
            
        except Exception as e:
            self._serving_stats['errors'] += 1
            raise Exception(f"Online feature serving failed: {str(e)}")
    
    def get_offline_features(
        self,
        feature_group: str,
        entity_ids: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        version: Optional[str] = None
    ) -> pd.DataFrame:
        """Get features for offline serving (batch processing)"""
        
        request_id = self._generate_request_id()
        
        try:
            # Read from storage
            feature_data = self._read_offline_features(
                feature_group=feature_group,
                entity_ids=entity_ids,
                features=features,
                start_time=start_time,
                end_time=end_time,
                version=version
            )
            
            return feature_data
            
        except Exception as e:
            self._serving_stats['errors'] += 1
            raise Exception(f"Offline feature serving failed: {str(e)}")
    
    def get_point_in_time_features(
        self,
        feature_group: str,
        entity_ids: List[str],
        timestamp: datetime,
        features: Optional[List[str]] = None,
        version: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get features as they existed at a specific point in time"""
        
        if not self.config.enable_point_in_time:
            raise ValueError("Point-in-time serving is not enabled")
        
        request_id = self._generate_request_id()
        
        try:
            # Create feature request with timestamp
            request = FeatureRequest(
                request_id=request_id,
                entity_ids=entity_ids,
                feature_names=features or [],
                feature_group=feature_group,
                timestamp=timestamp,
                version=version,
                serving_type="point_in_time"
            )
            
            # Serve features with time travel
            response = self._serve_point_in_time_features(request)
            return response.features
            
        except Exception as e:
            raise Exception(f"Point-in-time feature serving failed: {str(e)}")
    
    def update_online_store(
        self,
        feature_group: str,
        feature_data: pd.DataFrame,
        entity_column: str = "entity_id"
    ) -> bool:
        """Update online store with latest feature data"""
        
        if not self.online_store_enabled:
            return False
        
        try:
            # Process each row for online store
            for _, row in feature_data.iterrows():
                entity_id = str(row[entity_column])
                
                # Create feature vector
                feature_vector = {}
                for column in feature_data.columns:
                    if column != entity_column and not column.startswith('_'):
                        feature_vector[column] = self._serialize_feature_value(row[column])
                
                # Store in online store
                self._store_online_features(feature_group, entity_id, feature_vector)
            
            return True
            
        except Exception as e:
            raise Exception(f"Failed to update online store: {str(e)}")
    
    def serve_feature_batch(
        self,
        requests: List[FeatureRequest]
    ) -> List[FeatureResponse]:
        """Serve multiple feature requests in batch"""
        
        responses = []
        
        # Group requests by feature group for efficiency
        grouped_requests = {}
        for request in requests:
            group = request.feature_group
            if group not in grouped_requests:
                grouped_requests[group] = []
            grouped_requests[group].append(request)
        
        # Process each group
        for feature_group, group_requests in grouped_requests.items():
            group_responses = self._serve_feature_group_batch(feature_group, group_requests)
            responses.extend(group_responses)
        
        return responses
    
    def precompute_features(
        self,
        feature_group: str,
        entity_ids: List[str],
        features: Optional[List[str]] = None
    ) -> bool:
        """Precompute and cache features for faster serving"""
        
        try:
            # Read features from storage
            feature_data = self._read_offline_features(
                feature_group=feature_group,
                entity_ids=entity_ids,
                features=features
            )
            
            # Cache features
            for _, row in feature_data.iterrows():
                entity_id = str(row.get('entity_id', row.iloc[0]))
                cache_key = self._get_cache_key(feature_group, entity_id, features)
                
                feature_vector = {}
                for column in feature_data.columns:
                    if not column.startswith('_'):
                        feature_vector[column] = row[column]
                
                self._feature_cache[cache_key] = feature_vector
                self._cache_timestamps[cache_key] = datetime.now()
            
            return True
            
        except Exception as e:
            raise Exception(f"Feature precomputation failed: {str(e)}")
    
    def validate_feature_freshness(
        self,
        feature_group: str,
        max_age_minutes: int = 60
    ) -> Dict[str, Any]:
        """Validate feature freshness in online store"""
        
        validation_results = {
            'feature_group': feature_group,
            'validation_time': datetime.now().isoformat(),
            'max_age_minutes': max_age_minutes,
            'fresh_features': 0,
            'stale_features': 0,
            'missing_features': 0,
            'details': []
        }
        
        if not self.online_store_enabled:
            validation_results['error'] = 'Online store not enabled'
            return validation_results
        
        # Get expected features from metadata
        expected_features = self._get_feature_group_metadata(feature_group)
        
        # Check freshness for each feature
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        
        for feature_metadata in expected_features:
            feature_name = feature_metadata.name
            
            # Check online store for feature
            freshness_info = self._check_feature_freshness(feature_group, feature_name)
            
            if freshness_info['exists']:
                if freshness_info['last_updated'] >= cutoff_time:
                    validation_results['fresh_features'] += 1
                    status = 'fresh'
                else:
                    validation_results['stale_features'] += 1
                    status = 'stale'
            else:
                validation_results['missing_features'] += 1
                status = 'missing'
            
            validation_results['details'].append({
                'feature_name': feature_name,
                'status': status,
                'last_updated': freshness_info.get('last_updated', {}).isoformat() if freshness_info.get('last_updated') else None,
                'age_minutes': (datetime.now() - freshness_info['last_updated']).total_seconds() / 60 if freshness_info.get('last_updated') else None
            })
        
        return validation_results
    
    def get_serving_statistics(self) -> Dict[str, Any]:
        """Get feature serving statistics"""
        
        return {
            **self._serving_stats,
            'cache_hit_rate': (
                self._serving_stats['cache_hits'] / 
                max(self._serving_stats['cache_hits'] + self._serving_stats['cache_misses'], 1)
            ) * 100,
            'error_rate': (
                self._serving_stats['errors'] / 
                max(self._serving_stats['requests_served'], 1)
            ) * 100,
            'online_store_enabled': self.online_store_enabled,
            'cache_size': len(self._feature_cache)
        }
    
    def clear_cache(self, feature_group: Optional[str] = None) -> int:
        """Clear feature cache"""
        
        if feature_group:
            # Clear cache for specific feature group
            keys_to_remove = [
                key for key in self._feature_cache.keys()
                if key.startswith(f"{feature_group}:")
            ]
            
            for key in keys_to_remove:
                del self._feature_cache[key]
                if key in self._cache_timestamps:
                    del self._cache_timestamps[key]
            
            return len(keys_to_remove)
        else:
            # Clear entire cache
            cache_size = len(self._feature_cache)
            self._feature_cache.clear()
            self._cache_timestamps.clear()
            return cache_size
    
    def export_serving_config(self, output_path: str) -> str:
        """Export serving configuration"""
        
        config_data = {
            'serving_config': asdict(self.config),
            'serving_statistics': self.get_serving_statistics(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        return str(output_path)
    
    def _serve_features(self, request: FeatureRequest) -> FeatureResponse:
        """Internal method to serve features"""
        
        start_time = time.time()
        cache_hit = False
        
        # Check cache first
        if request.serving_type == "online":
            cached_features = self._get_cached_features(request)
            if cached_features:
                cache_hit = True
                return FeatureResponse(
                    request_id=request.request_id,
                    features=cached_features,
                    metadata={'source': 'cache'},
                    served_at=datetime.now(),
                    latency_ms=(time.time() - start_time) * 1000,
                    cache_hit=True
                )
        
        # Serve from storage/online store
        if self.online_store_enabled and request.serving_type == "online":
            features = self._serve_from_online_store(request)
        else:
            features = self._serve_from_offline_store(request)
        
        # Cache results for future requests
        if request.serving_type == "online":
            self._cache_features(request, features)
        
        return FeatureResponse(
            request_id=request.request_id,
            features=features,
            metadata={'source': 'online_store' if self.online_store_enabled else 'offline_store'},
            served_at=datetime.now(),
            latency_ms=(time.time() - start_time) * 1000,
            cache_hit=cache_hit
        )
    
    def _serve_point_in_time_features(self, request: FeatureRequest) -> FeatureResponse:
        """Serve features for a specific point in time"""
        
        start_time = time.time()
        
        # Read historical data up to the specified timestamp
        features = {}
        
        for entity_id in request.entity_ids:
            entity_features = self._get_point_in_time_entity_features(
                feature_group=request.feature_group,
                entity_id=entity_id,
                timestamp=request.timestamp,
                features=request.feature_names
            )
            features[entity_id] = entity_features
        
        return FeatureResponse(
            request_id=request.request_id,
            features=features,
            metadata={
                'source': 'point_in_time',
                'timestamp': request.timestamp.isoformat()
            },
            served_at=datetime.now(),
            latency_ms=(time.time() - start_time) * 1000,
            cache_hit=False
        )
    
    def _serve_feature_group_batch(
        self,
        feature_group: str,
        requests: List[FeatureRequest]
    ) -> List[FeatureResponse]:
        """Serve batch of requests for same feature group"""
        
        responses = []
        
        # Collect all entity IDs
        all_entity_ids = []
        for request in requests:
            all_entity_ids.extend(request.entity_ids)
        
        # Remove duplicates while preserving order
        unique_entity_ids = list(dict.fromkeys(all_entity_ids))
        
        # Batch read features
        if self.online_store_enabled:
            batch_features = self._batch_read_online_store(feature_group, unique_entity_ids)
        else:
            batch_features = self._batch_read_offline_store(feature_group, unique_entity_ids)
        
        # Create responses for each request
        for request in requests:
            request_features = {
                entity_id: batch_features.get(entity_id, {})
                for entity_id in request.entity_ids
            }
            
            response = FeatureResponse(
                request_id=request.request_id,
                features=request_features,
                metadata={'source': 'batch'},
                served_at=datetime.now(),
                latency_ms=0.0,  # Will be calculated separately
                cache_hit=False
            )
            
            responses.append(response)
        
        return responses
    
    def _setup_online_store(self) -> bool:
        """Setup online store connection"""
        
        try:
            # This would setup Redis or other online store
            # For now, we'll simulate successful setup
            return self.config.online_store_uri is not None
        except Exception:
            return False
    
    def _read_offline_features(
        self,
        feature_group: str,
        entity_ids: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        version: Optional[str] = None
    ) -> pd.DataFrame:
        """Read features from offline storage"""
        
        # Determine storage path
        if version and version != "latest":
            storage_path = self.storage_path / "snapshots" / feature_group / version / "data.parquet"
        else:
            storage_path = self.storage_path / feature_group
        
        if not storage_path.exists():
            return pd.DataFrame()
        
        # Read data
        if storage_path.is_dir():
            # Read partitioned dataset
            try:
                import pyarrow.parquet as pq
                dataset = pq.ParquetDataset(str(storage_path))
                table = dataset.read(columns=features)
                data = table.to_pandas()
            except ImportError:
                # Fallback to pandas
                data = pd.read_parquet(storage_path, columns=features)
        else:
            data = pd.read_parquet(storage_path, columns=features)
        
        # Apply filters
        if entity_ids:
            # Assume first column or 'entity_id' column contains entity IDs
            entity_column = 'entity_id' if 'entity_id' in data.columns else data.columns[0]
            data = data[data[entity_column].isin(entity_ids)]
        
        if start_time or end_time:
            timestamp_column = '_event_timestamp' if '_event_timestamp' in data.columns else None
            if timestamp_column:
                if start_time:
                    data = data[data[timestamp_column] >= start_time]
                if end_time:
                    data = data[data[timestamp_column] <= end_time]
        
        return data
    
    def _serve_from_online_store(self, request: FeatureRequest) -> Dict[str, Dict[str, Any]]:
        """Serve features from online store"""
        
        features = {}
        
        for entity_id in request.entity_ids:
            entity_features = self._get_online_entity_features(
                feature_group=request.feature_group,
                entity_id=entity_id,
                features=request.feature_names
            )
            features[entity_id] = entity_features
        
        return features
    
    def _serve_from_offline_store(self, request: FeatureRequest) -> Dict[str, Dict[str, Any]]:
        """Serve features from offline storage"""
        
        # Read features from storage
        feature_data = self._read_offline_features(
            feature_group=request.feature_group,
            entity_ids=request.entity_ids,
            features=request.feature_names
        )
        
        # Convert to response format
        features = {}
        entity_column = 'entity_id' if 'entity_id' in feature_data.columns else feature_data.columns[0]
        
        for _, row in feature_data.iterrows():
            entity_id = str(row[entity_column])
            entity_features = {}
            
            for column in feature_data.columns:
                if column != entity_column and not column.startswith('_'):
                    entity_features[column] = row[column]
            
            features[entity_id] = entity_features
        
        return features
    
    def _get_cached_features(self, request: FeatureRequest) -> Optional[Dict[str, Dict[str, Any]]]:
        """Get features from cache"""
        
        # Check if all requested features are cached and fresh
        cached_features = {}
        
        for entity_id in request.entity_ids:
            cache_key = self._get_cache_key(request.feature_group, entity_id, request.feature_names)
            
            if cache_key in self._feature_cache:
                # Check if cache is still fresh
                cache_time = self._cache_timestamps.get(cache_key)
                if cache_time and (datetime.now() - cache_time).total_seconds() < self.config.cache_ttl_seconds:
                    cached_features[entity_id] = self._feature_cache[cache_key].copy()
                else:
                    # Cache expired
                    return None
            else:
                # Feature not cached
                return None
        
        return cached_features if cached_features else None
    
    def _cache_features(self, request: FeatureRequest, features: Dict[str, Dict[str, Any]]):
        """Cache features for future requests"""
        
        for entity_id, entity_features in features.items():
            cache_key = self._get_cache_key(request.feature_group, entity_id, request.feature_names)
            self._feature_cache[cache_key] = entity_features
            self._cache_timestamps[cache_key] = datetime.now()
    
    def _get_cache_key(
        self,
        feature_group: str,
        entity_id: str,
        features: Optional[List[str]]
    ) -> str:
        """Generate cache key"""
        
        features_str = ",".join(sorted(features)) if features else "all"
        return f"{feature_group}:{entity_id}:{features_str}"
    
    def _store_online_features(
        self,
        feature_group: str,
        entity_id: str,
        feature_vector: Dict[str, Any]
    ):
        """Store features in online store"""
        
        # This would store in Redis or other online store
        # For now, we'll use a simple in-memory store
        store_key = f"{feature_group}:{entity_id}"
        # self.online_store.hset(store_key, mapping=feature_vector)
        # self.online_store.expire(store_key, self.config.cache_ttl_seconds)
    
    def _get_online_entity_features(
        self,
        feature_group: str,
        entity_id: str,
        features: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Get features for entity from online store"""
        
        # This would retrieve from Redis or other online store
        # For now, return empty dict
        return {}
    
    def _batch_read_online_store(
        self,
        feature_group: str,
        entity_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Batch read from online store"""
        
        batch_features = {}
        
        for entity_id in entity_ids:
            entity_features = self._get_online_entity_features(feature_group, entity_id, None)
            batch_features[entity_id] = entity_features
        
        return batch_features
    
    def _batch_read_offline_store(
        self,
        feature_group: str,
        entity_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Batch read from offline storage"""
        
        feature_data = self._read_offline_features(
            feature_group=feature_group,
            entity_ids=entity_ids
        )
        
        batch_features = {}
        entity_column = 'entity_id' if 'entity_id' in feature_data.columns else feature_data.columns[0]
        
        for _, row in feature_data.iterrows():
            entity_id = str(row[entity_column])
            entity_features = {}
            
            for column in feature_data.columns:
                if column != entity_column and not column.startswith('_'):
                    entity_features[column] = row[column]
            
            batch_features[entity_id] = entity_features
        
        return batch_features
    
    def _get_point_in_time_entity_features(
        self,
        feature_group: str,
        entity_id: str,
        timestamp: datetime,
        features: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Get features for entity at specific point in time"""
        
        # Read historical data
        feature_data = self._read_offline_features(
            feature_group=feature_group,
            entity_ids=[entity_id],
            features=features,
            end_time=timestamp
        )
        
        if feature_data.empty:
            return {}
        
        # Get latest record before timestamp
        if '_event_timestamp' in feature_data.columns:
            feature_data = feature_data.sort_values('_event_timestamp', ascending=False)
        
        latest_record = feature_data.iloc[0]
        entity_features = {}
        
        for column in feature_data.columns:
            if not column.startswith('_') and column != 'entity_id':
                entity_features[column] = latest_record[column]
        
        return entity_features
    
    def _get_feature_group_metadata(self, feature_group: str) -> List[FeatureMetadata]:
        """Get feature metadata for group"""
        
        return self.metadata_manager.list_features(feature_group=feature_group)
    
    def _check_feature_freshness(
        self,
        feature_group: str,
        feature_name: str
    ) -> Dict[str, Any]:
        """Check feature freshness in online store"""
        
        # This would check Redis or other online store
        # For now, return mock data
        return {
            'exists': True,
            'last_updated': datetime.now() - timedelta(minutes=30)
        }
    
    def _serialize_feature_value(self, value: Any) -> str:
        """Serialize feature value for storage"""
        
        if pd.isna(value):
            return "null"
        elif isinstance(value, (int, float, bool)):
            return str(value)
        elif isinstance(value, str):
            return value
        else:
            return json.dumps(value, default=str)
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        
        timestamp = int(time.time() * 1000)
        hash_input = f"request_{timestamp}_{len(self._serving_stats)}"
        request_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"req_{timestamp}_{request_hash}"
    
    def _update_serving_stats(self, latency_ms: float, cache_hit: bool):
        """Update serving statistics"""
        
        self._serving_stats['requests_served'] += 1
        
        if cache_hit:
            self._serving_stats['cache_hits'] += 1
        else:
            self._serving_stats['cache_misses'] += 1
        
        # Update average latency
        current_avg = self._serving_stats['average_latency_ms']
        request_count = self._serving_stats['requests_served']
        
        new_avg = ((current_avg * (request_count - 1)) + latency_ms) / request_count
        self._serving_stats['average_latency_ms'] = new_avg
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
