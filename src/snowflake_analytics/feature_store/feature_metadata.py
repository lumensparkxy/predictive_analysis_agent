"""
Feature Metadata Management

Comprehensive metadata management for features including schema definitions,
data lineage, versioning, and quality metrics.

Key capabilities:
- Feature schema and type management
- Version control and change tracking
- Data quality metrics and validation rules
- Feature dependencies and lineage
- Usage statistics and performance metrics
- Compliance and governance metadata
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from ...config.settings import SnowflakeSettings
from ...utils.logger import SnowflakeLogger


class FeatureType(Enum):
    """Feature data types"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    TEXT = "text"
    EMBEDDING = "embedding"
    DERIVED = "derived"


class FeatureStatus(Enum):
    """Feature lifecycle status"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    EXPERIMENTAL = "experimental"


class DataQualityStatus(Enum):
    """Data quality assessment status"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


@dataclass
class FeatureSchema:
    """Feature schema definition"""
    name: str
    feature_type: FeatureType
    data_type: str
    description: str
    nullable: bool = True
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Feature quality metrics"""
    completeness: float  # 0-1 (1 = no missing values)
    uniqueness: float    # 0-1 (1 = all values unique)
    consistency: float   # 0-1 (1 = consistent format/type)
    validity: float      # 0-1 (1 = all values valid)
    accuracy: float      # 0-1 (1 = accurate representation)
    timeliness: float    # 0-1 (1 = up-to-date)
    overall_score: float # 0-1 (weighted average)
    
    def __post_init__(self):
        if self.overall_score == 0:
            scores = [self.completeness, self.uniqueness, self.consistency, 
                     self.validity, self.accuracy, self.timeliness]
            self.overall_score = sum(scores) / len(scores)


@dataclass
class UsageStatistics:
    """Feature usage statistics"""
    total_reads: int = 0
    total_writes: int = 0
    unique_consumers: int = 0
    last_accessed: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    avg_query_time_ms: float = 0.0
    popular_queries: List[str] = field(default_factory=list)
    access_patterns: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureVersion:
    """Feature version information"""
    version_id: str
    version_number: str
    created_at: datetime
    created_by: str
    description: str
    schema_hash: str
    data_hash: Optional[str] = None
    parent_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_current: bool = False


@dataclass
class FeatureGroup:
    """Logical grouping of related features"""
    group_id: str
    name: str
    description: str
    features: List[str]
    created_at: datetime
    created_by: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureMetadata:
    """Comprehensive feature metadata"""
    feature_id: str
    name: str
    description: str
    feature_group: str
    schema: FeatureSchema
    status: FeatureStatus
    created_at: datetime
    created_by: str
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None
    
    # Versioning
    current_version: Optional[FeatureVersion] = None
    version_history: List[FeatureVersion] = field(default_factory=list)
    
    # Quality and validation
    quality_metrics: Optional[QualityMetrics] = None
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Usage and performance
    usage_stats: UsageStatistics = field(default_factory=UsageStatistics)
    
    # Lineage and dependencies
    upstream_dependencies: List[str] = field(default_factory=list)
    downstream_dependencies: List[str] = field(default_factory=list)
    
    # Governance and compliance
    owners: List[str] = field(default_factory=list)
    stewards: List[str] = field(default_factory=list)
    sensitivity_level: str = "public"  # public, internal, confidential, restricted
    compliance_tags: List[str] = field(default_factory=list)
    
    # Technical metadata
    storage_location: Optional[str] = None
    partitioning_scheme: Optional[Dict[str, Any]] = None
    refresh_frequency: Optional[str] = None
    retention_policy: Optional[Dict[str, Any]] = None
    
    # Business metadata
    business_definition: Optional[str] = None
    business_rules: List[str] = field(default_factory=list)
    kpis: List[str] = field(default_factory=list)
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


# SQLAlchemy models for persistent storage
Base = declarative_base()


class FeatureMetadataModel(Base):
    """SQLAlchemy model for feature metadata"""
    __tablename__ = 'feature_metadata'
    
    feature_id = Column(String, primary_key=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text)
    feature_group = Column(String, index=True)
    schema_json = Column(JSON)
    status = Column(String, index=True)
    created_at = Column(DateTime, nullable=False)
    created_by = Column(String)
    updated_at = Column(DateTime)
    updated_by = Column(String)
    
    # Versioning
    current_version_id = Column(String)
    
    # Quality metrics
    quality_score = Column(Float)
    completeness = Column(Float)
    uniqueness = Column(Float)
    consistency = Column(Float)
    validity = Column(Float)
    accuracy = Column(Float)
    timeliness = Column(Float)
    
    # Usage statistics
    total_reads = Column(Integer, default=0)
    total_writes = Column(Integer, default=0)
    unique_consumers = Column(Integer, default=0)
    last_accessed = Column(DateTime)
    last_updated = Column(DateTime)
    avg_query_time_ms = Column(Float, default=0.0)
    
    # Governance
    owners_json = Column(JSON)
    stewards_json = Column(JSON)
    sensitivity_level = Column(String, default='public')
    compliance_tags_json = Column(JSON)
    
    # Technical metadata
    storage_location = Column(String)
    partitioning_scheme_json = Column(JSON)
    refresh_frequency = Column(String)
    retention_policy_json = Column(JSON)
    
    # Business metadata
    business_definition = Column(Text)
    business_rules_json = Column(JSON)
    kpis_json = Column(JSON)
    
    # Additional metadata
    tags_json = Column(JSON)
    custom_metadata_json = Column(JSON)


class FeatureVersionModel(Base):
    """SQLAlchemy model for feature versions"""
    __tablename__ = 'feature_versions'
    
    version_id = Column(String, primary_key=True)
    feature_id = Column(String, nullable=False, index=True)
    version_number = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    created_by = Column(String)
    description = Column(Text)
    schema_hash = Column(String)
    data_hash = Column(String)
    parent_version = Column(String)
    tags_json = Column(JSON)
    metadata_json = Column(JSON)
    is_current = Column(Boolean, default=False)


class FeatureGroupModel(Base):
    """SQLAlchemy model for feature groups"""
    __tablename__ = 'feature_groups'
    
    group_id = Column(String, primary_key=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text)
    features_json = Column(JSON)
    created_at = Column(DateTime, nullable=False)
    created_by = Column(String)
    tags_json = Column(JSON)
    metadata_json = Column(JSON)


class FeatureMetadataManager:
    """
    Comprehensive feature metadata management system.
    
    Provides complete lifecycle management for feature metadata including
    creation, versioning, quality tracking, and governance.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        registry_uri: Optional[str] = None,
        enable_caching: bool = True
    ):
        """Initialize feature metadata manager"""
        
        # Core configuration
        self.settings = SnowflakeSettings(config_path)
        self.logger = SnowflakeLogger("FeatureMetadataManager").get_logger()
        
        # Database setup
        self.registry_uri = registry_uri or "sqlite:///data/feature_registry.db"
        self.engine = create_engine(self.registry_uri)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Caching
        self.enable_caching = enable_caching
        self.metadata_cache: Dict[str, FeatureMetadata] = {}
        self.cache_ttl = timedelta(minutes=30)
        self.cache_timestamps: Dict[str, datetime] = {}
        
        self.logger.info("FeatureMetadataManager initialized successfully")
    
    def create_feature_metadata(
        self,
        name: str,
        description: str,
        schema: FeatureSchema,
        feature_group: str,
        created_by: str,
        **kwargs
    ) -> FeatureMetadata:
        """Create new feature metadata"""
        
        feature_id = self._generate_feature_id(name, feature_group)
        
        metadata = FeatureMetadata(
            feature_id=feature_id,
            name=name,
            description=description,
            feature_group=feature_group,
            schema=schema,
            status=FeatureStatus.DRAFT,
            created_at=datetime.now(),
            created_by=created_by,
            **kwargs
        )
        
        # Create initial version
        initial_version = self._create_initial_version(metadata, created_by)
        metadata.current_version = initial_version
        metadata.version_history = [initial_version]
        
        # Store in database
        self._store_metadata(metadata)
        
        # Update cache
        if self.enable_caching:
            self.metadata_cache[feature_id] = metadata
            self.cache_timestamps[feature_id] = datetime.now()
        
        self.logger.info(f"Created feature metadata: {feature_id}")
        return metadata
    
    def get_feature_metadata(self, feature_id: str) -> Optional[FeatureMetadata]:
        """Get feature metadata by ID"""
        
        # Check cache first
        if self.enable_caching and feature_id in self.metadata_cache:
            cached_time = self.cache_timestamps.get(feature_id)
            if cached_time and datetime.now() - cached_time < self.cache_ttl:
                return self.metadata_cache[feature_id]
        
        # Load from database
        metadata = self._load_metadata(feature_id)
        
        # Update cache
        if metadata and self.enable_caching:
            self.metadata_cache[feature_id] = metadata
            self.cache_timestamps[feature_id] = datetime.now()
        
        return metadata
    
    def update_feature_metadata(
        self,
        feature_id: str,
        updates: Dict[str, Any],
        updated_by: str,
        create_version: bool = True
    ) -> FeatureMetadata:
        """Update feature metadata"""
        
        metadata = self.get_feature_metadata(feature_id)
        if not metadata:
            raise ValueError(f"Feature not found: {feature_id}")
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        metadata.updated_at = datetime.now()
        metadata.updated_by = updated_by
        
        # Create new version if requested
        if create_version:
            new_version = self._create_version(metadata, updated_by, "Updated metadata")
            metadata.version_history.append(new_version)
            metadata.current_version = new_version
        
        # Store in database
        self._store_metadata(metadata)
        
        # Update cache
        if self.enable_caching:
            self.metadata_cache[feature_id] = metadata
            self.cache_timestamps[feature_id] = datetime.now()
        
        self.logger.info(f"Updated feature metadata: {feature_id}")
        return metadata
    
    def update_quality_metrics(
        self,
        feature_id: str,
        quality_metrics: QualityMetrics
    ):
        """Update feature quality metrics"""
        
        metadata = self.get_feature_metadata(feature_id)
        if not metadata:
            raise ValueError(f"Feature not found: {feature_id}")
        
        metadata.quality_metrics = quality_metrics
        metadata.updated_at = datetime.now()
        
        # Store in database
        self._store_metadata(metadata)
        
        # Update cache
        if self.enable_caching:
            self.metadata_cache[feature_id] = metadata
            self.cache_timestamps[feature_id] = datetime.now()
        
        self.logger.info(f"Updated quality metrics for feature: {feature_id}")
    
    def update_usage_statistics(
        self,
        feature_id: str,
        usage_stats: UsageStatistics
    ):
        """Update feature usage statistics"""
        
        metadata = self.get_feature_metadata(feature_id)
        if not metadata:
            raise ValueError(f"Feature not found: {feature_id}")
        
        metadata.usage_stats = usage_stats
        metadata.updated_at = datetime.now()
        
        # Store in database
        self._store_metadata(metadata)
        
        # Update cache
        if self.enable_caching:
            self.metadata_cache[feature_id] = metadata
            self.cache_timestamps[feature_id] = datetime.now()
        
        self.logger.info(f"Updated usage statistics for feature: {feature_id}")
    
    def create_feature_group(
        self,
        name: str,
        description: str,
        features: List[str],
        created_by: str,
        **kwargs
    ) -> FeatureGroup:
        """Create feature group"""
        
        group_id = self._generate_group_id(name)
        
        group = FeatureGroup(
            group_id=group_id,
            name=name,
            description=description,
            features=features,
            created_at=datetime.now(),
            created_by=created_by,
            **kwargs
        )
        
        # Store in database
        self._store_feature_group(group)
        
        self.logger.info(f"Created feature group: {group_id}")
        return group
    
    def get_feature_group(self, group_id: str) -> Optional[FeatureGroup]:
        """Get feature group by ID"""
        
        return self._load_feature_group(group_id)
    
    def list_features(
        self,
        feature_group: Optional[str] = None,
        status: Optional[FeatureStatus] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[FeatureMetadata]:
        """List features with optional filtering"""
        
        return self._query_features(
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
        
        if search_fields is None:
            search_fields = ['name', 'description', 'business_definition']
        
        return self._search_features(query, search_fields, limit)
    
    def get_feature_lineage(
        self,
        feature_id: str,
        direction: str = 'both'  # 'upstream', 'downstream', 'both'
    ) -> Dict[str, List[str]]:
        """Get feature lineage (dependencies)"""
        
        metadata = self.get_feature_metadata(feature_id)
        if not metadata:
            return {'upstream': [], 'downstream': []}
        
        lineage = {}
        
        if direction in ['upstream', 'both']:
            lineage['upstream'] = metadata.upstream_dependencies
        
        if direction in ['downstream', 'both']:
            lineage['downstream'] = metadata.downstream_dependencies
        
        return lineage
    
    def add_feature_dependency(
        self,
        feature_id: str,
        dependency_id: str,
        dependency_type: str = 'upstream'
    ):
        """Add feature dependency"""
        
        metadata = self.get_feature_metadata(feature_id)
        if not metadata:
            raise ValueError(f"Feature not found: {feature_id}")
        
        if dependency_type == 'upstream':
            if dependency_id not in metadata.upstream_dependencies:
                metadata.upstream_dependencies.append(dependency_id)
            
            # Add reverse dependency
            dep_metadata = self.get_feature_metadata(dependency_id)
            if dep_metadata and feature_id not in dep_metadata.downstream_dependencies:
                dep_metadata.downstream_dependencies.append(feature_id)
                self._store_metadata(dep_metadata)
        
        elif dependency_type == 'downstream':
            if dependency_id not in metadata.downstream_dependencies:
                metadata.downstream_dependencies.append(dependency_id)
            
            # Add reverse dependency
            dep_metadata = self.get_feature_metadata(dependency_id)
            if dep_metadata and feature_id not in dep_metadata.upstream_dependencies:
                dep_metadata.upstream_dependencies.append(feature_id)
                self._store_metadata(dep_metadata)
        
        metadata.updated_at = datetime.now()
        self._store_metadata(metadata)
        
        # Update cache
        if self.enable_caching:
            self.metadata_cache[feature_id] = metadata
            self.cache_timestamps[feature_id] = datetime.now()
        
        self.logger.info(f"Added {dependency_type} dependency: {feature_id} -> {dependency_id}")
    
    def create_feature_version(
        self,
        feature_id: str,
        description: str,
        created_by: str,
        tags: Optional[List[str]] = None
    ) -> FeatureVersion:
        """Create new feature version"""
        
        metadata = self.get_feature_metadata(feature_id)
        if not metadata:
            raise ValueError(f"Feature not found: {feature_id}")
        
        new_version = self._create_version(metadata, created_by, description, tags)
        metadata.version_history.append(new_version)
        metadata.current_version = new_version
        metadata.updated_at = datetime.now()
        
        # Store in database
        self._store_metadata(metadata)
        
        # Update cache
        if self.enable_caching:
            self.metadata_cache[feature_id] = metadata
            self.cache_timestamps[feature_id] = datetime.now()
        
        self.logger.info(f"Created new version for feature: {feature_id}")
        return new_version
    
    def get_feature_versions(self, feature_id: str) -> List[FeatureVersion]:
        """Get all versions for a feature"""
        
        metadata = self.get_feature_metadata(feature_id)
        if not metadata:
            return []
        
        return metadata.version_history
    
    def export_metadata(
        self,
        output_path: str,
        format: str = 'json',
        include_versions: bool = True
    ):
        """Export feature metadata to file"""
        
        all_features = self.list_features()
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_features': len(all_features),
            'features': []
        }
        
        for feature in all_features:
            feature_data = asdict(feature)
            
            # Convert datetime objects
            for key, value in feature_data.items():
                if isinstance(value, datetime):
                    feature_data[key] = value.isoformat()
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, datetime):
                            feature_data[key][i] = item.isoformat()
            
            if not include_versions:
                feature_data.pop('version_history', None)
            
            export_data['features'].append(feature_data)
        
        output_path = Path(output_path)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format.lower() == 'csv':
            # Flatten data for CSV export
            flattened_data = []
            for feature_data in export_data['features']:
                flat_record = self._flatten_dict(feature_data)
                flattened_data.append(flat_record)
            
            df = pd.DataFrame(flattened_data)
            df.to_csv(output_path, index=False)
        
        self.logger.info(f"Exported {len(all_features)} features to: {output_path}")
    
    def _generate_feature_id(self, name: str, feature_group: str) -> str:
        """Generate unique feature ID"""
        
        base_id = f"{feature_group}.{name}".lower().replace(' ', '_')
        return base_id
    
    def _generate_group_id(self, name: str) -> str:
        """Generate unique group ID"""
        
        return name.lower().replace(' ', '_')
    
    def _create_initial_version(
        self,
        metadata: FeatureMetadata,
        created_by: str
    ) -> FeatureVersion:
        """Create initial version for new feature"""
        
        return FeatureVersion(
            version_id=str(uuid.uuid4()),
            version_number="1.0.0",
            created_at=datetime.now(),
            created_by=created_by,
            description="Initial version",
            schema_hash=self._calculate_schema_hash(metadata.schema),
            is_current=True
        )
    
    def _create_version(
        self,
        metadata: FeatureMetadata,
        created_by: str,
        description: str,
        tags: Optional[List[str]] = None
    ) -> FeatureVersion:
        """Create new version for existing feature"""
        
        # Calculate next version number
        current_version = metadata.current_version
        if current_version:
            version_parts = current_version.version_number.split('.')
            major, minor, patch = int(version_parts[0]), int(version_parts[1]), int(version_parts[2])
            new_version_number = f"{major}.{minor}.{patch + 1}"
        else:
            new_version_number = "1.0.0"
        
        return FeatureVersion(
            version_id=str(uuid.uuid4()),
            version_number=new_version_number,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
            schema_hash=self._calculate_schema_hash(metadata.schema),
            parent_version=current_version.version_id if current_version else None,
            tags=tags or [],
            is_current=True
        )
    
    def _calculate_schema_hash(self, schema: FeatureSchema) -> str:
        """Calculate hash for feature schema"""
        
        schema_str = json.dumps(asdict(schema), sort_keys=True, default=str)
        return hashlib.md5(schema_str.encode()).hexdigest()
    
    def _store_metadata(self, metadata: FeatureMetadata):
        """Store feature metadata in database"""
        
        session = self.SessionLocal()
        try:
            # Convert to database model
            db_metadata = FeatureMetadataModel(
                feature_id=metadata.feature_id,
                name=metadata.name,
                description=metadata.description,
                feature_group=metadata.feature_group,
                schema_json=asdict(metadata.schema),
                status=metadata.status.value,
                created_at=metadata.created_at,
                created_by=metadata.created_by,
                updated_at=metadata.updated_at,
                updated_by=metadata.updated_by,
                current_version_id=metadata.current_version.version_id if metadata.current_version else None,
                owners_json=metadata.owners,
                stewards_json=metadata.stewards,
                sensitivity_level=metadata.sensitivity_level,
                compliance_tags_json=metadata.compliance_tags,
                storage_location=metadata.storage_location,
                partitioning_scheme_json=metadata.partitioning_scheme,
                refresh_frequency=metadata.refresh_frequency,
                retention_policy_json=metadata.retention_policy,
                business_definition=metadata.business_definition,
                business_rules_json=metadata.business_rules,
                kpis_json=metadata.kpis,
                tags_json=metadata.tags,
                custom_metadata_json=metadata.custom_metadata
            )
            
            # Add quality metrics if available
            if metadata.quality_metrics:
                db_metadata.quality_score = metadata.quality_metrics.overall_score
                db_metadata.completeness = metadata.quality_metrics.completeness
                db_metadata.uniqueness = metadata.quality_metrics.uniqueness
                db_metadata.consistency = metadata.quality_metrics.consistency
                db_metadata.validity = metadata.quality_metrics.validity
                db_metadata.accuracy = metadata.quality_metrics.accuracy
                db_metadata.timeliness = metadata.quality_metrics.timeliness
            
            # Add usage statistics
            if metadata.usage_stats:
                db_metadata.total_reads = metadata.usage_stats.total_reads
                db_metadata.total_writes = metadata.usage_stats.total_writes
                db_metadata.unique_consumers = metadata.usage_stats.unique_consumers
                db_metadata.last_accessed = metadata.usage_stats.last_accessed
                db_metadata.last_updated = metadata.usage_stats.last_updated
                db_metadata.avg_query_time_ms = metadata.usage_stats.avg_query_time_ms
            
            session.merge(db_metadata)
            
            # Store versions
            for version in metadata.version_history:
                db_version = FeatureVersionModel(
                    version_id=version.version_id,
                    feature_id=metadata.feature_id,
                    version_number=version.version_number,
                    created_at=version.created_at,
                    created_by=version.created_by,
                    description=version.description,
                    schema_hash=version.schema_hash,
                    data_hash=version.data_hash,
                    parent_version=version.parent_version,
                    tags_json=version.tags,
                    metadata_json=version.metadata,
                    is_current=version.is_current
                )
                session.merge(db_version)
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to store metadata for {metadata.feature_id}: {str(e)}")
            raise
        finally:
            session.close()
    
    def _load_metadata(self, feature_id: str) -> Optional[FeatureMetadata]:
        """Load feature metadata from database"""
        
        session = self.SessionLocal()
        try:
            db_metadata = session.query(FeatureMetadataModel).filter_by(feature_id=feature_id).first()
            if not db_metadata:
                return None
            
            # Load versions
            db_versions = session.query(FeatureVersionModel).filter_by(feature_id=feature_id).all()
            
            # Convert from database model
            metadata = FeatureMetadata(
                feature_id=db_metadata.feature_id,
                name=db_metadata.name,
                description=db_metadata.description,
                feature_group=db_metadata.feature_group,
                schema=FeatureSchema(**db_metadata.schema_json) if db_metadata.schema_json else None,
                status=FeatureStatus(db_metadata.status),
                created_at=db_metadata.created_at,
                created_by=db_metadata.created_by,
                updated_at=db_metadata.updated_at,
                updated_by=db_metadata.updated_by,
                owners=db_metadata.owners_json or [],
                stewards=db_metadata.stewards_json or [],
                sensitivity_level=db_metadata.sensitivity_level,
                compliance_tags=db_metadata.compliance_tags_json or [],
                storage_location=db_metadata.storage_location,
                partitioning_scheme=db_metadata.partitioning_scheme_json,
                refresh_frequency=db_metadata.refresh_frequency,
                retention_policy=db_metadata.retention_policy_json,
                business_definition=db_metadata.business_definition,
                business_rules=db_metadata.business_rules_json or [],
                kpis=db_metadata.kpis_json or [],
                tags=db_metadata.tags_json or [],
                custom_metadata=db_metadata.custom_metadata_json or {}
            )
            
            # Add quality metrics
            if db_metadata.quality_score is not None:
                metadata.quality_metrics = QualityMetrics(
                    completeness=db_metadata.completeness or 0.0,
                    uniqueness=db_metadata.uniqueness or 0.0,
                    consistency=db_metadata.consistency or 0.0,
                    validity=db_metadata.validity or 0.0,
                    accuracy=db_metadata.accuracy or 0.0,
                    timeliness=db_metadata.timeliness or 0.0,
                    overall_score=db_metadata.quality_score
                )
            
            # Add usage statistics
            metadata.usage_stats = UsageStatistics(
                total_reads=db_metadata.total_reads or 0,
                total_writes=db_metadata.total_writes or 0,
                unique_consumers=db_metadata.unique_consumers or 0,
                last_accessed=db_metadata.last_accessed,
                last_updated=db_metadata.last_updated,
                avg_query_time_ms=db_metadata.avg_query_time_ms or 0.0
            )
            
            # Add versions
            version_history = []
            current_version = None
            
            for db_version in db_versions:
                version = FeatureVersion(
                    version_id=db_version.version_id,
                    version_number=db_version.version_number,
                    created_at=db_version.created_at,
                    created_by=db_version.created_by,
                    description=db_version.description,
                    schema_hash=db_version.schema_hash,
                    data_hash=db_version.data_hash,
                    parent_version=db_version.parent_version,
                    tags=db_version.tags_json or [],
                    metadata=db_version.metadata_json or {},
                    is_current=db_version.is_current
                )
                version_history.append(version)
                
                if db_version.is_current:
                    current_version = version
            
            metadata.version_history = version_history
            metadata.current_version = current_version
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load metadata for {feature_id}: {str(e)}")
            return None
        finally:
            session.close()
    
    def _store_feature_group(self, group: FeatureGroup):
        """Store feature group in database"""
        
        session = self.SessionLocal()
        try:
            db_group = FeatureGroupModel(
                group_id=group.group_id,
                name=group.name,
                description=group.description,
                features_json=group.features,
                created_at=group.created_at,
                created_by=group.created_by,
                tags_json=group.tags,
                metadata_json=group.metadata
            )
            
            session.merge(db_group)
            session.commit()
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to store feature group {group.group_id}: {str(e)}")
            raise
        finally:
            session.close()
    
    def _load_feature_group(self, group_id: str) -> Optional[FeatureGroup]:
        """Load feature group from database"""
        
        session = self.SessionLocal()
        try:
            db_group = session.query(FeatureGroupModel).filter_by(group_id=group_id).first()
            if not db_group:
                return None
            
            return FeatureGroup(
                group_id=db_group.group_id,
                name=db_group.name,
                description=db_group.description,
                features=db_group.features_json or [],
                created_at=db_group.created_at,
                created_by=db_group.created_by,
                tags=db_group.tags_json or [],
                metadata=db_group.metadata_json or {}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load feature group {group_id}: {str(e)}")
            return None
        finally:
            session.close()
    
    def _query_features(
        self,
        feature_group: Optional[str] = None,
        status: Optional[FeatureStatus] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[FeatureMetadata]:
        """Query features with filters"""
        
        session = self.SessionLocal()
        try:
            query = session.query(FeatureMetadataModel)
            
            if feature_group:
                query = query.filter_by(feature_group=feature_group)
            
            if status:
                query = query.filter_by(status=status.value)
            
            if limit:
                query = query.limit(limit)
            
            db_results = query.all()
            
            # Convert to FeatureMetadata objects
            results = []
            for db_metadata in db_results:
                metadata = self._load_metadata(db_metadata.feature_id)
                if metadata:
                    # Filter by tags if specified
                    if tags and not any(tag in metadata.tags for tag in tags):
                        continue
                    results.append(metadata)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to query features: {str(e)}")
            return []
        finally:
            session.close()
    
    def _search_features(
        self,
        query: str,
        search_fields: List[str],
        limit: int
    ) -> List[FeatureMetadata]:
        """Search features by text query"""
        
        session = self.SessionLocal()
        try:
            # Simple text search implementation
            db_query = session.query(FeatureMetadataModel)
            
            conditions = []
            query_lower = query.lower()
            
            if 'name' in search_fields:
                conditions.append(FeatureMetadataModel.name.contains(query_lower))
            
            if 'description' in search_fields:
                conditions.append(FeatureMetadataModel.description.contains(query_lower))
            
            if 'business_definition' in search_fields:
                conditions.append(FeatureMetadataModel.business_definition.contains(query_lower))
            
            if conditions:
                from sqlalchemy import or_
                db_query = db_query.filter(or_(*conditions))
            
            db_query = db_query.limit(limit)
            db_results = db_query.all()
            
            # Convert to FeatureMetadata objects
            results = []
            for db_metadata in db_results:
                metadata = self._load_metadata(db_metadata.feature_id)
                if metadata:
                    results.append(metadata)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search features: {str(e)}")
            return []
        finally:
            session.close()
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export"""
        
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)
