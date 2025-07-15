"""
Feature Registry

Comprehensive feature registry for managing feature definitions, metadata,
and discovery across the feature store ecosystem.

Key capabilities:
- Feature registration and discovery
- Feature group management
- Schema validation and evolution
- Feature dependencies tracking
- Access control and permissions
- Feature lifecycle management
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from dataclasses import dataclass, asdict

import pandas as pd

from .feature_metadata import (
    FeatureMetadata, FeatureMetadataManager, FeatureType, FeatureStatus
)


@dataclass
class FeatureRegistryConfig:
    """Feature registry configuration"""
    storage_path: str = "data/feature_registry"
    enable_versioning: bool = True
    enable_access_control: bool = False
    auto_schema_evolution: bool = True
    max_versions_per_feature: int = 10


class FeatureRegistry:
    """
    Comprehensive feature registry for feature discovery and management.
    
    Provides centralized registration, discovery, and management of features
    across the feature store ecosystem.
    """
    
    def __init__(
        self,
        metadata_manager: FeatureMetadataManager,
        storage_path: Path,
        config: Optional[FeatureRegistryConfig] = None
    ):
        """Initialize feature registry"""
        
        self.metadata_manager = metadata_manager
        self.storage_path = storage_path
        self.config = config or FeatureRegistryConfig()
        
        # Setup registry storage
        self.registry_path = storage_path / "registry"
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches
        self._feature_cache = {}
        self._group_cache = {}
        self._dependency_cache = {}
        
        # Load existing registry
        self._load_registry()
    
    def register_feature(self, metadata: FeatureMetadata) -> bool:
        """Register feature in the registry"""
        
        try:
            # Validate feature metadata
            validation_result = self._validate_feature_metadata(metadata)
            if not validation_result['valid']:
                raise ValueError(f"Feature validation failed: {validation_result['errors']}")
            
            # Check for conflicts
            existing_feature = self.get_feature(metadata.feature_id)
            if existing_feature and not self.config.enable_versioning:
                raise ValueError(f"Feature already exists: {metadata.feature_id}")
            
            # Store in metadata manager
            stored_metadata = self.metadata_manager.create_feature_metadata(
                name=metadata.name,
                description=metadata.description,
                schema=metadata.schema,
                feature_group=metadata.feature_group,
                created_by=metadata.created_by,
                tags=metadata.tags,
                properties=metadata.properties
            )
            
            # Update caches
            self._feature_cache[stored_metadata.feature_id] = stored_metadata
            
            # Update group cache
            group_name = stored_metadata.feature_group
            if group_name not in self._group_cache:
                self._group_cache[group_name] = set()
            self._group_cache[group_name].add(stored_metadata.name)
            
            # Save registry state
            self._save_registry()
            
            return True
            
        except Exception as e:
            raise Exception(f"Failed to register feature {metadata.feature_id}: {str(e)}")
    
    def get_feature(self, feature_id: str) -> Optional[FeatureMetadata]:
        """Get feature by ID"""
        
        # Check cache first
        if feature_id in self._feature_cache:
            return self._feature_cache[feature_id]
        
        # Query metadata manager
        metadata = self.metadata_manager.get_feature_metadata(feature_id)
        if metadata:
            self._feature_cache[feature_id] = metadata
        
        return metadata
    
    def get_feature_by_name(
        self,
        name: str,
        feature_group: Optional[str] = None
    ) -> Optional[FeatureMetadata]:
        """Get feature by name and optional group"""
        
        # Search in cache first
        for feature_id, metadata in self._feature_cache.items():
            if metadata.name == name:
                if feature_group is None or metadata.feature_group == feature_group:
                    return metadata
        
        # Search in metadata manager
        features = self.metadata_manager.list_features()
        for metadata in features:
            if metadata.name == name:
                if feature_group is None or metadata.feature_group == feature_group:
                    self._feature_cache[metadata.feature_id] = metadata
                    return metadata
        
        return None
    
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
    
    def get_feature_groups(self) -> List[str]:
        """Get list of all feature groups"""
        
        if not self._group_cache:
            # Build cache from metadata
            features = self.list_features()
            for feature in features:
                group_name = feature.feature_group
                if group_name not in self._group_cache:
                    self._group_cache[group_name] = set()
                self._group_cache[group_name].add(feature.name)
        
        return list(self._group_cache.keys())
    
    def get_feature_group_features(self, feature_group: str) -> List[str]:
        """Get features in a specific group"""
        
        if feature_group in self._group_cache:
            return list(self._group_cache[feature_group])
        
        # Build from metadata
        features = self.list_features(feature_group=feature_group)
        feature_names = [f.name for f in features]
        
        # Update cache
        self._group_cache[feature_group] = set(feature_names)
        
        return feature_names
    
    def add_feature_dependency(
        self,
        feature_id: str,
        depends_on: str,
        dependency_type: str = "transformation"
    ) -> bool:
        """Add feature dependency relationship"""
        
        try:
            # Validate both features exist
            feature = self.get_feature(feature_id)
            dependency = self.get_feature(depends_on)
            
            if not feature:
                raise ValueError(f"Feature not found: {feature_id}")
            if not dependency:
                raise ValueError(f"Dependency feature not found: {depends_on}")
            
            # Check for circular dependencies
            if self._would_create_cycle(feature_id, depends_on):
                raise ValueError("Adding dependency would create circular reference")
            
            # Add to dependency cache
            if feature_id not in self._dependency_cache:
                self._dependency_cache[feature_id] = {}
            
            self._dependency_cache[feature_id][depends_on] = {
                'dependency_type': dependency_type,
                'created_at': datetime.now().isoformat()
            }
            
            # Save registry state
            self._save_registry()
            
            return True
            
        except Exception as e:
            raise Exception(f"Failed to add dependency: {str(e)}")
    
    def get_feature_dependencies(
        self,
        feature_id: str,
        include_transitive: bool = False
    ) -> List[Dict[str, Any]]:
        """Get feature dependencies"""
        
        dependencies = []
        
        # Direct dependencies
        if feature_id in self._dependency_cache:
            for dep_id, dep_info in self._dependency_cache[feature_id].items():
                dep_feature = self.get_feature(dep_id)
                if dep_feature:
                    dependencies.append({
                        'feature_id': dep_id,
                        'feature_name': dep_feature.name,
                        'dependency_type': dep_info['dependency_type'],
                        'created_at': dep_info['created_at'],
                        'is_direct': True
                    })
        
        # Transitive dependencies
        if include_transitive:
            visited = set([feature_id])
            self._get_transitive_dependencies(feature_id, dependencies, visited)
        
        return dependencies
    
    def get_feature_dependents(self, feature_id: str) -> List[Dict[str, Any]]:
        """Get features that depend on this feature"""
        
        dependents = []
        
        for candidate_id, deps in self._dependency_cache.items():
            if feature_id in deps:
                candidate_feature = self.get_feature(candidate_id)
                if candidate_feature:
                    dependents.append({
                        'feature_id': candidate_id,
                        'feature_name': candidate_feature.name,
                        'dependency_type': deps[feature_id]['dependency_type'],
                        'created_at': deps[feature_id]['created_at']
                    })
        
        return dependents
    
    def validate_feature_schema(
        self,
        feature_id: str,
        new_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate feature schema changes"""
        
        feature = self.get_feature(feature_id)
        if not feature:
            return {
                'valid': False,
                'errors': [f"Feature not found: {feature_id}"]
            }
        
        current_schema = asdict(feature.schema)
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'changes': []
        }
        
        # Check for breaking changes
        if current_schema.get('data_type') != new_schema.get('data_type'):
            validation_result['errors'].append(
                f"Data type change not allowed: {current_schema.get('data_type')} -> {new_schema.get('data_type')}"
            )
            validation_result['valid'] = False
        
        # Check for compatible changes
        if current_schema.get('description') != new_schema.get('description'):
            validation_result['changes'].append("Description updated")
        
        if current_schema.get('constraints') != new_schema.get('constraints'):
            validation_result['warnings'].append("Constraints changed")
            validation_result['changes'].append("Constraints updated")
        
        # Validate new schema structure
        required_fields = ['feature_type', 'data_type']
        for field in required_fields:
            if field not in new_schema:
                validation_result['errors'].append(f"Missing required field: {field}")
                validation_result['valid'] = False
        
        return validation_result
    
    def update_feature_schema(
        self,
        feature_id: str,
        new_schema: Dict[str, Any],
        force: bool = False
    ) -> bool:
        """Update feature schema with validation"""
        
        if not force:
            validation_result = self.validate_feature_schema(feature_id, new_schema)
            if not validation_result['valid']:
                raise ValueError(f"Schema validation failed: {validation_result['errors']}")
        
        # Update feature metadata
        feature = self.get_feature(feature_id)
        if not feature:
            raise ValueError(f"Feature not found: {feature_id}")
        
        # Create new version if versioning is enabled
        if self.config.enable_versioning:
            # This would create a new version in the metadata manager
            pass
        
        # Update schema
        # This would update the schema in the metadata manager
        # For now, we'll update the cache
        updated_schema = feature.schema
        for key, value in new_schema.items():
            setattr(updated_schema, key, value)
        
        # Update cache
        self._feature_cache[feature_id] = feature
        
        # Save registry state
        self._save_registry()
        
        return True
    
    def retire_feature(
        self,
        feature_id: str,
        reason: str,
        retired_by: str
    ) -> bool:
        """Retire a feature"""
        
        feature = self.get_feature(feature_id)
        if not feature:
            raise ValueError(f"Feature not found: {feature_id}")
        
        # Check for dependencies
        dependents = self.get_feature_dependents(feature_id)
        if dependents:
            active_dependents = [
                d for d in dependents 
                if self.get_feature(d['feature_id']).status == FeatureStatus.ACTIVE
            ]
            if active_dependents:
                raise ValueError(
                    f"Cannot retire feature with active dependents: "
                    f"{[d['feature_name'] for d in active_dependents]}"
                )
        
        # Update status
        feature.status = FeatureStatus.DEPRECATED
        feature.updated_at = datetime.now()
        feature.properties = feature.properties or {}
        feature.properties.update({
            'retirement_reason': reason,
            'retired_by': retired_by,
            'retired_at': datetime.now().isoformat()
        })
        
        # Update cache
        self._feature_cache[feature_id] = feature
        
        # Save registry state
        self._save_registry()
        
        return True
    
    def get_feature_usage_stats(self, feature_id: str) -> Dict[str, Any]:
        """Get feature usage statistics"""
        
        feature = self.get_feature(feature_id)
        if not feature:
            return {}
        
        # This would typically come from the metadata manager
        # For now, return basic information
        return {
            'feature_id': feature_id,
            'feature_name': feature.name,
            'status': feature.status.value,
            'created_at': feature.created_at.isoformat(),
            'updated_at': feature.updated_at.isoformat(),
            'feature_group': feature.feature_group,
            'dependencies_count': len(self.get_feature_dependencies(feature_id)),
            'dependents_count': len(self.get_feature_dependents(feature_id))
        }
    
    def export_registry(
        self,
        output_path: str,
        include_dependencies: bool = True,
        format: str = 'json'
    ) -> str:
        """Export feature registry"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'registry_config': asdict(self.config),
            'features': [],
            'feature_groups': {},
            'dependencies': {} if include_dependencies else None
        }
        
        # Export all features
        all_features = self.list_features()
        for feature in all_features:
            feature_data = asdict(feature)
            # Convert datetime objects
            for key, value in feature_data.items():
                if isinstance(value, datetime):
                    feature_data[key] = value.isoformat()
            export_data['features'].append(feature_data)
        
        # Export feature groups
        for group in self.get_feature_groups():
            export_data['feature_groups'][group] = self.get_feature_group_features(group)
        
        # Export dependencies
        if include_dependencies:
            export_data['dependencies'] = self._dependency_cache
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        return str(output_path)
    
    def import_registry(
        self,
        import_path: str,
        merge_strategy: str = 'skip_existing'
    ) -> Dict[str, Any]:
        """Import feature registry from file"""
        
        import_path = Path(import_path)
        if not import_path.exists():
            raise ValueError(f"Import file not found: {import_path}")
        
        with open(import_path, 'r') as f:
            import_data = json.load(f)
        
        import_results = {
            'features_imported': 0,
            'features_skipped': 0,
            'features_failed': 0,
            'dependencies_imported': 0,
            'errors': []
        }
        
        # Import features
        for feature_data in import_data.get('features', []):
            try:
                feature_id = feature_data['feature_id']
                existing_feature = self.get_feature(feature_id)
                
                if existing_feature and merge_strategy == 'skip_existing':
                    import_results['features_skipped'] += 1
                    continue
                
                # Create FeatureMetadata object
                # This would require proper deserialization
                # For now, we'll track the import
                import_results['features_imported'] += 1
                
            except Exception as e:
                import_results['features_failed'] += 1
                import_results['errors'].append(f"Failed to import feature {feature_data.get('feature_id', 'unknown')}: {str(e)}")
        
        # Import dependencies
        if 'dependencies' in import_data and import_data['dependencies']:
            for feature_id, deps in import_data['dependencies'].items():
                for dep_id, dep_info in deps.items():
                    try:
                        self.add_feature_dependency(
                            feature_id=feature_id,
                            depends_on=dep_id,
                            dependency_type=dep_info.get('dependency_type', 'transformation')
                        )
                        import_results['dependencies_imported'] += 1
                    except Exception as e:
                        import_results['errors'].append(f"Failed to import dependency {feature_id}->{dep_id}: {str(e)}")
        
        return import_results
    
    def _validate_feature_metadata(self, metadata: FeatureMetadata) -> Dict[str, Any]:
        """Validate feature metadata"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Required fields
        if not metadata.name:
            validation_result['errors'].append("Feature name is required")
            validation_result['valid'] = False
        
        if not metadata.description:
            validation_result['warnings'].append("Feature description is empty")
        
        if not metadata.feature_group:
            validation_result['errors'].append("Feature group is required")
            validation_result['valid'] = False
        
        if not metadata.schema:
            validation_result['errors'].append("Feature schema is required")
            validation_result['valid'] = False
        
        # Name validation
        if metadata.name and not metadata.name.replace('_', '').isalnum():
            validation_result['errors'].append("Feature name must be alphanumeric (underscores allowed)")
            validation_result['valid'] = False
        
        return validation_result
    
    def _would_create_cycle(self, feature_id: str, depends_on: str) -> bool:
        """Check if adding dependency would create circular reference"""
        
        visited = set()
        
        def has_path(current: str, target: str) -> bool:
            if current == target:
                return True
            
            if current in visited:
                return False
            
            visited.add(current)
            
            # Check if current has dependencies that lead to target
            if current in self._dependency_cache:
                for dep_id in self._dependency_cache[current].keys():
                    if has_path(dep_id, target):
                        return True
            
            return False
        
        # Check if depends_on has a path to feature_id
        return has_path(depends_on, feature_id)
    
    def _get_transitive_dependencies(
        self,
        feature_id: str,
        dependencies: List[Dict[str, Any]],
        visited: Set[str]
    ):
        """Get transitive dependencies recursively"""
        
        if feature_id in self._dependency_cache:
            for dep_id, dep_info in self._dependency_cache[feature_id].items():
                if dep_id not in visited:
                    visited.add(dep_id)
                    
                    dep_feature = self.get_feature(dep_id)
                    if dep_feature:
                        dependencies.append({
                            'feature_id': dep_id,
                            'feature_name': dep_feature.name,
                            'dependency_type': dep_info['dependency_type'],
                            'created_at': dep_info['created_at'],
                            'is_direct': False
                        })
                    
                    # Recurse for transitive dependencies
                    self._get_transitive_dependencies(dep_id, dependencies, visited)
    
    def _load_registry(self):
        """Load registry state from storage"""
        
        # Load dependency cache
        deps_file = self.registry_path / "dependencies.json"
        if deps_file.exists():
            with open(deps_file, 'r') as f:
                self._dependency_cache = json.load(f)
        
        # Load group cache
        groups_file = self.registry_path / "groups.json"
        if groups_file.exists():
            with open(groups_file, 'r') as f:
                group_data = json.load(f)
                self._group_cache = {k: set(v) for k, v in group_data.items()}
    
    def _save_registry(self):
        """Save registry state to storage"""
        
        # Save dependency cache
        deps_file = self.registry_path / "dependencies.json"
        with open(deps_file, 'w') as f:
            json.dump(self._dependency_cache, f, indent=2, default=str)
        
        # Save group cache
        groups_file = self.registry_path / "groups.json"
        group_data = {k: list(v) for k, v in self._group_cache.items()}
        with open(groups_file, 'w') as f:
            json.dump(group_data, f, indent=2, default=str)
