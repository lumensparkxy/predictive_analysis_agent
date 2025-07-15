"""
Feature Lineage Tracker

Comprehensive feature lineage tracking system for maintaining data provenance,
impact analysis, and audit trails across the feature store ecosystem.

Key capabilities:
- Data lineage tracking and visualization
- Impact analysis for changes
- Audit trails for compliance
- Dependency graph management
- Change propagation tracking
- Root cause analysis support
"""

import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

import pandas as pd
import networkx as nx


@dataclass
class LineageNode:
    """Represents a node in the lineage graph"""
    node_id: str
    node_type: str  # feature, dataset, transformation, pipeline
    name: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class LineageEdge:
    """Represents an edge in the lineage graph"""
    source_id: str
    target_id: str
    relationship_type: str  # derives_from, transforms, depends_on
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class LineageEvent:
    """Represents a lineage tracking event"""
    event_id: str
    event_type: str
    timestamp: datetime
    actor: str
    nodes: List[str]
    edges: List[Tuple[str, str]]
    metadata: Dict[str, Any]


class FeatureLineageTracker:
    """
    Comprehensive feature lineage tracking system.
    
    Tracks data flow, transformations, and dependencies to provide
    complete visibility into feature evolution and relationships.
    """
    
    def __init__(
        self,
        storage_path: Path,
        enable_visualization: bool = True,
        retention_days: int = 365
    ):
        """Initialize lineage tracker"""
        
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.enable_visualization = enable_visualization
        self.retention_days = retention_days
        
        # Lineage graph
        self.lineage_graph = nx.DiGraph()
        
        # Storage for nodes, edges, and events
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: Dict[Tuple[str, str], LineageEdge] = {}
        self.events: List[LineageEvent] = []
        
        # Load existing lineage data
        self._load_lineage_data()
    
    def track_feature_creation(
        self,
        feature_id: str,
        created_by: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Track feature creation event"""
        
        event_id = self._generate_event_id("feature_creation")
        
        # Create feature node
        node = LineageNode(
            node_id=feature_id,
            node_type="feature",
            name=metadata.get('name', feature_id),
            metadata=metadata,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Add to graph and storage
        self.nodes[feature_id] = node
        self.lineage_graph.add_node(feature_id, **asdict(node))
        
        # Create event
        event = LineageEvent(
            event_id=event_id,
            event_type="feature_creation",
            timestamp=datetime.now(),
            actor=created_by,
            nodes=[feature_id],
            edges=[],
            metadata=metadata
        )
        
        self.events.append(event)
        self._save_lineage_data()
        
        return event_id
    
    def track_feature_transformation(
        self,
        source_features: List[str],
        target_feature: str,
        transformation_id: str,
        transformation_metadata: Dict[str, Any],
        actor: str
    ) -> str:
        """Track feature transformation event"""
        
        event_id = self._generate_event_id("feature_transformation")
        
        # Create transformation node
        transformation_node = LineageNode(
            node_id=transformation_id,
            node_type="transformation",
            name=transformation_metadata.get('name', transformation_id),
            metadata=transformation_metadata,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.nodes[transformation_id] = transformation_node
        self.lineage_graph.add_node(transformation_id, **asdict(transformation_node))
        
        # Create edges from source features to transformation
        edges_created = []
        for source_id in source_features:
            edge_key = (source_id, transformation_id)
            edge = LineageEdge(
                source_id=source_id,
                target_id=transformation_id,
                relationship_type="feeds_into",
                metadata={'transformation_type': transformation_metadata.get('type', 'unknown')},
                created_at=datetime.now()
            )
            
            self.edges[edge_key] = edge
            self.lineage_graph.add_edge(source_id, transformation_id, **asdict(edge))
            edges_created.append(edge_key)
        
        # Create edge from transformation to target feature
        if target_feature:
            edge_key = (transformation_id, target_feature)
            edge = LineageEdge(
                source_id=transformation_id,
                target_id=target_feature,
                relationship_type="produces",
                metadata={'transformation_type': transformation_metadata.get('type', 'unknown')},
                created_at=datetime.now()
            )
            
            self.edges[edge_key] = edge
            self.lineage_graph.add_edge(transformation_id, target_feature, **asdict(edge))
            edges_created.append(edge_key)
        
        # Create event
        event = LineageEvent(
            event_id=event_id,
            event_type="feature_transformation",
            timestamp=datetime.now(),
            actor=actor,
            nodes=source_features + [transformation_id, target_feature],
            edges=edges_created,
            metadata=transformation_metadata
        )
        
        self.events.append(event)
        self._save_lineage_data()
        
        return event_id
    
    def track_feature_write(
        self,
        feature_group: str,
        record_count: int,
        timestamp: datetime,
        storage_path: str,
        actor: str = "system"
    ) -> str:
        """Track feature write event"""
        
        event_id = self._generate_event_id("feature_write")
        
        # Create or update dataset node
        dataset_id = f"dataset_{feature_group}"
        if dataset_id in self.nodes:
            # Update existing node
            node = self.nodes[dataset_id]
            node.updated_at = timestamp
            node.metadata.update({
                'last_write': timestamp.isoformat(),
                'record_count': record_count,
                'storage_path': storage_path
            })
        else:
            # Create new dataset node
            node = LineageNode(
                node_id=dataset_id,
                node_type="dataset",
                name=f"Dataset: {feature_group}",
                metadata={
                    'feature_group': feature_group,
                    'storage_path': storage_path,
                    'record_count': record_count,
                    'first_write': timestamp.isoformat(),
                    'last_write': timestamp.isoformat()
                },
                created_at=timestamp,
                updated_at=timestamp
            )
            
            self.nodes[dataset_id] = node
            self.lineage_graph.add_node(dataset_id, **asdict(node))
        
        # Create write event
        event = LineageEvent(
            event_id=event_id,
            event_type="feature_write",
            timestamp=timestamp,
            actor=actor,
            nodes=[dataset_id],
            edges=[],
            metadata={
                'feature_group': feature_group,
                'record_count': record_count,
                'storage_path': storage_path
            }
        )
        
        self.events.append(event)
        self._save_lineage_data()
        
        return event_id
    
    def track_feature_read(
        self,
        feature_group: str,
        features: List[str],
        record_count: int,
        timestamp: datetime,
        actor: str = "system"
    ) -> str:
        """Track feature read event"""
        
        event_id = self._generate_event_id("feature_read")
        
        # Create read event
        event = LineageEvent(
            event_id=event_id,
            event_type="feature_read",
            timestamp=timestamp,
            actor=actor,
            nodes=features,
            edges=[],
            metadata={
                'feature_group': feature_group,
                'features': features,
                'record_count': record_count
            }
        )
        
        self.events.append(event)
        
        # Update usage statistics for features
        for feature_id in features:
            if feature_id in self.nodes:
                node = self.nodes[feature_id]
                if 'usage_stats' not in node.metadata:
                    node.metadata['usage_stats'] = {'read_count': 0, 'last_read': None}
                
                node.metadata['usage_stats']['read_count'] += 1
                node.metadata['usage_stats']['last_read'] = timestamp.isoformat()
                node.updated_at = timestamp
        
        self._save_lineage_data()
        return event_id
    
    def track_pipeline_execution(
        self,
        pipeline_id: str,
        input_features: List[str],
        output_features: List[str],
        execution_metadata: Dict[str, Any],
        actor: str
    ) -> str:
        """Track pipeline execution event"""
        
        event_id = self._generate_event_id("pipeline_execution")
        
        # Create or update pipeline node
        if pipeline_id in self.nodes:
            node = self.nodes[pipeline_id]
            node.updated_at = datetime.now()
            node.metadata.update(execution_metadata)
        else:
            node = LineageNode(
                node_id=pipeline_id,
                node_type="pipeline",
                name=execution_metadata.get('name', pipeline_id),
                metadata=execution_metadata,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.nodes[pipeline_id] = node
            self.lineage_graph.add_node(pipeline_id, **asdict(node))
        
        # Create edges from input features to pipeline
        edges_created = []
        for input_feature in input_features:
            edge_key = (input_feature, pipeline_id)
            if edge_key not in self.edges:
                edge = LineageEdge(
                    source_id=input_feature,
                    target_id=pipeline_id,
                    relationship_type="consumed_by",
                    metadata={'execution_id': event_id},
                    created_at=datetime.now()
                )
                
                self.edges[edge_key] = edge
                self.lineage_graph.add_edge(input_feature, pipeline_id, **asdict(edge))
                edges_created.append(edge_key)
        
        # Create edges from pipeline to output features
        for output_feature in output_features:
            edge_key = (pipeline_id, output_feature)
            if edge_key not in self.edges:
                edge = LineageEdge(
                    source_id=pipeline_id,
                    target_id=output_feature,
                    relationship_type="produces",
                    metadata={'execution_id': event_id},
                    created_at=datetime.now()
                )
                
                self.edges[edge_key] = edge
                self.lineage_graph.add_edge(pipeline_id, output_feature, **asdict(edge))
                edges_created.append(edge_key)
        
        # Create event
        event = LineageEvent(
            event_id=event_id,
            event_type="pipeline_execution",
            timestamp=datetime.now(),
            actor=actor,
            nodes=[pipeline_id] + input_features + output_features,
            edges=edges_created,
            metadata=execution_metadata
        )
        
        self.events.append(event)
        self._save_lineage_data()
        
        return event_id
    
    def track_snapshot_creation(
        self,
        feature_group: str,
        snapshot_id: str,
        created_by: str
    ) -> str:
        """Track snapshot creation event"""
        
        event_id = self._generate_event_id("snapshot_creation")
        
        # Create snapshot node
        snapshot_node = LineageNode(
            node_id=snapshot_id,
            node_type="snapshot",
            name=f"Snapshot: {feature_group}",
            metadata={
                'feature_group': feature_group,
                'snapshot_type': 'point_in_time'
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.nodes[snapshot_id] = snapshot_node
        self.lineage_graph.add_node(snapshot_id, **asdict(snapshot_node))
        
        # Create edge from feature group dataset to snapshot
        dataset_id = f"dataset_{feature_group}"
        if dataset_id in self.nodes:
            edge_key = (dataset_id, snapshot_id)
            edge = LineageEdge(
                source_id=dataset_id,
                target_id=snapshot_id,
                relationship_type="snapshot_of",
                metadata={'snapshot_type': 'point_in_time'},
                created_at=datetime.now()
            )
            
            self.edges[edge_key] = edge
            self.lineage_graph.add_edge(dataset_id, snapshot_id, **asdict(edge))
        
        # Create event
        event = LineageEvent(
            event_id=event_id,
            event_type="snapshot_creation",
            timestamp=datetime.now(),
            actor=created_by,
            nodes=[snapshot_id],
            edges=[(dataset_id, snapshot_id)] if dataset_id in self.nodes else [],
            metadata={'feature_group': feature_group}
        )
        
        self.events.append(event)
        self._save_lineage_data()
        
        return event_id
    
    def get_feature_lineage(
        self,
        feature_id: str,
        direction: str = 'both',
        depth: int = 5
    ) -> Dict[str, Any]:
        """Get feature lineage information"""
        
        if feature_id not in self.lineage_graph:
            return {
                'feature_id': feature_id,
                'error': 'Feature not found in lineage graph'
            }
        
        lineage_data = {
            'feature_id': feature_id,
            'direction': direction,
            'depth': depth,
            'nodes': {},
            'edges': [],
            'paths': [],
            'statistics': {}
        }
        
        # Get nodes within specified depth
        if direction in ['upstream', 'both']:
            upstream_nodes = self._get_upstream_nodes(feature_id, depth)
            for node_id in upstream_nodes:
                if node_id in self.nodes:
                    lineage_data['nodes'][node_id] = asdict(self.nodes[node_id])
        
        if direction in ['downstream', 'both']:
            downstream_nodes = self._get_downstream_nodes(feature_id, depth)
            for node_id in downstream_nodes:
                if node_id in self.nodes:
                    lineage_data['nodes'][node_id] = asdict(self.nodes[node_id])
        
        # Add the feature itself
        if feature_id in self.nodes:
            lineage_data['nodes'][feature_id] = asdict(self.nodes[feature_id])
        
        # Get edges between collected nodes
        for (source, target), edge in self.edges.items():
            if source in lineage_data['nodes'] and target in lineage_data['nodes']:
                lineage_data['edges'].append(asdict(edge))
        
        # Calculate statistics
        lineage_data['statistics'] = {
            'total_nodes': len(lineage_data['nodes']),
            'total_edges': len(lineage_data['edges']),
            'node_types': {},
            'relationship_types': {}
        }
        
        # Count node types
        for node_data in lineage_data['nodes'].values():
            node_type = node_data['node_type']
            lineage_data['statistics']['node_types'][node_type] = \
                lineage_data['statistics']['node_types'].get(node_type, 0) + 1
        
        # Count relationship types
        for edge_data in lineage_data['edges']:
            rel_type = edge_data['relationship_type']
            lineage_data['statistics']['relationship_types'][rel_type] = \
                lineage_data['statistics']['relationship_types'].get(rel_type, 0) + 1
        
        return lineage_data
    
    def analyze_impact(
        self,
        feature_id: str,
        change_type: str = 'schema_change'
    ) -> Dict[str, Any]:
        """Analyze impact of changes to a feature"""
        
        impact_analysis = {
            'feature_id': feature_id,
            'change_type': change_type,
            'timestamp': datetime.now().isoformat(),
            'directly_affected': [],
            'indirectly_affected': [],
            'affected_pipelines': [],
            'affected_models': [],
            'risk_assessment': 'low'
        }
        
        if feature_id not in self.lineage_graph:
            impact_analysis['error'] = 'Feature not found in lineage graph'
            return impact_analysis
        
        # Find directly affected features (immediate downstream)
        direct_successors = list(self.lineage_graph.successors(feature_id))
        impact_analysis['directly_affected'] = direct_successors
        
        # Find indirectly affected features (transitive downstream)
        all_downstream = self._get_downstream_nodes(feature_id, depth=10)
        impact_analysis['indirectly_affected'] = [
            node for node in all_downstream if node not in direct_successors
        ]
        
        # Find affected pipelines and models
        for node_id in all_downstream:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if node.node_type == 'pipeline':
                    impact_analysis['affected_pipelines'].append(node_id)
                elif node.node_type in ['model', 'ml_model']:
                    impact_analysis['affected_models'].append(node_id)
        
        # Assess risk based on number of affected components
        total_affected = (
            len(impact_analysis['directly_affected']) +
            len(impact_analysis['indirectly_affected']) +
            len(impact_analysis['affected_pipelines']) +
            len(impact_analysis['affected_models'])
        )
        
        if total_affected == 0:
            impact_analysis['risk_assessment'] = 'none'
        elif total_affected <= 5:
            impact_analysis['risk_assessment'] = 'low'
        elif total_affected <= 20:
            impact_analysis['risk_assessment'] = 'medium'
        else:
            impact_analysis['risk_assessment'] = 'high'
        
        return impact_analysis
    
    def get_root_cause_analysis(
        self,
        feature_id: str,
        issue_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Perform root cause analysis for feature issues"""
        
        analysis = {
            'feature_id': feature_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'issue_timestamp': issue_timestamp.isoformat() if issue_timestamp else None,
            'potential_causes': [],
            'upstream_changes': [],
            'recent_events': []
        }
        
        if feature_id not in self.lineage_graph:
            analysis['error'] = 'Feature not found in lineage graph'
            return analysis
        
        # Find upstream features
        upstream_nodes = self._get_upstream_nodes(feature_id, depth=5)
        
        # Look for recent changes in upstream features
        cutoff_time = issue_timestamp or (datetime.now() - timedelta(hours=24))
        
        for node_id in upstream_nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if node.updated_at >= cutoff_time:
                    analysis['upstream_changes'].append({
                        'node_id': node_id,
                        'node_type': node.node_type,
                        'name': node.name,
                        'updated_at': node.updated_at.isoformat(),
                        'metadata': node.metadata
                    })
        
        # Find recent events affecting this feature
        for event in self.events:
            if (feature_id in event.nodes and 
                event.timestamp >= cutoff_time):
                analysis['recent_events'].append({
                    'event_id': event.event_id,
                    'event_type': event.event_type,
                    'timestamp': event.timestamp.isoformat(),
                    'actor': event.actor,
                    'metadata': event.metadata
                })
        
        # Generate potential causes based on findings
        if analysis['upstream_changes']:
            analysis['potential_causes'].append(
                f"Upstream data changes detected in {len(analysis['upstream_changes'])} features"
            )
        
        if analysis['recent_events']:
            event_types = set(event['event_type'] for event in analysis['recent_events'])
            analysis['potential_causes'].append(
                f"Recent operations: {', '.join(event_types)}"
            )
        
        if not analysis['potential_causes']:
            analysis['potential_causes'].append("No obvious upstream changes detected")
        
        return analysis
    
    def get_feature_evolution(
        self,
        feature_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get feature evolution timeline"""
        
        evolution = {
            'feature_id': feature_id,
            'start_time': start_time.isoformat() if start_time else None,
            'end_time': end_time.isoformat() if end_time else None,
            'timeline': [],
            'summary': {
                'total_events': 0,
                'event_types': {},
                'actors': set(),
                'major_changes': []
            }
        }
        
        # Filter events for this feature
        feature_events = [
            event for event in self.events
            if feature_id in event.nodes and
            (not start_time or event.timestamp >= start_time) and
            (not end_time or event.timestamp <= end_time)
        ]
        
        # Sort by timestamp
        feature_events.sort(key=lambda x: x.timestamp)
        
        # Build timeline
        for event in feature_events:
            timeline_entry = {
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type,
                'actor': event.actor,
                'description': self._generate_event_description(event),
                'metadata': event.metadata
            }
            evolution['timeline'].append(timeline_entry)
            
            # Update summary
            evolution['summary']['total_events'] += 1
            evolution['summary']['event_types'][event.event_type] = \
                evolution['summary']['event_types'].get(event.event_type, 0) + 1
            evolution['summary']['actors'].add(event.actor)
            
            # Identify major changes
            if event.event_type in ['feature_transformation', 'schema_change', 'pipeline_execution']:
                evolution['summary']['major_changes'].append({
                    'timestamp': event.timestamp.isoformat(),
                    'type': event.event_type,
                    'description': timeline_entry['description']
                })
        
        # Convert actors set to list for JSON serialization
        evolution['summary']['actors'] = list(evolution['summary']['actors'])
        
        return evolution
    
    def export_lineage_graph(
        self,
        output_path: str,
        format: str = 'json',
        include_metadata: bool = True
    ) -> str:
        """Export lineage graph to file"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'graph_statistics': {
                'nodes': len(self.nodes),
                'edges': len(self.edges),
                'events': len(self.events)
            },
            'nodes': {},
            'edges': [],
            'events': [] if include_metadata else None
        }
        
        # Export nodes
        for node_id, node in self.nodes.items():
            node_data = asdict(node)
            # Convert datetime objects
            node_data['created_at'] = node.created_at.isoformat()
            node_data['updated_at'] = node.updated_at.isoformat()
            export_data['nodes'][node_id] = node_data
        
        # Export edges
        for edge in self.edges.values():
            edge_data = asdict(edge)
            edge_data['created_at'] = edge.created_at.isoformat()
            export_data['edges'].append(edge_data)
        
        # Export events
        if include_metadata:
            for event in self.events:
                event_data = asdict(event)
                event_data['timestamp'] = event.timestamp.isoformat()
                export_data['events'].append(event_data)
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        elif format.lower() == 'graphml' and self.enable_visualization:
            nx.write_graphml(self.lineage_graph, output_path)
        
        return str(output_path)
    
    def cleanup_old_events(
        self,
        retention_days: Optional[int] = None,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Clean up old lineage events"""
        
        retention_days = retention_days or self.retention_days
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        cleanup_results = {
            'cutoff_date': cutoff_date.isoformat(),
            'retention_days': retention_days,
            'dry_run': dry_run,
            'events_to_delete': 0,
            'events_kept': 0
        }
        
        if not dry_run:
            old_events = [e for e in self.events if e.timestamp < cutoff_date]
            self.events = [e for e in self.events if e.timestamp >= cutoff_date]
            cleanup_results['events_to_delete'] = len(old_events)
            cleanup_results['events_kept'] = len(self.events)
            
            # Save updated events
            self._save_lineage_data()
        else:
            cleanup_results['events_to_delete'] = sum(
                1 for e in self.events if e.timestamp < cutoff_date
            )
            cleanup_results['events_kept'] = sum(
                1 for e in self.events if e.timestamp >= cutoff_date
            )
        
        return cleanup_results
    
    def _get_upstream_nodes(self, node_id: str, depth: int) -> Set[str]:
        """Get upstream nodes within specified depth"""
        
        upstream = set()
        
        def traverse_upstream(current_id: str, remaining_depth: int):
            if remaining_depth <= 0:
                return
            
            for predecessor in self.lineage_graph.predecessors(current_id):
                if predecessor not in upstream:
                    upstream.add(predecessor)
                    traverse_upstream(predecessor, remaining_depth - 1)
        
        traverse_upstream(node_id, depth)
        return upstream
    
    def _get_downstream_nodes(self, node_id: str, depth: int) -> Set[str]:
        """Get downstream nodes within specified depth"""
        
        downstream = set()
        
        def traverse_downstream(current_id: str, remaining_depth: int):
            if remaining_depth <= 0:
                return
            
            for successor in self.lineage_graph.successors(current_id):
                if successor not in downstream:
                    downstream.add(successor)
                    traverse_downstream(successor, remaining_depth - 1)
        
        traverse_downstream(node_id, depth)
        return downstream
    
    def _generate_event_id(self, event_type: str) -> str:
        """Generate unique event ID"""
        
        timestamp = int(time.time() * 1000)
        hash_input = f"{event_type}_{timestamp}_{len(self.events)}"
        event_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{event_type}_{timestamp}_{event_hash}"
    
    def _generate_event_description(self, event: LineageEvent) -> str:
        """Generate human-readable event description"""
        
        descriptions = {
            'feature_creation': f"Feature created by {event.actor}",
            'feature_transformation': f"Feature transformation executed by {event.actor}",
            'feature_write': f"Data written to feature group",
            'feature_read': f"Features read by {event.actor}",
            'pipeline_execution': f"Pipeline executed by {event.actor}",
            'snapshot_creation': f"Snapshot created by {event.actor}"
        }
        
        return descriptions.get(event.event_type, f"Event: {event.event_type}")
    
    def _load_lineage_data(self):
        """Load lineage data from storage"""
        
        # Load nodes
        nodes_file = self.storage_path / "nodes.json"
        if nodes_file.exists():
            with open(nodes_file, 'r') as f:
                nodes_data = json.load(f)
                
            for node_id, node_data in nodes_data.items():
                node_data['created_at'] = datetime.fromisoformat(node_data['created_at'])
                node_data['updated_at'] = datetime.fromisoformat(node_data['updated_at'])
                self.nodes[node_id] = LineageNode(**node_data)
                self.lineage_graph.add_node(node_id, **node_data)
        
        # Load edges
        edges_file = self.storage_path / "edges.json"
        if edges_file.exists():
            with open(edges_file, 'r') as f:
                edges_data = json.load(f)
                
            for edge_data in edges_data:
                edge_data['created_at'] = datetime.fromisoformat(edge_data['created_at'])
                edge = LineageEdge(**edge_data)
                edge_key = (edge.source_id, edge.target_id)
                self.edges[edge_key] = edge
                self.lineage_graph.add_edge(edge.source_id, edge.target_id, **asdict(edge))
        
        # Load events
        events_file = self.storage_path / "events.json"
        if events_file.exists():
            with open(events_file, 'r') as f:
                events_data = json.load(f)
                
            for event_data in events_data:
                event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
                self.events.append(LineageEvent(**event_data))
    
    def _save_lineage_data(self):
        """Save lineage data to storage"""
        
        # Save nodes
        nodes_file = self.storage_path / "nodes.json"
        nodes_data = {}
        for node_id, node in self.nodes.items():
            node_data = asdict(node)
            node_data['created_at'] = node.created_at.isoformat()
            node_data['updated_at'] = node.updated_at.isoformat()
            nodes_data[node_id] = node_data
        
        with open(nodes_file, 'w') as f:
            json.dump(nodes_data, f, indent=2, default=str)
        
        # Save edges
        edges_file = self.storage_path / "edges.json"
        edges_data = []
        for edge in self.edges.values():
            edge_data = asdict(edge)
            edge_data['created_at'] = edge.created_at.isoformat()
            edges_data.append(edge_data)
        
        with open(edges_file, 'w') as f:
            json.dump(edges_data, f, indent=2, default=str)
        
        # Save events
        events_file = self.storage_path / "events.json"
        events_data = []
        for event in self.events:
            event_data = asdict(event)
            event_data['timestamp'] = event.timestamp.isoformat()
            events_data.append(event_data)
        
        with open(events_file, 'w') as f:
            json.dump(events_data, f, indent=2, default=str)
