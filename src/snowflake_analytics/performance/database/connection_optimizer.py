"""
Database connection optimizer for connection pooling and performance tuning.
"""

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import queue
import sqlite3
from contextlib import contextmanager


@dataclass
class ConnectionMetrics:
    """Connection performance metrics."""
    connection_id: str
    created_at: datetime
    last_used: datetime
    total_queries: int
    total_connection_time_ms: float
    avg_query_time_ms: float
    error_count: int
    is_active: bool


@dataclass
class PoolConfiguration:
    """Connection pool configuration."""
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout_seconds: int = 30
    idle_timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_connection_validation: bool = True
    validation_query: str = "SELECT 1"


@dataclass
class ConnectionOptimizationResult:
    """Connection optimization recommendation."""
    optimization_type: str
    current_value: Any
    recommended_value: Any
    description: str
    impact_level: str
    estimated_improvement: str


class ConnectionPool:
    """Simple database connection pool implementation."""
    
    def __init__(self, 
                 connection_factory: Callable,
                 config: PoolConfiguration):
        """
        Initialize connection pool.
        
        Args:
            connection_factory: Function that creates new connections
            config: Pool configuration
        """
        self.connection_factory = connection_factory
        self.config = config
        self._pool = queue.Queue(maxsize=config.max_connections)
        self._all_connections: Dict[str, Any] = {}
        self._connection_metrics: Dict[str, ConnectionMetrics] = {}
        self._lock = threading.Lock()
        
        # Statistics
        self.total_connections_created = 0
        self.total_connections_borrowed = 0
        self.total_connections_returned = 0
        self.connection_errors = 0
        
        # Initialize minimum connections
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize pool with minimum connections."""
        for _ in range(self.config.min_connections):
            try:
                conn = self._create_connection()
                self._pool.put(conn)
            except Exception as e:
                print(f"Error initializing connection pool: {e}")
    
    def _create_connection(self):
        """Create a new database connection."""
        conn_id = f"conn_{self.total_connections_created}_{int(time.time())}"
        connection = self.connection_factory()
        
        # Store connection reference
        self._all_connections[conn_id] = connection
        
        # Initialize metrics
        self._connection_metrics[conn_id] = ConnectionMetrics(
            connection_id=conn_id,
            created_at=datetime.now(),
            last_used=datetime.now(),
            total_queries=0,
            total_connection_time_ms=0,
            avg_query_time_ms=0,
            error_count=0,
            is_active=True
        )
        
        self.total_connections_created += 1
        setattr(connection, '_conn_id', conn_id)
        return connection
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool using context manager."""
        connection = None
        start_time = time.time()
        
        try:
            # Try to get existing connection
            try:
                connection = self._pool.get(timeout=self.config.connection_timeout_seconds)
            except queue.Empty:
                # Create new connection if pool is empty and under max limit
                with self._lock:
                    if len(self._all_connections) < self.config.max_connections:
                        connection = self._create_connection()
                    else:
                        raise Exception("Connection pool exhausted")
            
            # Update metrics
            conn_id = getattr(connection, '_conn_id', 'unknown')
            if conn_id in self._connection_metrics:
                self._connection_metrics[conn_id].last_used = datetime.now()
            
            self.total_connections_borrowed += 1
            
            # Validate connection if enabled
            if self.config.enable_connection_validation:
                self._validate_connection(connection)
            
            yield connection
            
        except Exception as e:
            self.connection_errors += 1
            if connection and hasattr(connection, '_conn_id'):
                conn_id = connection._conn_id
                if conn_id in self._connection_metrics:
                    self._connection_metrics[conn_id].error_count += 1
            raise
        
        finally:
            # Return connection to pool
            if connection:
                try:
                    connection_time = (time.time() - start_time) * 1000
                    self._update_connection_metrics(connection, connection_time)
                    self._pool.put(connection)
                    self.total_connections_returned += 1
                except Exception as e:
                    print(f"Error returning connection to pool: {e}")
    
    def _validate_connection(self, connection):
        """Validate connection is still usable."""
        try:
            if hasattr(connection, 'execute'):
                cursor = connection.cursor()
                cursor.execute(self.config.validation_query)
                cursor.fetchone()
                cursor.close()
        except Exception as e:
            raise Exception(f"Connection validation failed: {e}")
    
    def _update_connection_metrics(self, connection, connection_time_ms: float):
        """Update connection performance metrics."""
        conn_id = getattr(connection, '_conn_id', None)
        if conn_id and conn_id in self._connection_metrics:
            metrics = self._connection_metrics[conn_id]
            metrics.total_queries += 1
            metrics.total_connection_time_ms += connection_time_ms
            metrics.avg_query_time_ms = (
                metrics.total_connection_time_ms / metrics.total_queries
            )
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        active_connections = len(self._all_connections)
        available_connections = self._pool.qsize()
        
        return {
            'active_connections': active_connections,
            'available_connections': available_connections,
            'total_connections_created': self.total_connections_created,
            'total_connections_borrowed': self.total_connections_borrowed,
            'total_connections_returned': self.total_connections_returned,
            'connection_errors': self.connection_errors,
            'pool_utilization': (active_connections / self.config.max_connections) * 100,
            'error_rate': (self.connection_errors / max(self.total_connections_borrowed, 1)) * 100
        }
    
    def cleanup_idle_connections(self):
        """Remove idle connections from pool."""
        cutoff_time = datetime.now() - timedelta(seconds=self.config.idle_timeout_seconds)
        connections_to_close = []
        
        with self._lock:
            for conn_id, metrics in self._connection_metrics.items():
                if (metrics.last_used < cutoff_time and 
                    len(self._all_connections) > self.config.min_connections):
                    connections_to_close.append(conn_id)
        
        for conn_id in connections_to_close:
            try:
                connection = self._all_connections.pop(conn_id)
                self._connection_metrics.pop(conn_id)
                if hasattr(connection, 'close'):
                    connection.close()
            except Exception as e:
                print(f"Error closing idle connection {conn_id}: {e}")
    
    def close_all_connections(self):
        """Close all connections and cleanup pool."""
        with self._lock:
            for connection in self._all_connections.values():
                try:
                    if hasattr(connection, 'close'):
                        connection.close()
                except Exception as e:
                    print(f"Error closing connection: {e}")
            
            self._all_connections.clear()
            self._connection_metrics.clear()
            
            # Clear the pool queue
            while not self._pool.empty():
                try:
                    self._pool.get_nowait()
                except queue.Empty:
                    break


class ConnectionOptimizer:
    """
    Database connection optimizer for analyzing and optimizing connection patterns,
    pooling configurations, and connection lifecycle management.
    """
    
    def __init__(self):
        """Initialize connection optimizer."""
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self.optimization_history: List[ConnectionOptimizationResult] = []
        self._monitoring_data = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def register_connection_pool(self, 
                                pool_name: str, 
                                pool: ConnectionPool):
        """Register a connection pool for monitoring."""
        self.connection_pools[pool_name] = pool
    
    def analyze_connection_performance(self, 
                                     pool_name: str) -> Dict[str, Any]:
        """
        Analyze connection pool performance and identify bottlenecks.
        
        Args:
            pool_name: Name of the connection pool to analyze
            
        Returns:
            Performance analysis results
        """
        if pool_name not in self.connection_pools:
            return {'error': f'Pool {pool_name} not found'}
        
        pool = self.connection_pools[pool_name]
        stats = pool.get_pool_statistics()
        
        # Analyze connection metrics
        connection_analysis = self._analyze_connection_metrics(pool)
        
        # Identify performance issues
        performance_issues = self._identify_connection_issues(stats, connection_analysis)
        
        # Generate optimization recommendations
        recommendations = self._generate_connection_recommendations(
            stats, connection_analysis, performance_issues
        )
        
        return {
            'pool_name': pool_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'pool_statistics': stats,
            'connection_analysis': connection_analysis,
            'performance_issues': performance_issues,
            'optimization_recommendations': recommendations
        }
    
    def _analyze_connection_metrics(self, pool: ConnectionPool) -> Dict[str, Any]:
        """Analyze individual connection metrics."""
        metrics_list = list(pool._connection_metrics.values())
        
        if not metrics_list:
            return {'total_connections': 0}
        
        # Calculate aggregate metrics
        total_queries = sum(m.total_queries for m in metrics_list)
        total_errors = sum(m.error_count for m in metrics_list)
        
        query_times = [m.avg_query_time_ms for m in metrics_list if m.avg_query_time_ms > 0]
        avg_query_time = sum(query_times) / len(query_times) if query_times else 0
        
        # Connection age analysis
        now = datetime.now()
        connection_ages = [(now - m.created_at).total_seconds() for m in metrics_list]
        avg_connection_age = sum(connection_ages) / len(connection_ages)
        
        return {
            'total_connections': len(metrics_list),
            'total_queries': total_queries,
            'total_errors': total_errors,
            'error_rate': (total_errors / max(total_queries, 1)) * 100,
            'avg_query_time_ms': avg_query_time,
            'avg_connection_age_seconds': avg_connection_age,
            'oldest_connection_age_seconds': max(connection_ages) if connection_ages else 0,
            'connection_utilization_distribution': self._analyze_connection_utilization(metrics_list)
        }
    
    def _analyze_connection_utilization(self, metrics_list: List[ConnectionMetrics]) -> Dict[str, int]:
        """Analyze how connections are being utilized."""
        utilization_buckets = {
            'unused': 0,
            'light': 0,    # 1-10 queries
            'moderate': 0, # 11-100 queries  
            'heavy': 0     # 100+ queries
        }
        
        for metrics in metrics_list:
            if metrics.total_queries == 0:
                utilization_buckets['unused'] += 1
            elif metrics.total_queries <= 10:
                utilization_buckets['light'] += 1
            elif metrics.total_queries <= 100:
                utilization_buckets['moderate'] += 1
            else:
                utilization_buckets['heavy'] += 1
        
        return utilization_buckets
    
    def _identify_connection_issues(self, 
                                  stats: Dict[str, Any],
                                  analysis: Dict[str, Any]) -> List[str]:
        """Identify performance issues with connections."""
        issues = []
        
        # High error rate
        if stats.get('error_rate', 0) > 5:
            issues.append(f"High connection error rate: {stats['error_rate']:.1f}%")
        
        # Pool exhaustion
        if stats.get('pool_utilization', 0) > 90:
            issues.append(f"Pool utilization very high: {stats['pool_utilization']:.1f}%")
        
        # Slow average query time
        if analysis.get('avg_query_time_ms', 0) > 100:
            issues.append(f"Slow average query time: {analysis['avg_query_time_ms']:.1f}ms")
        
        # Old connections (potential memory leaks)
        if analysis.get('oldest_connection_age_seconds', 0) > 3600:  # 1 hour
            issues.append("Very old connections detected - potential memory leaks")
        
        # Uneven connection utilization
        utilization = analysis.get('connection_utilization_distribution', {})
        if utilization.get('unused', 0) > utilization.get('heavy', 0) * 2:
            issues.append("Uneven connection utilization - many unused connections")
        
        return issues
    
    def _generate_connection_recommendations(self,
                                           stats: Dict[str, Any],
                                           analysis: Dict[str, Any],
                                           issues: List[str]) -> List[ConnectionOptimizationResult]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Pool size optimization
        utilization = stats.get('pool_utilization', 0)
        if utilization > 85:
            recommendations.append(ConnectionOptimizationResult(
                optimization_type='pool_size',
                current_value=f"High utilization: {utilization:.1f}%",
                recommended_value="Increase max_connections by 25-50%",
                description="Pool is frequently exhausted, increase maximum connections",
                impact_level='high',
                estimated_improvement='30-50% reduction in connection wait times'
            ))
        elif utilization < 20:
            recommendations.append(ConnectionOptimizationResult(
                optimization_type='pool_size',
                current_value=f"Low utilization: {utilization:.1f}%",
                recommended_value="Decrease max_connections by 20-30%",
                description="Pool is over-provisioned, reduce maximum connections",
                impact_level='low',
                estimated_improvement='10-20% memory savings'
            ))
        
        # Connection timeout optimization
        error_rate = stats.get('error_rate', 0)
        if error_rate > 3:
            recommendations.append(ConnectionOptimizationResult(
                optimization_type='timeout',
                current_value=f"Error rate: {error_rate:.1f}%",
                recommended_value="Increase connection_timeout_seconds",
                description="High error rate may indicate connection timeouts",
                impact_level='medium',
                estimated_improvement='20-40% reduction in connection errors'
            ))
        
        # Query performance optimization
        avg_query_time = analysis.get('avg_query_time_ms', 0)
        if avg_query_time > 50:
            recommendations.append(ConnectionOptimizationResult(
                optimization_type='query_performance',
                current_value=f"Avg query time: {avg_query_time:.1f}ms",
                recommended_value="Review and optimize slow queries",
                description="Slow queries are holding connections longer than necessary",
                impact_level='high',
                estimated_improvement='40-70% improvement in connection throughput'
            ))
        
        # Connection lifecycle optimization
        oldest_connection = analysis.get('oldest_connection_age_seconds', 0)
        if oldest_connection > 1800:  # 30 minutes
            recommendations.append(ConnectionOptimizationResult(
                optimization_type='connection_lifecycle',
                current_value=f"Oldest connection: {oldest_connection/60:.1f} minutes",
                recommended_value="Implement connection recycling/refresh",
                description="Long-lived connections may accumulate resource issues",
                impact_level='medium',
                estimated_improvement='15-25% reduction in connection-related errors'
            ))
        
        return recommendations
    
    def optimize_pool_configuration(self, 
                                  pool_name: str,
                                  workload_pattern: str = 'balanced') -> PoolConfiguration:
        """
        Generate optimized pool configuration based on workload pattern.
        
        Args:
            pool_name: Name of the connection pool
            workload_pattern: 'light', 'balanced', 'heavy', 'burst'
            
        Returns:
            Optimized pool configuration
        """
        base_config = PoolConfiguration()
        
        # Adjust configuration based on workload pattern
        if workload_pattern == 'light':
            # Low concurrent usage
            base_config.min_connections = 2
            base_config.max_connections = 10
            base_config.idle_timeout_seconds = 180  # 3 minutes
            
        elif workload_pattern == 'balanced':
            # Moderate concurrent usage
            base_config.min_connections = 5
            base_config.max_connections = 20
            base_config.idle_timeout_seconds = 300  # 5 minutes
            
        elif workload_pattern == 'heavy':
            # High concurrent usage
            base_config.min_connections = 10
            base_config.max_connections = 50
            base_config.idle_timeout_seconds = 600  # 10 minutes
            base_config.connection_timeout_seconds = 60
            
        elif workload_pattern == 'burst':
            # Sporadic high usage
            base_config.min_connections = 3
            base_config.max_connections = 30
            base_config.idle_timeout_seconds = 120  # 2 minutes
            base_config.connection_timeout_seconds = 45
        
        # Apply any existing analysis-based optimizations
        if pool_name in self.connection_pools:
            analysis = self.analyze_connection_performance(pool_name)
            recommendations = analysis.get('optimization_recommendations', [])
            
            for rec in recommendations:
                if rec.optimization_type == 'pool_size':
                    if 'increase' in rec.recommended_value.lower():
                        base_config.max_connections = int(base_config.max_connections * 1.3)
                    elif 'decrease' in rec.recommended_value.lower():
                        base_config.max_connections = int(base_config.max_connections * 0.8)
                
                elif rec.optimization_type == 'timeout':
                    if 'increase' in rec.recommended_value.lower():
                        base_config.connection_timeout_seconds = int(base_config.connection_timeout_seconds * 1.5)
        
        return base_config
    
    def monitor_connection_health(self) -> Dict[str, Any]:
        """Monitor overall connection health across all pools."""
        health_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_pools': len(self.connection_pools),
            'pool_health': {},
            'overall_health_score': 0,
            'critical_issues': []
        }
        
        total_score = 0
        
        for pool_name, pool in self.connection_pools.items():
            analysis = self.analyze_connection_performance(pool_name)
            stats = analysis.get('pool_statistics', {})
            issues = analysis.get('performance_issues', [])
            
            # Calculate health score for this pool
            pool_score = 100
            pool_score -= len(issues) * 10  # -10 per issue
            pool_score -= min(stats.get('error_rate', 0) * 2, 30)  # Up to -30 for errors
            pool_score -= min(stats.get('pool_utilization', 0) - 80, 20) if stats.get('pool_utilization', 0) > 80 else 0
            
            pool_score = max(0, pool_score)  # Don't go below 0
            
            health_summary['pool_health'][pool_name] = {
                'health_score': pool_score,
                'error_rate': stats.get('error_rate', 0),
                'pool_utilization': stats.get('pool_utilization', 0),
                'issues': issues
            }
            
            # Track critical issues
            if pool_score < 50:
                health_summary['critical_issues'].append(f"Pool {pool_name} health critical: {pool_score}")
            
            total_score += pool_score
        
        # Calculate overall health score
        if self.connection_pools:
            health_summary['overall_health_score'] = total_score / len(self.connection_pools)
        
        return health_summary
    
    def generate_connection_report(self) -> Dict[str, Any]:
        """Generate comprehensive connection optimization report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'pools_analyzed': len(self.connection_pools),
            'health_summary': self.monitor_connection_health(),
            'pool_analyses': {},
            'top_recommendations': []
        }
        
        all_recommendations = []
        
        # Analyze each pool
        for pool_name in self.connection_pools.keys():
            analysis = self.analyze_connection_performance(pool_name)
            report['pool_analyses'][pool_name] = analysis
            
            recommendations = analysis.get('optimization_recommendations', [])
            for rec in recommendations:
                rec_dict = {
                    'pool': pool_name,
                    'type': rec.optimization_type,
                    'description': rec.description,
                    'impact_level': rec.impact_level,
                    'estimated_improvement': rec.estimated_improvement
                }
                all_recommendations.append(rec_dict)
        
        # Sort recommendations by impact
        impact_order = {'high': 3, 'medium': 2, 'low': 1}
        all_recommendations.sort(
            key=lambda x: impact_order.get(x['impact_level'], 0),
            reverse=True
        )
        
        report['top_recommendations'] = all_recommendations[:10]
        
        return report
    
    def export_connection_report(self, filepath: str) -> bool:
        """Export connection optimization report to file."""
        try:
            report = self.generate_connection_report()
            
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting connection report: {e}")
            return False