"""
Metrics Collection System for Snowflake Analytics
Comprehensive metrics collection, aggregation, and export for monitoring and observability.
"""

import os
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from collections import defaultdict, deque
from enum import Enum
import structlog
import psutil
import redis
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest

logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricsCollector:
    """
    Comprehensive metrics collection system.
    Collects application, system, and business metrics with Prometheus integration.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector."""
        self.registry = registry or CollectorRegistry()
        
        # Configuration
        self.collection_interval = int(os.getenv('METRICS_COLLECTION_INTERVAL', '10'))
        self.metrics_retention_hours = int(os.getenv('METRICS_RETENTION_HOURS', '24'))
        self.redis_enabled = os.getenv('REDIS_ENABLED', 'true').lower() == 'true'
        
        # Storage
        self.metrics_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.custom_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Redis connection for metrics storage
        self.redis_client = None
        if self.redis_enabled:
            try:
                self.redis_client = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'localhost'),
                    port=int(os.getenv('REDIS_PORT', '6379')),
                    db=int(os.getenv('REDIS_METRICS_DB', '1')),
                    decode_responses=True
                )
                self.redis_client.ping()
            except Exception as e:
                logger.warning("Redis not available for metrics storage", error=str(e))
                self.redis_client = None
        
        # Prometheus metrics
        self._init_prometheus_metrics()
        
        # Background collection
        self._collection_active = False
        self._collection_thread = None
        
        # Metric aggregators
        self.aggregators: Dict[str, Callable] = {
            'sum': sum,
            'avg': lambda x: sum(x) / len(x) if x else 0,
            'min': min,
            'max': max,
            'count': len,
            'last': lambda x: x[-1] if x else None
        }
        
        logger.info("MetricsCollector initialized", 
                   registry_metrics=len(self.registry._collector_to_names),
                   redis_enabled=self.redis_client is not None)
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # System metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_percent', 
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage', 
            ['path'],
            registry=self.registry
        )
        
        self.system_load_average = Gauge(
            'system_load_average',
            'System load average',
            ['period'],
            registry=self.registry
        )
        
        # Application metrics
        self.app_requests_total = Counter(
            'app_requests_total',
            'Total application requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.app_request_duration = Histogram(
            'app_request_duration_seconds',
            'Application request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.app_active_sessions = Gauge(
            'app_active_sessions',
            'Number of active user sessions',
            registry=self.registry
        )
        
        self.app_database_connections = Gauge(
            'app_database_connections',
            'Number of active database connections',
            ['database'],
            registry=self.registry
        )
        
        # Business metrics
        self.business_users_active = Gauge(
            'business_users_active',
            'Number of active users',
            ['period'],
            registry=self.registry
        )
        
        self.business_data_processed = Counter(
            'business_data_processed_bytes',
            'Total data processed',
            ['source', 'type'],
            registry=self.registry
        )
        
        self.business_queries_executed = Counter(
            'business_queries_executed_total',
            'Total queries executed',
            ['type', 'status'],
            registry=self.registry
        )
        
        # Analytics-specific metrics
        self.analytics_models_trained = Counter(
            'analytics_models_trained_total',
            'Total ML models trained',
            ['model_type'],
            registry=self.registry
        )
        
        self.analytics_predictions_made = Counter(
            'analytics_predictions_made_total',
            'Total predictions made',
            ['model_type', 'accuracy_tier'],
            registry=self.registry
        )
        
        self.analytics_data_quality_score = Gauge(
            'analytics_data_quality_score',
            'Data quality score (0-100)',
            ['dataset'],
            registry=self.registry
        )
    
    def start_collection(self):
        """Start background metrics collection."""
        if self._collection_active:
            logger.warning("Metrics collection already active")
            return
        
        self._collection_active = True
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collection_thread.start()
        
        logger.info("Metrics collection started", interval=self.collection_interval)
    
    def stop_collection(self):
        """Stop background metrics collection."""
        if not self._collection_active:
            return
        
        self._collection_active = False
        if self._collection_thread:
            self._collection_thread.join(timeout=10)
        
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Background metrics collection loop."""
        while self._collection_active:
            try:
                self.collect_system_metrics()
                self.collect_application_metrics()
                self.collect_business_metrics()
                self._cleanup_old_metrics()
            except Exception as e:
                logger.error("Error in metrics collection loop", error=str(e))
            
            time.sleep(self.collection_interval)
    
    def collect_system_metrics(self):
        """Collect system-level metrics."""
        timestamp = datetime.utcnow()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            self._store_metric('system.cpu.usage_percent', cpu_percent, timestamp)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.system_memory_usage.set(memory_percent)
            self._store_metric('system.memory.usage_percent', memory_percent, timestamp)
            self._store_metric('system.memory.total_bytes', memory.total, timestamp)
            self._store_metric('system.memory.available_bytes', memory.available, timestamp)
            
            # Disk usage
            app_root = os.getenv('APP_ROOT', '/opt/analytics')
            disk_usage = psutil.disk_usage(app_root)
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            self.system_disk_usage.labels(path=app_root).set(disk_percent)
            self._store_metric('system.disk.usage_percent', disk_percent, timestamp)
            self._store_metric('system.disk.total_bytes', disk_usage.total, timestamp)
            self._store_metric('system.disk.free_bytes', disk_usage.free, timestamp)
            
            # Load average
            load_avg = psutil.getloadavg()
            self.system_load_average.labels(period='1min').set(load_avg[0])
            self.system_load_average.labels(period='5min').set(load_avg[1])
            self.system_load_average.labels(period='15min').set(load_avg[2])
            self._store_metric('system.load.1min', load_avg[0], timestamp)
            self._store_metric('system.load.5min', load_avg[1], timestamp)
            self._store_metric('system.load.15min', load_avg[2], timestamp)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self._store_metric('system.network.bytes_sent', net_io.bytes_sent, timestamp)
            self._store_metric('system.network.bytes_recv', net_io.bytes_recv, timestamp)
            self._store_metric('system.network.packets_sent', net_io.packets_sent, timestamp)
            self._store_metric('system.network.packets_recv', net_io.packets_recv, timestamp)
            
            # Process information
            process = psutil.Process()
            self._store_metric('process.memory.rss_bytes', process.memory_info().rss, timestamp)
            self._store_metric('process.memory.vms_bytes', process.memory_info().vms, timestamp)
            self._store_metric('process.cpu.percent', process.cpu_percent(), timestamp)
            self._store_metric('process.threads.count', process.num_threads(), timestamp)
            self._store_metric('process.connections.count', len(process.connections()), timestamp)
            
            logger.debug("System metrics collected", cpu=cpu_percent, memory=memory_percent, disk=disk_percent)
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
    
    def collect_application_metrics(self):
        """Collect application-specific metrics."""
        timestamp = datetime.utcnow()
        
        try:
            # Database connections
            self._collect_database_metrics(timestamp)
            
            # Cache metrics
            self._collect_cache_metrics(timestamp)
            
            # Queue metrics
            self._collect_queue_metrics(timestamp)
            
            # Session metrics
            self._collect_session_metrics(timestamp)
            
        except Exception as e:
            logger.error("Failed to collect application metrics", error=str(e))
    
    def collect_business_metrics(self):
        """Collect business-specific metrics."""
        timestamp = datetime.utcnow()
        
        try:
            # Data processing metrics
            self._collect_data_processing_metrics(timestamp)
            
            # Analytics metrics
            self._collect_analytics_metrics(timestamp)
            
            # User activity metrics
            self._collect_user_activity_metrics(timestamp)
            
        except Exception as e:
            logger.error("Failed to collect business metrics", error=str(e))
    
    def _collect_database_metrics(self, timestamp: datetime):
        """Collect database-related metrics."""
        try:
            # PostgreSQL metrics
            pg_connections = self._get_postgresql_connections()
            if pg_connections is not None:
                self.app_database_connections.labels(database='postgresql').set(pg_connections)
                self._store_metric('database.postgresql.connections', pg_connections, timestamp)
            
            # Snowflake metrics
            sf_connections = self._get_snowflake_connections()
            if sf_connections is not None:
                self.app_database_connections.labels(database='snowflake').set(sf_connections)
                self._store_metric('database.snowflake.connections', sf_connections, timestamp)
            
        except Exception as e:
            logger.error("Failed to collect database metrics", error=str(e))
    
    def _collect_cache_metrics(self, timestamp: datetime):
        """Collect cache-related metrics."""
        try:
            if self.redis_client:
                info = self.redis_client.info()
                self._store_metric('cache.redis.used_memory', info.get('used_memory', 0), timestamp)
                self._store_metric('cache.redis.connected_clients', info.get('connected_clients', 0), timestamp)
                self._store_metric('cache.redis.total_commands_processed', info.get('total_commands_processed', 0), timestamp)
                self._store_metric('cache.redis.keyspace_hits', info.get('keyspace_hits', 0), timestamp)
                self._store_metric('cache.redis.keyspace_misses', info.get('keyspace_misses', 0), timestamp)
                
                # Calculate hit rate
                hits = info.get('keyspace_hits', 0)
                misses = info.get('keyspace_misses', 0)
                hit_rate = (hits / (hits + misses)) * 100 if (hits + misses) > 0 else 0
                self._store_metric('cache.redis.hit_rate_percent', hit_rate, timestamp)
            
        except Exception as e:
            logger.error("Failed to collect cache metrics", error=str(e))
    
    def _collect_queue_metrics(self, timestamp: datetime):
        """Collect queue-related metrics."""
        try:
            # This would integrate with actual queue system (Celery, RQ, etc.)
            # For now, using mock data
            queue_size = 5  # Mock queue size
            self._store_metric('queue.tasks.pending', queue_size, timestamp)
            self._store_metric('queue.tasks.processed_total', 1000, timestamp)
            self._store_metric('queue.workers.active', 4, timestamp)
            
        except Exception as e:
            logger.error("Failed to collect queue metrics", error=str(e))
    
    def _collect_session_metrics(self, timestamp: datetime):
        """Collect session-related metrics."""
        try:
            # This would integrate with actual session store
            # For now, using mock data
            active_sessions = 25  # Mock active sessions
            self.app_active_sessions.set(active_sessions)
            self._store_metric('sessions.active', active_sessions, timestamp)
            
        except Exception as e:
            logger.error("Failed to collect session metrics", error=str(e))
    
    def _collect_data_processing_metrics(self, timestamp: datetime):
        """Collect data processing metrics."""
        try:
            # This would integrate with actual data processing pipeline
            # For now, using mock data
            self._store_metric('data.snowflake.rows_processed', 10000, timestamp)
            self._store_metric('data.snowflake.bytes_processed', 1024 * 1024 * 100, timestamp)  # 100MB
            self._store_metric('data.collection.runs_successful', 95, timestamp)
            self._store_metric('data.collection.runs_failed', 5, timestamp)
            
        except Exception as e:
            logger.error("Failed to collect data processing metrics", error=str(e))
    
    def _collect_analytics_metrics(self, timestamp: datetime):
        """Collect analytics-specific metrics."""
        try:
            # This would integrate with actual analytics pipeline
            # For now, using mock data
            self._store_metric('analytics.models.total', 15, timestamp)
            self._store_metric('analytics.predictions.daily', 500, timestamp)
            self._store_metric('analytics.accuracy.average_percent', 87.5, timestamp)
            
            # Data quality metrics
            self.analytics_data_quality_score.labels(dataset='snowflake_usage').set(92.5)
            self.analytics_data_quality_score.labels(dataset='cost_data').set(88.0)
            self._store_metric('analytics.data_quality.snowflake_usage', 92.5, timestamp)
            self._store_metric('analytics.data_quality.cost_data', 88.0, timestamp)
            
        except Exception as e:
            logger.error("Failed to collect analytics metrics", error=str(e))
    
    def _collect_user_activity_metrics(self, timestamp: datetime):
        """Collect user activity metrics."""
        try:
            # This would integrate with actual user activity tracking
            # For now, using mock data
            self.business_users_active.labels(period='daily').set(50)
            self.business_users_active.labels(period='weekly').set(120)
            self.business_users_active.labels(period='monthly').set(300)
            
            self._store_metric('users.active.daily', 50, timestamp)
            self._store_metric('users.active.weekly', 120, timestamp)
            self._store_metric('users.active.monthly', 300, timestamp)
            self._store_metric('users.total', 500, timestamp)
            
        except Exception as e:
            logger.error("Failed to collect user activity metrics", error=str(e))
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record an API request metric."""
        try:
            # Prometheus metrics
            status_class = f"{status_code // 100}xx"
            self.app_requests_total.labels(method=method, endpoint=endpoint, status=status_class).inc()
            self.app_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
            
            # Custom storage
            timestamp = datetime.utcnow()
            self._store_metric(f'requests.{method.lower()}.{endpoint}.count', 1, timestamp)
            self._store_metric(f'requests.{method.lower()}.{endpoint}.duration', duration, timestamp)
            
            logger.debug("Request metric recorded", method=method, endpoint=endpoint, status=status_code, duration=duration)
            
        except Exception as e:
            logger.error("Failed to record request metric", error=str(e))
    
    def record_business_event(self, event_type: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Record a business metric event."""
        try:
            timestamp = datetime.utcnow()
            metric_name = f'business.{event_type}'
            
            if labels:
                for key, label_value in labels.items():
                    self._store_metric(f'{metric_name}.{key}.{label_value}', value, timestamp)
            else:
                self._store_metric(metric_name, value, timestamp)
            
            logger.debug("Business metric recorded", event_type=event_type, value=value, labels=labels)
            
        except Exception as e:
            logger.error("Failed to record business metric", error=str(e))
    
    def record_custom_metric(self, name: str, value: Union[int, float], 
                           metric_type: MetricType = MetricType.GAUGE,
                           labels: Optional[Dict[str, str]] = None,
                           help_text: Optional[str] = None):
        """Record a custom metric."""
        try:
            timestamp = datetime.utcnow()
            
            # Store in custom metrics
            if name not in self.custom_metrics:
                self.custom_metrics[name] = {
                    'type': metric_type.value,
                    'help': help_text or f'Custom metric: {name}',
                    'labels': labels or {},
                    'created_at': timestamp
                }
            
            # Store the value
            self._store_metric(name, value, timestamp, labels)
            
            logger.debug("Custom metric recorded", name=name, value=value, type=metric_type.value)
            
        except Exception as e:
            logger.error("Failed to record custom metric", name=name, error=str(e))
    
    def _store_metric(self, name: str, value: Union[int, float], timestamp: datetime, 
                     labels: Optional[Dict[str, str]] = None):
        """Store metric data point."""
        metric_data = {
            'name': name,
            'value': value,
            'timestamp': timestamp.isoformat(),
            'labels': labels or {}
        }
        
        # Store in memory
        self.metrics_data[name].append(metric_data)
        
        # Store in Redis if available
        if self.redis_client:
            try:
                key = f"metrics:{name}"
                if labels:
                    label_str = ','.join([f"{k}={v}" for k, v in labels.items()])
                    key = f"{key}[{label_str}]"
                
                self.redis_client.zadd(
                    key,
                    {json.dumps(metric_data): timestamp.timestamp()}
                )
                
                # Set expiration
                self.redis_client.expire(key, self.metrics_retention_hours * 3600)
                
            except Exception as e:
                logger.warning("Failed to store metric in Redis", name=name, error=str(e))
    
    def get_metric_data(self, name: str, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       labels: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Get metric data points."""
        data = []
        
        # Try Redis first
        if self.redis_client:
            try:
                key = f"metrics:{name}"
                if labels:
                    label_str = ','.join([f"{k}={v}" for k, v in labels.items()])
                    key = f"{key}[{label_str}]"
                
                start_score = start_time.timestamp() if start_time else 0
                end_score = end_time.timestamp() if end_time else time.time()
                
                redis_data = self.redis_client.zrangebyscore(key, start_score, end_score)
                data = [json.loads(item) for item in redis_data]
                
            except Exception as e:
                logger.warning("Failed to get metric data from Redis", name=name, error=str(e))
        
        # Fallback to memory
        if not data and name in self.metrics_data:
            memory_data = list(self.metrics_data[name])
            
            for item in memory_data:
                item_time = datetime.fromisoformat(item['timestamp'])
                
                if start_time and item_time < start_time:
                    continue
                if end_time and item_time > end_time:
                    continue
                if labels and not all(item['labels'].get(k) == v for k, v in labels.items()):
                    continue
                
                data.append(item)
        
        return sorted(data, key=lambda x: x['timestamp'])
    
    def get_aggregated_metric(self, name: str, aggregator: str = 'avg',
                            period_minutes: int = 60,
                            labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get aggregated metric value."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=period_minutes)
        
        data = self.get_metric_data(name, start_time, end_time, labels)
        values = [item['value'] for item in data]
        
        if not values:
            return None
        
        if aggregator in self.aggregators:
            return self.aggregators[aggregator](values)
        else:
            logger.warning("Unknown aggregator", aggregator=aggregator)
            return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        timestamp = datetime.utcnow()
        
        # Get recent system metrics
        cpu_usage = self.get_aggregated_metric('system.cpu.usage_percent', 'avg', 5)
        memory_usage = self.get_aggregated_metric('system.memory.usage_percent', 'avg', 5)
        disk_usage = self.get_aggregated_metric('system.disk.usage_percent', 'last', 5)
        
        # Get application metrics
        active_sessions = self.get_aggregated_metric('sessions.active', 'last', 5)
        
        # Calculate request rate
        request_count = len(self.get_metric_data('requests.get./health.count', 
                                               timestamp - timedelta(minutes=5), timestamp))
        request_rate = request_count / 5.0 if request_count else 0  # requests per minute
        
        return {
            'timestamp': timestamp.isoformat(),
            'system': {
                'cpu_usage_percent': cpu_usage,
                'memory_usage_percent': memory_usage,
                'disk_usage_percent': disk_usage
            },
            'application': {
                'active_sessions': active_sessions,
                'request_rate_per_minute': request_rate
            },
            'collection': {
                'active': self._collection_active,
                'interval_seconds': self.collection_interval,
                'stored_metrics': len(self.metrics_data),
                'redis_enabled': self.redis_client is not None
            }
        }
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry)
    
    def _cleanup_old_metrics(self):
        """Clean up old metric data."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.metrics_retention_hours)
        
        # Clean up memory storage
        for name, data in self.metrics_data.items():
            while data and datetime.fromisoformat(data[0]['timestamp']) < cutoff_time:
                data.popleft()
        
        # Redis cleanup is handled by expiration
    
    def _get_postgresql_connections(self) -> Optional[int]:
        """Get PostgreSQL connection count."""
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=int(os.getenv('DB_PORT', '5432')),
                dbname=os.getenv('DB_NAME', 'analytics'),
                user=os.getenv('DB_USER', 'analytics'),
                password=os.getenv('DB_PASSWORD', ''),
                connect_timeout=5
            )
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
            result = cur.fetchone()[0]
            cur.close()
            conn.close()
            return result
        except Exception:
            return None
    
    def _get_snowflake_connections(self) -> Optional[int]:
        """Get Snowflake connection count."""
        try:
            # This would query Snowflake for active sessions
            # For now, return a mock value
            return 3
        except Exception:
            return None