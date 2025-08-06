"""
System resource profiling for CPU, memory, I/O, and network performance monitoring.
"""

import time
import psutil
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


@dataclass
class SystemMetrics:
    """System performance metrics data structure."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_io_read_bytes: int
    disk_io_write_bytes: int
    network_bytes_sent: int
    network_bytes_recv: int
    load_average: Optional[List[float]] = None
    process_count: int = 0


class SystemProfiler:
    """
    Comprehensive system resource profiler for monitoring CPU, memory, I/O, and network.
    
    Provides real-time monitoring and historical data collection for performance analysis.
    """
    
    def __init__(self, 
                 collection_interval: float = 1.0,
                 history_size: int = 3600):  # 1 hour at 1-second intervals
        """
        Initialize system profiler.
        
        Args:
            collection_interval: Seconds between metric collections
            history_size: Maximum number of metrics to store in history
        """
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.metrics_history: List[SystemMetrics] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Initialize baseline counters
        self._baseline_disk_io = psutil.disk_io_counters()
        self._baseline_network = psutil.net_io_counters()
    
    def get_current_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        current_time = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory metrics  
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available = memory.available
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        disk_read_bytes = disk_io.read_bytes if disk_io else 0
        disk_write_bytes = disk_io.write_bytes if disk_io else 0
        
        # Network metrics
        network = psutil.net_io_counters()
        network_sent = network.bytes_sent if network else 0
        network_recv = network.bytes_recv if network else 0
        
        # Load average (Unix-like systems)
        load_avg = None
        try:
            load_avg = list(psutil.getloadavg())
        except (AttributeError, OSError):
            pass  # Not available on all platforms
        
        # Process count
        process_count = len(psutil.pids())
        
        return SystemMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available=memory_available,
            disk_io_read_bytes=disk_read_bytes,
            disk_io_write_bytes=disk_write_bytes,
            network_bytes_sent=network_sent,
            network_bytes_recv=network_recv,
            load_average=load_avg,
            process_count=process_count
        )
    
    def start_monitoring(self) -> bool:
        """Start continuous system monitoring."""
        if self.is_monitoring:
            return False
        
        self.is_monitoring = True
        self._stop_event.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        return True
    
    def stop_monitoring(self) -> bool:
        """Stop continuous system monitoring."""
        if not self.is_monitoring:
            return False
        
        self.is_monitoring = False
        self._stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        return True
    
    def _monitoring_loop(self):
        """Internal monitoring loop."""
        while not self._stop_event.is_set():
            try:
                metrics = self.get_current_metrics()
                self._add_to_history(metrics)
                time.sleep(self.collection_interval)
            except Exception as e:
                # Log error but continue monitoring
                print(f"Error collecting system metrics: {e}")
                time.sleep(self.collection_interval)
    
    def _add_to_history(self, metrics: SystemMetrics):
        """Add metrics to history with size limit."""
        self.metrics_history.append(metrics)
        
        # Maintain history size limit
        if len(self.metrics_history) > self.history_size:
            self.metrics_history = self.metrics_history[-self.history_size:]
    
    def get_metrics_summary(self, 
                           duration_minutes: int = 10) -> Dict[str, Any]:
        """
        Get performance metrics summary for specified duration.
        
        Args:
            duration_minutes: Number of minutes to analyze
            
        Returns:
            Dictionary with min, max, avg metrics
        """
        if not self.metrics_history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        return {
            'duration_minutes': duration_minutes,
            'sample_count': len(recent_metrics),
            'cpu': {
                'min': min(cpu_values),
                'max': max(cpu_values),
                'avg': sum(cpu_values) / len(cpu_values)
            },
            'memory': {
                'min': min(memory_values),
                'max': max(memory_values), 
                'avg': sum(memory_values) / len(memory_values)
            },
            'disk_io': {
                'total_read_gb': recent_metrics[-1].disk_io_read_bytes / (1024**3),
                'total_write_gb': recent_metrics[-1].disk_io_write_bytes / (1024**3)
            },
            'network': {
                'total_sent_gb': recent_metrics[-1].network_bytes_sent / (1024**3),
                'total_recv_gb': recent_metrics[-1].network_bytes_recv / (1024**3)
            }
        }
    
    def detect_resource_pressure(self) -> Dict[str, Any]:
        """
        Detect current resource pressure and potential bottlenecks.
        
        Returns:
            Dictionary with pressure indicators and recommendations
        """
        current = self.get_current_metrics()
        pressure_indicators = {}
        
        # CPU pressure
        if current.cpu_percent > 80:
            pressure_indicators['cpu'] = {
                'level': 'high',
                'value': current.cpu_percent,
                'recommendation': 'Consider CPU optimization or scaling'
            }
        elif current.cpu_percent > 60:
            pressure_indicators['cpu'] = {
                'level': 'medium',
                'value': current.cpu_percent,
                'recommendation': 'Monitor CPU usage trends'
            }
        
        # Memory pressure
        if current.memory_percent > 85:
            pressure_indicators['memory'] = {
                'level': 'high',
                'value': current.memory_percent,
                'recommendation': 'Optimize memory usage or add more RAM'
            }
        elif current.memory_percent > 70:
            pressure_indicators['memory'] = {
                'level': 'medium',
                'value': current.memory_percent,
                'recommendation': 'Monitor memory usage patterns'
            }
        
        # Load average pressure (if available)
        if current.load_average:
            cpu_count = psutil.cpu_count()
            if cpu_count and current.load_average[0] > cpu_count * 0.8:
                pressure_indicators['load'] = {
                    'level': 'high',
                    'value': current.load_average[0],
                    'cpu_count': cpu_count,
                    'recommendation': 'System under heavy load'
                }
        
        return {
            'timestamp': current.timestamp,
            'pressure_indicators': pressure_indicators,
            'overall_health': 'good' if not pressure_indicators else 'degraded'
        }
    
    def get_historical_data(self, hours: int = 1) -> List[SystemMetrics]:
        """Get historical metrics for specified number of hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]
    
    def export_metrics_csv(self, filepath: str, hours: int = 24) -> bool:
        """Export metrics to CSV file."""
        try:
            import csv
            
            data = self.get_historical_data(hours)
            if not data:
                return False
            
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'cpu_percent', 'memory_percent', 
                    'memory_available', 'disk_io_read_bytes', 
                    'disk_io_write_bytes', 'network_bytes_sent',
                    'network_bytes_recv', 'process_count'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for metrics in data:
                    writer.writerow({
                        'timestamp': metrics.timestamp.isoformat(),
                        'cpu_percent': metrics.cpu_percent,
                        'memory_percent': metrics.memory_percent,
                        'memory_available': metrics.memory_available,
                        'disk_io_read_bytes': metrics.disk_io_read_bytes,
                        'disk_io_write_bytes': metrics.disk_io_write_bytes,
                        'network_bytes_sent': metrics.network_bytes_sent,
                        'network_bytes_recv': metrics.network_bytes_recv,
                        'process_count': metrics.process_count
                    })
            
            return True
        except Exception as e:
            print(f"Error exporting metrics: {e}")
            return False