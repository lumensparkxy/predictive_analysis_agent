"""
Bottleneck analyzer for identifying system performance bottlenecks across all components.
"""

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

from system_profiler import SystemProfiler, SystemMetrics
from application_profiler import ApplicationProfiler, PerformanceStats
from database_profiler import DatabaseProfiler, QueryStats
from api_profiler import APIProfiler, EndpointStats


class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK = "network"
    DATABASE = "database"
    APPLICATION = "application"
    API = "api"


@dataclass
class BottleneckAlert:
    """Performance bottleneck alert."""
    type: BottleneckType
    severity: str  # 'low', 'medium', 'high', 'critical'
    component: str
    description: str
    current_value: float
    threshold_value: float
    impact_score: float
    detected_at: datetime
    recommendations: List[str]


@dataclass
class PerformanceBaseline:
    """Performance baseline measurements."""
    component: str
    metric_name: str
    baseline_value: float
    sample_count: int
    created_at: datetime
    last_updated: datetime


class BottleneckAnalyzer:
    """
    Comprehensive bottleneck analyzer that correlates performance data across
    system, application, database, and API components to identify root causes.
    """
    
    def __init__(self,
                 system_profiler: Optional[SystemProfiler] = None,
                 app_profiler: Optional[ApplicationProfiler] = None,
                 db_profiler: Optional[DatabaseProfiler] = None,
                 api_profiler: Optional[APIProfiler] = None):
        """
        Initialize bottleneck analyzer.
        
        Args:
            system_profiler: System resource profiler instance
            app_profiler: Application profiler instance
            db_profiler: Database profiler instance
            api_profiler: API profiler instance
        """
        self.system_profiler = system_profiler or SystemProfiler()
        self.app_profiler = app_profiler or ApplicationProfiler()
        self.db_profiler = db_profiler or DatabaseProfiler()
        self.api_profiler = api_profiler or APIProfiler()
        
        self.bottleneck_history: List[BottleneckAlert] = []
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self._lock = threading.Lock()
        
        # Analysis thresholds
        self.thresholds = {
            BottleneckType.CPU: {
                'medium': 70.0,
                'high': 85.0,
                'critical': 95.0
            },
            BottleneckType.MEMORY: {
                'medium': 75.0,
                'high': 85.0,
                'critical': 95.0
            },
            BottleneckType.DISK_IO: {
                'medium': 70.0,  # Percentage utilization
                'high': 85.0,
                'critical': 95.0
            },
            BottleneckType.DATABASE: {
                'medium': 1000.0,  # ms
                'high': 2000.0,
                'critical': 5000.0
            },
            BottleneckType.APPLICATION: {
                'medium': 500.0,  # ms
                'high': 1000.0,
                'critical': 2000.0
            },
            BottleneckType.API: {
                'medium': 500.0,  # ms
                'high': 1000.0,
                'critical': 2000.0
            }
        }
        
        # Baseline learning parameters
        self.baseline_learning_enabled = True
        self.baseline_min_samples = 100
        self.baseline_max_deviation = 2.0  # Standard deviations
    
    def analyze_current_performance(self) -> List[BottleneckAlert]:
        """
        Analyze current performance across all components and identify bottlenecks.
        
        Returns:
            List of detected bottleneck alerts
        """
        alerts = []
        current_time = datetime.now()
        
        # System resource analysis
        system_alerts = self._analyze_system_bottlenecks(current_time)
        alerts.extend(system_alerts)
        
        # Application performance analysis
        app_alerts = self._analyze_application_bottlenecks(current_time)
        alerts.extend(app_alerts)
        
        # Database performance analysis
        db_alerts = self._analyze_database_bottlenecks(current_time)
        alerts.extend(db_alerts)
        
        # API performance analysis
        api_alerts = self._analyze_api_bottlenecks(current_time)
        alerts.extend(api_alerts)
        
        # Store alerts in history
        with self._lock:
            self.bottleneck_history.extend(alerts)
            # Keep only recent alerts (last 7 days)
            cutoff_time = current_time - timedelta(days=7)
            self.bottleneck_history = [
                alert for alert in self.bottleneck_history
                if alert.detected_at >= cutoff_time
            ]
        
        return sorted(alerts, key=lambda x: x.impact_score, reverse=True)
    
    def _analyze_system_bottlenecks(self, current_time: datetime) -> List[BottleneckAlert]:
        """Analyze system resource bottlenecks."""
        alerts = []
        
        try:
            current_metrics = self.system_profiler.get_current_metrics()
            
            # CPU bottleneck analysis
            cpu_alert = self._check_threshold_alert(
                BottleneckType.CPU,
                "CPU Utilization",
                current_metrics.cpu_percent,
                current_time,
                "High CPU utilization detected",
                [
                    "Identify CPU-intensive processes",
                    "Consider process optimization or scaling",
                    "Review application algorithm efficiency"
                ]
            )
            if cpu_alert:
                alerts.append(cpu_alert)
            
            # Memory bottleneck analysis
            memory_alert = self._check_threshold_alert(
                BottleneckType.MEMORY,
                "Memory Utilization",
                current_metrics.memory_percent,
                current_time,
                "High memory utilization detected",
                [
                    "Identify memory-intensive processes",
                    "Look for memory leaks",
                    "Consider increasing available memory",
                    "Optimize data structures and caching"
                ]
            )
            if memory_alert:
                alerts.append(memory_alert)
            
        except Exception as e:
            print(f"Error analyzing system bottlenecks: {e}")
        
        return alerts
    
    def _analyze_application_bottlenecks(self, current_time: datetime) -> List[BottleneckAlert]:
        """Analyze application performance bottlenecks."""
        alerts = []
        
        try:
            slow_functions = self.app_profiler.get_top_slow_functions(10)
            anomalies = self.app_profiler.detect_performance_anomalies()
            
            # Analyze slow functions
            for func_stats in slow_functions:
                if func_stats.avg_duration_ms > self.thresholds[BottleneckType.APPLICATION]['medium']:
                    severity = self._get_severity(
                        BottleneckType.APPLICATION,
                        func_stats.avg_duration_ms
                    )
                    
                    alert = BottleneckAlert(
                        type=BottleneckType.APPLICATION,
                        severity=severity,
                        component=func_stats.function_name,
                        description=f"Slow function execution: {func_stats.function_name}",
                        current_value=func_stats.avg_duration_ms,
                        threshold_value=self.thresholds[BottleneckType.APPLICATION]['medium'],
                        impact_score=self._calculate_impact_score(
                            func_stats.avg_duration_ms,
                            func_stats.total_calls
                        ),
                        detected_at=current_time,
                        recommendations=[
                            "Profile function execution to identify bottlenecks",
                            "Optimize algorithm complexity",
                            "Consider caching frequently computed results",
                            "Review data access patterns"
                        ]
                    )
                    alerts.append(alert)
            
            # Analyze performance anomalies
            for anomaly in anomalies[:5]:  # Top 5 anomalies
                alert = BottleneckAlert(
                    type=BottleneckType.APPLICATION,
                    severity='high',
                    component=anomaly['function'],
                    description=f"Performance anomaly detected: {anomaly['multiplier']:.1f}x slower than average",
                    current_value=anomaly['actual_duration_ms'],
                    threshold_value=anomaly['average_duration_ms'],
                    impact_score=anomaly['multiplier'] * 10,
                    detected_at=current_time,
                    recommendations=[
                        "Investigate recent changes to the function",
                        "Check for resource contention",
                        "Review input data characteristics",
                        "Monitor for recurring patterns"
                    ]
                )
                alerts.append(alert)
                
        except Exception as e:
            print(f"Error analyzing application bottlenecks: {e}")
        
        return alerts
    
    def _analyze_database_bottlenecks(self, current_time: datetime) -> List[BottleneckAlert]:
        """Analyze database performance bottlenecks."""
        alerts = []
        
        try:
            slow_queries = self.db_profiler.get_slow_queries(10)
            
            for query_stats in slow_queries:
                severity = self._get_severity(
                    BottleneckType.DATABASE,
                    query_stats.avg_duration_ms
                )
                
                alert = BottleneckAlert(
                    type=BottleneckType.DATABASE,
                    severity=severity,
                    component=f"Query Pattern: {query_stats.query_pattern[:100]}...",
                    description=f"Slow database query detected",
                    current_value=query_stats.avg_duration_ms,
                    threshold_value=self.thresholds[BottleneckType.DATABASE]['medium'],
                    impact_score=self._calculate_impact_score(
                        query_stats.avg_duration_ms,
                        query_stats.total_executions
                    ),
                    detected_at=current_time,
                    recommendations=[
                        "Review query execution plan",
                        "Add or optimize database indexes",
                        "Consider query rewriting",
                        "Evaluate data partitioning strategies",
                        "Check for missing WHERE clause optimizations"
                    ]
                )
                alerts.append(alert)
                
        except Exception as e:
            print(f"Error analyzing database bottlenecks: {e}")
        
        return alerts
    
    def _analyze_api_bottlenecks(self, current_time: datetime) -> List[BottleneckAlert]:
        """Analyze API performance bottlenecks."""
        alerts = []
        
        try:
            slow_endpoints = self.api_profiler.get_slow_endpoints(10)
            high_error_endpoints = self.api_profiler.get_high_error_endpoints(10)
            
            # Analyze slow endpoints
            for endpoint_stats in slow_endpoints:
                severity = self._get_severity(
                    BottleneckType.API,
                    endpoint_stats.avg_duration_ms
                )
                
                alert = BottleneckAlert(
                    type=BottleneckType.API,
                    severity=severity,
                    component=f"{endpoint_stats.method}:{endpoint_stats.endpoint}",
                    description=f"Slow API endpoint detected",
                    current_value=endpoint_stats.avg_duration_ms,
                    threshold_value=self.thresholds[BottleneckType.API]['medium'],
                    impact_score=self._calculate_impact_score(
                        endpoint_stats.avg_duration_ms,
                        endpoint_stats.total_calls
                    ),
                    detected_at=current_time,
                    recommendations=[
                        "Profile endpoint execution path",
                        "Implement response caching",
                        "Optimize database queries in endpoint",
                        "Consider request/response compression",
                        "Review business logic efficiency"
                    ]
                )
                alerts.append(alert)
            
            # Analyze high error rate endpoints
            for error_endpoint in high_error_endpoints:
                alert = BottleneckAlert(
                    type=BottleneckType.API,
                    severity='high' if error_endpoint['error_rate'] > 0.1 else 'medium',
                    component=error_endpoint['endpoint'],
                    description=f"High error rate: {error_endpoint['error_rate']:.1%}",
                    current_value=error_endpoint['error_rate'] * 100,
                    threshold_value=5.0,  # 5% error rate threshold
                    impact_score=error_endpoint['error_rate'] * error_endpoint['total_calls'],
                    detected_at=current_time,
                    recommendations=[
                        "Review error logs for root cause analysis",
                        "Improve input validation",
                        "Add proper error handling",
                        "Check for dependency failures",
                        "Implement circuit breaker patterns"
                    ]
                )
                alerts.append(alert)
                
        except Exception as e:
            print(f"Error analyzing API bottlenecks: {e}")
        
        return alerts
    
    def _check_threshold_alert(self,
                              bottleneck_type: BottleneckType,
                              component: str,
                              current_value: float,
                              current_time: datetime,
                              description: str,
                              recommendations: List[str]) -> Optional[BottleneckAlert]:
        """Check if metric exceeds threshold and create alert."""
        thresholds = self.thresholds[bottleneck_type]
        
        severity = None
        threshold_value = None
        
        if current_value >= thresholds['critical']:
            severity = 'critical'
            threshold_value = thresholds['critical']
        elif current_value >= thresholds['high']:
            severity = 'high'
            threshold_value = thresholds['high']
        elif current_value >= thresholds['medium']:
            severity = 'medium'
            threshold_value = thresholds['medium']
        
        if severity:
            return BottleneckAlert(
                type=bottleneck_type,
                severity=severity,
                component=component,
                description=description,
                current_value=current_value,
                threshold_value=threshold_value,
                impact_score=current_value / threshold_value * 100,
                detected_at=current_time,
                recommendations=recommendations
            )
        
        return None
    
    def _get_severity(self, bottleneck_type: BottleneckType, value: float) -> str:
        """Get severity level based on value and thresholds."""
        thresholds = self.thresholds[bottleneck_type]
        
        if value >= thresholds['critical']:
            return 'critical'
        elif value >= thresholds['high']:
            return 'high'
        elif value >= thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_impact_score(self, duration_ms: float, frequency: int) -> float:
        """Calculate impact score based on duration and frequency."""
        # Normalize impact: duration impact * frequency impact
        duration_impact = min(duration_ms / 1000, 10)  # Cap at 10 seconds
        frequency_impact = min(frequency / 100, 10)    # Cap at 100 calls
        return duration_impact * frequency_impact
    
    def get_bottleneck_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze bottleneck trends over time."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_alerts = [
                alert for alert in self.bottleneck_history
                if alert.detected_at >= cutoff_time
            ]
        
        # Group by type and severity
        trend_analysis = defaultdict(lambda: defaultdict(int))
        component_frequency = defaultdict(int)
        
        for alert in recent_alerts:
            trend_analysis[alert.type.value][alert.severity] += 1
            component_frequency[alert.component] += 1
        
        return {
            'analysis_period_hours': hours,
            'total_alerts': len(recent_alerts),
            'alerts_by_type': dict(trend_analysis),
            'most_problematic_components': dict(
                sorted(component_frequency.items(),
                      key=lambda x: x[1], reverse=True)[:10]
            ),
            'avg_impact_score': sum(alert.impact_score for alert in recent_alerts) / len(recent_alerts) if recent_alerts else 0
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report."""
        current_alerts = self.analyze_current_performance()
        trend_analysis = self.get_bottleneck_trends()
        
        # System overview
        system_summary = self.system_profiler.get_metrics_summary(10)
        app_summary = self.app_profiler.get_performance_summary()
        db_summary = self.db_profiler.get_query_performance_summary()
        api_summary = self.api_profiler.get_endpoint_performance_summary()
        
        return {
            'generated_at': datetime.now().isoformat(),
            'current_alerts': [
                {
                    'type': alert.type.value,
                    'severity': alert.severity,
                    'component': alert.component,
                    'description': alert.description,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value,
                    'impact_score': alert.impact_score,
                    'recommendations': alert.recommendations
                }
                for alert in current_alerts
            ],
            'trend_analysis': trend_analysis,
            'component_summaries': {
                'system': system_summary,
                'application': app_summary,
                'database': db_summary,
                'api': api_summary
            },
            'overall_health_score': self._calculate_health_score(current_alerts),
            'top_recommendations': self._generate_top_recommendations(current_alerts)
        }
    
    def _calculate_health_score(self, alerts: List[BottleneckAlert]) -> float:
        """Calculate overall system health score (0-100)."""
        if not alerts:
            return 100.0
        
        # Penalty based on alert severity and impact
        penalty = 0
        for alert in alerts:
            severity_multiplier = {
                'low': 1,
                'medium': 3,
                'high': 7,
                'critical': 15
            }
            penalty += severity_multiplier.get(alert.severity, 1) * min(alert.impact_score / 100, 5)
        
        health_score = max(0, 100 - penalty)
        return round(health_score, 1)
    
    def _generate_top_recommendations(self, alerts: List[BottleneckAlert]) -> List[str]:
        """Generate top performance optimization recommendations."""
        recommendation_counts = defaultdict(int)
        
        for alert in alerts:
            for recommendation in alert.recommendations:
                recommendation_counts[recommendation] += alert.impact_score
        
        # Sort by weighted frequency (impact * count)
        top_recommendations = sorted(
            recommendation_counts.items(),
            key=lambda x: x[1], reverse=True
        )[:10]
        
        return [rec[0] for rec in top_recommendations]
    
    def export_bottleneck_analysis(self, filepath: str) -> bool:
        """Export comprehensive bottleneck analysis to file."""
        try:
            report = self.generate_performance_report()
            
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting bottleneck analysis: {e}")
            return False
    
    def set_thresholds(self, 
                      bottleneck_type: BottleneckType,
                      thresholds: Dict[str, float]):
        """Set custom thresholds for bottleneck detection."""
        self.thresholds[bottleneck_type] = thresholds
    
    def reset_analysis(self):
        """Reset bottleneck analysis history."""
        with self._lock:
            self.bottleneck_history.clear()
            self.baselines.clear()