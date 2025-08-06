"""
API endpoint performance profiler for response time tracking and optimization.
"""

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import functools
from urllib.parse import urlparse, parse_qs


@dataclass
class APICall:
    """API endpoint call performance record."""
    endpoint: str
    method: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status_code: Optional[int] = None
    request_size_bytes: Optional[int] = None
    response_size_bytes: Optional[int] = None
    user_agent: Optional[str] = None
    client_ip: Optional[str] = None
    query_params: Optional[Dict[str, Any]] = None
    exception: Optional[str] = None


@dataclass
class EndpointStats:
    """Aggregated API endpoint performance statistics."""
    endpoint: str
    method: str
    total_calls: int
    total_duration_ms: float
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    p95_duration_ms: float
    success_count: int
    error_count: int
    total_request_bytes: int
    total_response_bytes: int
    last_called: datetime


class APIProfiler:
    """
    API endpoint performance profiler for monitoring response times,
    throughput, and identifying performance bottlenecks.
    """
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize API profiler.
        
        Args:
            max_history: Maximum number of API calls to keep in history
        """
        self.max_history = max_history
        self.api_calls: deque = deque(maxlen=max_history)
        self.endpoint_stats: Dict[str, EndpointStats] = {}
        self.active_calls: Dict[str, APICall] = {}
        self._lock = threading.Lock()
        self.enabled = True
        
        # Performance thresholds
        self.slow_response_threshold_ms = 1000
        self.error_rate_threshold = 0.05  # 5%
    
    def profile_api_call(self, func: Callable) -> Callable:
        """
        Decorator to profile API endpoint calls.
        
        Args:
            func: API handler function to profile
            
        Returns:
            Wrapped function with profiling
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)
            
            # Extract request information
            request = None
            endpoint = func.__name__
            method = 'UNKNOWN'
            
            # Try to extract request info from common frameworks
            if args:
                if hasattr(args[0], 'method'):  # Flask/FastAPI request
                    request = args[0]
                    endpoint = getattr(request, 'endpoint', request.url.path if hasattr(request, 'url') else endpoint)
                    method = request.method
                elif hasattr(args[0], 'path'):  # Some framework request objects
                    endpoint = args[0].path
                    method = getattr(args[0], 'method', 'GET')
            
            call_id = f"{threading.current_thread().ident}_{time.time()}"
            
            # Create API call record
            api_call = APICall(
                endpoint=endpoint,
                method=method,
                start_time=datetime.now()
            )
            
            # Extract additional request info if available
            if request:
                api_call.user_agent = getattr(request, 'user_agent', None)
                api_call.client_ip = getattr(request, 'remote_addr', None)
                
                # Extract query parameters
                if hasattr(request, 'args'):
                    api_call.query_params = dict(request.args)
                elif hasattr(request, 'query_params'):
                    api_call.query_params = dict(request.query_params)
                
                # Estimate request size
                if hasattr(request, 'content_length'):
                    api_call.request_size_bytes = request.content_length
                elif hasattr(request, 'data'):
                    try:
                        api_call.request_size_bytes = len(str(request.data).encode())
                    except:
                        pass
            
            with self._lock:
                self.active_calls[call_id] = api_call
            
            start_time = time.perf_counter()
            status_code = 200
            exception_occurred = None
            response_size = None
            
            try:
                result = func(*args, **kwargs)
                
                # Try to extract response information
                if hasattr(result, 'status_code'):
                    status_code = result.status_code
                elif isinstance(result, tuple) and len(result) >= 2:
                    # Flask-style return (data, status_code)
                    if isinstance(result[1], int):
                        status_code = result[1]
                
                # Estimate response size
                if hasattr(result, 'content'):
                    try:
                        response_size = len(result.content)
                    except:
                        pass
                elif isinstance(result, (str, bytes)):
                    response_size = len(str(result).encode())
                elif isinstance(result, dict):
                    try:
                        import json
                        response_size = len(json.dumps(result).encode())
                    except:
                        pass
                
                return result
                
            except Exception as e:
                exception_occurred = str(e)
                status_code = 500
                raise
                
            finally:
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                
                # Update API call record
                api_call.end_time = datetime.now()
                api_call.duration_ms = duration_ms
                api_call.status_code = status_code
                api_call.response_size_bytes = response_size
                api_call.exception = exception_occurred
                
                with self._lock:
                    # Remove from active calls
                    self.active_calls.pop(call_id, None)
                    
                    # Add to history
                    self.api_calls.append(api_call)
                    
                    # Update endpoint statistics
                    self._update_endpoint_stats(api_call)
        
        return wrapper
    
    def start_api_call_tracking(self, 
                               endpoint: str,
                               method: str = 'GET',
                               request_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Manually start tracking an API call.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            request_info: Additional request information
            
        Returns:
            Call tracking ID
        """
        if not self.enabled:
            return ""
        
        call_id = f"{threading.current_thread().ident}_{time.time()}"
        api_call = APICall(
            endpoint=endpoint,
            method=method,
            start_time=datetime.now()
        )
        
        if request_info:
            api_call.user_agent = request_info.get('user_agent')
            api_call.client_ip = request_info.get('client_ip')
            api_call.query_params = request_info.get('query_params')
            api_call.request_size_bytes = request_info.get('request_size')
        
        with self._lock:
            self.active_calls[call_id] = api_call
        
        return call_id
    
    def end_api_call_tracking(self,
                             call_id: str,
                             status_code: int = 200,
                             response_size: Optional[int] = None,
                             exception: Optional[str] = None):
        """
        End tracking an API call.
        
        Args:
            call_id: Call tracking ID from start_api_call_tracking
            status_code: HTTP response status code
            response_size: Response size in bytes
            exception: Exception message if call failed
        """
        if not self.enabled or not call_id:
            return
        
        with self._lock:
            api_call = self.active_calls.pop(call_id, None)
            
            if api_call is None:
                return
            
            # Update call record
            api_call.end_time = datetime.now()
            api_call.duration_ms = (
                (api_call.end_time - api_call.start_time).total_seconds() * 1000
            )
            api_call.status_code = status_code
            api_call.response_size_bytes = response_size
            api_call.exception = exception
            
            # Add to history
            self.api_calls.append(api_call)
            
            # Update statistics
            self._update_endpoint_stats(api_call)
    
    def _update_endpoint_stats(self, call: APICall):
        """Update aggregated endpoint statistics."""
        endpoint_key = f"{call.method}:{call.endpoint}"
        
        if endpoint_key in self.endpoint_stats:
            stats = self.endpoint_stats[endpoint_key]
            stats.total_calls += 1
            stats.total_duration_ms += call.duration_ms or 0
            stats.avg_duration_ms = stats.total_duration_ms / stats.total_calls
            stats.min_duration_ms = min(stats.min_duration_ms, call.duration_ms or 0)
            stats.max_duration_ms = max(stats.max_duration_ms, call.duration_ms or 0)
            
            if call.status_code and call.status_code < 400:
                stats.success_count += 1
            else:
                stats.error_count += 1
                
            stats.total_request_bytes += call.request_size_bytes or 0
            stats.total_response_bytes += call.response_size_bytes or 0
            stats.last_called = call.end_time or call.start_time
            
            # Update P95 (simplified calculation)
            self._update_p95(stats, endpoint_key)
            
        else:
            self.endpoint_stats[endpoint_key] = EndpointStats(
                endpoint=call.endpoint,
                method=call.method,
                total_calls=1,
                total_duration_ms=call.duration_ms or 0,
                avg_duration_ms=call.duration_ms or 0,
                min_duration_ms=call.duration_ms or 0,
                max_duration_ms=call.duration_ms or 0,
                p95_duration_ms=call.duration_ms or 0,
                success_count=1 if call.status_code and call.status_code < 400 else 0,
                error_count=1 if call.status_code and call.status_code >= 400 else 0,
                total_request_bytes=call.request_size_bytes or 0,
                total_response_bytes=call.response_size_bytes or 0,
                last_called=call.end_time or call.start_time
            )
    
    def _update_p95(self, stats: EndpointStats, endpoint_key: str):
        """Update P95 response time calculation."""
        # Get recent durations for this endpoint
        endpoint_calls = [
            call for call in list(self.api_calls)[-1000:]  # Last 1000 calls
            if f"{call.method}:{call.endpoint}" == endpoint_key and call.duration_ms is not None
        ]
        
        if len(endpoint_calls) >= 20:  # Need sufficient data points
            durations = sorted([call.duration_ms for call in endpoint_calls])
            p95_index = int(len(durations) * 0.95)
            stats.p95_duration_ms = durations[p95_index]
    
    def get_slow_endpoints(self, 
                          limit: int = 10,
                          threshold_ms: Optional[float] = None) -> List[EndpointStats]:
        """Get slowest API endpoints by average response time."""
        threshold = threshold_ms or self.slow_response_threshold_ms
        
        with self._lock:
            slow_endpoints = [
                stats for stats in self.endpoint_stats.values()
                if stats.avg_duration_ms >= threshold
            ]
            
            return sorted(slow_endpoints,
                         key=lambda x: x.avg_duration_ms,
                         reverse=True)[:limit]
    
    def get_high_error_endpoints(self, 
                               limit: int = 10,
                               min_calls: int = 10) -> List[Dict[str, Any]]:
        """Get endpoints with high error rates."""
        with self._lock:
            high_error_endpoints = []
            
            for stats in self.endpoint_stats.values():
                if stats.total_calls < min_calls:
                    continue
                
                error_rate = stats.error_count / stats.total_calls
                if error_rate >= self.error_rate_threshold:
                    high_error_endpoints.append({
                        'endpoint': f"{stats.method}:{stats.endpoint}",
                        'error_rate': error_rate,
                        'total_calls': stats.total_calls,
                        'error_count': stats.error_count,
                        'avg_duration_ms': stats.avg_duration_ms
                    })
            
            return sorted(high_error_endpoints,
                         key=lambda x: x['error_rate'],
                         reverse=True)[:limit]
    
    def get_throughput_analysis(self, minutes: int = 10) -> Dict[str, Any]:
        """Analyze API throughput for specified time period."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            recent_calls = [
                call for call in self.api_calls
                if call.start_time >= cutoff_time
            ]
        
        if not recent_calls:
            return {
                'total_requests': 0,
                'requests_per_minute': 0,
                'avg_response_time_ms': 0,
                'error_rate': 0
            }
        
        total_requests = len(recent_calls)
        successful_requests = len([c for c in recent_calls if c.status_code and c.status_code < 400])
        error_requests = total_requests - successful_requests
        
        durations = [c.duration_ms for c in recent_calls if c.duration_ms is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            'total_requests': total_requests,
            'requests_per_minute': total_requests / minutes,
            'successful_requests': successful_requests,
            'error_requests': error_requests,
            'error_rate': error_requests / total_requests,
            'avg_response_time_ms': avg_duration,
            'p95_response_time_ms': sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
            'time_period_minutes': minutes
        }
    
    def get_endpoint_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive API performance summary."""
        with self._lock:
            if not self.endpoint_stats:
                return {
                    'total_endpoints': 0,
                    'total_calls': 0,
                    'avg_response_time_ms': 0,
                    'overall_error_rate': 0,
                    'active_calls': 0
                }
            
            total_calls = sum(stats.total_calls for stats in self.endpoint_stats.values())
            total_errors = sum(stats.error_count for stats in self.endpoint_stats.values())
            total_time = sum(stats.total_duration_ms for stats in self.endpoint_stats.values())
            
            return {
                'total_endpoints': len(self.endpoint_stats),
                'total_calls': total_calls,
                'avg_response_time_ms': total_time / total_calls if total_calls > 0 else 0,
                'overall_error_rate': total_errors / total_calls if total_calls > 0 else 0,
                'active_calls': len(self.active_calls),
                'slow_endpoints_count': len(self.get_slow_endpoints(limit=1000)),
                'high_error_endpoints_count': len(self.get_high_error_endpoints(limit=1000))
            }
    
    def analyze_traffic_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze API traffic patterns over time."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_calls = [
                call for call in self.api_calls
                if call.start_time >= cutoff_time
            ]
        
        # Group calls by hour
        hourly_stats = defaultdict(lambda: {'count': 0, 'total_duration': 0})
        endpoint_popularity = defaultdict(int)
        method_distribution = defaultdict(int)
        
        for call in recent_calls:
            hour_key = call.start_time.strftime('%Y-%m-%d %H:00')
            hourly_stats[hour_key]['count'] += 1
            hourly_stats[hour_key]['total_duration'] += call.duration_ms or 0
            
            endpoint_popularity[call.endpoint] += 1
            method_distribution[call.method] += 1
        
        # Calculate hourly averages
        for hour_data in hourly_stats.values():
            hour_data['avg_duration'] = (
                hour_data['total_duration'] / hour_data['count'] 
                if hour_data['count'] > 0 else 0
            )
        
        return {
            'analysis_period_hours': hours,
            'hourly_traffic': dict(hourly_stats),
            'endpoint_popularity': dict(sorted(endpoint_popularity.items(), 
                                             key=lambda x: x[1], reverse=True)[:20]),
            'method_distribution': dict(method_distribution),
            'peak_hour': max(hourly_stats.items(), 
                           key=lambda x: x[1]['count'])[0] if hourly_stats else None
        }
    
    def export_api_performance_report(self, filepath: str) -> bool:
        """Export comprehensive API performance report."""
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'performance_summary': self.get_endpoint_performance_summary(),
                'throughput_analysis': self.get_throughput_analysis(60),  # Last hour
                'slow_endpoints': [
                    {
                        'endpoint': f"{stats.method}:{stats.endpoint}",
                        'avg_duration_ms': stats.avg_duration_ms,
                        'p95_duration_ms': stats.p95_duration_ms,
                        'total_calls': stats.total_calls
                    }
                    for stats in self.get_slow_endpoints(20)
                ],
                'high_error_endpoints': self.get_high_error_endpoints(20),
                'traffic_patterns': self.analyze_traffic_patterns(24)
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting API performance report: {e}")
            return False
    
    def reset_statistics(self):
        """Reset all API profiling statistics."""
        with self._lock:
            self.api_calls.clear()
            self.endpoint_stats.clear()
            self.active_calls.clear()
    
    def set_thresholds(self, 
                      slow_response_ms: Optional[float] = None,
                      error_rate: Optional[float] = None):
        """Set performance thresholds."""
        if slow_response_ms is not None:
            self.slow_response_threshold_ms = slow_response_ms
        if error_rate is not None:
            self.error_rate_threshold = error_rate
    
    def enable(self):
        """Enable API profiling."""
        self.enabled = True
    
    def disable(self):
        """Disable API profiling."""
        self.enabled = False