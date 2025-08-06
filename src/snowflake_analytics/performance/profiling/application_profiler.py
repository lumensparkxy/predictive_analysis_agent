"""
Application performance profiler for function execution time and call stack analysis.
"""

import time
import functools
import threading
import cProfile
import pstats
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import traceback


@dataclass
class FunctionCall:
    """Function call performance data."""
    function_name: str
    module_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    args_count: int = 0
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    exception: Optional[str] = None


@dataclass
class PerformanceStats:
    """Aggregated performance statistics for a function."""
    function_name: str
    total_calls: int
    total_duration_ms: float
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    last_called: datetime


class ApplicationProfiler:
    """
    Application performance profiler for monitoring function execution times,
    call patterns, and identifying performance bottlenecks.
    """
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize application profiler.
        
        Args:
            max_history: Maximum number of function calls to keep in history
        """
        self.max_history = max_history
        self.function_calls: deque = deque(maxlen=max_history)
        self.performance_stats: Dict[str, PerformanceStats] = {}
        self.active_calls: Dict[str, FunctionCall] = {}
        self._lock = threading.Lock()
        self.enabled = True
        
        # Profiler state
        self._profiler: Optional[cProfile.Profile] = None
        self._profiling_active = False
    
    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator to profile function execution time and performance.
        
        Args:
            func: Function to profile
            
        Returns:
            Wrapped function with profiling
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)
            
            # Create function call record
            call_id = f"{id(threading.current_thread())}_{time.time()}"
            function_call = FunctionCall(
                function_name=func.__name__,
                module_name=func.__module__ if hasattr(func, '__module__') else 'unknown',
                start_time=datetime.now(),
                args_count=len(args) + len(kwargs)
            )
            
            # Get memory usage before (if psutil available)
            try:
                import psutil
                process = psutil.Process()
                function_call.memory_before = process.memory_info().rss / 1024 / 1024  # MB
            except ImportError:
                pass
            
            with self._lock:
                self.active_calls[call_id] = function_call
            
            start_time = time.perf_counter()
            exception_occurred = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                exception_occurred = str(e)
                raise
            finally:
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                
                # Update function call record
                function_call.end_time = datetime.now()
                function_call.duration_ms = duration_ms
                function_call.exception = exception_occurred
                
                # Get memory usage after
                try:
                    import psutil
                    process = psutil.Process()
                    function_call.memory_after = process.memory_info().rss / 1024 / 1024  # MB
                except ImportError:
                    pass
                
                with self._lock:
                    # Remove from active calls
                    self.active_calls.pop(call_id, None)
                    
                    # Add to history
                    self.function_calls.append(function_call)
                    
                    # Update performance statistics
                    self._update_performance_stats(function_call)
        
        return wrapper
    
    def _update_performance_stats(self, call: FunctionCall):
        """Update aggregated performance statistics."""
        func_key = f"{call.module_name}.{call.function_name}"
        
        if func_key in self.performance_stats:
            stats = self.performance_stats[func_key]
            stats.total_calls += 1
            stats.total_duration_ms += call.duration_ms or 0
            stats.avg_duration_ms = stats.total_duration_ms / stats.total_calls
            stats.min_duration_ms = min(stats.min_duration_ms, call.duration_ms or 0)
            stats.max_duration_ms = max(stats.max_duration_ms, call.duration_ms or 0)
            stats.last_called = call.end_time or call.start_time
        else:
            self.performance_stats[func_key] = PerformanceStats(
                function_name=func_key,
                total_calls=1,
                total_duration_ms=call.duration_ms or 0,
                avg_duration_ms=call.duration_ms or 0,
                min_duration_ms=call.duration_ms or 0,
                max_duration_ms=call.duration_ms or 0,
                last_called=call.end_time or call.start_time
            )
    
    def start_profiling(self) -> bool:
        """Start cProfile-based detailed profiling."""
        if self._profiling_active:
            return False
        
        self._profiler = cProfile.Profile()
        self._profiler.enable()
        self._profiling_active = True
        return True
    
    def stop_profiling(self) -> Optional[pstats.Stats]:
        """Stop profiling and return statistics."""
        if not self._profiling_active or not self._profiler:
            return None
        
        self._profiler.disable()
        self._profiling_active = False
        
        stats = pstats.Stats(self._profiler)
        stats.sort_stats('cumulative')
        return stats
    
    def get_top_slow_functions(self, limit: int = 10) -> List[PerformanceStats]:
        """Get top slowest functions by average execution time."""
        with self._lock:
            sorted_stats = sorted(
                self.performance_stats.values(),
                key=lambda x: x.avg_duration_ms,
                reverse=True
            )
        return sorted_stats[:limit]
    
    def get_most_called_functions(self, limit: int = 10) -> List[PerformanceStats]:
        """Get most frequently called functions."""
        with self._lock:
            sorted_stats = sorted(
                self.performance_stats.values(),
                key=lambda x: x.total_calls,
                reverse=True
            )
        return sorted_stats[:limit]
    
    def get_function_calls_in_timeframe(self, 
                                       minutes: int = 10) -> List[FunctionCall]:
        """Get function calls within specified timeframe."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        with self._lock:
            return [
                call for call in self.function_calls
                if call.start_time >= cutoff_time
            ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            total_calls = sum(stats.total_calls for stats in self.performance_stats.values())
            total_time = sum(stats.total_duration_ms for stats in self.performance_stats.values())
            
            if not self.performance_stats:
                return {
                    'total_functions': 0,
                    'total_calls': 0,
                    'total_time_ms': 0,
                    'avg_call_time_ms': 0,
                    'active_calls': 0
                }
            
            return {
                'total_functions': len(self.performance_stats),
                'total_calls': total_calls,
                'total_time_ms': total_time,
                'avg_call_time_ms': total_time / total_calls if total_calls > 0 else 0,
                'active_calls': len(self.active_calls),
                'top_slow_functions': [
                    {
                        'name': stat.function_name,
                        'avg_duration_ms': stat.avg_duration_ms,
                        'total_calls': stat.total_calls
                    }
                    for stat in self.get_top_slow_functions(5)
                ],
                'most_called_functions': [
                    {
                        'name': stat.function_name,
                        'total_calls': stat.total_calls,
                        'avg_duration_ms': stat.avg_duration_ms
                    }
                    for stat in self.get_most_called_functions(5)
                ]
            }
    
    def detect_performance_anomalies(self, 
                                   threshold_multiplier: float = 3.0) -> List[Dict[str, Any]]:
        """
        Detect performance anomalies where recent calls are significantly slower.
        
        Args:
            threshold_multiplier: How many times slower than average to flag
            
        Returns:
            List of anomalous function calls
        """
        anomalies = []
        recent_calls = self.get_function_calls_in_timeframe(5)  # Last 5 minutes
        
        for call in recent_calls:
            func_key = f"{call.module_name}.{call.function_name}"
            if func_key in self.performance_stats and call.duration_ms:
                avg_duration = self.performance_stats[func_key].avg_duration_ms
                if call.duration_ms > avg_duration * threshold_multiplier:
                    anomalies.append({
                        'function': func_key,
                        'actual_duration_ms': call.duration_ms,
                        'average_duration_ms': avg_duration,
                        'multiplier': call.duration_ms / avg_duration,
                        'timestamp': call.start_time,
                        'exception': call.exception
                    })
        
        return sorted(anomalies, key=lambda x: x['multiplier'], reverse=True)
    
    def get_memory_usage_analysis(self) -> Dict[str, Any]:
        """Analyze memory usage patterns from profiled functions."""
        memory_stats = defaultdict(list)
        
        with self._lock:
            for call in self.function_calls:
                if call.memory_before is not None and call.memory_after is not None:
                    func_key = f"{call.module_name}.{call.function_name}"
                    memory_delta = call.memory_after - call.memory_before
                    memory_stats[func_key].append(memory_delta)
        
        analysis = {}
        for func_key, deltas in memory_stats.items():
            if deltas:
                analysis[func_key] = {
                    'avg_memory_delta_mb': sum(deltas) / len(deltas),
                    'max_memory_delta_mb': max(deltas),
                    'min_memory_delta_mb': min(deltas),
                    'sample_count': len(deltas)
                }
        
        # Sort by average memory usage
        sorted_analysis = dict(sorted(
            analysis.items(),
            key=lambda x: x[1]['avg_memory_delta_mb'],
            reverse=True
        ))
        
        return sorted_analysis
    
    def export_profiling_report(self, filepath: str) -> bool:
        """Export comprehensive profiling report to file."""
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'performance_summary': self.get_performance_summary(),
                'top_slow_functions': [
                    {
                        'function': stat.function_name,
                        'total_calls': stat.total_calls,
                        'avg_duration_ms': stat.avg_duration_ms,
                        'max_duration_ms': stat.max_duration_ms,
                        'total_duration_ms': stat.total_duration_ms
                    }
                    for stat in self.get_top_slow_functions(20)
                ],
                'performance_anomalies': self.detect_performance_anomalies(),
                'memory_usage_analysis': self.get_memory_usage_analysis()
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting profiling report: {e}")
            return False
    
    def reset_statistics(self):
        """Reset all profiling statistics."""
        with self._lock:
            self.function_calls.clear()
            self.performance_stats.clear()
            self.active_calls.clear()
    
    def enable(self):
        """Enable profiling."""
        self.enabled = True
    
    def disable(self):
        """Disable profiling."""
        self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self.enabled