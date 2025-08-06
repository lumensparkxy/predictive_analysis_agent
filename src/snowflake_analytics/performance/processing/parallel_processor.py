"""
Parallel processor for concurrent data processing.
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Any, Callable, Dict
import multiprocessing


class ParallelProcessor:
    """Parallel processor for concurrent operations."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.execution_stats = {'tasks_completed': 0, 'total_time': 0}
    
    def process_parallel_threads(self, func: Callable, data_list: List[Any]) -> List[Any]:
        """Process data using thread pool."""
        import time
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(func, data_list))
        
        # Update stats
        self.execution_stats['tasks_completed'] += len(data_list)
        self.execution_stats['total_time'] += time.time() - start_time
        
        return results
    
    def process_parallel_processes(self, func: Callable, data_list: List[Any]) -> List[Any]:
        """Process data using process pool."""
        import time
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(func, data_list))
        
        # Update stats  
        self.execution_stats['tasks_completed'] += len(data_list)
        self.execution_stats['total_time'] += time.time() - start_time
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'max_workers': self.max_workers,
            'cpu_count': multiprocessing.cpu_count(),
            **self.execution_stats
        }