"""
Pipeline optimizer for data processing pipelines.
"""

from typing import List, Dict, Any, Callable
import time
from concurrent.futures import ThreadPoolExecutor


class PipelineOptimizer:
    """Optimizer for data processing pipelines."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.pipeline_stats = {}
    
    def optimize_pipeline(self, steps: List[Callable], data: Any) -> Any:
        """Optimize pipeline execution."""
        start_time = time.time()
        
        # Sequential execution with timing
        result = data
        step_times = []
        
        for i, step in enumerate(steps):
            step_start = time.time()
            result = step(result)
            step_duration = time.time() - step_start
            step_times.append(step_duration)
        
        total_time = time.time() - start_time
        
        # Store stats
        self.pipeline_stats['last_execution'] = {
            'total_time': total_time,
            'step_times': step_times,
            'steps_count': len(steps)
        }
        
        return result
    
    def parallel_pipeline(self, steps: List[Callable], data_batches: List[Any]) -> List[Any]:
        """Execute pipeline in parallel for multiple data batches."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            def process_batch(batch):
                result = batch
                for step in steps:
                    result = step(result)
                return result
            
            futures = [executor.submit(process_batch, batch) for batch in data_batches]
            return [future.result() for future in futures]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline optimization statistics."""
        return self.pipeline_stats.copy()