"""
API response optimizer for compression, serialization, and response optimization.
"""

import gzip
import json
import time
from typing import Any, Dict, Optional, Union, List
from datetime import datetime
from dataclasses import dataclass


@dataclass
class CompressionResult:
    """Compression result with metrics."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time_ms: float


class ResponseOptimizer:
    """
    API response optimizer for compression, efficient serialization,
    and response time optimization.
    """
    
    def __init__(self):
        """Initialize response optimizer."""
        self.compression_stats = {
            'total_compressions': 0,
            'total_bytes_saved': 0,
            'avg_compression_ratio': 0.0
        }
    
    def optimize_response(self, 
                         data: Any,
                         compress: bool = True,
                         compression_threshold: int = 1024) -> Dict[str, Any]:
        """
        Optimize API response with compression and efficient serialization.
        
        Args:
            data: Response data to optimize
            compress: Whether to compress response
            compression_threshold: Minimum size in bytes to trigger compression
            
        Returns:
            Optimized response with metadata
        """
        start_time = time.time()
        
        # Serialize data efficiently
        serialized = self._efficient_serialize(data)
        serialized_size = len(serialized.encode('utf-8'))
        
        result = {
            'data': serialized,
            'original_size': serialized_size,
            'compressed': False,
            'compression_ratio': 1.0,
            'optimization_time_ms': 0
        }
        
        # Apply compression if beneficial
        if compress and serialized_size > compression_threshold:
            compression_result = self._compress_data(serialized)
            if compression_result.compression_ratio > 1.1:  # At least 10% savings
                result.update({
                    'data': compression_result,
                    'compressed': True,
                    'compression_ratio': compression_result.compression_ratio,
                    'compressed_size': compression_result.compressed_size
                })
                
                # Update stats
                self.compression_stats['total_compressions'] += 1
                self.compression_stats['total_bytes_saved'] += (
                    compression_result.original_size - compression_result.compressed_size
                )
        
        result['optimization_time_ms'] = (time.time() - start_time) * 1000
        return result
    
    def _efficient_serialize(self, data: Any) -> str:
        """Efficiently serialize data to JSON."""
        try:
            return json.dumps(data, separators=(',', ':'), default=str)
        except Exception:
            return json.dumps(str(data))
    
    def _compress_data(self, data: str) -> CompressionResult:
        """Compress string data using gzip."""
        start_time = time.time()
        original_bytes = data.encode('utf-8')
        compressed_bytes = gzip.compress(original_bytes)
        
        return CompressionResult(
            original_size=len(original_bytes),
            compressed_size=len(compressed_bytes),
            compression_ratio=len(original_bytes) / len(compressed_bytes),
            compression_time_ms=(time.time() - start_time) * 1000
        )
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get response optimization statistics."""
        return {
            'compression_stats': self.compression_stats.copy(),
            'timestamp': datetime.now().isoformat()
        }