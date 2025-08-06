"""
Compression manager for request/response compression.
"""

import gzip
import zlib
from typing import Union, Dict, Any


class CompressionManager:
    """Manager for request/response compression."""
    
    def __init__(self):
        self.compression_stats = {
            'total_compressions': 0,
            'bytes_saved': 0
        }
    
    def compress(self, data: Union[str, bytes], method: str = 'gzip') -> bytes:
        """Compress data using specified method."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if method == 'gzip':
            compressed = gzip.compress(data)
        elif method == 'zlib':
            compressed = zlib.compress(data)
        else:
            compressed = data
        
        # Update stats
        self.compression_stats['total_compressions'] += 1
        self.compression_stats['bytes_saved'] += len(data) - len(compressed)
        
        return compressed
    
    def decompress(self, data: bytes, method: str = 'gzip') -> bytes:
        """Decompress data."""
        if method == 'gzip':
            return gzip.decompress(data)
        elif method == 'zlib':
            return zlib.decompress(data)
        return data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return self.compression_stats.copy()