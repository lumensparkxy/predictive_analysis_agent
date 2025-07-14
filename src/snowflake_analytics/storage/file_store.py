"""
File-based storage system for time-series data and cache management.

Handles Parquet file operations, data compression, retention policies,
and automatic cleanup of expired files.
"""

import gzip
import hashlib
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from diskcache import Cache

logger = logging.getLogger(__name__)


class FileStore:
    """File-based storage manager for time-series data."""
    
    def __init__(self, base_path: str = "data"):
        """Initialize file store with base directory path."""
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.models_path = self.base_path / "models"
        self.exports_path = self.base_path / "exports"
        
        # Create directories
        for path in [self.raw_path, self.processed_path, self.models_path, self.exports_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def save_dataframe(self, df: pd.DataFrame, file_path: str, 
                      data_type: str = "raw", compress: bool = True) -> Dict[str, Any]:
        """Save DataFrame to Parquet format with metadata."""
        # Determine full path based on data type
        type_path_map = {
            "raw": self.raw_path,
            "processed": self.processed_path,
            "exports": self.exports_path
        }
        
        if data_type not in type_path_map:
            raise ValueError(f"Invalid data_type: {data_type}. Must be one of {list(type_path_map.keys())}")
        
        full_path = type_path_map[data_type] / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp columns if not present
        if 'created_at' not in df.columns:
            df = df.copy()
            df['created_at'] = datetime.utcnow()
        
        # Save to Parquet with compression
        compression_type = 'snappy' if compress else None
        df.to_parquet(full_path, compression=compression_type, index=False)
        
        # Calculate file metadata
        file_size = full_path.stat().st_size
        checksum = self._calculate_checksum(full_path)
        
        metadata = {
            "file_path": str(full_path),
            "file_size": file_size,
            "checksum": checksum,
            "rows": len(df),
            "columns": list(df.columns),
            "data_type": data_type,
            "compressed": compress,
            "created_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Saved DataFrame to {full_path}: {len(df)} rows, {file_size} bytes")
        return metadata
    
    def load_dataframe(self, file_path: str, data_type: str = "raw", 
                      columns: Optional[List[str]] = None,
                      filters: Optional[List] = None) -> pd.DataFrame:
        """Load DataFrame from Parquet file."""
        type_path_map = {
            "raw": self.raw_path,
            "processed": self.processed_path,
            "exports": self.exports_path
        }
        
        full_path = type_path_map[data_type] / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
        
        # Load with optional column selection and filtering
        df = pd.read_parquet(full_path, columns=columns, filters=filters)
        
        logger.info(f"Loaded DataFrame from {full_path}: {len(df)} rows")
        return df
    
    def save_model(self, model: Any, model_name: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Save ML model using joblib with metadata."""
        import joblib
        
        model_path = self.models_path / f"{model_name}.joblib"
        metadata_path = self.models_path / f"{model_name}_metadata.json"
        
        # Save model
        joblib.dump(model, model_path)
        
        # Prepare metadata
        model_metadata = {
            "model_name": model_name,
            "model_path": str(model_path),
            "file_size": model_path.stat().st_size,
            "checksum": self._calculate_checksum(model_path),
            "created_at": datetime.utcnow().isoformat(),
            "model_type": type(model).__name__,
        }
        
        if metadata:
            model_metadata.update(metadata)
        
        # Save metadata
        import json
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        logger.info(f"Saved model to {model_path}")
        return model_metadata
    
    def load_model(self, model_name: str) -> tuple[Any, Dict]:
        """Load ML model and its metadata."""
        import joblib
        import json
        
        model_path = self.models_path / f"{model_name}.joblib"
        metadata_path = self.models_path / f"{model_name}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load metadata if available
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        logger.info(f"Loaded model from {model_path}")
        return model, metadata
    
    def list_files(self, data_type: str = "raw", pattern: str = "*.parquet") -> List[Dict[str, Any]]:
        """List files in a directory with metadata."""
        type_path_map = {
            "raw": self.raw_path,
            "processed": self.processed_path,
            "models": self.models_path,
            "exports": self.exports_path
        }
        
        directory = type_path_map[data_type]
        files = []
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                stat = file_path.stat()
                file_info = {
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                }
                files.append(file_info)
        
        return sorted(files, key=lambda x: x['modified'], reverse=True)
    
    def compress_file(self, file_path: Path) -> Path:
        """Compress a file using gzip."""
        compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
        
        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove original file
        file_path.unlink()
        
        logger.info(f"Compressed {file_path} to {compressed_path}")
        return compressed_path
    
    def decompress_file(self, compressed_path: Path) -> Path:
        """Decompress a gzipped file."""
        if not compressed_path.name.endswith('.gz'):
            raise ValueError("File is not gzipped")
        
        original_path = compressed_path.with_suffix('')
        
        with gzip.open(compressed_path, 'rb') as f_in:
            with open(original_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        logger.info(f"Decompressed {compressed_path} to {original_path}")
        return original_path
    
    def cleanup_old_files(self, data_type: str, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up files older than specified days."""
        type_path_map = {
            "raw": self.raw_path,
            "processed": self.processed_path,
            "models": self.models_path,
            "exports": self.exports_path
        }
        
        directory = type_path_map[data_type]
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        deleted_count = 0
        deleted_size = 0
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if modified_time < cutoff_date:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    deleted_count += 1
                    deleted_size += file_size
                    logger.debug(f"Deleted old file: {file_path}")
        
        stats = {
            "files_deleted": deleted_count,
            "bytes_freed": deleted_size,
            "cleanup_date": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Cleaned up {deleted_count} files ({deleted_size} bytes) from {directory}")
        return stats
    
    def get_storage_stats(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get storage statistics for all directories."""
        stats = {}
        
        for data_type, directory in [
            ("raw", self.raw_path),
            ("processed", self.processed_path),
            ("models", self.models_path),
            ("exports", self.exports_path)
        ]:
            file_count = 0
            total_size = 0
            
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    file_count += 1
                    total_size += file_path.stat().st_size
            
            stats[data_type] = {
                "file_count": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "directory": str(directory)
            }
        
        return stats
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class CacheStore:
    """File-based cache manager using diskcache."""
    
    def __init__(self, cache_dir: str = "cache", size_limit: int = 1_000_000_000):  # 1GB default
        """Initialize cache store."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache = Cache(str(self.cache_dir), size_limit=size_limit)
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        return self.cache.get(key, default)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        try:
            if ttl:
                return self.cache.set(key, value, expire=ttl)
            else:
                return self.cache.set(key, value)
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        return self.cache.delete(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": self.cache.volume(),
            "count": len(self.cache),
            "hits": getattr(self.cache, 'hits', 0),
            "misses": getattr(self.cache, 'misses', 0),
            "directory": str(self.cache_dir)
        }
    
    def cleanup_expired(self) -> int:
        """Clean up expired cache entries."""
        expired_count = 0
        try:
            expired_count = self.cache.expire()
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache entries: {e}")
        
        logger.info(f"Cleaned up {expired_count} expired cache entries")
        return expired_count


if __name__ == "__main__":
    # Test the file store
    import numpy as np
    
    # Create test data
    test_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'value': np.random.randn(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Test file store
    file_store = FileStore("test_data")
    
    # Save and load DataFrame
    metadata = file_store.save_dataframe(test_df, "test_data.parquet", "raw")
    print("Save metadata:", metadata)
    
    loaded_df = file_store.load_dataframe("test_data.parquet", "raw")
    print(f"Loaded DataFrame: {len(loaded_df)} rows")
    
    # Test cache store
    cache_store = CacheStore("test_cache")
    cache_store.set("test_key", {"data": "test_value"}, ttl=3600)
    cached_value = cache_store.get("test_key")
    print("Cached value:", cached_value)
    
    # Get storage stats
    stats = file_store.get_storage_stats()
    print("Storage stats:", stats)
    
    cache_stats = cache_store.get_stats()
    print("Cache stats:", cache_stats)
    
    # Cleanup test files
    shutil.rmtree("test_data", ignore_errors=True)
    shutil.rmtree("test_cache", ignore_errors=True)
    
    print("✅ File store test completed successfully")
