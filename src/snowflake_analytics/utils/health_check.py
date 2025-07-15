"""
Health check utilities for monitoring system status.

Provides comprehensive health checks for all system components
including database, file storage, cache, and external connections.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import sqlite3


class HealthChecker:
    """System health monitoring and diagnostics."""
    
    def __init__(self):
        """Initialize health checker."""
        self.logger = logging.getLogger(__name__)
    
    def check_all(self) -> Dict[str, Dict[str, Any]]:
        """Run all health checks and return comprehensive status."""
        health_status = {}
        
        # Check each component
        health_status["database"] = self.check_database()
        health_status["file_storage"] = self.check_file_storage()
        health_status["cache"] = self.check_cache()
        health_status["configuration"] = self.check_configuration()
        health_status["directories"] = self.check_directories()
        health_status["permissions"] = self.check_permissions()
        health_status["dependencies"] = self.check_dependencies()
        
        # Overall health status
        overall_healthy = all(
            component["healthy"] for component in health_status.values()
        )
        
        health_status["overall"] = {
            "healthy": overall_healthy,
            "message": "All systems operational" if overall_healthy else "Some systems have issues",
            "timestamp": self._get_timestamp()
        }
        
        return health_status
    
    def check_database(self) -> Dict[str, Any]:
        """Check SQLite database connectivity and integrity."""
        try:
            from ..storage.sqlite_store import SQLiteStore
            
            store = SQLiteStore()
            
            # Test database connection
            with store.get_connection() as conn:
                cursor = conn.execute("SELECT 1")
                cursor.fetchone()
            
            # Get database stats
            stats = store.get_database_stats()
            
            return {
                "healthy": True,
                "message": "Database operational",
                "details": {
                    "database_size_mb": stats.get("database_size_bytes", 0) / (1024 * 1024),
                    "total_records": sum(v for k, v in stats.items() if k.endswith("_count")),
                    "tables": [k.replace("_count", "") for k in stats.keys() if k.endswith("_count")]
                }
            }
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return {
                "healthy": False,
                "message": f"Database error: {str(e)[:100]}",
                "details": {"error": str(e)}
            }
    
    def check_file_storage(self) -> Dict[str, Any]:
        """Check file storage system health."""
        try:
            from ..storage.file_store import FileStore
            
            file_store = FileStore()
            stats = file_store.get_storage_stats()
            
            # Calculate total storage used
            total_size_mb = sum(
                storage["total_size_mb"] for storage in stats.values()
            )
            
            return {
                "healthy": True,
                "message": "File storage operational",
                "details": {
                    "total_size_mb": total_size_mb,
                    "storage_breakdown": stats,
                    "directories_available": len(stats)
                }
            }
        except Exception as e:
            self.logger.error(f"File storage health check failed: {e}")
            return {
                "healthy": False,
                "message": f"File storage error: {str(e)[:100]}",
                "details": {"error": str(e)}
            }
    
    def check_cache(self) -> Dict[str, Any]:
        """Check cache system health."""
        try:
            from ..storage.cache_store import CacheStore
            
            cache = CacheStore()
            
            # Test cache operations
            test_key = "__health_check__"
            test_value = {"timestamp": self._get_timestamp()}
            
            cache.set(test_key, test_value)
            retrieved = cache.get(test_key)
            cache.delete(test_key)
            
            if retrieved != test_value:
                raise Exception("Cache read/write test failed")
            
            stats = cache.get_stats()
            
            return {
                "healthy": True,
                "message": "Cache operational",
                "details": {
                    "size_mb": stats.get("size_mb", 0),
                    "entry_count": stats.get("count", 0),
                    "hit_rate": stats.get("hit_rate", 0.0)
                }
            }
        except Exception as e:
            self.logger.error(f"Cache health check failed: {e}")
            return {
                "healthy": False,
                "message": f"Cache error: {str(e)[:100]}",
                "details": {"error": str(e)}
            }
    
    def check_configuration(self) -> Dict[str, Any]:
        """Check configuration loading and validation."""
        try:
            from ..config.settings import get_settings, validate_configuration
            
            # Load settings
            settings = get_settings()
            
            # Validate configuration
            validation_results = validate_configuration()
            
            all_valid = all(validation_results.values())
            
            return {
                "healthy": all_valid,
                "message": "Configuration valid" if all_valid else "Configuration has issues",
                "details": {
                    "app_name": settings.app.name,
                    "app_version": settings.app.version,
                    "snowflake_configured": settings.snowflake is not None,
                    "validation_results": validation_results
                }
            }
        except Exception as e:
            self.logger.error(f"Configuration health check failed: {e}")
            return {
                "healthy": False,
                "message": f"Configuration error: {str(e)[:100]}",
                "details": {"error": str(e)}
            }
    
    def check_directories(self) -> Dict[str, Any]:
        """Check required directory structure."""
        required_dirs = [
            "data/raw", "data/processed", "data/models", "data/exports",
            "cache", "logs", "config"
        ]
        
        missing_dirs = []
        existing_dirs = []
        
        for directory in required_dirs:
            dir_path = Path(directory)
            if dir_path.exists() and dir_path.is_dir():
                existing_dirs.append(directory)
            else:
                missing_dirs.append(directory)
        
        all_present = len(missing_dirs) == 0
        
        return {
            "healthy": all_present,
            "message": "All directories present" if all_present else f"Missing {len(missing_dirs)} directories",
            "details": {
                "existing_directories": existing_dirs,
                "missing_directories": missing_dirs,
                "total_required": len(required_dirs)
            }
        }
    
    def check_permissions(self) -> Dict[str, Any]:
        """Check file system permissions for required operations."""
        test_paths = [
            ("data", "data/health_check_test.tmp"),
            ("cache", "cache/health_check_test.tmp"),
            ("logs", "logs/health_check_test.tmp")
        ]
        
        permission_issues = []
        successful_tests = []
        
        for directory, test_file in test_paths:
            try:
                # Create directory if it doesn't exist
                Path(directory).mkdir(parents=True, exist_ok=True)
                
                # Test write permission
                test_path = Path(test_file)
                test_path.write_text("health check test")
                
                # Test read permission
                content = test_path.read_text()
                if content != "health check test":
                    raise Exception("Read test failed")
                
                # Test delete permission
                test_path.unlink()
                
                successful_tests.append(directory)
                
            except Exception as e:
                permission_issues.append({
                    "directory": directory,
                    "error": str(e)
                })
        
        all_permissions_ok = len(permission_issues) == 0
        
        return {
            "healthy": all_permissions_ok,
            "message": "All permissions OK" if all_permissions_ok else f"Permission issues in {len(permission_issues)} directories",
            "details": {
                "successful_tests": successful_tests,
                "permission_issues": permission_issues
            }
        }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check critical Python dependencies."""
        critical_packages = [
            "pandas", "numpy", "scikit-learn", "snowflake-connector-python",
            "pyarrow", "joblib", "fastapi", "uvicorn", "pydantic",
            "python-dotenv", "click", "diskcache"
        ]
        
        available_packages = []
        missing_packages = []
        package_versions = {}
        
        for package in critical_packages:
            try:
                if package == "snowflake-connector-python":
                    import snowflake.connector
                    available_packages.append(package)
                    package_versions[package] = getattr(snowflake.connector, "__version__", "unknown")
                elif package == "python-dotenv":
                    import dotenv
                    available_packages.append(package)
                    package_versions[package] = getattr(dotenv, "__version__", "unknown")
                else:
                    module = __import__(package.replace("-", "_"))
                    available_packages.append(package)
                    package_versions[package] = getattr(module, "__version__", "unknown")
            except ImportError:
                missing_packages.append(package)
        
        all_available = len(missing_packages) == 0
        
        return {
            "healthy": all_available,
            "message": "All dependencies available" if all_available else f"Missing {len(missing_packages)} packages",
            "details": {
                "available_packages": available_packages,
                "missing_packages": missing_packages,
                "package_versions": package_versions,
                "total_required": len(critical_packages)
            }
        }
    
    def check_snowflake_connection(self) -> Dict[str, Any]:
        """Check Snowflake database connectivity (optional)."""
        try:
            from ..config.settings import get_settings
            
            settings = get_settings()
            if not settings.snowflake:
                return {
                    "healthy": True,
                    "message": "Snowflake not configured (optional)",
                    "details": {"configured": False}
                }
            
            # Test connection (without importing snowflake-connector-python if not available)
            try:
                import snowflake.connector
                
                conn = snowflake.connector.connect(
                    account=settings.snowflake.account,
                    user=settings.snowflake.user,
                    password=settings.snowflake.password,
                    warehouse=settings.snowflake.warehouse,
                    database=settings.snowflake.database,
                    schema=settings.snowflake.schema,
                    role=settings.snowflake.role
                )
                
                # Test simple query
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                
                cursor.close()
                conn.close()
                
                return {
                    "healthy": True,
                    "message": "Snowflake connection successful",
                    "details": {
                        "account": settings.snowflake.account,
                        "database": settings.snowflake.database,
                        "test_query_result": result[0] if result else None
                    }
                }
            except ImportError:
                return {
                    "healthy": False,
                    "message": "Snowflake connector not available",
                    "details": {"error": "snowflake-connector-python not installed"}
                }
        except Exception as e:
            self.logger.error(f"Snowflake connection check failed: {e}")
            return {
                "healthy": False,
                "message": f"Snowflake connection failed: {str(e)[:100]}",
                "details": {"error": str(e)}
            }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status."""
        results = {
            'overall_status': 'healthy',
            'checks': {},
            'timestamp': self._get_timestamp(),
            'errors': []
        }
        
        try:
            # Run the comprehensive check_all method
            check_results = self.check_all()
            
            # Transform the results for compatibility
            results['checks'] = check_results
            results['overall_status'] = 'healthy' if check_results.get('overall', {}).get('healthy', False) else 'issues'
            
            return results
        except Exception as e:
            results['overall_status'] = 'error'
            results['errors'].append(str(e))
            return results


def quick_health_check() -> bool:
    """Perform a quick health check and return overall status."""
    try:
        checker = HealthChecker()
        status = checker.check_all()
        return status["overall"]["healthy"]
    except Exception:
        return False


if __name__ == "__main__":
    # Test health checker
    checker = HealthChecker()
    
    print("Running comprehensive health check...")
    health_status = checker.check_all()
    
    print("\nHealth Check Results:")
    print("=" * 50)
    
    for component, status in health_status.items():
        icon = "✅" if status["healthy"] else "❌"
        print(f"{icon} {component.upper()}: {status['message']}")
        
        if "details" in status and status["details"]:
            for key, value in status["details"].items():
                if isinstance(value, (list, dict)) and len(str(value)) > 100:
                    print(f"    {key}: {type(value).__name__} with {len(value) if hasattr(value, '__len__') else 'N/A'} items")
                else:
                    print(f"    {key}: {value}")
    
    overall_status = health_status["overall"]["healthy"]
    print(f"\nOverall System Status: {'✅ HEALTHY' if overall_status else '❌ ISSUES DETECTED'}")
    
    # Test quick health check
    quick_status = quick_health_check()
    print(f"Quick Health Check: {'✅ OK' if quick_status else '❌ FAILED'}")
    
    print("\n✅ Health check test completed")
