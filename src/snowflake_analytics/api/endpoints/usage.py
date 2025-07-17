"""
Usage metrics API endpoints for the dashboard.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class UsageEndpoints:
    """Handles usage metrics API endpoints."""
    
    def __init__(self):
        self.logger = logger
        
    async def get_usage_metrics(self, time_range: str = "24h") -> Dict[str, Any]:
        """
        Get usage metrics for the specified time range.
        
        Args:
            time_range: Time range for usage metrics (1h, 24h, 7d, 30d)
            
        Returns:
            Dictionary with usage metrics data
        """
        try:
            # Mock usage data
            mock_data = {
                "total_queries": 12450,
                "active_users": 67,
                "active_warehouses": 4,
                "data_processed_gb": 2348.7,
                "avg_query_duration": 4.2,
                "query_success_rate": 97.8,
                "peak_concurrent_queries": 45,
                "query_types": [
                    {"type": "SELECT", "count": 8950, "percentage": 71.9},
                    {"type": "INSERT", "count": 1890, "percentage": 15.2},
                    {"type": "UPDATE", "count": 980, "percentage": 7.9},
                    {"type": "DELETE", "count": 430, "percentage": 3.5},
                    {"type": "DDL", "count": 200, "percentage": 1.6}
                ],
                "user_activity": [
                    {"user": "analytics_team", "queries": 3200, "data_gb": 890.5},
                    {"user": "data_engineers", "queries": 2800, "data_gb": 1200.2},
                    {"user": "reporting_team", "queries": 1950, "data_gb": 156.8},
                    {"user": "dev_team", "queries": 1200, "data_gb": 78.3}
                ],
                "hourly_usage": [
                    {"hour": "00:00", "queries": 89, "users": 12},
                    {"hour": "01:00", "queries": 45, "users": 8},
                    {"hour": "02:00", "queries": 23, "users": 4},
                    {"hour": "03:00", "queries": 12, "users": 2},
                    {"hour": "04:00", "queries": 34, "users": 6},
                    {"hour": "05:00", "queries": 67, "users": 15},
                    {"hour": "06:00", "queries": 123, "users": 23},
                    {"hour": "07:00", "queries": 189, "users": 31},
                    {"hour": "08:00", "queries": 245, "users": 45},
                    {"hour": "09:00", "queries": 312, "users": 67},
                    {"hour": "10:00", "queries": 398, "users": 78},
                    {"hour": "11:00", "queries": 445, "users": 89}
                ],
                "time_range": time_range,
                "last_updated": datetime.now().isoformat()
            }
            
            return {
                "status": "success",
                "data": mock_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get usage metrics: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_query_performance(self, limit: int = 100) -> Dict[str, Any]:
        """
        Get query performance metrics.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            Dictionary with query performance data
        """
        try:
            # Mock query performance data
            queries = []
            for i in range(limit):
                query = {
                    "query_id": f"query_{i+1:04d}",
                    "duration_seconds": round(1.5 + (i % 30) * 0.3, 2),
                    "rows_returned": (i % 1000) * 100,
                    "bytes_scanned": (i % 500) * 1024 * 1024,
                    "warehouse": ["ANALYTICS_WH", "ETL_WH", "REPORTING_WH"][i % 3],
                    "user": ["analytics_team", "data_engineers", "reporting_team"][i % 3],
                    "status": "success" if i % 50 != 0 else "failed",
                    "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat()
                }
                queries.append(query)
            
            # Performance statistics
            successful_queries = [q for q in queries if q["status"] == "success"]
            avg_duration = sum(q["duration_seconds"] for q in successful_queries) / len(successful_queries)
            
            return {
                "status": "success",
                "data": {
                    "queries": queries,
                    "statistics": {
                        "total_queries": len(queries),
                        "successful_queries": len(successful_queries),
                        "failed_queries": len(queries) - len(successful_queries),
                        "success_rate": round(len(successful_queries) / len(queries) * 100, 2),
                        "avg_duration": round(avg_duration, 2),
                        "median_duration": round(sorted([q["duration_seconds"] for q in successful_queries])[len(successful_queries)//2], 2),
                        "total_data_scanned_gb": round(sum(q["bytes_scanned"] for q in queries) / (1024**3), 2)
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get query performance: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_warehouse_utilization(self) -> Dict[str, Any]:
        """
        Get warehouse utilization metrics.
        
        Returns:
            Dictionary with warehouse utilization data
        """
        try:
            # Mock warehouse utilization data
            warehouses = [
                {
                    "warehouse": "ANALYTICS_WH",
                    "status": "running",
                    "utilization": 78.5,
                    "queue_depth": 3,
                    "avg_queue_time": 2.3,
                    "active_queries": 12,
                    "credits_used": 145.7,
                    "credits_remaining": 854.3,
                    "auto_suspend_time": 300,
                    "size": "MEDIUM",
                    "scaling_policy": "AUTO"
                },
                {
                    "warehouse": "ETL_WH",
                    "status": "running",
                    "utilization": 65.2,
                    "queue_depth": 1,
                    "avg_queue_time": 0.8,
                    "active_queries": 8,
                    "credits_used": 98.4,
                    "credits_remaining": 901.6,
                    "auto_suspend_time": 180,
                    "size": "LARGE",
                    "scaling_policy": "MANUAL"
                },
                {
                    "warehouse": "REPORTING_WH",
                    "status": "suspended",
                    "utilization": 0.0,
                    "queue_depth": 0,
                    "avg_queue_time": 0.0,
                    "active_queries": 0,
                    "credits_used": 67.2,
                    "credits_remaining": 932.8,
                    "auto_suspend_time": 60,
                    "size": "SMALL",
                    "scaling_policy": "AUTO"
                },
                {
                    "warehouse": "DEV_WH",
                    "status": "running",
                    "utilization": 23.1,
                    "queue_depth": 0,
                    "avg_queue_time": 0.0,
                    "active_queries": 2,
                    "credits_used": 23.8,
                    "credits_remaining": 976.2,
                    "auto_suspend_time": 300,
                    "size": "XSMALL",
                    "scaling_policy": "AUTO"
                }
            ]
            
            return {
                "status": "success",
                "data": {
                    "warehouses": warehouses,
                    "total_active_queries": sum(w["active_queries"] for w in warehouses),
                    "total_credits_used": round(sum(w["credits_used"] for w in warehouses), 2),
                    "avg_utilization": round(sum(w["utilization"] for w in warehouses) / len(warehouses), 2),
                    "running_warehouses": len([w for w in warehouses if w["status"] == "running"])
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get warehouse utilization: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Global instance
usage_endpoints = UsageEndpoints()