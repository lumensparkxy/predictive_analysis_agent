"""
SQLite storage layer for metadata and system information.

Provides database operations for tracking data collection runs,
model training sessions, alerts, and system metrics.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SQLiteStore:
    """SQLite database manager for application metadata."""
    
    def __init__(self, db_path: str = "storage.db"):
        """Initialize SQLite store with database path."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database on first use
        if not self.db_path.exists():
            self.initialize_database()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def initialize_database(self) -> None:
        """Create database tables with schema."""
        schema_sql = """
        -- Data collection tracking
        CREATE TABLE IF NOT EXISTS data_collection_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            data_source TEXT NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'running',
            records_collected INTEGER DEFAULT 0,
            file_path TEXT,
            error_message TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Model training tracking
        CREATE TABLE IF NOT EXISTS model_training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            model_type TEXT NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'running',
            training_records INTEGER DEFAULT 0,
            validation_score REAL,
            test_score REAL,
            model_path TEXT,
            hyperparameters TEXT,
            feature_importance TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Alert history
        CREATE TABLE IF NOT EXISTS alert_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id TEXT UNIQUE NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            details TEXT,
            triggered_at TIMESTAMP NOT NULL,
            resolved_at TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'active',
            notification_sent BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- System performance metrics
        CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            metric_unit TEXT,
            tags TEXT,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Configuration change history
        CREATE TABLE IF NOT EXISTS configuration_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_section TEXT NOT NULL,
            config_key TEXT NOT NULL,
            old_value TEXT,
            new_value TEXT NOT NULL,
            changed_by TEXT,
            change_reason TEXT,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- File metadata and tracking
        CREATE TABLE IF NOT EXISTS file_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            checksum TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP,
            retention_until TIMESTAMP,
            compressed BOOLEAN DEFAULT FALSE
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_data_collection_source_time 
            ON data_collection_runs(data_source, start_time);
        CREATE INDEX IF NOT EXISTS idx_model_training_type_time 
            ON model_training_runs(model_type, start_time);
        CREATE INDEX IF NOT EXISTS idx_alert_type_time 
            ON alert_history(alert_type, triggered_at);
        CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time 
            ON system_metrics(metric_name, timestamp);
        CREATE INDEX IF NOT EXISTS idx_file_metadata_type 
            ON file_metadata(file_type, created_at);
        """
        
        with self.get_connection() as conn:
            # Split and execute each statement
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
            for statement in statements:
                conn.execute(statement)
            conn.commit()
            logger.info("Database schema initialized successfully")
    
    # Data Collection Operations
    def start_data_collection_run(self, run_id: str, data_source: str, metadata: Optional[Dict] = None) -> None:
        """Start a new data collection run."""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO data_collection_runs 
                (run_id, data_source, start_time, status, metadata)
                VALUES (?, ?, ?, 'running', ?)
            """, (run_id, data_source, datetime.utcnow(), str(metadata) if metadata else None))
            conn.commit()
    
    def complete_data_collection_run(self, run_id: str, records_collected: int, 
                                   file_path: Optional[str] = None, error_message: Optional[str] = None) -> None:
        """Complete a data collection run."""
        status = 'failed' if error_message else 'completed'
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE data_collection_runs 
                SET end_time = ?, status = ?, records_collected = ?, file_path = ?, error_message = ?
                WHERE run_id = ?
            """, (datetime.utcnow(), status, records_collected, file_path, error_message, run_id))
            conn.commit()
    
    def get_data_collection_history(self, data_source: Optional[str] = None, 
                                  limit: int = 100) -> List[Dict]:
        """Get data collection run history."""
        query = """
            SELECT * FROM data_collection_runs 
            WHERE (? IS NULL OR data_source = ?)
            ORDER BY start_time DESC LIMIT ?
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, (data_source, data_source, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    # Model Training Operations
    def start_model_training_run(self, run_id: str, model_type: str, training_records: int,
                               hyperparameters: Optional[Dict] = None) -> None:
        """Start a new model training run."""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO model_training_runs 
                (run_id, model_type, start_time, status, training_records, hyperparameters)
                VALUES (?, ?, ?, 'running', ?, ?)
            """, (run_id, model_type, datetime.utcnow(), training_records, 
                  str(hyperparameters) if hyperparameters else None))
            conn.commit()
    
    def complete_model_training_run(self, run_id: str, validation_score: Optional[float],
                                  test_score: Optional[float], model_path: Optional[str],
                                  feature_importance: Optional[Dict] = None,
                                  error_message: Optional[str] = None) -> None:
        """Complete a model training run."""
        status = 'failed' if error_message else 'completed'
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE model_training_runs 
                SET end_time = ?, status = ?, validation_score = ?, test_score = ?, 
                    model_path = ?, feature_importance = ?, error_message = ?
                WHERE run_id = ?
            """, (datetime.utcnow(), status, validation_score, test_score, model_path,
                  str(feature_importance) if feature_importance else None, error_message, run_id))
            conn.commit()
    
    # Alert Operations
    def create_alert(self, alert_id: str, alert_type: str, severity: str, 
                    message: str, details: Optional[Dict] = None) -> None:
        """Create a new alert."""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO alert_history 
                (alert_id, alert_type, severity, message, details, triggered_at, status)
                VALUES (?, ?, ?, ?, ?, ?, 'active')
            """, (alert_id, alert_type, severity, message, 
                  str(details) if details else None, datetime.utcnow()))
            conn.commit()
    
    def resolve_alert(self, alert_id: str) -> None:
        """Mark an alert as resolved."""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE alert_history 
                SET resolved_at = ?, status = 'resolved'
                WHERE alert_id = ? AND status = 'active'
            """, (datetime.utcnow(), alert_id))
            conn.commit()
    
    def get_active_alerts(self, alert_type: Optional[str] = None) -> List[Dict]:
        """Get active alerts."""
        query = """
            SELECT * FROM alert_history 
            WHERE status = 'active' 
            AND (? IS NULL OR alert_type = ?)
            ORDER BY triggered_at DESC
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, (alert_type, alert_type))
            return [dict(row) for row in cursor.fetchall()]
    
    # System Metrics Operations
    def record_metric(self, metric_name: str, metric_value: float, 
                     metric_unit: Optional[str] = None, tags: Optional[Dict] = None) -> None:
        """Record a system metric."""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO system_metrics 
                (metric_name, metric_value, metric_unit, tags, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (metric_name, metric_value, metric_unit,
                  str(tags) if tags else None, datetime.utcnow()))
            conn.commit()
    
    def get_metrics(self, metric_name: str, hours_back: int = 24) -> List[Dict]:
        """Get metrics for the specified time period."""
        query = """
            SELECT * FROM system_metrics 
            WHERE metric_name = ? 
            AND timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp ASC
        """.format(hours_back)
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, (metric_name,))
            return [dict(row) for row in cursor.fetchall()]
    
    # File Metadata Operations
    def register_file(self, file_path: str, file_type: str, file_size: int,
                     checksum: Optional[str] = None, retention_days: Optional[int] = None) -> None:
        """Register a file in the metadata store."""
        retention_until = None
        if retention_days:
            from datetime import timedelta
            retention_until = datetime.utcnow() + timedelta(days=retention_days)
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO file_metadata 
                (file_path, file_type, file_size, checksum, retention_until, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (file_path, file_type, file_size, checksum, retention_until, datetime.utcnow()))
            conn.commit()
    
    def get_files_for_cleanup(self) -> List[Dict]:
        """Get files that are eligible for cleanup."""
        query = """
            SELECT * FROM file_metadata 
            WHERE retention_until IS NOT NULL 
            AND retention_until < datetime('now')
            ORDER BY retention_until ASC
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]
    
    # Health and Statistics
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        stats = {}
        tables = [
            'data_collection_runs', 'model_training_runs', 'alert_history',
            'system_metrics', 'configuration_history', 'file_metadata'
        ]
        
        with self.get_connection() as conn:
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Database size
            cursor = conn.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor = conn.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            stats["database_size_bytes"] = page_count * page_size
        
        return stats
    
    def cleanup_old_records(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old records beyond retention period."""
        cleanup_stats = {}
        cutoff_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_to_keep)
        
        cleanup_queries = {
            'data_collection_runs': "DELETE FROM data_collection_runs WHERE created_at < ?",
            'model_training_runs': "DELETE FROM model_training_runs WHERE created_at < ?",
            'alert_history': "DELETE FROM alert_history WHERE created_at < ? AND status = 'resolved'",
            'system_metrics': "DELETE FROM system_metrics WHERE created_at < ?",
            'configuration_history': "DELETE FROM configuration_history WHERE created_at < ?"
        }
        
        with self.get_connection() as conn:
            for table, query in cleanup_queries.items():
                cursor = conn.execute(query, (cutoff_date,))
                cleanup_stats[f"{table}_deleted"] = cursor.rowcount
            conn.commit()
        
        return cleanup_stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Alias for get_database_stats for API compatibility."""
        return self.get_database_stats()


if __name__ == "__main__":
    # Test the SQLite store
    store = SQLiteStore("test_storage.db")
    
    # Test data collection tracking
    run_id = f"test_run_{datetime.utcnow().isoformat()}"
    store.start_data_collection_run(run_id, "warehouse_usage", {"test": True})
    store.complete_data_collection_run(run_id, 1000, "data/raw/test.parquet")
    
    # Test metrics recording
    store.record_metric("test_metric", 42.0, "count", {"component": "test"})
    
    # Get statistics
    stats = store.get_database_stats()
    print("Database Statistics:", stats)
    
    # Cleanup test database
    Path("test_storage.db").unlink(missing_ok=True)
    print("✅ SQLite store test completed successfully")
