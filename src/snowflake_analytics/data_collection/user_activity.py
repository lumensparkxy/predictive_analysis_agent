"""
User Activity Collector - Tracks user login patterns and behavior analysis.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

from ..connectors.connection_pool import ConnectionPool
from ..storage.sqlite_store import SQLiteStore
from ..utils.logger import get_logger

logger = get_logger(__name__)


class UserActivityCollector:
    """Collector for user activity and login patterns."""
    
    def __init__(self, connection_pool: ConnectionPool, storage: SQLiteStore, config: Optional[Dict] = None):
        self.connection_pool = connection_pool
        self.storage = storage
        self.config = config or {}
        self.lookback_days = self.config.get('lookback_days', 7)
    
    def collect_user_activity(self, days_back: Optional[int] = None) -> Dict[str, Any]:
        """Collect user activity and login data."""
        logger.info("Collecting user activity data")
        start_time = datetime.now()
        
        days_back = days_back or self.lookback_days
        from_time = datetime.now() - timedelta(days=days_back)
        
        try:
            # Collect login history
            login_data = self._collect_login_history(from_time)
            
            if not login_data:
                return {'success': True, 'records_collected': 0, 'message': 'No login data available'}
            
            df = pd.DataFrame(login_data)
            df = self._clean_and_analyze_activity(df)
            
            records_stored = self._store_activity_data(df)
            insights = self._analyze_user_patterns(df)
            
            return {
                'success': True,
                'records_collected': records_stored,
                'collection_time_seconds': (datetime.now() - start_time).total_seconds(),
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error collecting user activity: {e}")
            return {'success': False, 'error_message': str(e), 'records_collected': 0}
    
    def _collect_login_history(self, from_time: datetime) -> List[Dict]:
        """Collect login history from Snowflake."""
        query = f"""
        SELECT 
            EVENT_ID, EVENT_TIMESTAMP, EVENT_TYPE, USER_NAME, CLIENT_IP,
            REPORTED_CLIENT_TYPE, REPORTED_CLIENT_VERSION,
            FIRST_AUTHENTICATION_FACTOR, SECOND_AUTHENTICATION_FACTOR,
            IS_SUCCESS, ERROR_CODE, ERROR_MESSAGE
        FROM SNOWFLAKE.ACCOUNT_USAGE.LOGIN_HISTORY 
        WHERE EVENT_TIMESTAMP >= '{from_time.strftime('%Y-%m-%d %H:%M:%S')}'::timestamp
        ORDER BY EVENT_TIMESTAMP DESC
        """
        
        with self.connection_pool.get_connection() as conn:
            return conn.execute_query(query)
    
    def _clean_and_analyze_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean activity data and add analysis fields."""
        # Remove duplicates and clean data
        df = df.drop_duplicates(subset=['EVENT_ID'])
        df['EVENT_TIMESTAMP'] = pd.to_datetime(df['EVENT_TIMESTAMP'])
        
        # Add time-based analysis fields
        df['HOUR_OF_DAY'] = df['EVENT_TIMESTAMP'].dt.hour
        df['DAY_OF_WEEK'] = df['EVENT_TIMESTAMP'].dt.dayofweek
        df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6])
        df['IS_BUSINESS_HOURS'] = df['HOUR_OF_DAY'].between(8, 18)
        
        # Categorize client types
        df['CLIENT_CATEGORY'] = df['REPORTED_CLIENT_TYPE'].map({
            'JDBC_DRIVER': 'Application',
            'ODBC_DRIVER': 'Application', 
            'PYTHON_CONNECTOR': 'Script/API',
            'WEB_UI': 'Web Interface',
            'SNOWSQL': 'CLI Tool'
        }).fillna('Other')
        
        return df
    
    def _store_activity_data(self, df: pd.DataFrame) -> int:
        """Store user activity data."""
        if df.empty:
            return 0
        df['_collected_at'] = datetime.now()
        return self.storage.store_dataframe(df, 'user_activity_metrics')
    
    def _analyze_user_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user activity patterns."""
        if df.empty:
            return {}
        
        return {
            'summary': {
                'total_logins': len(df),
                'unique_users': df['USER_NAME'].nunique(),
                'success_rate': (df['IS_SUCCESS'] == 'YES').mean(),
                'weekend_activity_ratio': df['IS_WEEKEND'].mean(),
                'business_hours_ratio': df['IS_BUSINESS_HOURS'].mean()
            },
            'top_users': df['USER_NAME'].value_counts().head(10).to_dict(),
            'client_usage': df['CLIENT_CATEGORY'].value_counts().to_dict(),
            'hourly_pattern': df.groupby('HOUR_OF_DAY').size().to_dict(),
            'failed_logins': df[df['IS_SUCCESS'] == 'NO'].groupby('USER_NAME').size().head(10).to_dict()
        }
