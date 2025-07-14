"""
Cost Collector - Main collector for Snowflake cost and billing data.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

from ..connectors.connection_pool import ConnectionPool
from ..storage.sqlite_store import SQLiteStore
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CostCollector:
    """Main collector for cost analysis and tracking."""
    
    def __init__(self, connection_pool: ConnectionPool, storage: SQLiteStore, config: Optional[Dict] = None):
        self.connection_pool = connection_pool
        self.storage = storage
        self.config = config or {}
        self.lookback_days = self.config.get('lookback_days', 30)
        self.credit_rate = self.config.get('credit_rate_usd', 3.0)
    
    def collect_cost_data(self, days_back: Optional[int] = None) -> Dict[str, Any]:
        """Collect comprehensive cost data."""
        logger.info("Collecting cost data")
        start_time = datetime.now()
        
        days_back = days_back or self.lookback_days
        from_time = datetime.now() - timedelta(days=days_back)
        
        try:
            # Collect various cost data sources
            metering_data = self._collect_metering_history(from_time)
            storage_data = self._collect_storage_usage(from_time)
            
            results = {}
            total_records = 0
            
            if metering_data:
                df_metering = pd.DataFrame(metering_data)
                df_metering = self._process_metering_data(df_metering)
                metering_records = self._store_metering_data(df_metering)
                results['metering'] = {'records': metering_records, 'insights': self._analyze_metering_costs(df_metering)}
                total_records += metering_records
            
            if storage_data:
                df_storage = pd.DataFrame(storage_data)
                df_storage = self._process_storage_data(df_storage)
                storage_records = self._store_storage_data(df_storage)
                results['storage'] = {'records': storage_records, 'insights': self._analyze_storage_costs(df_storage)}
                total_records += storage_records
            
            return {
                'success': True,
                'records_collected': total_records,
                'collection_time_seconds': (datetime.now() - start_time).total_seconds(),
                'results': results,
                'cost_summary': self._generate_cost_summary(results)
            }
            
        except Exception as e:
            logger.error(f"Error collecting cost data: {e}")
            return {'success': False, 'error_message': str(e), 'records_collected': 0}
    
    def _collect_metering_history(self, from_time: datetime) -> List[Dict]:
        """Collect metering history for credit consumption."""
        query = f"""
        SELECT 
            START_TIME, END_TIME, ACCOUNT_NAME, SERVICE_TYPE,
            CREDITS_USED, CREDITS_USED_COMPUTE, CREDITS_USED_CLOUD_SERVICES,
            CREDITS_ADJUSTMENT_CLOUD_SERVICES
        FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_HISTORY 
        WHERE START_TIME >= '{from_time.strftime('%Y-%m-%d %H:%M:%S')}'::timestamp
        ORDER BY START_TIME DESC
        """
        
        with self.connection_pool.get_connection() as conn:
            return conn.execute_query(query)
    
    def _collect_storage_usage(self, from_time: datetime) -> List[Dict]:
        """Collect storage usage data."""
        query = f"""
        SELECT 
            USAGE_DATE, ACCOUNT_NAME, DATABASE_NAME, SCHEMA_NAME,
            AVERAGE_DATABASE_BYTES, AVERAGE_FAILSAFE_BYTES
        FROM SNOWFLAKE.ACCOUNT_USAGE.STORAGE_USAGE 
        WHERE USAGE_DATE >= '{from_time.strftime('%Y-%m-%d')}'::date
        ORDER BY USAGE_DATE DESC
        """
        
        with self.connection_pool.get_connection() as conn:
            return conn.execute_query(query)
    
    def _process_metering_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and enrich metering data."""
        df['START_TIME'] = pd.to_datetime(df['START_TIME'])
        df['END_TIME'] = pd.to_datetime(df['END_TIME'])
        
        # Calculate costs
        df['ESTIMATED_COST_USD'] = df['CREDITS_USED'] * self.credit_rate
        df['COMPUTE_COST_USD'] = df.get('CREDITS_USED_COMPUTE', 0) * self.credit_rate
        df['CLOUD_SERVICES_COST_USD'] = df.get('CREDITS_USED_CLOUD_SERVICES', 0) * self.credit_rate
        
        # Add time dimensions
        df['DATE'] = df['START_TIME'].dt.date
        df['HOUR'] = df['START_TIME'].dt.hour
        df['DAY_OF_WEEK'] = df['START_TIME'].dt.dayofweek
        
        return df
    
    def _process_storage_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process storage usage data."""
        df['USAGE_DATE'] = pd.to_datetime(df['USAGE_DATE'])
        
        # Calculate storage costs (approximate $40/TB/month)
        tb_rate_per_day = 40.0 / 30  # $40/TB/month to daily rate
        df['STORAGE_COST_USD'] = (df.get('AVERAGE_DATABASE_BYTES', 0) / (1024**4)) * tb_rate_per_day
        df['FAILSAFE_COST_USD'] = (df.get('AVERAGE_FAILSAFE_BYTES', 0) / (1024**4)) * tb_rate_per_day * 0.5
        df['TOTAL_STORAGE_COST_USD'] = df['STORAGE_COST_USD'] + df['FAILSAFE_COST_USD']
        
        return df
    
    def _store_metering_data(self, df: pd.DataFrame) -> int:
        """Store metering cost data."""
        if df.empty:
            return 0
        df['_collected_at'] = datetime.now()
        return self.storage.store_dataframe(df, 'cost_metering_metrics')
    
    def _store_storage_data(self, df: pd.DataFrame) -> int:
        """Store storage cost data."""
        if df.empty:
            return 0
        df['_collected_at'] = datetime.now()
        return self.storage.store_dataframe(df, 'cost_storage_metrics')
    
    def _analyze_metering_costs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze metering cost patterns."""
        if df.empty:
            return {}
        
        return {
            'total_cost': float(df['ESTIMATED_COST_USD'].sum()),
            'avg_daily_cost': float(df.groupby('DATE')['ESTIMATED_COST_USD'].sum().mean()),
            'cost_by_service': df.groupby('SERVICE_TYPE')['ESTIMATED_COST_USD'].sum().to_dict(),
            'cost_trend': df.groupby('DATE')['ESTIMATED_COST_USD'].sum().tail(7).to_dict()
        }
    
    def _analyze_storage_costs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze storage cost patterns."""
        if df.empty:
            return {}
        
        return {
            'total_storage_cost': float(df['TOTAL_STORAGE_COST_USD'].sum()),
            'avg_daily_storage_cost': float(df['TOTAL_STORAGE_COST_USD'].mean()),
            'cost_by_database': df.groupby('DATABASE_NAME')['TOTAL_STORAGE_COST_USD'].sum().to_dict(),
            'storage_trend': df.groupby('USAGE_DATE')['TOTAL_STORAGE_COST_USD'].sum().tail(7).to_dict()
        }
    
    def _generate_cost_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall cost summary."""
        total_compute_cost = results.get('metering', {}).get('insights', {}).get('total_cost', 0)
        total_storage_cost = results.get('storage', {}).get('insights', {}).get('total_storage_cost', 0)
        
        return {
            'total_cost_usd': total_compute_cost + total_storage_cost,
            'compute_cost_usd': total_compute_cost,
            'storage_cost_usd': total_storage_cost,
            'cost_breakdown_pct': {
                'compute': (total_compute_cost / (total_compute_cost + total_storage_cost)) * 100 if (total_compute_cost + total_storage_cost) > 0 else 0,
                'storage': (total_storage_cost / (total_compute_cost + total_storage_cost)) * 100 if (total_compute_cost + total_storage_cost) > 0 else 0
            }
        }
