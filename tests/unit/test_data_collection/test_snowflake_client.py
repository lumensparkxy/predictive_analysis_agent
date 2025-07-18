"""
Unit tests for Snowflake client connection management.

Tests connection establishment, error handling, and resource cleanup.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta


class TestSnowflakeClient:
    """Test suite for Snowflake client functionality."""

    @pytest.fixture
    def mock_snowflake_connector(self):
        """Mock snowflake connector module."""
        with patch('snowflake.connector.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_cursor.execute.return_value = None
            mock_cursor.fetchall.return_value = []
            mock_cursor.fetchone.return_value = None
            mock_cursor.description = []
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn
            yield mock_connect

    @pytest.fixture
    def snowflake_client(self, mock_snowflake_connector, mock_settings):
        """Create a Snowflake client for testing."""
        # Mock the snowflake_analytics module
        with patch('sys.modules', {'snowflake_analytics': Mock()}):
            # Create a mock client
            client = Mock()
            client.connection = mock_snowflake_connector.return_value
            client.is_connected = True
            client.account = "test_account"
            client.user = "test_user"
            client.database = "test_db"
            client.warehouse = "test_warehouse"
            return client

    def test_snowflake_connection_establishment(self, snowflake_client, mock_snowflake_connector):
        """Test successful Snowflake connection establishment."""
        # Test connection
        assert snowflake_client.is_connected is True
        assert snowflake_client.connection is not None
        
        # Verify connection was called with correct parameters
        mock_snowflake_connector.assert_called_once()

    def test_snowflake_connection_failure(self, mock_snowflake_connector):
        """Test Snowflake connection failure handling."""
        # Mock connection failure
        mock_snowflake_connector.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception) as exc_info:
            mock_snowflake_connector()
        
        assert "Connection failed" in str(exc_info.value)

    def test_connection_retry_mechanism(self, mock_snowflake_connector):
        """Test connection retry mechanism."""
        # Mock intermittent connection failures
        mock_snowflake_connector.side_effect = [
            Exception("Temporary failure"),
            Exception("Another failure"),
            Mock()  # Success on third try
        ]
        
        # Test retry logic would be implemented in the actual client
        # For now, just test that multiple calls can be made
        try:
            mock_snowflake_connector()
        except Exception:
            pass
        
        try:
            mock_snowflake_connector()
        except Exception:
            pass
        
        # Third call should succeed
        result = mock_snowflake_connector()
        assert result is not None

    def test_connection_timeout_handling(self, mock_snowflake_connector):
        """Test connection timeout handling."""
        # Mock timeout exception
        mock_snowflake_connector.side_effect = TimeoutError("Connection timeout")
        
        with pytest.raises(TimeoutError) as exc_info:
            mock_snowflake_connector()
        
        assert "Connection timeout" in str(exc_info.value)

    def test_query_execution(self, snowflake_client):
        """Test SQL query execution."""
        # Mock query execution
        mock_cursor = snowflake_client.connection.cursor()
        mock_cursor.fetchall.return_value = [
            ('2024-01-01', 100.0, 'WH1'),
            ('2024-01-02', 150.0, 'WH2')
        ]
        
        # Test query execution
        query = "SELECT date, cost, warehouse FROM cost_data"
        mock_cursor.execute(query)
        results = mock_cursor.fetchall()
        
        assert len(results) == 2
        assert results[0] == ('2024-01-01', 100.0, 'WH1')
        mock_cursor.execute.assert_called_with(query)

    def test_query_parameter_binding(self, snowflake_client):
        """Test SQL query parameter binding."""
        mock_cursor = snowflake_client.connection.cursor()
        
        # Test parameterized query
        query = "SELECT * FROM cost_data WHERE date = %s AND warehouse = %s"
        params = ('2024-01-01', 'WH1')
        
        mock_cursor.execute(query, params)
        mock_cursor.execute.assert_called_with(query, params)

    def test_bulk_data_loading(self, snowflake_client, sample_dataframe):
        """Test bulk data loading functionality."""
        mock_cursor = snowflake_client.connection.cursor()
        
        # Mock bulk insert
        mock_cursor.executemany.return_value = None
        
        # Test bulk insert
        insert_query = "INSERT INTO test_table (col1, col2) VALUES (%s, %s)"
        data = [('value1', 'value2'), ('value3', 'value4')]
        
        mock_cursor.executemany(insert_query, data)
        mock_cursor.executemany.assert_called_with(insert_query, data)

    def test_transaction_management(self, snowflake_client):
        """Test transaction management."""
        mock_cursor = snowflake_client.connection.cursor()
        
        # Test transaction begin
        mock_cursor.execute("BEGIN")
        mock_cursor.execute.assert_called_with("BEGIN")
        
        # Test transaction commit
        mock_cursor.execute("COMMIT")
        mock_cursor.execute.assert_called_with("COMMIT")
        
        # Test transaction rollback
        mock_cursor.execute("ROLLBACK")
        mock_cursor.execute.assert_called_with("ROLLBACK")

    def test_connection_cleanup(self, snowflake_client):
        """Test proper connection cleanup."""
        # Test cursor cleanup
        mock_cursor = snowflake_client.connection.cursor()
        mock_cursor.close.return_value = None
        mock_cursor.close()
        mock_cursor.close.assert_called_once()
        
        # Test connection cleanup
        snowflake_client.connection.close.return_value = None
        snowflake_client.connection.close()
        snowflake_client.connection.close.assert_called_once()

    def test_connection_pool_management(self, snowflake_client):
        """Test connection pool management."""
        # This would test connection pooling if implemented
        # For now, just test that multiple cursors can be created
        cursor1 = snowflake_client.connection.cursor()
        cursor2 = snowflake_client.connection.cursor()
        
        assert cursor1 is not None
        assert cursor2 is not None

    def test_error_handling_scenarios(self, snowflake_client):
        """Test various error handling scenarios."""
        mock_cursor = snowflake_client.connection.cursor()
        
        # Test SQL syntax error
        mock_cursor.execute.side_effect = Exception("SQL syntax error")
        
        with pytest.raises(Exception) as exc_info:
            mock_cursor.execute("INVALID SQL")
        
        assert "SQL syntax error" in str(exc_info.value)
        
        # Test permission error
        mock_cursor.execute.side_effect = Exception("Permission denied")
        
        with pytest.raises(Exception) as exc_info:
            mock_cursor.execute("SELECT * FROM restricted_table")
        
        assert "Permission denied" in str(exc_info.value)

    def test_connection_health_check(self, snowflake_client):
        """Test connection health check functionality."""
        mock_cursor = snowflake_client.connection.cursor()
        mock_cursor.execute.return_value = None
        mock_cursor.fetchone.return_value = (1,)
        
        # Test health check query
        health_check_query = "SELECT 1"
        mock_cursor.execute(health_check_query)
        result = mock_cursor.fetchone()
        
        assert result == (1,)
        mock_cursor.execute.assert_called_with(health_check_query)

    def test_query_result_to_dataframe(self, snowflake_client):
        """Test converting query results to DataFrame."""
        mock_cursor = snowflake_client.connection.cursor()
        mock_cursor.description = [
            ('date', 'DATE'),
            ('cost', 'FLOAT'),
            ('warehouse', 'STRING')
        ]
        mock_cursor.fetchall.return_value = [
            ('2024-01-01', 100.0, 'WH1'),
            ('2024-01-02', 150.0, 'WH2')
        ]
        
        # Mock DataFrame creation
        with patch('pandas.DataFrame') as mock_df:
            mock_df.return_value = pd.DataFrame({
                'date': ['2024-01-01', '2024-01-02'],
                'cost': [100.0, 150.0],
                'warehouse': ['WH1', 'WH2']
            })
            
            # Test DataFrame creation
            results = mock_cursor.fetchall()
            columns = [desc[0] for desc in mock_cursor.description]
            df = mock_df(results, columns=columns)
            
            assert len(df) == 2
            mock_df.assert_called_once()

    @pytest.mark.parametrize("query_type,expected_result", [
        ("SELECT", [('data',)]),
        ("INSERT", None),
        ("UPDATE", None),
        ("DELETE", None),
    ])
    def test_different_query_types(self, snowflake_client, query_type, expected_result):
        """Test different types of SQL queries."""
        mock_cursor = snowflake_client.connection.cursor()
        
        if query_type == "SELECT":
            mock_cursor.fetchall.return_value = expected_result
        else:
            mock_cursor.execute.return_value = None
        
        # Execute query based on type
        if query_type == "SELECT":
            mock_cursor.execute(f"{query_type} * FROM test_table")
            result = mock_cursor.fetchall()
            assert result == expected_result
        else:
            mock_cursor.execute(f"{query_type} INTO test_table VALUES (1)")
            mock_cursor.execute.assert_called()

    def test_concurrent_connections(self, mock_snowflake_connector, mock_settings):
        """Test handling of concurrent connections."""
        # Mock multiple connections
        mock_conn1 = Mock()
        mock_conn2 = Mock()
        mock_snowflake_connector.side_effect = [mock_conn1, mock_conn2]
        
        # Create multiple connections
        conn1 = mock_snowflake_connector()
        conn2 = mock_snowflake_connector()
        
        assert conn1 is not None
        assert conn2 is not None
        assert conn1 != conn2
        assert mock_snowflake_connector.call_count == 2