"""
Schema Validator - Validates data schemas and ensures data consistency.

This module provides schema validation for all collected Snowflake data,
ensuring data integrity and consistency across collections.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FieldSchema:
    """Schema definition for a single field."""
    name: str
    data_type: str
    required: bool = True
    nullable: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    description: Optional[str] = None


@dataclass
class TableSchema:
    """Schema definition for a table."""
    table_name: str
    fields: List[FieldSchema] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    unique_constraints: List[List[str]] = field(default_factory=list)
    foreign_keys: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    table_name: str
    total_rows: int
    valid_rows: int
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.now)


class SchemaValidator:
    """
    Validates data schemas and ensures data consistency.
    
    Features:
    - Predefined schemas for Snowflake data tables
    - Field-level validation (type, range, pattern)
    - Table-level validation (keys, constraints)
    - Detailed error reporting and suggestions
    - Performance optimization for large datasets
    """
    
    def __init__(self):
        """Initialize schema validator with predefined schemas."""
        self.schemas = self._initialize_schemas()
        
    def _initialize_schemas(self) -> Dict[str, TableSchema]:
        """Initialize predefined schemas for Snowflake data tables."""
        schemas = {}
        
        # Warehouse usage schema
        schemas['warehouse_usage_raw'] = TableSchema(
            table_name='warehouse_usage_raw',
            fields=[
                FieldSchema('START_TIME', 'datetime', required=True),
                FieldSchema('END_TIME', 'datetime', required=True),
                FieldSchema('WAREHOUSE_ID', 'int', required=True, min_value=0),
                FieldSchema('WAREHOUSE_NAME', 'string', required=True),
                FieldSchema('CREDITS_USED', 'float', required=True, min_value=0),
                FieldSchema('CREDITS_USED_COMPUTE', 'float', required=False, min_value=0),
                FieldSchema('CREDITS_USED_CLOUD_SERVICES', 'float', required=False, min_value=0),
                FieldSchema('BYTES_SCANNED', 'int', required=False, min_value=0),
                FieldSchema('BYTES_WRITTEN', 'int', required=False, min_value=0),
            ],
            primary_keys=['START_TIME', 'WAREHOUSE_ID'],
            description='Warehouse usage and credit consumption data'
        )
        
        # Query history schema
        schemas['query_history_raw'] = TableSchema(
            table_name='query_history_raw',
            fields=[
                FieldSchema('QUERY_ID', 'string', required=True),
                FieldSchema('START_TIME', 'datetime', required=True),
                FieldSchema('END_TIME', 'datetime', nullable=True),
                FieldSchema('USER_NAME', 'string', required=True),
                FieldSchema('WAREHOUSE_NAME', 'string', nullable=True),
                FieldSchema('EXECUTION_STATUS', 'string', required=True, 
                          allowed_values=['SUCCESS', 'FAIL', 'CANCELLED', 'RUNNING']),
                FieldSchema('TOTAL_ELAPSED_TIME', 'int', required=False, min_value=0),
                FieldSchema('BYTES_SCANNED', 'int', required=False, min_value=0),
                FieldSchema('ROWS_PRODUCED', 'int', required=False, min_value=0),
            ],
            primary_keys=['QUERY_ID'],
            description='Query execution history and performance metrics'
        )
        
        # User activity schema
        schemas['user_activity_raw'] = TableSchema(
            table_name='user_activity_raw',
            fields=[
                FieldSchema('EVENT_ID', 'string', required=True),
                FieldSchema('EVENT_TIMESTAMP', 'datetime', required=True),
                FieldSchema('EVENT_TYPE', 'string', required=True),
                FieldSchema('USER_NAME', 'string', required=True),
                FieldSchema('IS_SUCCESS', 'string', required=True, allowed_values=['YES', 'NO']),
                FieldSchema('CLIENT_IP', 'string', nullable=True),
            ],
            primary_keys=['EVENT_ID'],
            description='User login and activity patterns'
        )
        
        # Query metrics schema
        schemas['query_metrics'] = TableSchema(
            table_name='query_metrics',
            fields=[
                FieldSchema('QUERY_ID', 'string', required=True),
                FieldSchema('QUERY_DURATION_MS', 'float', required=True, min_value=0),
                FieldSchema('PERFORMANCE_CLASS', 'string', nullable=True,
                          allowed_values=['Fast', 'Normal', 'Slow', 'Very Slow', 'Extremely Slow']),
                FieldSchema('CACHE_HIT_RATIO', 'float', required=False, min_value=0, max_value=1),
                FieldSchema('SPILLAGE_RATIO', 'float', required=False, min_value=0),
                FieldSchema('IS_RESOURCE_INTENSIVE', 'bool', required=False),
            ],
            primary_keys=['QUERY_ID'],
            description='Processed query performance metrics'
        )
        
        return schemas
    
    def validate_dataframe(self, df: pd.DataFrame, table_name: str) -> ValidationResult:
        """
        Validate a DataFrame against a predefined schema.
        
        Args:
            df: DataFrame to validate
            table_name: Name of the table schema to use
            
        Returns:
            ValidationResult: Detailed validation results
        """
        logger.debug(f"Validating DataFrame for table: {table_name}")
        
        if table_name not in self.schemas:
            return ValidationResult(
                is_valid=False,
                table_name=table_name,
                total_rows=len(df),
                valid_rows=0,
                errors=[{
                    'type': 'schema_not_found',
                    'message': f"No schema defined for table '{table_name}'"
                }]
            )
        
        schema = self.schemas[table_name]
        result = ValidationResult(
            is_valid=True,
            table_name=table_name,
            total_rows=len(df),
            valid_rows=len(df)
        )
        
        # Validate schema structure
        self._validate_table_structure(df, schema, result)
        
        # Validate field-level data
        if result.is_valid:
            self._validate_field_data(df, schema, result)
        
        # Validate table-level constraints
        if result.is_valid:
            self._validate_table_constraints(df, schema, result)
        
        # Calculate final validity
        result.is_valid = len(result.errors) == 0
        
        logger.debug(
            f"Validation complete: {result.valid_rows}/{result.total_rows} valid rows, "
            f"{len(result.errors)} errors, {len(result.warnings)} warnings"
        )
        
        return result
    
    def _validate_table_structure(self, df: pd.DataFrame, schema: TableSchema, result: ValidationResult):
        """Validate table structure (columns, etc.)."""
        required_fields = [f.name for f in schema.fields if f.required]
        missing_fields = [f for f in required_fields if f not in df.columns]
        extra_fields = [c for c in df.columns if c not in [f.name for f in schema.fields]]
        
        if missing_fields:
            result.errors.append({
                'type': 'missing_required_fields',
                'message': f"Missing required fields: {missing_fields}",
                'fields': missing_fields
            })
        
        if extra_fields:
            # Extra fields are warnings, not errors
            result.warnings.append({
                'type': 'extra_fields',
                'message': f"Extra fields found: {extra_fields}",
                'fields': extra_fields
            })
    
    def _validate_field_data(self, df: pd.DataFrame, schema: TableSchema, result: ValidationResult):
        """Validate field-level data types and constraints."""
        for field in schema.fields:
            if field.name not in df.columns:
                continue  # Already handled in structure validation
            
            field_data = df[field.name]
            field_errors = []
            
            # Validate data type
            type_errors = self._validate_field_type(field_data, field)
            field_errors.extend(type_errors)
            
            # Validate nullability
            null_errors = self._validate_field_nulls(field_data, field)
            field_errors.extend(null_errors)
            
            # Validate value constraints
            value_errors = self._validate_field_values(field_data, field)
            field_errors.extend(value_errors)
            
            if field_errors:
                result.errors.extend([{
                    'type': 'field_validation',
                    'field': field.name,
                    'message': error,
                    'affected_rows': self._count_affected_rows(field_data, error)
                } for error in field_errors])
    
    def _validate_field_type(self, field_data: pd.Series, field: FieldSchema) -> List[str]:
        """Validate field data type."""
        errors = []
        
        if field.data_type == 'datetime':
            try:
                pd.to_datetime(field_data, errors='coerce')
                invalid_count = field_data.isna().sum()
                if invalid_count > 0:
                    errors.append(f"Invalid datetime values: {invalid_count} rows")
            except Exception as e:
                errors.append(f"Datetime conversion failed: {e}")
        
        elif field.data_type == 'int':
            try:
                numeric_data = pd.to_numeric(field_data, errors='coerce')
                invalid_count = numeric_data.isna().sum() - field_data.isna().sum()
                if invalid_count > 0:
                    errors.append(f"Non-integer values: {invalid_count} rows")
            except Exception as e:
                errors.append(f"Integer validation failed: {e}")
        
        elif field.data_type == 'float':
            try:
                numeric_data = pd.to_numeric(field_data, errors='coerce')
                invalid_count = numeric_data.isna().sum() - field_data.isna().sum()
                if invalid_count > 0:
                    errors.append(f"Non-numeric values: {invalid_count} rows")
            except Exception as e:
                errors.append(f"Float validation failed: {e}")
        
        elif field.data_type == 'bool':
            if not field_data.dtype == bool:
                bool_values = field_data.astype(str).str.lower().isin(['true', 'false', '1', '0', 'yes', 'no'])
                invalid_count = (~bool_values & ~field_data.isna()).sum()
                if invalid_count > 0:
                    errors.append(f"Non-boolean values: {invalid_count} rows")
        
        return errors
    
    def _validate_field_nulls(self, field_data: pd.Series, field: FieldSchema) -> List[str]:
        """Validate field nullability constraints."""
        errors = []
        
        if field.required and not field.nullable:
            null_count = field_data.isna().sum()
            if null_count > 0:
                errors.append(f"Required field has null values: {null_count} rows")
        
        return errors
    
    def _validate_field_values(self, field_data: pd.Series, field: FieldSchema) -> List[str]:
        """Validate field value constraints."""
        errors = []
        
        # Validate allowed values
        if field.allowed_values:
            invalid_values = ~field_data.isin(field.allowed_values + [None, np.nan])
            invalid_count = invalid_values.sum()
            if invalid_count > 0:
                errors.append(f"Invalid values (not in allowed list): {invalid_count} rows")
        
        # Validate numeric ranges
        if field.min_value is not None:
            try:
                numeric_data = pd.to_numeric(field_data, errors='coerce')
                below_min = (numeric_data < field.min_value) & ~numeric_data.isna()
                below_min_count = below_min.sum()
                if below_min_count > 0:
                    errors.append(f"Values below minimum ({field.min_value}): {below_min_count} rows")
            except:
                pass  # Type validation will catch this
        
        if field.max_value is not None:
            try:
                numeric_data = pd.to_numeric(field_data, errors='coerce')
                above_max = (numeric_data > field.max_value) & ~numeric_data.isna()
                above_max_count = above_max.sum()
                if above_max_count > 0:
                    errors.append(f"Values above maximum ({field.max_value}): {above_max_count} rows")
            except:
                pass  # Type validation will catch this
        
        # Validate patterns (for string fields)
        if field.pattern and field.data_type == 'string':
            try:
                import re
                pattern_match = field_data.astype(str).str.match(field.pattern, na=False)
                invalid_pattern_count = (~pattern_match & ~field_data.isna()).sum()
                if invalid_pattern_count > 0:
                    errors.append(f"Values not matching pattern: {invalid_pattern_count} rows")
            except Exception as e:
                errors.append(f"Pattern validation failed: {e}")
        
        return errors
    
    def _validate_table_constraints(self, df: pd.DataFrame, schema: TableSchema, result: ValidationResult):
        """Validate table-level constraints."""
        # Validate primary key uniqueness
        if schema.primary_keys:
            pk_columns = [col for col in schema.primary_keys if col in df.columns]
            if pk_columns:
                duplicate_count = df.duplicated(subset=pk_columns).sum()
                if duplicate_count > 0:
                    result.errors.append({
                        'type': 'primary_key_violation',
                        'message': f"Duplicate primary key values: {duplicate_count} rows",
                        'affected_rows': duplicate_count
                    })
        
        # Validate unique constraints
        for unique_constraint in schema.unique_constraints:
            constraint_columns = [col for col in unique_constraint if col in df.columns]
            if constraint_columns:
                duplicate_count = df.duplicated(subset=constraint_columns).sum()
                if duplicate_count > 0:
                    result.warnings.append({
                        'type': 'unique_constraint_violation',
                        'message': f"Duplicate values in unique constraint {constraint_columns}: {duplicate_count} rows",
                        'affected_rows': duplicate_count
                    })
    
    def _count_affected_rows(self, field_data: pd.Series, error_message: str) -> int:
        """Count affected rows for an error (simplified implementation)."""
        # This could be more sophisticated to actually count the specific issue
        return 1
    
    def add_custom_schema(self, schema: TableSchema):
        """Add a custom schema definition."""
        self.schemas[schema.table_name] = schema
        logger.info(f"Added custom schema for table: {schema.table_name}")
    
    def get_schema(self, table_name: str) -> Optional[TableSchema]:
        """Get schema definition for a table."""
        return self.schemas.get(table_name)
    
    def list_available_schemas(self) -> List[str]:
        """List all available schema names."""
        return list(self.schemas.keys())
    
    def validate_and_clean_dataframe(self, df: pd.DataFrame, table_name: str) -> tuple[pd.DataFrame, ValidationResult]:
        """
        Validate DataFrame and return cleaned version with validation results.
        
        Args:
            df: DataFrame to validate and clean
            table_name: Table schema to use
            
        Returns:
            Tuple of (cleaned_dataframe, validation_result)
        """
        validation_result = self.validate_dataframe(df, table_name)
        
        if not validation_result.is_valid:
            logger.warning(f"Validation failed for {table_name}: {len(validation_result.errors)} errors")
            return df, validation_result
        
        # Perform cleaning based on schema
        cleaned_df = df.copy()
        
        if table_name in self.schemas:
            schema = self.schemas[table_name]
            
            # Convert data types
            for field in schema.fields:
                if field.name in cleaned_df.columns:
                    cleaned_df = self._convert_field_type(cleaned_df, field)
        
        validation_result.valid_rows = len(cleaned_df)
        
        return cleaned_df, validation_result
    
    def _convert_field_type(self, df: pd.DataFrame, field: FieldSchema) -> pd.DataFrame:
        """Convert field to proper data type."""
        try:
            if field.data_type == 'datetime':
                df[field.name] = pd.to_datetime(df[field.name], errors='coerce')
            elif field.data_type == 'int':
                df[field.name] = pd.to_numeric(df[field.name], errors='coerce').astype('Int64')
            elif field.data_type == 'float':
                df[field.name] = pd.to_numeric(df[field.name], errors='coerce')
            elif field.data_type == 'bool':
                df[field.name] = df[field.name].astype(bool)
            elif field.data_type == 'string':
                df[field.name] = df[field.name].astype(str)
        except Exception as e:
            logger.warning(f"Failed to convert {field.name} to {field.data_type}: {e}")
        
        return df
