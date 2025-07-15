"""
ML Pipeline Orchestrator

Comprehensive orchestrator for managing the end-to-end ML data processing pipeline,
coordinating all stages from data ingestion through ML-ready feature preparation.

Key capabilities:
- End-to-end pipeline execution coordination
- Stage-by-stage progress tracking
- Resource and performance monitoring
- Error handling and recovery
- Checkpointing and resume capabilities
- Parallel and sequential execution modes
- Comprehensive logging and metrics
- Integration with all pipeline components
"""

import os
import json
import time
import asyncio
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np
from snowflake.connector import connect

from ...config.settings import SnowflakeSettings
from ...utils.logger import SnowflakeLogger
from ..cleaning.data_cleaner import DataCleaner
from ..feature_engineering.feature_pipeline import FeaturePipeline
from ..aggregation.aggregation_pipeline import AggregationPipeline
from ..validation.validation_pipeline import ValidationPipeline


class PipelineStage(Enum):
    """Pipeline execution stages"""
    DATA_INGESTION = "data_ingestion"
    DATA_CLEANING = "data_cleaning"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_AGGREGATION = "data_aggregation"
    DATA_VALIDATION = "data_validation"
    ML_PREPARATION = "ml_preparation"


class ExecutionMode(Enum):
    """Pipeline execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


class PipelineStatus(Enum):
    """Pipeline execution status"""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    RESUMED = "resumed"


@dataclass
class StageResult:
    """Result from pipeline stage execution"""
    stage: PipelineStage
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    records_processed: int
    output_path: Optional[str]
    metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


@dataclass
class PipelineExecution:
    """Complete pipeline execution tracking"""
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_duration_seconds: Optional[float]
    status: PipelineStatus
    mode: ExecutionMode
    stages_completed: List[PipelineStage]
    current_stage: Optional[PipelineStage]
    stage_results: Dict[PipelineStage, StageResult]
    total_records_processed: int
    overall_metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


class MLPipelineOrchestrator:
    """
    Comprehensive orchestrator for ML data processing pipeline.
    
    Coordinates execution of all pipeline stages with monitoring,
    error handling, and performance optimization.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        max_workers: int = 4,
        checkpoint_dir: Optional[str] = None,
        metrics_dir: Optional[str] = None
    ):
        """Initialize pipeline orchestrator"""
        
        # Core configuration
        self.settings = SnowflakeSettings(config_path)
        self.logger = SnowflakeLogger("MLPipelineOrchestrator").get_logger()
        
        # Execution configuration
        self.execution_mode = execution_mode
        self.max_workers = max_workers
        
        # Directory setup
        self.checkpoint_dir = Path(checkpoint_dir or "data/checkpoints")
        self.metrics_dir = Path(metrics_dir or "data/metrics")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline components
        self._initialize_components()
        
        # Execution tracking
        self.current_execution: Optional[PipelineExecution] = None
        self.execution_history: List[PipelineExecution] = []
        
        # Stage configuration
        self.stage_config = {
            PipelineStage.DATA_INGESTION: {
                'component': None,  # Will be set when implemented
                'parallel_capable': True,
                'resource_intensive': False,
                'dependencies': []
            },
            PipelineStage.DATA_CLEANING: {
                'component': self.data_cleaner,
                'parallel_capable': True,
                'resource_intensive': True,
                'dependencies': [PipelineStage.DATA_INGESTION]
            },
            PipelineStage.FEATURE_ENGINEERING: {
                'component': self.feature_pipeline,
                'parallel_capable': True,
                'resource_intensive': True,
                'dependencies': [PipelineStage.DATA_CLEANING]
            },
            PipelineStage.DATA_AGGREGATION: {
                'component': self.aggregation_pipeline,
                'parallel_capable': True,
                'resource_intensive': True,
                'dependencies': [PipelineStage.FEATURE_ENGINEERING]
            },
            PipelineStage.DATA_VALIDATION: {
                'component': self.validation_pipeline,
                'parallel_capable': False,
                'resource_intensive': False,
                'dependencies': [PipelineStage.DATA_AGGREGATION]
            },
            PipelineStage.ML_PREPARATION: {
                'component': None,  # Will be implemented in feature store
                'parallel_capable': False,
                'resource_intensive': False,
                'dependencies': [PipelineStage.DATA_VALIDATION]
            }
        }
        
        self.logger.info("MLPipelineOrchestrator initialized successfully")
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            # Initialize pipeline components
            self.data_cleaner = DataCleaner()
            self.feature_pipeline = FeaturePipeline()
            self.aggregation_pipeline = AggregationPipeline()
            self.validation_pipeline = ValidationPipeline()
            
            self.logger.info("All pipeline components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {str(e)}")
            raise
    
    def execute_pipeline(
        self,
        input_data: Union[pd.DataFrame, str, Dict[str, Any]],
        stages: Optional[List[PipelineStage]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        resume_from_checkpoint: bool = False,
        save_checkpoints: bool = True
    ) -> PipelineExecution:
        """
        Execute complete ML pipeline or specific stages
        
        Args:
            input_data: Input data (DataFrame, file path, or config)
            stages: Specific stages to execute (None for all)
            config_overrides: Override default configurations
            resume_from_checkpoint: Resume from last checkpoint
            save_checkpoints: Save intermediate results
            
        Returns:
            PipelineExecution: Complete execution results
        """
        
        execution_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Initialize execution tracking
            self.current_execution = PipelineExecution(
                execution_id=execution_id,
                start_time=datetime.now(),
                end_time=None,
                total_duration_seconds=None,
                status=PipelineStatus.RUNNING,
                mode=self.execution_mode,
                stages_completed=[],
                current_stage=None,
                stage_results={},
                total_records_processed=0,
                overall_metrics={},
                errors=[],
                warnings=[]
            )
            
            self.logger.info(f"Starting pipeline execution: {execution_id}")
            
            # Determine stages to execute
            if stages is None:
                stages = list(PipelineStage)
            
            # Resume from checkpoint if requested
            if resume_from_checkpoint:
                checkpoint = self._load_checkpoint(execution_id)
                if checkpoint:
                    self._resume_from_checkpoint(checkpoint)
            
            # Execute stages based on mode
            if self.execution_mode == ExecutionMode.SEQUENTIAL:
                self._execute_sequential(input_data, stages, config_overrides, save_checkpoints)
            elif self.execution_mode == ExecutionMode.PARALLEL:
                self._execute_parallel(input_data, stages, config_overrides, save_checkpoints)
            else:  # HYBRID
                self._execute_hybrid(input_data, stages, config_overrides, save_checkpoints)
            
            # Finalize execution
            self.current_execution.end_time = datetime.now()
            self.current_execution.total_duration_seconds = (
                self.current_execution.end_time - self.current_execution.start_time
            ).total_seconds()
            self.current_execution.status = PipelineStatus.COMPLETED
            
            # Calculate overall metrics
            self._calculate_overall_metrics()
            
            # Save execution results
            self._save_execution_results()
            
            self.logger.info(f"Pipeline execution completed successfully: {execution_id}")
            
        except Exception as e:
            self._handle_execution_error(e)
            
        finally:
            # Add to history
            if self.current_execution:
                self.execution_history.append(self.current_execution)
        
        return self.current_execution
    
    def _execute_sequential(
        self,
        input_data: Union[pd.DataFrame, str, Dict[str, Any]],
        stages: List[PipelineStage],
        config_overrides: Optional[Dict[str, Any]],
        save_checkpoints: bool
    ):
        """Execute pipeline stages sequentially"""
        
        current_data = input_data
        
        for stage in stages:
            try:
                self.current_execution.current_stage = stage
                self.logger.info(f"Executing stage: {stage.value}")
                
                # Execute stage
                stage_result = self._execute_stage(stage, current_data, config_overrides)
                
                # Update execution tracking
                self.current_execution.stage_results[stage] = stage_result
                self.current_execution.stages_completed.append(stage)
                self.current_execution.total_records_processed += stage_result.records_processed
                
                # Use output as input for next stage
                if stage_result.output_path:
                    current_data = stage_result.output_path
                
                # Save checkpoint if enabled
                if save_checkpoints:
                    self._save_checkpoint(stage, stage_result)
                
                self.logger.info(f"Stage {stage.value} completed successfully")
                
            except Exception as e:
                self.logger.error(f"Stage {stage.value} failed: {str(e)}")
                self.current_execution.errors.append(f"Stage {stage.value}: {str(e)}")
                self.current_execution.status = PipelineStatus.FAILED
                raise
    
    def _execute_parallel(
        self,
        input_data: Union[pd.DataFrame, str, Dict[str, Any]],
        stages: List[PipelineStage],
        config_overrides: Optional[Dict[str, Any]],
        save_checkpoints: bool
    ):
        """Execute pipeline stages in parallel where possible"""
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(stages)
        
        # Execute stages in dependency order with parallelization
        executed_stages = set()
        current_data = input_data
        
        while len(executed_stages) < len(stages):
            # Find stages ready for execution
            ready_stages = []
            for stage in stages:
                if stage not in executed_stages:
                    dependencies = self.stage_config[stage]['dependencies']
                    if all(dep in executed_stages for dep in dependencies):
                        ready_stages.append(stage)
            
            if not ready_stages:
                raise RuntimeError("Circular dependency detected in pipeline stages")
            
            # Execute ready stages in parallel if possible
            if len(ready_stages) == 1 or not all(
                self.stage_config[stage]['parallel_capable'] for stage in ready_stages
            ):
                # Sequential execution for non-parallel stages
                for stage in ready_stages:
                    stage_result = self._execute_stage(stage, current_data, config_overrides)
                    self._process_stage_result(stage, stage_result, save_checkpoints)
                    executed_stages.add(stage)
                    if stage_result.output_path:
                        current_data = stage_result.output_path
            else:
                # Parallel execution
                with ThreadPoolExecutor(max_workers=min(len(ready_stages), self.max_workers)) as executor:
                    futures = {
                        executor.submit(self._execute_stage, stage, current_data, config_overrides): stage
                        for stage in ready_stages
                    }
                    
                    for future in futures:
                        stage = futures[future]
                        try:
                            stage_result = future.result()
                            self._process_stage_result(stage, stage_result, save_checkpoints)
                            executed_stages.add(stage)
                        except Exception as e:
                            self.logger.error(f"Parallel stage {stage.value} failed: {str(e)}")
                            raise
    
    def _execute_hybrid(
        self,
        input_data: Union[pd.DataFrame, str, Dict[str, Any]],
        stages: List[PipelineStage],
        config_overrides: Optional[Dict[str, Any]],
        save_checkpoints: bool
    ):
        """Execute pipeline with hybrid approach (parallel where beneficial)"""
        
        # Identify resource-intensive stages for parallel execution
        parallel_stages = [
            stage for stage in stages 
            if self.stage_config[stage]['parallel_capable'] and 
               self.stage_config[stage]['resource_intensive']
        ]
        
        sequential_stages = [stage for stage in stages if stage not in parallel_stages]
        
        # Execute based on stage characteristics
        current_data = input_data
        
        for stage in stages:
            if stage in parallel_stages and self._should_execute_parallel(stage, current_data):
                # Use parallel execution for resource-intensive stages
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future = executor.submit(self._execute_stage, stage, current_data, config_overrides)
                    stage_result = future.result()
            else:
                # Use sequential execution
                stage_result = self._execute_stage(stage, current_data, config_overrides)
            
            self._process_stage_result(stage, stage_result, save_checkpoints)
            
            if stage_result.output_path:
                current_data = stage_result.output_path
    
    def _execute_stage(
        self,
        stage: PipelineStage,
        input_data: Union[pd.DataFrame, str, Dict[str, Any]],
        config_overrides: Optional[Dict[str, Any]]
    ) -> StageResult:
        """Execute individual pipeline stage"""
        
        start_time = datetime.now()
        stage_result = StageResult(
            stage=stage,
            status=PipelineStatus.RUNNING,
            start_time=start_time,
            end_time=None,
            duration_seconds=None,
            records_processed=0,
            output_path=None,
            metrics={},
            errors=[],
            warnings=[]
        )
        
        try:
            self.logger.info(f"Executing stage: {stage.value}")
            
            # Get stage component
            component = self.stage_config[stage]['component']
            if component is None:
                raise NotImplementedError(f"Stage {stage.value} not yet implemented")
            
            # Prepare stage-specific configuration
            stage_config = self._prepare_stage_config(stage, config_overrides)
            
            # Execute stage based on type
            if stage == PipelineStage.DATA_CLEANING:
                result = self._execute_cleaning_stage(component, input_data, stage_config)
            elif stage == PipelineStage.FEATURE_ENGINEERING:
                result = self._execute_feature_engineering_stage(component, input_data, stage_config)
            elif stage == PipelineStage.DATA_AGGREGATION:
                result = self._execute_aggregation_stage(component, input_data, stage_config)
            elif stage == PipelineStage.DATA_VALIDATION:
                result = self._execute_validation_stage(component, input_data, stage_config)
            else:
                raise NotImplementedError(f"Stage execution not implemented: {stage.value}")
            
            # Process results
            stage_result.records_processed = result.get('records_processed', 0)
            stage_result.output_path = result.get('output_path')
            stage_result.metrics = result.get('metrics', {})
            stage_result.warnings = result.get('warnings', [])
            stage_result.status = PipelineStatus.COMPLETED
            
        except Exception as e:
            stage_result.status = PipelineStatus.FAILED
            stage_result.errors.append(str(e))
            self.logger.error(f"Stage {stage.value} execution failed: {str(e)}")
            raise
        
        finally:
            stage_result.end_time = datetime.now()
            stage_result.duration_seconds = (
                stage_result.end_time - stage_result.start_time
            ).total_seconds()
        
        return stage_result
    
    def _execute_cleaning_stage(
        self,
        component: DataCleaner,
        input_data: Union[pd.DataFrame, str],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute data cleaning stage"""
        
        # Load data if path provided
        if isinstance(input_data, str):
            if input_data.endswith('.parquet'):
                data = pd.read_parquet(input_data)
            elif input_data.endswith('.csv'):
                data = pd.read_csv(input_data)
            else:
                raise ValueError(f"Unsupported file format: {input_data}")
        else:
            data = input_data
        
        # Execute cleaning
        cleaned_data = component.clean_data(data)
        
        # Save results
        output_path = self.checkpoint_dir / f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        cleaned_data.to_parquet(output_path)
        
        return {
            'records_processed': len(cleaned_data),
            'output_path': str(output_path),
            'metrics': {
                'input_records': len(data),
                'output_records': len(cleaned_data),
                'records_removed': len(data) - len(cleaned_data)
            }
        }
    
    def _execute_feature_engineering_stage(
        self,
        component: FeaturePipeline,
        input_data: Union[pd.DataFrame, str],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute feature engineering stage"""
        
        # Load data if path provided
        if isinstance(input_data, str):
            data = pd.read_parquet(input_data)
        else:
            data = input_data
        
        # Execute feature engineering
        features = component.generate_features(data)
        
        # Save results
        output_path = self.checkpoint_dir / f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        features.to_parquet(output_path)
        
        return {
            'records_processed': len(features),
            'output_path': str(output_path),
            'metrics': {
                'input_features': len(data.columns),
                'output_features': len(features.columns),
                'features_added': len(features.columns) - len(data.columns)
            }
        }
    
    def _execute_aggregation_stage(
        self,
        component: AggregationPipeline,
        input_data: Union[pd.DataFrame, str],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute data aggregation stage"""
        
        # Load data if path provided
        if isinstance(input_data, str):
            data = pd.read_parquet(input_data)
        else:
            data = input_data
        
        # Execute aggregation
        aggregated_data = component.run_aggregation_pipeline(data)
        
        # Save results
        output_path = self.checkpoint_dir / f"aggregated_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        aggregated_data.to_parquet(output_path)
        
        return {
            'records_processed': len(aggregated_data),
            'output_path': str(output_path),
            'metrics': {
                'input_records': len(data),
                'output_records': len(aggregated_data),
                'aggregation_ratio': len(data) / max(len(aggregated_data), 1)
            }
        }
    
    def _execute_validation_stage(
        self,
        component: ValidationPipeline,
        input_data: Union[pd.DataFrame, str],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute data validation stage"""
        
        # Load data if path provided
        if isinstance(input_data, str):
            data = pd.read_parquet(input_data)
        else:
            data = input_data
        
        # Execute validation
        validation_results = component.validate_data(data)
        
        # Save validation report
        output_path = self.checkpoint_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        return {
            'records_processed': len(data),
            'output_path': str(output_path),
            'metrics': {
                'validation_passed': validation_results.get('overall_validation_passed', False),
                'quality_score': validation_results.get('overall_quality_score', 0.0),
                'validation_issues': len(validation_results.get('validation_issues', []))
            }
        }
    
    def _prepare_stage_config(
        self,
        stage: PipelineStage,
        config_overrides: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare configuration for specific stage"""
        
        base_config = self.settings.__dict__.copy()
        
        if config_overrides:
            stage_overrides = config_overrides.get(stage.value, {})
            base_config.update(stage_overrides)
        
        return base_config
    
    def _process_stage_result(
        self,
        stage: PipelineStage,
        stage_result: StageResult,
        save_checkpoints: bool
    ):
        """Process stage execution result"""
        
        # Update execution tracking
        self.current_execution.stage_results[stage] = stage_result
        self.current_execution.stages_completed.append(stage)
        self.current_execution.total_records_processed += stage_result.records_processed
        
        # Collect errors and warnings
        self.current_execution.errors.extend(stage_result.errors)
        self.current_execution.warnings.extend(stage_result.warnings)
        
        # Save checkpoint if enabled
        if save_checkpoints:
            self._save_checkpoint(stage, stage_result)
        
        self.logger.info(f"Stage {stage.value} completed in {stage_result.duration_seconds:.2f}s")
    
    def _should_execute_parallel(
        self,
        stage: PipelineStage,
        input_data: Union[pd.DataFrame, str, Dict[str, Any]]
    ) -> bool:
        """Determine if stage should be executed in parallel"""
        
        # Check if data size justifies parallel execution
        data_size = 0
        if isinstance(input_data, pd.DataFrame):
            data_size = len(input_data)
        elif isinstance(input_data, str) and os.path.exists(input_data):
            data_size = os.path.getsize(input_data)
        
        # Use parallel execution for large datasets
        return data_size > 100000  # Threshold for parallel execution
    
    def _build_dependency_graph(self, stages: List[PipelineStage]) -> Dict[PipelineStage, List[PipelineStage]]:
        """Build dependency graph for stages"""
        
        graph = {}
        for stage in stages:
            dependencies = self.stage_config[stage]['dependencies']
            graph[stage] = [dep for dep in dependencies if dep in stages]
        
        return graph
    
    def _calculate_overall_metrics(self):
        """Calculate overall pipeline metrics"""
        
        if not self.current_execution:
            return
        
        # Aggregate metrics from all stages
        total_duration = self.current_execution.total_duration_seconds or 0
        total_records = self.current_execution.total_records_processed
        
        stage_durations = [
            result.duration_seconds for result in self.current_execution.stage_results.values()
            if result.duration_seconds is not None
        ]
        
        self.current_execution.overall_metrics = {
            'total_duration_seconds': total_duration,
            'total_records_processed': total_records,
            'average_stage_duration': np.mean(stage_durations) if stage_durations else 0,
            'records_per_second': total_records / max(total_duration, 1),
            'stages_executed': len(self.current_execution.stages_completed),
            'success_rate': len(self.current_execution.stages_completed) / max(len(self.stage_config), 1),
            'error_count': len(self.current_execution.errors),
            'warning_count': len(self.current_execution.warnings)
        }
    
    def _save_checkpoint(self, stage: PipelineStage, stage_result: StageResult):
        """Save execution checkpoint"""
        
        checkpoint_data = {
            'execution_id': self.current_execution.execution_id,
            'stage': stage.value,
            'stage_result': asdict(stage_result),
            'timestamp': datetime.now().isoformat(),
            'stages_completed': [s.value for s in self.current_execution.stages_completed],
            'current_execution': asdict(self.current_execution)
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.current_execution.execution_id}_{stage.value}.json"
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _load_checkpoint(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Load execution checkpoint"""
        
        checkpoint_files = list(self.checkpoint_dir.glob(f"checkpoint_{execution_id}_*.json"))
        
        if not checkpoint_files:
            return None
        
        # Load latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        
        with open(latest_checkpoint, 'r') as f:
            return json.load(f)
    
    def _resume_from_checkpoint(self, checkpoint: Dict[str, Any]):
        """Resume execution from checkpoint"""
        
        # Restore execution state
        execution_data = checkpoint['current_execution']
        self.current_execution = PipelineExecution(**execution_data)
        
        self.logger.info(f"Resumed from checkpoint: {checkpoint['stage']}")
    
    def _save_execution_results(self):
        """Save complete execution results"""
        
        results_path = self.metrics_dir / f"execution_{self.current_execution.execution_id}.json"
        
        with open(results_path, 'w') as f:
            json.dump(asdict(self.current_execution), f, indent=2, default=str)
        
        self.logger.info(f"Execution results saved: {results_path}")
    
    def _handle_execution_error(self, error: Exception):
        """Handle pipeline execution error"""
        
        if self.current_execution:
            self.current_execution.status = PipelineStatus.FAILED
            self.current_execution.end_time = datetime.now()
            self.current_execution.total_duration_seconds = (
                self.current_execution.end_time - self.current_execution.start_time
            ).total_seconds()
            self.current_execution.errors.append(f"Pipeline execution failed: {str(error)}")
        
        self.logger.error(f"Pipeline execution failed: {str(error)}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def get_execution_status(self, execution_id: Optional[str] = None) -> Optional[PipelineExecution]:
        """Get execution status"""
        
        if execution_id is None:
            return self.current_execution
        
        # Search in history
        for execution in self.execution_history:
            if execution.execution_id == execution_id:
                return execution
        
        return None
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[PipelineExecution]:
        """Get execution history"""
        
        history = self.execution_history
        if limit:
            history = history[-limit:]
        
        return history
    
    def pause_execution(self):
        """Pause current execution"""
        
        if self.current_execution and self.current_execution.status == PipelineStatus.RUNNING:
            self.current_execution.status = PipelineStatus.PAUSED
            self.logger.info("Pipeline execution paused")
    
    def resume_execution(self):
        """Resume paused execution"""
        
        if self.current_execution and self.current_execution.status == PipelineStatus.PAUSED:
            self.current_execution.status = PipelineStatus.RESUMED
            self.logger.info("Pipeline execution resumed")
    
    def cancel_execution(self):
        """Cancel current execution"""
        
        if self.current_execution and self.current_execution.status in [
            PipelineStatus.RUNNING, PipelineStatus.PAUSED
        ]:
            self.current_execution.status = PipelineStatus.FAILED
            self.current_execution.end_time = datetime.now()
            self.current_execution.errors.append("Execution cancelled by user")
            self.logger.info("Pipeline execution cancelled")
    
    def get_stage_statistics(self) -> Dict[str, Any]:
        """Get statistics across all executions"""
        
        if not self.execution_history:
            return {}
        
        stage_stats = {}
        
        for stage in PipelineStage:
            durations = []
            success_count = 0
            failure_count = 0
            
            for execution in self.execution_history:
                if stage in execution.stage_results:
                    result = execution.stage_results[stage]
                    if result.duration_seconds:
                        durations.append(result.duration_seconds)
                    
                    if result.status == PipelineStatus.COMPLETED:
                        success_count += 1
                    else:
                        failure_count += 1
            
            if durations:
                stage_stats[stage.value] = {
                    'average_duration': np.mean(durations),
                    'min_duration': np.min(durations),
                    'max_duration': np.max(durations),
                    'success_count': success_count,
                    'failure_count': failure_count,
                    'success_rate': success_count / (success_count + failure_count)
                }
        
        return stage_stats
