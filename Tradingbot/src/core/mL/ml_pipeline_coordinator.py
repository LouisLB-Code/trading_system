# src/core/ml/pipeline_coordinator.py

from typing import Dict, Any, Optional
import logging
from datetime import datetime
import asyncio

class MLPipelineCoordinator:
    """Coordinates ML pipeline components and workflow"""
    
    def __init__(self, config):
        self.config = config
        self.feature_processor = FeatureProcessor(config)
        self.model_evaluator = ModelEvaluator(config)
        self.hyperopt = HyperparameterOptimizer(config)
        self.model_registry = ModelRegistry(config)
        self.logger = logging.getLogger(__name__)
        
    async def run_pipeline(self,
                          data: Dict[str, Any],
                          model_type: str,
                          pipeline_config: Optional[Dict] = None) -> Dict:
        """Run complete ML pipeline"""
        try:
            pipeline_id = self._generate_pipeline_id()
            self.logger.info(f"Starting ML pipeline: {pipeline_id}")
            
            # Process features
            features = await self.feature_processor.process_features(
                data['market_data'],
                data.get('additional_data')
            )
            
            # Split data
            train_data, val_data, test_data = self._split_data(
                features.features,
                data['target']
            )
            
            # Select and train model
            model = await self._train_model(
                train_data,
                model_type,
                pipeline_config
            )
            
            # Evaluate model
            evaluation = await self.model_evaluator.evaluate_model(
                model,
                val_data['X'],
                val_data['y'],
                features.features.columns
            )
            
            # Optimize if needed
            if self._should_optimize(evaluation):
                optimization = await self.hyperopt.optimize_parameters(
                    model.__class__,
                    train_data['X'],
                    train_data['y'],
                    self._get_param_space(model_type)
                )
                
                # Retrain with optimal parameters
                model = await self._train_model(
                    train_data,
                    model_type,
                    optimization.best_params
                )
            
            # Register model
            model_info = await self.model_registry.register_model(
                model,
                {
                    'pipeline_id': pipeline_id,
                    'features': features,
                    'evaluation': evaluation,
                    'optimization': optimization if 'optimization' in locals() else None,
                    'config': pipeline_config
                }
            )
            
            return {
                'model_info': model_info,
                'features': features,
                'evaluation': evaluation,
                'pipeline_id': pipeline_id
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            raise
            
    async def _train_model(self,
                          train_data: Dict,
                          model_type: str,
                          config: Optional[Dict]) -> Any:
        """Train model with given configuration"""
        try:
            # Get model class
            model_class = self._get_model_class(model_type)
            
            # Initialize model
            model = model_class(**config) if config else model_class()
            
            # Train model
            await self._async_train(model, train_data)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Model training error: {str(e)}")
            raise
            
    def _should_optimize(self, evaluation: ModelEvaluation) -> bool:
        """Determine if optimization is needed"""
        # Check if performance meets thresholds
        if evaluation.metrics['validation_score'] < self.config.OPTIMIZATION_THRESHOLD:
            return True
            
        # Check if performance is significantly different from baseline
        if self._check_baseline_difference(evaluation):
            return True
            
        return False
        
    async def _async_train(self, model: Any, train_data: Dict):
        """Asynchronously train model"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: model.fit(train_data['X'], train_data['y'])
        )
