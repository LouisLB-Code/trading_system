# src/core/ml/feature_engineering/feature_processor.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

@dataclass
class FeatureSet:
    """Processed feature set"""
    features: pd.DataFrame
    feature_importance: Dict[str, float]
    metadata: Dict[str, any]
    timestamp: datetime

class FeatureProcessor:
    """Advanced feature engineering and processing"""
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.feature_history = []
        self.feature_stats = {}
        self.logger = logging.getLogger(__name__)
        
    async def process_features(self,
                             market_data: pd.DataFrame,
                             additional_data: Dict = None) -> FeatureSet:
        """Process and engineer features"""
        try:
            # Generate base features
            base_features = self._generate_base_features(market_data)
            
            # Generate advanced features
            advanced_features = self._generate_advanced_features(
                base_features,
                additional_data
            )
            
            # Select relevant features
            selected_features = self._select_features(advanced_features)
            
            # Scale features
            scaled_features = self._scale_features(selected_features)
            
            # Calculate feature importance
            importance = self._calculate_feature_importance(scaled_features)
            
            return FeatureSet(
                features=scaled_features,
                feature_importance=importance,
                metadata=self._generate_feature_metadata(scaled_features),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Feature processing error: {str(e)}")
            raise
    
    def _generate_advanced_features(self,
                                  base_features: pd.DataFrame,
                                  additional_data: Dict) -> pd.DataFrame:
        """Generate advanced engineered features"""
        features = base_features.copy()
        
        # Add interaction features
        interactions = self._create_feature_interactions(features)
        features = pd.concat([features, interactions], axis=1)
        
        # Add time-based features
        time_features = self._create_time_features(features)
        features = pd.concat([features, time_features], axis=1)
        
        # Add market microstructure features
        if additional_data and 'order_book' in additional_data:
            micro_features = self._create_microstructure_features(
                features,
                additional_data['order_book']
            )
            features = pd.concat([features, micro_features], axis=1)
        
        # Add sentiment features
        if additional_data and 'sentiment' in additional_data:
            sentiment_features = self._create_sentiment_features(
                features,
                additional_data['sentiment']
            )
            features = pd.concat([features, sentiment_features], axis=1)
        
        return features

# src/core/ml/model_selection/model_evaluator.py

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_squared_error
)

@dataclass
class ModelEvaluation:
    """Model evaluation results"""
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    validation_predictions: np.ndarray
    cross_val_scores: List[float]
    confusion_matrix: np.ndarray
    performance_analysis: Dict

class ModelEvaluator:
    """Evaluates and compares model performance"""
    
    def __init__(self, config):
        self.config = config
        self.evaluation_history = []
        self.baseline_metrics = {}
        
    async def evaluate_model(self,
                           model: Any,
                           X_val: np.ndarray,
                           y_val: np.ndarray,
                           feature_names: List[str]) -> ModelEvaluation:
        """Comprehensive model evaluation"""
        try:
            # Get predictions
            predictions = model.predict(X_val)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_val, predictions)
            
            # Calculate feature importance
            importance = self._calculate_feature_importance(
                model,
                feature_names
            )
            
            # Perform cross validation
            cv_scores = self._perform_cross_validation(
                model,
                X_val,
                y_val
            )
            
            # Generate confusion matrix
            conf_matrix = self._generate_confusion_matrix(
                y_val,
                predictions
            )
            
            # Analyze performance
            analysis = self._analyze_performance(
                metrics,
                cv_scores,
                conf_matrix
            )
            
            return ModelEvaluation(
                metrics=metrics,
                feature_importance=importance,
                validation_predictions=predictions,
                cross_val_scores=cv_scores,
                confusion_matrix=conf_matrix,
                performance_analysis=analysis
            )
            
        except Exception as e:
            self.logger.error(f"Model evaluation error: {str(e)}")
            raise

# src/core/ml/hyperopt/parameter_optimizer.py

from typing import Dict, List, Optional, Union
import optuna
from optuna.trial import Trial
import numpy as np

@dataclass
class OptimizationResult:
    """Hyperparameter optimization results"""
    best_params: Dict[str, Union[float, int, str]]
    best_score: float
    optimization_history: List[Dict]
    parameter_importance: Dict[str, float]
    convergence_plot: Dict

class HyperparameterOptimizer:
    """Optimizes model hyperparameters"""
    
    def __init__(self, config):
        self.config = config
        self.study_pruner = optuna.pruners.MedianPruner()
        self.optimization_history = []
        
    async def optimize_parameters(self,
                                model_class: type,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                param_space: Dict) -> OptimizationResult:
        """Optimize model hyperparameters"""
        try:
            # Create study
            study = optuna.create_study(
                direction="maximize",
                pruner=self.study_pruner
            )
            
            # Define objective
            def objective(trial: Trial) -> float:
                params = self._sample_parameters(trial, param_space)
                return self._evaluate_parameters(
                    model_class,
                    params,
                    X_train,
                    y_train,
                    trial
                )
            
            # Run optimization
            study.optimize(
                objective,
                n_trials=self.config.N_TRIALS,
                timeout=self.config.OPTIMIZATION_TIMEOUT
            )
            
            # Analyze results
            importance = optuna.importance.get_param_importances(study)
            
            # Store history
            self.optimization_history.append({
                'study': study,
                'best_params': study.best_params,
                'best_score': study.best_value,
                'parameter_importance': importance
            })
            
            return OptimizationResult(
                best_params=study.best_params,
                best_score=study.best_value,
                optimization_history=study.trials,
                parameter_importance=importance,
                convergence_plot=self._create_convergence_plot(study)
            )
            
        except Exception as e:
            self.logger.error(f"Parameter optimization error: {str(e)}")
            raise
