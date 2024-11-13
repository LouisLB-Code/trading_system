import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

@dataclass
class OptimizationMetrics:
    """Metrics for meta-model optimization"""
    loss: float
    accuracy: float
    adaptation_score: float
    generalization_score: float
    resource_efficiency: float

class MetaModelOptimizer:
    """Optimizes meta-learning models based on performance and adaptation"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.meta_model = self._build_meta_model().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.meta_model.parameters(),
            lr=config.LEARNING_RATE
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        self.logger = logging.getLogger(__name__)
        self.best_metrics = None
        self.no_improvement_count = 0
        
    async def optimize(self,
                      performance_data: Dict,
                      adaptation_data: Dict) -> OptimizationMetrics:
        """Optimize meta-model based on performance and adaptation data"""
        try:
            # Prepare training data
            train_data, val_data = self._prepare_training_data(
                performance_data,
                adaptation_data
            )
            
            # Train model
            train_metrics = await self._train_model(train_data)
            
            # Evaluate model
            val_metrics = self._evaluate_model(val_data)
            
            # Update model if improved
            if self._should_update_model(val_metrics):
                self._update_model()
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            
            # Adapt learning rate
            self._adapt_learning_rate(val_metrics)
            
            # Store best metrics
            if self.best_metrics is None or val_metrics.loss < self.best_metrics.loss:
                self.best_metrics = val_metrics
            
            return val_metrics
            
        except Exception as e:
            self.logger.error(f"Model optimization error: {str(e)}")
            raise
            
    def _build_meta_model(self) -> nn.Module:
        """Build meta-learning neural network"""
        try:
            model = nn.Sequential(
                # Feature extraction layers
                nn.Linear(self.config.INPUT_FEATURES, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                
                # Hidden layers
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                
                # Output layer
                nn.Linear(64, self.config.OUTPUT_FEATURES)
            )
            
            # Initialize weights
            self._initialize_weights(model)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Model building error: {str(e)}")
            raise
            
    def _initialize_weights(self, model: nn.Module):
        """Initialize network weights"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
    def _prepare_training_data(self,
                             performance_data: Dict,
                             adaptation_data: Dict) -> Tuple[Dict, Dict]:
        """Prepare data for model training"""
        try:
            # Extract features
            features = self._extract_features(
                performance_data,
                adaptation_data
            )
            
            # Extract targets
            targets = self._extract_targets(
                performance_data,
                adaptation_data
            )
            
            # Convert to tensors
            features = torch.FloatTensor(features).to(self.device)
            targets = torch.FloatTensor(targets).to(self.device)
            
            # Split into train/val
            train_idx, val_idx = self._split_data(len(features))
            
            train_data = {
                'features': features[train_idx],
                'targets': targets[train_idx]
            }
            
            val_data = {
                'features': features[val_idx],
                'targets': targets[val_idx]
            }
            
            return train_data, val_data
            
        except Exception as e:
            self.logger.error(f"Data preparation error: {str(e)}")
            raise
            
    def _extract_features(self,
                         performance_data: Dict,
                         adaptation_data: Dict) -> np.ndarray:
        """Extract features for meta-learning"""
        try:
            features = []
            
            # Performance features
            features.extend(self._extract_performance_features(performance_data))
            
            # Adaptation features
            features.extend(self._extract_adaptation_features(adaptation_data))
            
            # Strategy features
            features.extend(self._extract_strategy_features(performance_data))
            
            # Risk features
            features.extend(self._extract_risk_features(performance_data))
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {str(e)}")
            raise
            
    def _extract_targets(self,
                        performance_data: Dict,
                        adaptation_data: Dict) -> np.ndarray:
        """Extract target values for training"""
        try:
            targets = []
            
            # Adaptation success
            targets.append(adaptation_data.get('success_rate', 0.0))
            
            # Performance improvement
            targets.append(adaptation_data.get('improvement_rate', 0.0))
            
            # Strategy effectiveness
            targets.append(performance_data.get('strategy_effectiveness', 0.0))
            
            # Risk management effectiveness
            targets.append(performance_data.get('risk_effectiveness', 0.0))
            
            return np.array(targets)
            
        except Exception as e:
            self.logger.error(f"Target extraction error: {str(e)}")
            raise
            
    async def _train_model(self, train_data: Dict) -> OptimizationMetrics:
        """Train meta-model"""
        try:
            self.meta_model.train()
            total_loss = 0
            
            for epoch in range(self.config.TRAIN_EPOCHS):
                epoch_loss = 0
                
                # Train in batches
                for batch in self._get_batches(train_data, self.config.BATCH_SIZE):
                    # Forward pass
                    outputs = self.meta_model(batch['features'])
                    loss = F.mse_loss(outputs, batch['targets'])
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.meta_model.parameters(),
                        self.config.MAX_GRAD_NORM
                    )
                    
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                
                total_loss += epoch_loss / self.config.BATCH_SIZE
                
            # Calculate metrics
            metrics = self._calculate_metrics(
                train_data['features'],
                train_data['targets']
            )
            
            metrics.loss = total_loss / self.config.TRAIN_EPOCHS
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model training error: {str(e)}")
            raise
            
    def _evaluate_model(self, val_data: Dict) -> OptimizationMetrics:
        """Evaluate meta-model performance"""
        try:
            self.meta_model.eval()
            
            with torch.no_grad():
                # Forward pass
                outputs = self.meta_model(val_data['features'])
                loss = F.mse_loss(outputs, val_data['targets'])
                
                # Calculate metrics
                metrics = self._calculate_metrics(
                    val_data['features'],
                    val_data['targets']
                )
                
                metrics.loss = loss.item()
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model evaluation error: {str(e)}")
            raise
            
    def _calculate_metrics(self,
                         features: torch.Tensor,
                         targets: torch.Tensor) -> OptimizationMetrics:
        """Calculate optimization metrics"""
        try:
            with torch.no_grad():
                # Get predictions
                predictions = self.meta_model(features)
                
                # Calculate accuracy
                accuracy = self._calculate_accuracy(predictions, targets)
                
                # Calculate adaptation score
                adaptation_score = self._calculate_adaptation_score(
                    predictions,
                    targets
                )
                
                # Calculate generalization score
                generalization_score = self._calculate_generalization_score(
                    predictions,
                    targets
                )
                
                # Calculate resource efficiency
                resource_efficiency = self._calculate_resource_efficiency()
                
                return OptimizationMetrics(
                    loss=0.0,  # Will be set by caller
                    accuracy=accuracy,
                    adaptation_score=adaptation_score,
                    generalization_score=generalization_score,
                    resource_efficiency=resource_efficiency
                )
                
        except Exception as e:
            self.logger.error(f"Metrics calculation error: {str(e)}")
            raise
            
    def _calculate_accuracy(self,
                          predictions: torch.Tensor,
                          targets: torch.Tensor) -> float:
        """Calculate prediction accuracy"""
        try:
            with torch.no_grad():
                # Calculate relative error
                relative_error = torch.abs(predictions - targets) / torch.abs(targets + 1e-8)
                
                # Calculate accuracy
                accuracy = torch.mean((1 - relative_error).clamp(0, 1))
                
                return float(accuracy)
                
        except Exception as e:
            self.logger.error(f"Accuracy calculation error: {str(e)}")
            return 0.0
            
    def _calculate_adaptation_score(self,
                                  predictions: torch.Tensor,
                                  targets: torch.Tensor) -> float:
        """Calculate adaptation capability score"""
        try:
            with torch.no_grad():
                # Calculate adaptation accuracy
                adaptation_accuracy = self._calculate_accuracy(
                    predictions[:, 0],  # Adaptation success prediction
                    targets[:, 0]       # Actual adaptation success
                )
                
                # Calculate adaptation speed
                adaptation_speed = self._calculate_adaptation_speed(
                    predictions,
                    targets
                )
                
                # Combine scores
                adaptation_score = (
                    self.config.ACCURACY_WEIGHT * adaptation_accuracy +
                    self.config.SPEED_WEIGHT * adaptation_speed
                )
                
                return float(adaptation_score)
                
        except Exception as e:
            self.logger.error(f"Adaptation score calculation error: {str(e)}")
            return 0.0
            
    def _calculate_generalization_score(self,
                                      predictions: torch.Tensor,
                                      targets: torch.Tensor) -> float:
        """Calculate generalization capability score"""
        try:
            with torch.no_grad():
                # Calculate performance on different market conditions
                market_scores = []
                for condition in self.config.MARKET_CONDITIONS:
                    condition_mask = self._get_condition_mask(condition)
                    if condition_mask.any():
                        score = self._calculate_accuracy(
                            predictions[condition_mask],
                            targets[condition_mask]
                        )
                        market_scores.append(score)
                
                # Calculate generalization score
                generalization_score = (
                    np.mean(market_scores) if market_scores else 0.0
                )
                
                return float(generalization_score)
                
        except Exception as e:
            self.logger.error(f"Generalization score calculation error: {str(e)}")
            return 0.0
            
    def _calculate_resource_efficiency(self) -> float:
        """Calculate resource utilization efficiency"""
        try:
            # Calculate memory efficiency
            memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_efficiency = 1.0 - (memory_usage / torch.cuda.max_memory_allocated() 
                                     if torch.cuda.is_available() else 0)
            
            # Calculate computational efficiency
            compute_efficiency = self._calculate_compute_efficiency()
            
            # Combine efficiencies
            resource_efficiency = (
                self.config.MEMORY_WEIGHT * memory_efficiency +
                self.config.COMPUTE_WEIGHT * compute_efficiency
            )
            
            return float(resource_efficiency)
            
        except Exception as e:
            self.logger.error(f"Resource efficiency calculation error: {str(e)}")
            return 0.0
            
    def _should_update_model(self, metrics: OptimizationMetrics) -> bool:
        """Determine if model should be updated"""
        if self.best_metrics is None:
            return True
            
        # Check if current metrics are better
        if metrics.loss < self.best_metrics.loss:
            return True
            
        # Check if adaptation score improved significantly
        if metrics.adaptation_score > self.best_metrics.adaptation_score + self.config.IMPROVEMENT_THRESHOLD:
            return True
            
        return False
        
    def _adapt_learning_rate(self, metrics: OptimizationMetrics):
        """Adapt learning rate based on performance"""
        self.scheduler.step(metrics.loss)
        
        # Check for learning rate adjustment
        if self.no_improvement_count > self.config.PATIENCE:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.config.LR_DECAY
