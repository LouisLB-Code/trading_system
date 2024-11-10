# File: src/core/models/model_manager.py

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from dataclasses import dataclass
import joblib
from sklearn.preprocessing import StandardScaler

@dataclass
class ModelMetrics:
    accuracy: float
    loss: float
    sharpe_ratio: float
    max_drawdown: float
    prediction_score: float
    regime_accuracy: float

class ModelManager:
    """Manages deep learning models for the trading system"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.training_history = {}
        
    async def initialize_models(self):
        """Initialize all required models"""
        try:
            # Risk prediction model
            self.models['risk'] = self._build_risk_model()
            
            # Pattern recognition model
            self.models['pattern'] = self._build_pattern_model()
            
            # Regime prediction model
            self.models['regime'] = self._build_regime_model()
            
            # Strategy generation model
            self.models['strategy'] = self._build_strategy_model()
            
            # Initialize scalers
            self._initialize_scalers()
            
        except Exception as e:
            logging.error(f"Model initialization error: {str(e)}")
            raise
    
    async def train_models(self,
                         training_data: Dict[str, pd.DataFrame],
                         market_conditions: Dict):
        """Train or update all models"""
        try:
            # Train risk model
            risk_metrics = await self._train_risk_model(
                training_data['risk'],
                market_conditions
            )
            
            # Train pattern model
            pattern_metrics = await self._train_pattern_model(
                training_data['pattern']
            )
            
            # Train regime model
            regime_metrics = await self._train_regime_model(
                training_data['regime']
            )
            
            # Train strategy model
            strategy_metrics = await self._train_strategy_model(
                training_data['strategy']
            )
            
            # Update performance metrics
            self._update_performance_metrics({
                'risk': risk_metrics,
                'pattern': pattern_metrics,
                'regime': regime_metrics,
                'strategy': strategy_metrics
            })
            
        except Exception as e:
            logging.error(f"Model training error: {str(e)}")
            raise
    
    def _build_risk_model(self) -> nn.Module:
        """Build neural network for risk prediction"""
        return nn.Sequential(
            nn.Linear(self.config.RISK_INPUT_FEATURES, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.config.RISK_OUTPUT_FEATURES)
        )
    
    def _build_pattern_model(self) -> nn.Module:
        """Build neural network for pattern recognition"""
        return nn.Sequential(
            nn.Conv1d(
                in_channels=self.config.PATTERN_INPUT_CHANNELS,
                out_channels=64,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(128 * 30, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.config.PATTERN_OUTPUT_FEATURES)
        )
    
    def _build_regime_model(self) -> nn.Module:
        """Build neural network for regime prediction"""
        return nn.Sequential(
            nn.LSTM(
                input_size=self.config.REGIME_INPUT_FEATURES,
                hidden_size=128,
                num_layers=2,
                dropout=0.3,
                batch_first=True
            ),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.config.REGIME_OUTPUT_FEATURES)
        )
    
    def _build_strategy_model(self) -> nn.Module:
        """Build neural network for strategy generation"""
        return nn.Sequential(
            nn.Linear(self.config.STRATEGY_INPUT_FEATURES, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.config.STRATEGY_OUTPUT_FEATURES)
        )
    
    async def _train_risk_model(self,
                              data: pd.DataFrame,
                              market_conditions: Dict) -> ModelMetrics:
        """Train risk prediction model"""
        try:
            # Prepare training data
            X, y = self._prepare_risk_data(data, market_conditions)
            
            # Scale features
            X_scaled = self.scalers['risk'].fit_transform(X)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y)
            
            # Train model
            optimizer = torch.optim.Adam(
                self.models['risk'].parameters(),
                lr=self.config.LEARNING_RATE
            )
            criterion = nn.MSELoss()
            
            for epoch in range(self.config.EPOCHS):
                # Forward pass
                outputs = self.models['risk'](X_tensor)
                loss = criterion(outputs, y_tensor)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Calculate metrics
            metrics = self._calculate_risk_metrics(outputs, y_tensor)
            
            return metrics
            
        except Exception as e:
            logging.error(f"Risk model training error: {str(e)}")
            raise
    
    def _calculate_risk_metrics(self,
                              predictions: torch.Tensor,
                              targets: torch.Tensor) -> ModelMetrics:
        """Calculate model performance metrics"""
        try:
            # Convert to numpy
            pred_np = predictions.detach().numpy()
            target_np = targets.detach().numpy()
            
            # Calculate metrics
            metrics = ModelMetrics(
                accuracy=np.mean(np.abs(pred_np - target_np) < 0.1),
                loss=float(nn.MSELoss()(predictions, targets)),
                sharpe_ratio=self._calculate_prediction_sharpe(pred_np, target_np),
                max_drawdown=self._calculate_prediction_drawdown(pred_np),
                prediction_score=self._calculate_prediction_score(pred_np, target_np),
                regime_accuracy=self._calculate_regime_accuracy(pred_np, target_np)
            )
            
            return metrics
            
        except Exception as e:
            logging.error(f"Metrics calculation error: {str(e)}")
            raise
    
    def save_models(self, path: str):
        """Save all models and scalers"""
        try:
            # Save models
            for name, model in self.models.items():
                torch.save(
                    model.state_dict(),
                    f"{path}/{name}_model.pth"
                )
            
            # Save scalers
            for name, scaler in self.scalers.items():
                joblib.dump(
                    scaler,
                    f"{path}/{name}_scaler.pkl"
                )
            
            # Save metrics
            joblib.dump(
                self.performance_metrics,
                f"{path}/metrics.pkl"
            )
            
        except Exception as e:
            logging.error(f"Model saving error: {str(e)}")
            raise
    
    def load_models(self, path: str):
        """Load all models and scalers"""
        try:
            # Load models
            for name in self.models.keys():
                self.models[name].load_state_dict(
                    torch.load(f"{path}/{name}_model.pth")
                )
            
            # Load scalers
            for name in self.scalers.keys():
                self.scalers[name] = joblib.load(
                    f"{path}/{name}_scaler.pkl"
                )
            
            # Load metrics
            self.performance_metrics = joblib.load(
                f"{path}/metrics.pkl"
            )
            
        except Exception as e:
            logging.error(f"Model loading error: {str(e)}")
            raise
