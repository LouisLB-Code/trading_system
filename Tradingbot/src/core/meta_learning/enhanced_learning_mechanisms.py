import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

@dataclass
class LearningState:
    """State representation for meta-learning"""
    market_features: torch.Tensor
    strategy_state: torch.Tensor
    performance_metrics: torch.Tensor
    adaptation_history: torch.Tensor
    regime_info: torch.Tensor

class MetaReinforcementLearner:
    """Meta-reinforcement learning for trading strategies"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.policy_network = self._build_policy_network()
        self.value_network = self._build_value_network()
        self.meta_network = self._build_meta_network()
        
        # Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=config.POLICY_LR
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=config.VALUE_LR
        )
        self.meta_optimizer = torch.optim.Adam(
            self.meta_network.parameters(),
            lr=config.META_LR
        )
        
        self.experience_buffer = []
        self.logger = logging.getLogger(__name__)
        
    def _build_policy_network(self) -> nn.Module:
        """Build policy network for action selection"""
        return nn.Sequential(
            nn.Linear(self.config.STATE_DIM, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, self.config.ACTION_DIM),
            nn.Tanh()  # For continuous action space
        ).to(self.device)
        
    def _build_value_network(self) -> nn.Module:
        """Build value network for state evaluation"""
        return nn.Sequential(
            nn.Linear(self.config.STATE_DIM, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1)
        ).to(self.device)
        
    def _build_meta_network(self) -> nn.Module:
        """Build meta-network for strategy adaptation"""
        return nn.Sequential(
            # Meta-learning encoder
            nn.Linear(self.config.META_INPUT_DIM, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            # Task-specific adaptation layers
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            # Strategy parameter generation
            nn.Linear(256, self.config.STRATEGY_PARAM_DIM)
        ).to(self.device)
        
    async def update(self, state: LearningState, reward: float) -> Dict:
        """Update meta-reinforcement learning system"""
        try:
            # Convert state to tensor
            state_tensor = self._prepare_state_tensor(state)
            
            # Get action from policy network
            action = self.policy_network(state_tensor)
            
            # Get value estimate
            value = self.value_network(state_tensor)
            
            # Get meta-parameters
            meta_params = self.meta_network(state_tensor)
            
            # Calculate advantages
            advantages = self._calculate_advantages(reward, value)
            
            # Update networks
            policy_loss = self._update_policy(state_tensor, action, advantages)
            value_loss = self._update_value(state_tensor, reward)
            meta_loss = self._update_meta(state_tensor, meta_params, advantages)
            
            # Store experience
            self._store_experience(state, action, reward, meta_params)
            
            return {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'meta_loss': meta_loss.item(),
                'action': action.detach().cpu().numpy(),
                'meta_params': meta_params.detach().cpu().numpy()
            }
            
        except Exception as e:
            self.logger.error(f"Meta-RL update error: {str(e)}")
            raise
            
    def _update_policy(self,
                      state: torch.Tensor,
                      action: torch.Tensor,
                      advantages: torch.Tensor) -> torch.Tensor:
        """Update policy network"""
        self.policy_optimizer.zero_grad()
        
        # Calculate policy loss with entropy regularization
        policy_output = self.policy_network(state)
        log_prob = self._calculate_log_prob(policy_output, action)
        entropy = self._calculate_entropy(policy_output)
        
        policy_loss = -(log_prob * advantages).mean() - self.config.ENTROPY_COEF * entropy
        
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            self.config.MAX_GRAD_NORM
        )
        self.policy_optimizer.step()
        
        return policy_loss
        
    def _update_value(self,
                     state: torch.Tensor,
                     reward: float) -> torch.Tensor:
        """Update value network"""
        self.value_optimizer.zero_grad()
        
        value_pred = self.value_network(state)
        value_loss = F.mse_loss(value_pred, torch.tensor([reward]).to(self.device))
        
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.value_network.parameters(),
            self.config.MAX_GRAD_NORM
        )
        self.value_optimizer.step()
        
        return value_loss
        
    def _update_meta(self,
                    state: torch.Tensor,
                    meta_params: torch.Tensor,
                    advantages: torch.Tensor) -> torch.Tensor:
        """Update meta-network"""
        self.meta_optimizer.zero_grad()
        
        # Calculate meta-learning loss
        meta_output = self.meta_network(state)
        meta_loss = F.mse_loss(meta_output, meta_params)
        
        # Add advantage-weighted loss
        meta_loss += self.config.ADVANTAGE_WEIGHT * (meta_loss * advantages).mean()
        
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.meta_network.parameters(),
            self.config.MAX_GRAD_NORM
        )
        self.meta_optimizer.step()
        
        return meta_loss

class TransferLearningModule:
    """Transfer learning for strategy adaptation"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.feature_extractor = self._build_feature_extractor()
        self.regime_adapter = self._build_regime_adapter()
        self.strategy_adapter = self._build_strategy_adapter()
        
        self.logger = logging.getLogger(__name__)
        
    def _build_feature_extractor(self) -> nn.Module:
        """Build feature extraction network"""
        return nn.Sequential(
            # Convolutional layers for temporal features
            nn.Conv1d(self.config.INPUT_CHANNELS, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            # Flatten and dense layers
            nn.Flatten(),
            nn.Linear(256 * self.config.SEQUENCE_LENGTH // 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        ).to(self.device)
        
    def _build_regime_adapter(self) -> nn.Module:
        """Build regime adaptation network"""
        return nn.Sequential(
            nn.Linear(512 + self.config.REGIME_DIM, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, self.config.REGIME_ADAPTATION_DIM)
        ).to(self.device)
        
    def _build_strategy_adapter(self) -> nn.Module:
        """Build strategy adaptation network"""
        return nn.Sequential(
            nn.Linear(512 + self.config.STRATEGY_DIM, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, self.config.STRATEGY_ADAPTATION_DIM)
        ).to(self.device)
        
    async def adapt_strategy(self,
                           source_strategy: Dict,
                           target_regime: Dict,
                           market_data: torch.Tensor) -> Dict:
        """Adapt strategy to new market regime"""
        try:
            # Extract features
            features = self.feature_extractor(market_data)
            
            # Adapt to new regime
            regime_features = self._get_regime_features(target_regime)
            regime_adaptation = self.regime_adapter(
                torch.cat([features, regime_features], dim=1)
            )
            
            # Adapt strategy
            strategy_features = self._get_strategy_features(source_strategy)
            strategy_adaptation = self.strategy_adapter(
                torch.cat([features, strategy_features], dim=1)
            )
            
            # Combine adaptations
            adapted_strategy = self._combine_adaptations(
                source_strategy,
                regime_adaptation,
                strategy_adaptation
            )
            
            return adapted_strategy
            
        except Exception as e:
            self.logger.error(f"Strategy adaptation error: {str(e)}")
            raise
            
    def _get_regime_features(self, regime: Dict) -> torch.Tensor:
        """Extract regime features"""
        features = []
        
        # Extract regime characteristics
        features.extend([
            regime.get('volatility', 0),
            regime.get('trend_strength', 0),
            regime.get('liquidity', 0),
            regime.get('correlation', 0)
        ])
        
        # One-hot encode regime type
        regime_type = regime.get('type', 'unknown')
        regime_encoding = torch.zeros(len(self.config.REGIME_TYPES))
        if regime_type in self.config.REGIME_TYPES:
            regime_encoding[self.config.REGIME_TYPES.index(regime_type)] = 1
            
        features.extend(regime_encoding.tolist())
        
        return torch.tensor(features, device=self.device)
        
    def _get_strategy_features(self, strategy: Dict) -> torch.Tensor:
        """Extract strategy features"""
        features = []
        
        # Extract strategy parameters
        features.extend([
            param for param in strategy.get('parameters', {}).values()
        ])
        
        # Extract strategy type encoding
        strategy_type = strategy.get('type', 'unknown')
        strategy_encoding = torch.zeros(len(self.config.STRATEGY_TYPES))
        if strategy_type in self.config.STRATEGY_TYPES:
            strategy_encoding[self.config.STRATEGY_TYPES.index(strategy_type)] = 1
            
        features.extend(strategy_encoding.tolist())
        
        return torch.tensor(features, device=self.device)
        
    def _combine_adaptations(self,
                           source_strategy: Dict,
                           regime_adaptation: torch.Tensor,
                           strategy_adaptation: torch.Tensor) -> Dict:
        """Combine adaptations to create new strategy"""
        # Implementation specific to your strategy structure
        pass

class CurriculumLearner:
    """Curriculum learning for gradual strategy complexity"""
    
    def __init__(self, config):
        self.config = config
        self.current_stage = 0
        self.stage_metrics = []
        self.logger = logging.getLogger(__name__)
        
    async def update_curriculum(self, performance_metrics: Dict) -> Dict:
        """Update learning curriculum based on performance"""
        try:
            # Store stage metrics
            self.stage_metrics.append(performance_metrics)
            
            # Check if ready for next stage
            if self._ready_for_advancement(performance_metrics):
                self.current_stage += 1
                
                # Get new curriculum parameters
                new_params = self._get_stage_parameters(self.current_stage)
                
                return {
                    'advance_stage': True,
                    'new_parameters': new_params,
                    'stage_metrics': self.stage_metrics[-1]
                }
            
            return {
                'advance_stage': False,
                'current_parameters': self._get_stage_parameters(self.current_stage),
                'stage_metrics': self.stage_metrics[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Curriculum update error: {str(e)}")
            raise
            
    def _ready_for_advancement(self, metrics: Dict) -> bool:
        """Check if ready to advance to next stage"""
        if len(self.stage_metrics) < self.config.MIN_STAGE_SAMPLES:
            return False
            
        # Check performance thresholds
        return all([
            metrics['sharpe_ratio'] > self.config.STAGE_THRESHOLDS['sharpe_ratio'],
            metrics['win_rate'] > self.config.STAGE_THRESHOLDS['win_rate'],
            metrics['max_drawdown'] < self.config.STAGE_THRESHOLDS['max_drawdown']
        ])
        
    def _get_stage_parameters(self, stage: int) -> Dict:
        """Get parameters for current learning stage"""
        # Implementation specific to your strategy complexity levels
        pass
