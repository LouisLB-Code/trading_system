import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging
from datetime import datetime

from ..meta_learning.advanced_adaptation import AdaptationCoordinator
from ..meta_learning.enhanced_learning import MetaReinforcementLearner
from ..market_analysis.regime_detector import MarketRegime

@dataclass
class RiskBounds:
    """Risk boundaries for different market regimes"""
    position_size: Tuple[float, float]
    leverage: Tuple[float, float]
    stop_loss: Tuple[float, float]
    correlation: Tuple[float, float]
    var_limit: Tuple[float, float]
    drawdown_limit: Tuple[float, float]

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    var: float
    cvar: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    beta: float
    correlation_risk: float
    liquidity_risk: float
    regime_risk: float
    timestamp: datetime

class AdvancedRiskEngine:
    """Advanced risk management system with meta-learning integration"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.risk_learner = RiskMetaLearner(config)
        self.risk_adapter = RiskAdapter(config)
        self.regime_risk_manager = RegimeRiskManager(config)
        
        # Initialize state
        self.current_risk_state = None
        self.risk_history = []
        self.adaptation_coordinator = AdaptationCoordinator(config)
        
        self.logger = logging.getLogger(__name__)
        
    async def update_risk_assessment(self,
                                   market_data: Dict,
                                   positions: Dict,
                                   regime: MarketRegime) -> Dict:
        """Update risk assessment and adapt risk parameters"""
        try:
            # Calculate current risk metrics
            risk_metrics = await self._calculate_risk_metrics(
                market_data,
                positions
            )
            
            # Get regime-specific risk bounds
            risk_bounds = self.regime_risk_manager.get_risk_bounds(regime)
            
            # Check for risk violations
            violations = self._check_risk_violations(
                risk_metrics,
                risk_bounds
            )
            
            if violations:
                # Generate risk adaptation
                adaptation = await self._generate_risk_adaptation(
                    risk_metrics,
                    violations,
                    regime
                )
                
                # Apply risk adaptation
                await self._apply_risk_adaptation(adaptation)
                
                # Update risk state
                self.current_risk_state = {
                    'metrics': risk_metrics,
                    'bounds': risk_bounds,
                    'adaptations': adaptation,
                    'timestamp': datetime.now()
                }
                
            # Store risk history
            self._update_risk_history(risk_metrics)
            
            return {
                'metrics': risk_metrics,
                'violations': violations,
                'adaptation': adaptation if violations else None,
                'state': self.current_risk_state
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment error: {str(e)}")
            raise
            
    async def _calculate_risk_metrics(self,
                                    market_data: Dict,
                                    positions: Dict) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Calculate basic metrics
            var = self._calculate_var(market_data, positions)
            cvar = self._calculate_cvar(market_data, positions)
            
            # Calculate ratio metrics
            sharpe = self._calculate_sharpe_ratio(market_data)
            sortino = self._calculate_sortino_ratio(market_data)
            
            # Calculate drawdown metrics
            max_dd, current_dd = self._calculate_drawdowns(market_data)
            
            # Calculate risk factors
            beta = self._calculate_beta(market_data)
            correlation_risk = self._calculate_correlation_risk(positions)
            liquidity_risk = self._calculate_liquidity_risk(market_data)
            regime_risk = self._calculate_regime_risk(market_data)
            
            return RiskMetrics(
                var=var,
                cvar=cvar,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_dd,
                current_drawdown=current_dd,
                beta=beta,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                regime_risk=regime_risk,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation error: {str(e)}")
            raise
            
    async def _generate_risk_adaptation(self,
                                      metrics: RiskMetrics,
                                      violations: Dict,
                                      regime: MarketRegime) -> Dict:
        """Generate risk parameter adaptations"""
        try:
            # Prepare current state
            current_state = {
                'metrics': metrics.__dict__,
                'violations': violations,
                'regime': regime.__dict__
            }
            
            # Prepare target state based on bounds
            target_state = self._get_target_risk_state(
                metrics,
                regime
            )
            
            # Generate adaptation using coordinator
            adaptation = await self.adaptation_coordinator.generate_adaptation(
                current_state,
                target_state
            )
            
            # Validate adaptation
            if not self._validate_risk_adaptation(adaptation, metrics):
                raise ValueError("Risk adaptation validation failed")
                
            return adaptation
            
        except Exception as e:
            self.logger.error(f"Risk adaptation generation error: {str(e)}")
            raise
            
    async def _apply_risk_adaptation(self, adaptation: Dict):
        """Apply risk parameter adaptations"""
        try:
            # Update position size limits
            if 'position_size' in adaptation:
                await self._update_position_limits(
                    adaptation['position_size']
                )
                
            # Update leverage limits
            if 'leverage' in adaptation:
                await self._update_leverage_limits(
                    adaptation['leverage']
                )
                
            # Update stop loss settings
            if 'stop_loss' in adaptation:
                await self._update_stop_loss_settings(
                    adaptation['stop_loss']
                )
                
            # Update correlation limits
            if 'correlation' in adaptation:
                await self._update_correlation_limits(
                    adaptation['correlation']
                )
                
            # Log adaptation
            self.logger.info(
                "Risk adaptation applied",
                extra={'adaptation': adaptation}
            )
            
        except Exception as e:
            self.logger.error(f"Risk adaptation application error: {str(e)}")
            raise

class RiskMetaLearner(MetaReinforcementLearner):
    """Meta-learning for risk management"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize risk-specific components
        self.risk_encoder = self._build_risk_encoder()
        self.risk_decoder = self._build_risk_decoder()
        
    def _build_risk_encoder(self) -> nn.Module:
        """Build risk-specific encoder network"""
        return nn.Sequential(
            nn.Linear(self.config.RISK_INPUT_DIM, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, self.config.RISK_LATENT_DIM)
        ).to(self.device)
        
    def _build_risk_decoder(self) -> nn.Module:
        """Build risk-specific decoder network"""
        return nn.Sequential(
            nn.Linear(self.config.RISK_LATENT_DIM, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, self.config.RISK_OUTPUT_DIM)
        ).to(self.device)
        
    async def learn_risk_patterns(self, risk_history: List[Dict]) -> Dict:
        """Learn patterns in risk behavior"""
        try:
            # Encode risk history
            encoded_history = self._encode_risk_history(risk_history)
            
            # Extract risk patterns
            patterns = self._extract_risk_patterns(encoded_history)
            
            # Update learning policy
            policy_update = await self._update_risk_policy(patterns)
            
            return {
                'patterns': patterns,
                'policy_update': policy_update,
                'encoded_state': encoded_history[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Risk pattern learning error: {str(e)}")
            raise

class RegimeRiskManager:
    """Manages regime-specific risk parameters"""
    
    def __init__(self, config):
        self.config = config
        self.regime_bounds = self._initialize_regime_bounds()
        self.logger = logging.getLogger(__name__)
        
    def get_risk_bounds(self, regime: MarketRegime) -> RiskBounds:
        """Get risk boundaries for current regime"""
        try:
            # Get base bounds for regime
            base_bounds = self.regime_bounds.get(
                regime.name,
                self.regime_bounds['default']
            )
            
            # Adjust bounds based on regime confidence
            adjusted_bounds = self._adjust_bounds_for_confidence(
                base_bounds,
                regime.confidence
            )
            
            return adjusted_bounds
            
        except Exception as e:
            self.logger.error(f"Risk bounds error: {str(e)}")
            raise
            
    def _initialize_regime_bounds(self) -> Dict[str, RiskBounds]:
        """Initialize risk bounds for different regimes"""
        return {
            'trending': RiskBounds(
                position_size=(0.1, 0.3),
                leverage=(1.0, 2.0),
                stop_loss=(0.02, 0.05),
                correlation=(0.0, 0.7),
                var_limit=(0.02, 0.05),
                drawdown_limit=(0.1, 0.2)
            ),
            'volatile': RiskBounds(
                position_size=(0.05, 0.15),
                leverage=(1.0, 1.5),
                stop_loss=(0.03, 0.07),
                correlation=(0.0, 0.5),
                var_limit=(0.03, 0.07),
                drawdown_limit=(0.15, 0.25)
            ),
            'default': RiskBounds(
                position_size=(0.05, 0.2),
                leverage=(1.0, 1.5),
                stop_loss=(0.02, 0.05),
                correlation=(0.0, 0.6),
                var_limit=(0.02, 0.05),
                drawdown_limit=(0.1, 0.2)
            )
        }
        
    def _adjust_bounds_for_confidence(self,
                                    bounds: RiskBounds,
                                    confidence: float) -> RiskBounds:
        """Adjust risk bounds based on regime confidence"""
        try:
            # Lower confidence should result in tighter bounds
            adjustment_factor = 0.5 + (0.5 * confidence)
            
            def adjust_range(bound_range: Tuple[float, float]) -> Tuple[float, float]:
                center = (bound_range[0] + bound_range[1]) / 2
                half_width = (bound_range[1] - bound_range[0]) / 2
                adjusted_width = half_width * adjustment_factor
                return (center - adjusted_width, center + adjusted_width)
                
            return RiskBounds(
                position_size=adjust_range(bounds.position_size),
                leverage=adjust_range(bounds.leverage),
                stop_loss=adjust_range(bounds.stop_loss),
                correlation=adjust_range(bounds.correlation),
                var_limit=adjust_range(bounds.var_limit),
                drawdown_limit=adjust_range(bounds.drawdown_limit)
            )
            
        except Exception as e:
            self.logger.error(f"Bounds adjustment error: {str(e)}")
            return bounds

class RiskAdapter:
    """Adapts risk parameters based on market conditions"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.risk_policy_network = self._build_policy_network()
        self.risk_value_network = self._build_value_network()
        
        self.logger = logging.getLogger(__name__)
        
    def _build_policy_network(self) -> nn.Module:
        """Build risk policy network"""
        return nn.Sequential(
            nn.Linear(self.config.RISK_STATE_DIM, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, self.config.RISK_ACTION_DIM)
        ).to(self.device)
        
    def _build_value_network(self) -> nn.Module:
        """Build risk value network"""
        return nn.Sequential(
            nn.Linear(self.config.RISK_STATE_DIM, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1)
        ).to(self.device)
        
    async def adapt_risk_parameters(self,
                                  current_state: Dict,
                                  target_state: Dict) -> Dict:
        """Adapt risk parameters to target state"""
        try:
            # Get policy action
            state_tensor = self._prepare_state_tensor(current_state)
            action = self.risk_policy_network(state_tensor)
            
            # Get value estimate
            value = self.risk_value_network(state_tensor)
            
            # Calculate adaptation parameters
            adaptation = self._calculate_adaptation_params(
                action,
                current_state,
                target_state
            )
            
            # Validate adaptation
            if not self._validate_adaptation(adaptation, current_state):
                raise ValueError("Risk adaptation validation failed")
                
            return adaptation
            
        except Exception as e:
            self.logger.error(f"Risk parameter adaptation error: {str(e)}")
            raise
