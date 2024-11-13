# src/core/market_microstructure/liquidity_modeler.py

import numpy as np
from scipy import stats
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

@dataclass
class LiquidityModel:
    """Liquidity model parameters and metrics"""
    resilience: float
    elasticity: float
    immediacy: float
    tightness: float
    market_impact: float
    execution_cost: float
    
class LiquidityModeler:
    """Models market liquidity and impact"""
    
    def __init__(self, config):
        self.config = config
        self.history = []
        self.logger = logging.getLogger(__name__)
        
    async def model_liquidity(self,
                            order_book: Dict,
                            trading_history: pd.DataFrame) -> LiquidityModel:
        """Build comprehensive liquidity model"""
        try:
            # Calculate liquidity measures
            resilience = self._calculate_resilience(trading_history)
            elasticity = self._calculate_price_elasticity(order_book)
            immediacy = self._calculate_immediacy(order_book)
            tightness = self._calculate_tightness(order_book)
            
            # Calculate impact measures
            market_impact = self._estimate_market_impact(order_book)
            execution_cost = self._estimate_execution_cost(order_book)
            
            return LiquidityModel(
                resilience=resilience,
                elasticity=elasticity,
                immediacy=immediacy,
                tightness=tightness,
                market_impact=market_impact,
                execution_cost=execution_cost
            )
            
        except Exception as e:
            self.logger.error(f"Liquidity modeling error: {str(e)}")
            raise
    
    def _calculate_resilience(self, trading_history: pd.DataFrame) -> float:
        """Calculate market resilience"""
        # Time to revert to equilibrium after large trades
        price_changes = trading_history['price'].diff()
        large_changes = price_changes[abs(price_changes) > self.config.LARGE_TRADE_THRESHOLD]
        
        if len(large_changes) < 2:
            return 0.0
            
        reversion_times = []
        for i in large_changes.index:
            reversion_time = self._find_reversion_time(trading_history, i)
            if reversion_time is not None:
                reversion_times.append(reversion_time)
                
        return np.mean(reversion_times) if reversion_times else 0.0
    
    def _estimate_market_impact(self, order_book: Dict) -> float:
        """Estimate market impact function"""
        try:
            # Calculate average impact using square root model
            volumes = np.array([size for _, size in order_book['asks']])
            prices = np.array([price for price, _ in order_book['asks']])
            
            # Fit impact model
            impact_coef = np.polyfit(np.sqrt(volumes), (prices - prices[0])/prices[0], 1)[0]
            
            return impact_coef
            
        except Exception as e:
            self.logger.error(f"Market impact estimation error: {str(e)}")
            return 0.0
