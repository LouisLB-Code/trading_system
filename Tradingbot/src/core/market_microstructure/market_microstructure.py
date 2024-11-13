# src/core/market_microstructure/order_book_analyzer.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

@dataclass
class OrderBookMetrics:
    """Order book analysis metrics"""
    spread: float
    depth: float
    imbalance: float
    pressure: float
    liquidity_score: float
    resistance_levels: List[float]
    support_levels: List[float]
    timestamp: datetime

class OrderBookAnalyzer:
    """Analyzes order book structure and dynamics"""
    
    def __init__(self, config):
        self.config = config
        self.history = []
        self.logger = logging.getLogger(__name__)
        
    async def analyze_order_book(self, order_book: Dict) -> OrderBookMetrics:
        """Analyze order book structure"""
        try:
            # Calculate basic metrics
            spread = self._calculate_spread(order_book)
            depth = self._calculate_market_depth(order_book)
            imbalance = self._calculate_order_imbalance(order_book)
            
            # Calculate advanced metrics
            pressure = self._calculate_price_pressure(order_book)
            liquidity = self._calculate_liquidity_score(order_book)
            
            # Find support/resistance
            support, resistance = self._find_price_levels(order_book)
            
            return OrderBookMetrics(
                spread=spread,
                depth=depth,
                imbalance=imbalance,
                pressure=pressure,
                liquidity_score=liquidity,
                resistance_levels=resistance,
                support_levels=support,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Order book analysis error: {str(e)}")
            raise
            
    def _calculate_spread(self, order_book: Dict) -> float:
        """Calculate bid-ask spread"""
        best_bid = order_book['bids'][0][0] if order_book['bids'] else 0
        best_ask = order_book['asks'][0][0] if order_book['asks'] else 0
        return (best_ask - best_bid) / best_bid if best_bid > 0 else 0
    
    def _calculate_market_depth(self, order_book: Dict) -> float:
        """Calculate market depth"""
        depth = 0
        for price, size in order_book['bids'][:self.config.DEPTH_LEVELS]:
            depth += price * size
        for price, size in order_book['asks'][:self.config.DEPTH_LEVELS]:
            depth += price * size
        return depth
    
    def _calculate_order_imbalance(self, order_book: Dict) -> float:
        """Calculate order book imbalance"""
        bid_volume = sum(size for _, size in order_book['bids'][:self.config.IMBALANCE_LEVELS])
        ask_volume = sum(size for _, size in order_book['asks'][:self.config.IMBALANCE_LEVELS])
        total_volume = bid_volume + ask_volume
        return (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
