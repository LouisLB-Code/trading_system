# File: src/core/execution/position_manager.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging

@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    current_price: float
    size: float
    leverage: float
    unrealized_pnl: float
    realized_pnl: float
    entry_time: datetime
    stop_loss: Optional[float]
    take_profit: Optional[float]
    strategy: str
    metadata: Dict

class PositionManager:
    """Manages trading positions and their lifecycle"""
    
    def __init__(self, config):
        self.config = config
        self.positions = {}
        self.position_history = []
        self.trade_history = []
        
    async def update_positions(self, market_data: pd.DataFrame):
        """Update all position states"""
        try:
            for symbol, position in self.positions.items():
                if symbol in market_data.index:
                    current_price = market_data.loc[symbol, 'close']
                    
                    # Update position status
                    await self._update_position_status(
                        position,
                        current_price
                    )
                    
                    # Check for stop loss/take profit
                    await self._check_exit_conditions(
                        position,
                        current_price
                    )
                    
        except Exception as e:
            logging.error(f"Position update error: {str(e)}")
            raise
    
    async def add_position(self, execution_result: Dict) -> Position:
        """Add new position"""
        try:
            position = Position(
                symbol=execution_result['symbol'],
                side=execution_result['side'],
                entry_price=execution_result['price'],
                current_price=execution_result['price'],
                size=execution_result['quantity'],
                leverage=execution_result['leverage'],
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_time=datetime.now(),
                stop_loss=execution_result.get('stop_loss'),
                take_profit=execution_result.get('take_profit'),
                strategy=execution_result['strategy'],
                metadata=execution_result.get('metadata', {})
            )
            
            self.positions[position.symbol] = position
            return position
            
        except Exception as e:
            logging.error(f"Position addition error: {str(e)}")
            raise
    
    async def close_position(self,
                           symbol: str,
                           price: float,
                           reason: str) -> Optional[Dict]:
        """Close an existing position"""
        try:
            if symbol not in self.positions:
                return None
                
            position = self.positions[symbol]
            
            # Calculate final PnL
            realized_pnl = self._calculate_pnl(
                position,
                price
            )
            
            # Record trade
            trade_record = {
                'symbol': symbol,
                'entry_price': position.entry_price,
                'exit_price': price,
                'size': position.size,
                'side': position.side,
                'entry_time': position.entry_time,
                'exit_time': datetime.now(),
                'realized_pnl': realized_pnl,
                'strategy': position.strategy,
                'close_reason': reason,
                'metadata': position.metadata
            }
            
            self.trade_history.append(trade_record)
            
            # Remove position
            del self.positions[symbol]
            
            return trade_record
            
        except Exception as e:
            logging.error(f"Position closing error: {str(e)}")
            raise

    async def _update_position_status(self,
                                    position: Position,
                                    current_price: float):
        """Update position status with current market price"""
        try:
            position.current_price = current_price
            
            # Calculate unrealized PnL
            position.unrealized_pnl = self._calculate_pnl(
                position,
                current_price
            )
            
        except Exception as e:
            logging.error(f"Position status update error: {str(e)}")
            raise
    
    async def _check_exit_conditions(self,
                                   position: Position,
                                   current_price: float):
        """Check if position should be closed based on conditions"""
        try:
            # Check stop loss
            if position.stop_loss:
                if position.side == 'long' and current_price <= position.stop_loss:
                    await self.close_position(
                        position.symbol,
                        current_price,
                        'stop_loss'
                    )
                elif position.side == 'short' and current_price >= position.stop_loss:
                    await self.close_position(
                        position.symbol,
                        current_price,
                        'stop_loss'
                    )
            
            # Check take profit
            if position.take_profit:
                if position.side == 'long' and current_price >= position.take_profit:
                    await self.close_position(
                        position.symbol,
                        current_price,
                        'take_profit'
                    )
                elif position.side == 'short' and current_price <= position.take_profit:
                    await self.close_position(
                        position.symbol,
                        current_price,
                        'take_profit'
                    )
                    
        except Exception as e:
            logging.error(f"Exit condition check error: {str(e)}")
            raise
    
    def _calculate_pnl(self, position: Position, current_price: float) -> float:
        """Calculate position PnL"""
        price_diff = current_price - position.entry_price
        if position.side == 'short':
            price_diff = -price_diff
            
        return price_diff * position.size * position.leverage

# File: src/core/execution/order_manager.py

@dataclass
class Order:
    id: str
    symbol: str
    side: str
    type: str
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: str
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    creation_time: datetime = datetime.now()
    last_update: datetime = datetime.now()

class OrderManager:
    """Manages order lifecycle and execution"""
    
    def __init__(self, config, exchange_client):
        self.config = config
        self.exchange = exchange_client
        self.active_orders = {}
        self.order_history = []
        
    async def place_order(self, order_details: 'OrderDetails') -> Optional[Order]:
        """Place new order"""
        try:
            # Create order object
            order = Order(
                id=self._generate_order_id(),
                symbol=order_details.symbol,
                side=order_details.side,
                type=order_details.type,
                quantity=order_details.quantity,
                price=order_details.price,
                stop_price=order_details.stop_loss,
                status='new'
            )
            
            # Place order on exchange
            exchange_order = await self.exchange.create_order(
                order_details.symbol,
                order_details.type,
                order_details.side,
                order_details.quantity,
                order_details.price
            )
            
            # Update order with exchange data
            order.id = exchange_order['id']
            order.status = exchange_order['status']
            
            # Store order
            self.active_orders[order.id] = order
            
            return order
            
        except Exception as e:
            logging.error(f"Order placement error: {str(e)}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel active order"""
        try:
            if order_id not in self.active_orders:
                return False
                
            # Cancel on exchange
            await self.exchange.cancel_order(order_id)
            
            # Update order status
            order = self.active_orders[order_id]
            order.status = 'cancelled'
            
            # Move to history
            self.order_history.append(order)
            del self.active_orders[order_id]
            
            return True
            
        except Exception as e:
            logging.error(f"Order cancellation error: {str(e)}")
            return False
    
    async def update_orders(self):
        """Update status of all active orders"""
        try:
            for order_id, order in list(self.active_orders.items()):
                # Get order status from exchange
                exchange_order = await self.exchange.get_order(order_id)
                
                # Update order status
                order.status = exchange_order['status']
                order.filled_quantity = exchange_order['filled']
                order.average_price = exchange_order['average']
                order.last_update = datetime.now()
                
                # Move completed orders to history
                if order.status in ['filled', 'cancelled']:
                    self.order_history.append(order)
                    del self.active_orders[order_id]
                    
        except Exception as e:
            logging.error(f"Order update error: {str(e)}")
            raise
