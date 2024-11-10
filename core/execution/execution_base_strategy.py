from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import asyncio
from .order_details import OrderDetails

class ExecutionStrategy(ABC):
    """Base class for all execution strategies"""
    
    def __init__(self, config, exchange_client=None):
        self.config = config
        self.exchange = exchange_client
        
    @abstractmethod
    async def execute(self, order_details: OrderDetails) -> Optional[Dict]:
        """Execute order using strategy"""
        pass

    async def _execute_order(self, order_details: OrderDetails, price: float) -> Optional[Dict]:
        """Base method for order execution"""
        try:
            if self.exchange:
                # Live trading
                order = await self.exchange.create_order(
                    symbol=order_details.symbol,
                    type=order_details.order_type,
                    side=order_details.side,
                    amount=order_details.quantity,
                    price=price if order_details.order_type == 'limit' else None
                )
                
                return {
                    'symbol': order_details.symbol,
                    'side': order_details.side,
                    'quantity': order_details.quantity,
                    'price': price,
                    'timestamp': datetime.now()
                }
            else:
                # Backtesting
                return {
                    'symbol': order_details.symbol,
                    'side': order_details.side,
                    'quantity': order_details.quantity,
                    'price': price,
                    'timestamp': datetime.now()
                }
                
        except Exception as e:
            logging.error(f"Order execution error: {str(e)}")
            return None
    
    async def _wait_until(self, target_time: datetime):
        """Wait until specific time"""
        try:
            now = datetime.now()
            if target_time > now:
                await asyncio.sleep((target_time - now).total_seconds())
        except Exception as e:
            logging.error(f"Wait error: {str(e)}")
    
    def _combine_executions(self, executions: List[Dict]) -> Optional[Dict]:
        """Combine multiple executions into single result"""
        try:
            if not executions:
                return None
                
            total_quantity = sum(e['quantity'] for e in executions)
            total_value = sum(e['quantity'] * e['price'] for e in executions)
            
            return {
                'symbol': executions[0]['symbol'],
                'side': executions[0]['side'],
                'quantity': total_quantity,
                'price': total_value / total_quantity if total_quantity > 0 else 0,
                'executions': executions,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Execution combination error: {str(e)}")
            return None