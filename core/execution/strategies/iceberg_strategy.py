import asyncio
from datetime import datetime
import logging
from typing import Dict, List, Optional
from ..execution_base_strategy import ExecutionStrategy
from ..order_details import OrderDetails

class IcebergStrategy(ExecutionStrategy):
    """Iceberg order execution strategy"""
    
    async def execute(self, order_details: OrderDetails) -> Optional[Dict]:
        """Execute order using iceberg orders"""
        try:
            display_size = self._calculate_display_size(
                order_details.quantity
            )
            
            executions = []
            remaining_quantity = order_details.quantity
            
            while remaining_quantity > 0:
                # Calculate current slice size
                slice_size = min(display_size, remaining_quantity)
                
                # Execute slice
                execution = await self._execute_slice(
                    order_details,
                    slice_size
                )
                
                if execution:
                    executions.append(execution)
                    remaining_quantity -= slice_size
                else:
                    break
            
            return self._combine_executions(executions)
            
        except Exception as e:
            logging.error(f"Iceberg execution error: {str(e)}")
            return None

    def _calculate_display_size(self, total_quantity: float) -> float:
        """Calculate visible order size"""
        display_size = total_quantity * self.config.ICEBERG_DISPLAY_SIZE
        return max(display_size, self.config.ICEBERG_MIN_SIZE)

    async def _execute_slice(self,
                           order_details: OrderDetails,
                           slice_size: float) -> Optional[Dict]:
        """Execute single iceberg slice"""
        try:
            # Get current market data
            market_data = await self._get_market_data(order_details.symbol)
            
            # Calculate execution price
            execution_price = self._calculate_execution_price(
                market_data,
                order_details
            )
            
            # Create slice order
            slice_order = OrderDetails(
                symbol=order_details.symbol,
                side=order_details.side,
                quantity=slice_size,
                order_type='limit',
                price=execution_price,
                time_in_force='GTC'
            )
            
            # Execute slice
            execution = await self._execute_order(slice_order, execution_price)
            
            return execution
            
        except Exception as e:
            logging.error(f"Iceberg slice execution error: {str(e)}")
            return None

    async def _get_market_data(self, symbol: str) -> Dict:
        """Get current market data"""
        try:
            if self.exchange:
                return await self.exchange.get_ticker(symbol)
            else:
                return {'bid': None, 'ask': None, 'last': None}
                
        except Exception as e:
            logging.error(f"Market data error: {str(e)}")
            raise

    def _calculate_execution_price(self,
                                market_data: Dict,
                                order_details: OrderDetails) -> float:
        """Calculate execution price for iceberg slice"""
        try:
            if order_details.side == 'buy':
                return market_data.get('ask', market_data.get('last'))
            else:
                return market_data.get('bid', market_data.get('last'))
                
        except Exception as e:
            logging.error(f"Price calculation error: {str(e)}")
            raise