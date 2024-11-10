import asyncio
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional
import numpy as np
from ..execution_base_strategy import ExecutionStrategy
from ..order_details import OrderDetails

class POVStrategy(ExecutionStrategy):
    """Percentage of Volume execution strategy"""
    
    async def execute(self, order_details: OrderDetails) -> Optional[Dict]:
        """Execute order using POV"""
        try:
            executions = []
            remaining_quantity = order_details.quantity
            
            while remaining_quantity > 0:
                # Get current market volume
                market_volume = await self._get_market_volume(
                    order_details.symbol
                )
                
                # Calculate execution quantity
                quantity = min(
                    remaining_quantity,
                    market_volume * self.config.POV_MAX_PARTICIPATION
                )
                
                if quantity >= self.config.MIN_TRADE_SIZE:
                    execution = await self._execute_slice(
                        order_details,
                        quantity
                    )
                    
                    if execution:
                        executions.append(execution)
                        remaining_quantity -= quantity
                
                # Wait before next check
                await asyncio.sleep(self.config.POV_INTERVAL * 60)
            
            return self._combine_executions(executions)
            
        except Exception as e:
            logging.error(f"POV execution error: {str(e)}")
            return None

    async def _get_market_volume(self, symbol: str) -> float:
        """Get current market volume"""
        try:
            # Get recent trades
            if self.exchange:
                trades = await self.exchange.get_recent_trades(
                    symbol,
                    limit=self.config.POV_VOLUME_WINDOW
                )
                
                # Calculate volume in specified time window
                window_start = datetime.now() - timedelta(
                    minutes=self.config.POV_INTERVAL
                )
                
                volume = sum(
                    trade['amount'] for trade in trades
                    if datetime.fromtimestamp(trade['timestamp']/1000) > window_start
                )
                
                return volume
            else:
                # For backtesting
                return 1000.0  # Default volume for testing
                
        except Exception as e:
            logging.error(f"Market volume error: {str(e)}")
            raise

    async def _execute_slice(self,
                           order_details: OrderDetails,
                           quantity: float) -> Optional[Dict]:
        """Execute single POV slice"""
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
                quantity=quantity,
                order_type='market',  # POV typically uses market orders
                time_in_force='GTC'
            )
            
            # Execute slice
            execution = await self._execute_order(slice_order, execution_price)
            
            return execution
            
        except Exception as e:
            logging.error(f"POV slice execution error: {str(e)}")
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
        """Calculate execution price for POV slice"""
        try:
            if order_details.side == 'buy':
                return market_data.get('ask', market_data.get('last'))
            else:
                return market_data.get('bid', market_data.get('last'))
                
        except Exception as e:
            logging.error(f"Price calculation error: {str(e)}")
            raise