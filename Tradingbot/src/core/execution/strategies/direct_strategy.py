from datetime import datetime
import logging
from typing import Dict, Optional
from ..execution_base_strategy import ExecutionStrategy
from ..order_details import OrderDetails

class DirectStrategy(ExecutionStrategy):
    """Direct market execution strategy"""
    
    async def execute(self, order_details: OrderDetails) -> Optional[Dict]:
        """Execute order immediately"""
        try:
            # Get current market data
            market_data = await self._get_market_data(order_details.symbol)
            
            # Calculate execution price
            execution_price = self._calculate_execution_price(
                market_data,
                order_details
            )
            
            # Execute order
            execution = await self._execute_order(
                order_details,
                execution_price
            )
            
            return execution
            
        except Exception as e:
            logging.error(f"Direct execution error: {str(e)}")
            return None
    
    async def _get_market_data(self, symbol: str) -> Dict:
        """Get current market data"""
        try:
            if self.exchange:
                return await self.exchange.get_ticker(symbol)
            else:
                # For backtesting
                return {'bid': None, 'ask': None, 'last': None}
                
        except Exception as e:
            logging.error(f"Market data error: {str(e)}")
            raise
    
    def _calculate_execution_price(self,
                                 market_data: Dict,
                                 order_details: OrderDetails) -> float:
        """Calculate execution price"""
        try:
            if order_details.side == 'buy':
                return market_data.get('ask', market_data.get('last'))
            else:
                return market_data.get('bid', market_data.get('last'))
                
        except Exception as e:
            logging.error(f"Price calculation error: {str(e)}")
            raise
    
    async def _execute_order(self,
                           order_details: OrderDetails,
                           price: float) -> Dict:
        """Execute single order"""
        try:
            if self.exchange:
                # Live trading
                order = await self.exchange.create_order(
                    symbol=order_details.symbol,
                    type='market',
                    side=order_details.side,
                    amount=order_details.quantity
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
            raise

# File: src/core/execution/strategies/twap_strategy.py

class TWAPStrategy(ExecutionStrategy):
    """Time-Weighted Average Price execution strategy"""
    
    async def execute(self, order_details: OrderDetails) -> Optional[Dict]:
        """Execute order using TWAP"""
        try:
            # Calculate execution schedule
            schedule = self._create_schedule(order_details)
            
            executions = []
            for execution_time, quantity in schedule.items():
                # Wait until scheduled time
                await self._wait_until(execution_time)
                
                # Execute slice
                execution = await self._execute_slice(
                    order_details,
                    quantity
                )
                
                if execution:
                    executions.append(execution)
                
            return self._combine_executions(executions)
            
        except Exception as e:
            logging.error(f"TWAP execution error: {str(e)}")
            return None
    
    def _create_schedule(self, order_details: OrderDetails) -> Dict[datetime, float]:
        """Create time-based execution schedule"""
        try:
            # Get schedule parameters
            duration = timedelta(minutes=self.config.TWAP_DURATION)
            num_intervals = self.config.TWAP_INTERVALS
            
            # Calculate interval parameters
            interval_duration = duration / num_intervals
            interval_quantity = order_details.quantity / num_intervals
            
            # Create schedule
            schedule = {}
            current_time = datetime.now()
            
            for i in range(num_intervals):
                execution_time = current_time + (interval_duration * i)
                schedule[execution_time] = interval_quantity
                
            return schedule
            
        except Exception as e:
            logging.error(f"Schedule creation error: {str(e)}")
            raise
    
    async def _execute_slice(self,
                           order_details: OrderDetails,
                           quantity: float) -> Optional[Dict]:
        """Execute single TWAP slice"""
        try:
            # Get current market data
            market_data = await self._get_market_data(order_details.symbol)
            
            # Calculate execution price
            execution_price = self._calculate_execution_price(
                market_data,
                order_details
            )
            
            # Execute slice
            execution = await self._execute_order(
                OrderDetails(
                    symbol=order_details.symbol,
                    side=order_details.side,
                    quantity=quantity,
                    type='market',
                    price=None
                ),
                execution_price
            )
            
            return execution
            
        except Exception as e:
            logging.error(f"Slice execution error: {str(e)}")
            return None
