
import asyncio
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional
from ..execution_base_strategy import ExecutionStrategy
from ..order_details import OrderDetails

class TWAPStrategy(ExecutionStrategy):
    """Time Weighted Average Price execution strategy"""
    
    async def execute(self, order_details: OrderDetails) -> Optional[Dict]:
        """Execute order using TWAP"""
        try:
            # Calculate execution schedule
            schedule = self._create_schedule(order_details)
            
            executions = []
            for time, quantity in schedule.items():
                # Wait until scheduled time
                await self._wait_until(time)
                
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
            duration = timedelta(minutes=self.config.TWAP_DURATION)
            intervals = self.config.TWAP_INTERVALS
            
            interval_duration = duration / intervals
            interval_quantity = order_details.quantity / intervals
            
            schedule = {}
            current_time = datetime.now()
            
            for i in range(intervals):
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
            
            # Create slice order
            slice_order = OrderDetails(
                symbol=order_details.symbol,
                side=order_details.side,
                quantity=quantity,
                order_type='market',
                time_in_force='GTC'
            )
            
            # Execute slice
            execution = await self._execute_order(slice_order, execution_price)
            
            return execution
            
        except Exception as e:
            logging.error(f"TWAP slice execution error: {str(e)}")
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
        """Calculate execution price for TWAP slice"""
        try:
            if order_details.side == 'buy':
                return market_data.get('ask', market_data.get('last'))
            else:
                return market_data.get('bid', market_data.get('last'))
                
        except Exception as e:
            logging.error(f"Price calculation error: {str(e)}")
            raise

    async def _validate_slice_execution(self,
                                     execution: Dict,
                                     target_price: float) -> bool:
        """Validate slice execution results"""
        try:
            if not execution:
                return False

            price_deviation = abs(execution['price'] - target_price) / target_price
            
            # Check if execution price is within acceptable range
            if price_deviation > self.config.TWAP_MAX_DEVIATION:
                logging.warning(f"TWAP slice price deviation too high: {price_deviation}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Slice validation error: {str(e)}")
            return False