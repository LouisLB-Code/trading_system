from datetime import datetime, timedelta
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np
from ..execution_base_strategy import ExecutionStrategy
from ..order_details import OrderDetails

class VWAPStrategy(ExecutionStrategy):
    """Volume Weighted Average Price execution strategy"""
    
    async def execute(self, order_details: OrderDetails) -> Optional[Dict]:
        """Execute order using VWAP"""
        try:
            # Calculate VWAP parameters
            vwap_data = await self._calculate_vwap_data(order_details.symbol)
            
            # Create volume-based schedule
            schedule = self._create_vwap_schedule(
                order_details.quantity,
                vwap_data
            )
            
            executions = []
            for slice_time, slice_qty in schedule.items():
                # Wait until scheduled time
                await self._wait_until(slice_time)
                
                # Execute slice
                execution = await self._execute_vwap_slice(
                    order_details,
                    slice_qty,
                    vwap_data
                )
                
                if execution:
                    executions.append(execution)
                
            return self._combine_executions(executions)
            
        except Exception as e:
            logging.error(f"VWAP execution error: {str(e)}")
            return None

    async def _calculate_vwap_data(self, symbol: str) -> Dict:
        """Calculate VWAP data for symbol"""
        try:
            # Get historical data
            market_data = await self._get_historical_data(
                symbol,
                self.config.VWAP_LOOKBACK
            )
            
            # Calculate typical price
            typical_price = (
                market_data['high'] + 
                market_data['low'] + 
                market_data['close']
            ) / 3
            
            # Calculate VWAP
            vwap = (typical_price * market_data['volume']).cumsum() / market_data['volume'].cumsum()
            
            return {
                'vwap': vwap,
                'volume_profile': self._calculate_volume_profile(market_data),
                'price_levels': self._calculate_price_levels(market_data)
            }
            
        except Exception as e:
            logging.error(f"VWAP data calculation error: {str(e)}")
            raise

    async def _get_historical_data(self, symbol: str, lookback: int) -> pd.DataFrame:
        """Get historical market data"""
        try:
            if self.exchange:
                # Get historical data from exchange
                bars = await self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe='1m',
                    limit=lookback
                )
                
                return pd.DataFrame(
                    bars,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                ).set_index('timestamp')
            else:
                # For backtesting
                return pd.DataFrame()  # Return empty DataFrame
                
        except Exception as e:
            logging.error(f"Historical data error: {str(e)}")
            raise
            
    def _calculate_volume_profile(self, market_data: pd.DataFrame) -> Dict:
        """Calculate volume profile"""
        try:
            # Group data into time intervals
            intervals = pd.Grouper(freq=f'{self.config.VWAP_INTERVAL}T')
            volume_profile = market_data.groupby(intervals)['volume'].sum()
            
            # Convert to dictionary with time intervals as keys
            return {
                interval: float(volume)
                for interval, volume in volume_profile.items()
                if volume > 0
            }
            
        except Exception as e:
            logging.error(f"Volume profile calculation error: {str(e)}")
            raise

    def _calculate_price_levels(self, market_data: pd.DataFrame) -> Dict:
        """Calculate significant price levels"""
        try:
            price_levels = {}
            
            # Calculate volume-weighted price levels
            price_bins = pd.qcut(market_data['close'], 10)
            volume_by_price = market_data.groupby(price_bins)['volume'].sum()
            
            for price_bin, volume in volume_by_price.items():
                price_levels[float(price_bin.right)] = float(volume)
            
            return price_levels
            
        except Exception as e:
            logging.error(f"Price levels calculation error: {str(e)}")
            raise

    def _create_vwap_schedule(self, quantity: float, vwap_data: Dict) -> Dict[datetime, float]:
        """Create volume-based execution schedule"""
        try:
            # Get volume profile
            volume_profile = vwap_data['volume_profile']
            total_volume = sum(volume_profile.values())
            
            schedule = {}
            current_time = datetime.now()
            
            for interval, volume in volume_profile.items():
                # Calculate quantity for interval based on volume distribution
                interval_ratio = volume / total_volume
                interval_quantity = quantity * interval_ratio
                
                # Add to schedule
                execution_time = current_time + (interval - current_time)
                schedule[execution_time] = interval_quantity
            
            return schedule
            
        except Exception as e:
            logging.error(f"VWAP schedule creation error: {str(e)}")
            raise

    async def _execute_vwap_slice(self,
                                order_details: OrderDetails,
                                quantity: float,
                                vwap_data: Dict) -> Optional[Dict]:
        """Execute single VWAP slice"""
        try:
            # Get current market data
            market_data = await self._get_market_data(order_details.symbol)
            
            # Calculate target price based on VWAP
            target_price = self._calculate_target_price(
                market_data,
                vwap_data,
                order_details
            )
            
            # Create slice order
            slice_order = OrderDetails(
                symbol=order_details.symbol,
                side=order_details.side,
                quantity=quantity,
                order_type='limit',
                price=target_price,
                time_in_force='GTC'
            )
            
            # Execute slice
            execution = await self._execute_order(slice_order, target_price)
            
            return execution
            
        except Exception as e:
            logging.error(f"VWAP slice execution error: {str(e)}")
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

    def _calculate_target_price(self,
                              market_data: Dict,
                              vwap_data: Dict,
                              order_details: OrderDetails) -> float:
        """Calculate target execution price based on VWAP"""
        try:
            current_price = market_data.get(
                'last',
                market_data.get('bid' if order_details.side == 'sell' else 'ask')
            )
            
            # Use VWAP as reference price
            vwap_price = float(vwap_data['vwap'].iloc[-1])
            
            # Calculate target price as weighted average of current price and VWAP
            target_price = (current_price + vwap_price) / 2
            
            return target_price
            
        except Exception as e:
            logging.error(f"Target price calculation error: {str(e)}")
            raise