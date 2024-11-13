# File: src/core/backtesting/execution_simulator.py

class BacktestExecutionSimulator:
    """Simulates order execution in backtesting"""
    
    def __init__(self, config, market_data: Dict[str, pd.DataFrame]):
        self.config = config
        self.market_data = market_data
        self.slippage_model = SlippageModel(config)
        self.fee_model = FeeModel(config)
        
    async def execute_signal(self,
                           signal: Dict,
                           timestamp: pd.Timestamp) -> Optional[Dict]:
        """Simulate order execution"""
        try:
            # Get execution price
            execution_price = self._calculate_execution_price(
                signal,
                timestamp
            )
            
            # Apply slippage
            final_price = self.slippage_model.apply_slippage(
                execution_price,
                signal,
                self._get_market_impact(signal)
            )
            
            # Calculate fees
            fees = self.fee_model.calculate_fees(
                signal['quantity'],
                final_price,
                signal.get('type', 'market')
            )
            
            # Create execution result
            execution = {
                'symbol': signal['symbol'],
                'side': signal['side'],
                'quantity': signal['quantity'],
                'price': final_price,
                'fees': fees,
                'timestamp': timestamp,
                'strategy': signal['strategy'],
                'type': signal.get('type', 'market'),
                'metadata': signal.get('metadata', {})
            }
            
            return execution
            
        except Exception as e:
            logging.error(f"Execution simulation error: {str(e)}")
            return None
    
    def _calculate_execution_price(self,
                                 signal: Dict,
                                 timestamp: pd.Timestamp) -> float:
        """Calculate realistic execution price"""
        try:
            symbol = signal['symbol']
            df = self.market_data[symbol]
            
            if signal.get('type') == 'limit':
                return signal['price']
            
            # For market orders, use VWAP if available, else use close
            if 'vwap' in df.columns:
                return df.loc[timestamp, 'vwap']
            
            return df.loc[timestamp, 'close']
            
        except Exception as e:
            logging.error(f"Price calculation error: {str(e)}")
            raise

class SlippageModel:
    """Models price slippage in backtesting"""
    
    def __init__(self, config):
        self.config = config
        
    def apply_slippage(self,
                      price: float,
                      signal: Dict,
                      market_impact: float) -> float:
        """Apply price slippage to execution"""
        try:
            # Calculate base slippage
            base_slippage = self._calculate_base_slippage(price)
            
            # Calculate volume-based slippage
            volume_slippage = self._calculate_volume_slippage(
                signal['quantity'],
                market_impact
            )
            
            # Calculate volatility-based slippage
            volatility_slippage = self._calculate_volatility_slippage(
                price,
                signal['symbol']
            )
            
            # Combine slippage components
            total_slippage = (
                base_slippage +
                volume_slippage +
                volatility_slippage
            )
            
            # Apply slippage to price
            direction = 1 if signal['side'] == 'buy' else -1
            return price * (1 + direction * total_slippage)
            
        except Exception as e:
            logging.error(f"Slippage calculation error: {str(e)}")
            return price

class FeeModel:
    """Models trading fees in backtesting"""
    
    def __init__(self, config):
        self.config = config
        
    def calculate_fees(self,
                      quantity: float,
                      price: float,
                      order_type: str) -> float:
        """Calculate trading fees"""
        try:
            # Calculate base fee
            base_fee = quantity * price * self.config.BASE_FEE_RATE
            
            # Add exchange-specific fees
            if order_type == 'market':
                base_fee *= self.config.MARKET_FEE_MULTIPLIER
            elif order_type == 'limit':
                base_fee *= self.config.LIMIT_FEE_MULTIPLIER
            
            return base_fee
            
        except Exception as e:
            logging.error(f"Fee calculation error: {str(e)}")
            return 0.0
