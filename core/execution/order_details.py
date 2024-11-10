
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime

@dataclass
class OrderDetails:
    """Details for order execution"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str  # 'market', 'limit', etc.
    price: Optional[float] = None
    time_in_force: str = 'GTC'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: float = 1.0
    reduce_only: bool = False
    post_only: bool = False
    client_order_id: Optional[str] = None
    metadata: Optional[Dict] = None
    timestamp: datetime = datetime.now()
    
    def validate(self) -> bool:
        """Validate order details"""
        if not all([self.symbol, self.side, self.quantity, self.order_type]):
            return False
        
        if self.side not in ['buy', 'sell']:
            return False
            
        if self.order_type not in ['market', 'limit']:
            return False
            
        if self.order_type == 'limit' and self.price is None:
            return False
            
        return True