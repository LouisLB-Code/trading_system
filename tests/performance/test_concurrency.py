# tests/performance/test_concurrency.py
import asyncio
import pytest
from datetime import datetime
from src.core.trading_system import TradingSystem
from tests.utils.test_data_generator import create_test_market_data

class TestConcurrency:
    @pytest.fixture
    async def trading_system(self):
        """Setup trading system for tests"""
        config = TestConfig()
        system = TradingSystem(config)
        yield system
        await system.shutdown()

    @pytest.mark.asyncio
    async def test_parallel_processing(self, trading_system):
        """Test system performance with parallel data processing"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        market_data = {sym: create_test_market_data() for sym in symbols}
        
        start_time = datetime.now()
        tasks = [
            trading_system.process_symbol(sym, data) 
            for sym, data in market_data.items()
        ]
        
        results = await asyncio.gather(*tasks)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        assert all(r['success'] for r in results)
        assert processing_time < len(symbols) * 0.1  # Expected 100ms per symbol
        
    @pytest.mark.asyncio
    async def test_concurrent_order_execution(self, trading_system):
        """Test concurrent order execution"""
        orders = [
            {'symbol': 'BTC/USDT', 'side': 'buy', 'amount': 0.1, 'price': 35000},
            {'symbol': 'ETH/USDT', 'side': 'buy', 'amount': 1.0, 'price': 2000},
            {'symbol': 'SOL/USDT', 'side': 'sell', 'amount': 10.0, 'price': 100}
        ]
        
        tasks = [trading_system.execute_order(order) for order in orders]
        results = await asyncio.gather(*tasks)
        
        assert all(r['status'] == 'executed' for r in results)
        
    @pytest.mark.asyncio
    async def test_data_stream_processing(self, trading_system):
        """Test processing of multiple data streams"""
        async def generate_market_updates(symbol, count):
            for _ in range(count):
                yield {'symbol': symbol, 'price': 100 + _ * 0.1, 'volume': 1000}
                await asyncio.sleep(0.01)
        
        streams = [
            generate_market_updates('BTC/USDT', 100),
            generate_market_updates('ETH/USDT', 100),
            generate_market_updates('SOL/USDT', 100)
        ]
        
        results = []
        async def process_stream(stream):
            async for update in stream:
                result = await trading_system.process_update(update)
                results.append(result)
        
        await asyncio.gather(*[process_stream(stream) for stream in streams])
        assert len(results) == 300  # Total updates processed
