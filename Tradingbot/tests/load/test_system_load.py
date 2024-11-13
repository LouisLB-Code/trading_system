# tests/load/test_system_load.py
import pytest
import time
import asyncio
from src.core.trading_system import TradingSystem

class TestSystemLoad:
    @pytest.fixture
    def trading_system(self):
        config = TestConfig()
        return TradingSystem(config)

    @pytest.mark.load
    def test_sustained_load(self, trading_system):
        """Test system under sustained load"""
        start_time = time.time()
        updates_processed = 0
        errors = []
        
        while time.time() - start_time < 60:  # Run for 1 minute
            try:
                trading_system.process_update({
                    'price': 100 + updates_processed * 0.01,
                    'volume': 1000,
                    'timestamp': time.time()
                })
                updates_processed += 1
            except Exception as e:
                errors.append(e)
                
        processing_rate = updates_processed / 60
        assert processing_rate > 100  # Minimum 100 updates per second
        assert len(errors) == 0  # No errors during load test
        
    @pytest.mark.asyncio
    async def test_concurrent_users(self, trading_system):
        """Test system with multiple concurrent users"""
        user_count = 50
        requests_per_user = 20
        
        async def simulate_user(user_id):
            results = []
            for i in range(requests_per_user):
                result = await trading_system.process_user_request({
                    'user_id': user_id,
                    'action': 'place_order',
                    'order': {'symbol': 'BTC/USDT', 'side': 'buy', 'amount': 0.1}
                })
                results.append(result)
            return results
            
        user_tasks = [simulate_user(i) for i in range(user_count)]
        all_results = await asyncio.gather(*user_tasks)
        
        # Verify results
        total_requests = sum(len(results) for results in all_results)
        successful_requests = sum(
            sum(1 for r in results if r['success'])
            for results in all_results
        )
        
        assert total_requests == user_count * requests_per_user
        assert successful_requests / total_requests > 0.95  # 95% success rate
