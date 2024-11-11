# tests/unit/test_edge_cases.py
import pytest
from src.core.risk_management import RiskManager
from src.core.market_analysis import MarketConditionAnalyzer
from tests.utils.test_market_conditions import (
    create_flash_crash_data,
    create_pump_dump_data,
    create_low_liquidity_data
)

class TestEdgeCases:
    @pytest.fixture
    def setup(self):
        """Setup test environment"""
        self.config = TestConfig()
        self.risk_manager = RiskManager(self.config)
        self.market_analyzer = MarketConditionAnalyzer(self.config)
    
    def test_flash_crash_detection(self, setup):
        """Test system behavior during flash crash"""
        crash_data = create_flash_crash_data()
        
        # Test market condition detection
        market_state = self.market_analyzer.analyze_market_conditions(crash_data)
        assert market_state.is_crash_detected
        assert market_state.volatility > self.config.HIGH_VOLATILITY_THRESHOLD
        
        # Test risk management response
        risk_assessment = self.risk_manager.assess_market_risk(crash_data)
        assert risk_assessment.should_halt_trading
        assert risk_assessment.position_size_multiplier < 0.5  # Reduced position size
    
    def test_pump_dump_handling(self, setup):
        """Test system behavior during pump and dump"""
        pump_dump_data = create_pump_dump_data()
        
        # Test pump detection
        market_state = self.market_analyzer.analyze_market_conditions(pump_dump_data)
        assert market_state.is_pump_detected
        
        # Test risk management during pump
        risk_assessment = self.risk_manager.assess_market_risk(pump_dump_data)
        assert risk_assessment.increased_stop_loss_distance
        assert not risk_assessment.should_open_new_positions
    
    def test_low_liquidity_handling(self, setup):
        """Test system behavior in low liquidity conditions"""
        low_liq_data = create_low_liquidity_data()
        
        # Test liquidity detection
        market_state = self.market_analyzer.analyze_market_conditions(low_liq_data)
        assert market_state.is_low_liquidity
        
        # Test risk management in low liquidity
        risk_assessment = self.risk_manager.assess_market_risk(low_liq_data)
        assert risk_assessment.should_reduce_position_size
        assert risk_assessment.max_position_size < self.config.DEFAULT_POSITION_SIZE
