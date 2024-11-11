# tests/unit/test_market_analysis.py
import pytest
from src.core.market_analysis import MarketConditionAnalyzer, EnhancedRegimeDetector

class TestMarketAnalysis:
    @pytest.fixture
    def setup(self):
        self.config = TestConfig()
        self.analyzer = MarketConditionAnalyzer(self.config)
        self.regime_detector = EnhancedRegimeDetector(self.config)
        
    def test_market_conditions(self, setup):
        """Test market condition calculations"""
        market_data = create_test_market_data()
        
        conditions = self.analyzer.analyze(market_data)
        assert 0 <= conditions.volatility <= 1
        assert 0 <= conditions.trend_strength <= 1
        assert 0 <= conditions.liquidity <= 1
        
    def test_regime_detection(self, setup):
        """Test regime detection accuracy"""
        # Test trending market
        trending_data = create_trending_market_data()
        regime = self.regime_detector.detect_regime(trending_data)
        assert regime.name in ["TRENDING_UP", "TRENDING_DOWN"]
        assert regime.confidence >= 0.7
        
        # Test ranging market
        ranging_data = create_ranging_market_data()
        regime = self.regime_detector.detect_regime(ranging_data)
        assert regime.name == "RANGING"
        
        # Test volatile market
        volatile_data = create_volatile_market_data()
        regime = self.regime_detector.detect_regime(volatile_data)
        assert regime.name == "VOLATILE"
