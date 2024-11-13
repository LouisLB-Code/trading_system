import pytest
import torch
import numpy as np
from datetime import datetime, timedelta
from src.core.meta_learning import (
    MetaModelOptimizer,
    ExperienceMemoryBank,
    AdaptationTracker,
    PatternExtractor
)
from src.core.meta_learning.experience_bank import Experience

class TestMetaLearning:
    @pytest.fixture
    def setup(self):
        """Setup test environment"""
        self.config = TestConfig()
        self.meta_optimizer = MetaModelOptimizer(self.config)
        self.memory_bank = ExperienceMemoryBank(self.config)
        self.adaptation_tracker = AdaptationTracker(self.config)
        self.pattern_extractor = PatternExtractor(self.config)
        
    @pytest.mark.asyncio
    async def test_meta_model_optimization(self, setup):
        """Test meta-model optimization process"""
        # Prepare test data
        performance_data = self._create_test_performance_data()
        adaptation_data = self._create_test_adaptation_data()
        
        # Test optimization
        metrics = await self.meta_optimizer.optimize(
            performance_data,
            adaptation_data
        )
        
        # Verify metrics
        assert isinstance(metrics.loss, float)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.adaptation_score <= 1
        assert 0 <= metrics.generalization_score <= 1
        assert 0 <= metrics.resource_efficiency <= 1
        
    def test_experience_memory(self, setup):
        """Test experience memory bank"""
        # Create test experience
        experience = Experience(
            state={'market_state': 'trending'},
            action={'strategy_update': True},
            reward=1.0,
            next_state={'market_state': 'volatile'},
            metadata={'confidence': 0.8}
        )
        
        # Add experience
        self.memory_bank.add_experience(experience)
        
        # Test retrieval
        relevant_experiences = self.memory_bank.get_relevant_experiences(
            current_state={'market_state': 'trending'},
            k=5
        )
        
        assert len(relevant_experiences) > 0
        assert isinstance(relevant_experiences[0], Experience)
        
    @pytest.mark.asyncio
    async def test_adaptation_tracking(self, setup):
        """Test adaptation tracking"""
        # Create test event
        event = AdaptationEvent(
            trigger={'market_change': 'regime_shift'},
            action={'strategy_adjustment': True},
            result={'success': True}
        )
        
        # Track adaptation
        analysis = await self.adaptation_tracker.track_adaptation(event)
        
        # Verify analysis
        assert 'effectiveness' in analysis
        assert 'speed' in analysis
        assert 'stability' in analysis
        assert 'learning' in analysis
        
    @pytest.mark.asyncio
    async def test_pattern_extraction(self, setup):
        """Test learning pattern extraction"""
        # Create test data
        performance_history = self._create_test_performance_history()
        market_conditions = self._create_test_market_conditions()
        
        # Extract patterns
        patterns = await self.pattern_extractor.extract_patterns(
            performance_history,
            market_conditions
        )
        
        # Verify patterns
        assert 'strategy_patterns' in patterns
        assert 'adaptation_patterns' in patterns
        assert 'market_response_patterns' in patterns
        
    def _create_test_performance_data(self):
        """Create test performance data"""
        return {
            'returns': np.random.normal(0.001, 0.02, 100),
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.1,
            'win_rate': 0.6,
            'strategy_effectiveness': 0.75,
            'risk_effectiveness': 0.8
        }
        
    def _create_test_adaptation_data(self):
        """Create test adaptation data"""
        return {
            'success_rate': 0.7,
            'improvement_rate': 0.15,
            'adaptation_speed': 0.8,
            'stability_impact': 0.9
        }
        
    def _create_test_performance_history(self):
        """Create test performance history"""
        history = []
        
        for i in range(100):
            history.append({
                'timestamp': datetime.now() - timedelta(hours=i),
                'returns': np.random.normal(0.001, 0.02),
                'strategy': {
                    'name': 'test_strategy',
                    'parameters': {'param1': 0.1, 'param2': 0.2}
                },
                'market_conditions': {
                    'regime': 'trending' if i % 2 == 0 else 'volatile',
                    'volatility': np.random.uniform(0.1, 0.3)
                }
            })
            
        return history
        
    def _create_test_market_conditions(self):
        """Create test market conditions"""
        return {
            'current_regime': 'trending',
            'volatility': 0.2,
            'trend_strength': 0.7,
            'liquidity': 0.8,
            'correlation': 0.3
        }

class TestMetaModelComponents:
    """Test individual components of meta-model"""
    
    def test_model_architecture(self):
        """Test meta-model architecture"""
        config = TestConfig()
        model = MetaModelOptimizer(config).meta_model
        
        # Test forward pass
        batch_size = 32
        input_tensor = torch.randn(batch_size, config.INPUT_FEATURES)
        output = model(input_tensor)
        
        assert output.shape == (batch_size, config.OUTPUT_FEATURES)
        
    def test_experience_similarity(self):
        """Test experience similarity calculation"""
        config = TestConfig()
        memory_bank = ExperienceMemoryBank(config)
        
        state1 = {'market_state': 'trending', 'volatility': 0.2}
        state2 = {'market_state': 'trending', 'volatility': 0.3}
        
        similarity = memory_bank._calculate_similarity(state1, state2)
        assert 0 <= similarity <= 1
        
    def test_pattern_clustering(self):
        """Test pattern clustering"""
        config = TestConfig()
        pattern_extractor = PatternExtractor(config)
        
        # Create test sequences
        sequences = [
            {'adaptation_speed': 0.8, 'stability_impact': 0.7},
            {'adaptation_speed': 0.7, 'stability_impact': 0.8},
            {'adaptation_speed': 0.2, 'stability_impact': 0.3}
        ]
        
        clusters = pattern_extractor._cluster_adaptations(sequences)
        assert len(clusters) > 0

class TestMetaLearningIntegration:
    """Test meta-learning system integration"""
    
    @pytest.fixture
    def setup(self):
        self.config = TestConfig()
        self.meta_learning_system = {
            'optimizer': MetaModelOptimizer(self.config),
            'memory_bank': ExperienceMemoryBank(self.config),
            'adaptation_tracker': AdaptationTracker(self.config),
            'pattern_extractor': PatternExtractor(self.config)
        }
        
    @pytest.mark.asyncio
    async def test_full_learning_cycle(self, setup):
        """Test complete meta-learning cycle"""
        # Create test data
        market_data = self._create_test_market_data()
        performance_data = self._create_test_performance_data()
        
        # Extract patterns
        patterns = await self.meta_learning_system['pattern_extractor'].extract_patterns(
            performance_data['history'],
            market_data
        )
        
        # Track adaptation
        adaptation_event = self._create_adaptation_event(patterns)
        analysis = await self.meta_learning_system['adaptation_tracker'].track_adaptation(
            adaptation_event
        )
        
        # Store experience
        experience = Experience(
            state=market_data,
            action=adaptation_event.action,
            reward=analysis['effectiveness'],
            next_state=self._get_current_market_data(),
            metadata={'patterns': patterns}
        )
        self.meta_learning_system['memory_bank'].add_experience(experience)
        
        # Optimize model
        metrics = await self.meta_learning_system['optimizer'].optimize(
            performance_data,
            {'adaptations': [analysis]}
        )
        
        # Verify complete cycle
        assert metrics.adaptation_score > 0
        assert len(self.meta_learning_system['memory_bank'].experiences) > 0
