from dataclasses import dataclass
from typing import Dict, List
from datetime import timedelta

@dataclass
class MetaModelConfig:
    """Meta-model neural network configuration"""
    INPUT_FEATURES: int = 128
    OUTPUT_FEATURES: int = 64
    HIDDEN_LAYERS: List[int] = (256, 128, 64)
    DROPOUT_RATE: float = 0.3
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    TRAIN_EPOCHS: int = 100
    MAX_GRAD_NORM: float = 1.0
    
    # Model architecture settings
    USE_ATTENTION: bool = True
    USE_RESIDUAL: bool = True
    ATTENTION_HEADS: int = 8
    TRANSFORMER_LAYERS: int = 3
    
    # Training settings
    LR_DECAY: float = 0.5
    PATIENCE: int = 10
    EARLY_STOPPING_PATIENCE: int = 20
    MIN_LEARNING_RATE: float = 1e-6
    WEIGHT_DECAY: float = 1e-5
    
    # Loss weights
    PERFORMANCE_LOSS_WEIGHT: float = 0.4
    ADAPTATION_LOSS_WEIGHT: float = 0.3
    GENERALIZATION_LOSS_WEIGHT: float = 0.3

@dataclass
class ExperienceConfig:
    """Experience memory configuration"""
    MAX_EXPERIENCES: int = 10000
    MIN_EXPERIENCES_FOR_TRAINING: int = 1000
    EXPERIENCE_BATCH_SIZE: int = 64
    IMPORTANCE_SAMPLING_EXPONENT: float = 0.6
    
    # Experience priorities
    REWARD_WEIGHT: float = 0.4
    NOVELTY_WEIGHT: float = 0.3
    IMPACT_WEIGHT: float = 0.3
    
    # Sampling settings
    SIMILARITY_THRESHOLD: float = 0.7
    MIN_SIMILARITY_COUNT: int = 5
    MAX_EXPERIENCE_AGE: timedelta = timedelta(days=30)
    
    # Pattern detection
    PATTERN_WINDOW: int = 100
    PATTERN_UPDATE_INTERVAL: int = 10
    MIN_PATTERN_OCCURRENCES: int = 3

@dataclass
class AdaptationConfig:
    """Adaptation system configuration"""
    ADAPTATION_THRESHOLD: float = 0.7
    MIN_ADAPTATION_INTERVAL: timedelta = timedelta(minutes=5)
    MAX_ADAPTATIONS_PER_HOUR: int = 12
    
    # Adaptation weights
    STRATEGY_WEIGHT: float = 0.4
    RISK_WEIGHT: float = 0.3
    EXECUTION_WEIGHT: float = 0.2
    RESOURCE_WEIGHT: float = 0.1
    
    # Validation thresholds
    MIN_CONFIDENCE_THRESHOLD: float = 0.8
    MAX_RISK_INCREASE: float = 0.2
    MAX_RESOURCE_USAGE: float = 0.9
    
    # Adaptation types
    ALLOWED_ADAPTATIONS: List[str] = [
        "strategy_parameters",
        "risk_parameters",
        "execution_parameters",
        "resource_allocation",
        "portfolio_weights"
    ]

@dataclass
class OptimizationConfig:
    """System optimization configuration"""
    OPTIMIZATION_INTERVAL: timedelta = timedelta(hours=1)
    MIN_OPTIMIZATION_SAMPLES: int = 1000
    PERFORMANCE_BUFFER_SIZE: int = 5000
    
    # Improvement thresholds
    MIN_IMPROVEMENT_THRESHOLD: float = 0.05
    SIGNIFICANT_IMPROVEMENT_THRESHOLD: float = 0.1
    
    # Resource constraints
    MAX_MEMORY_USAGE: float = 0.9
    MAX_CPU_USAGE: float = 0.8
    MAX_GPU_USAGE: float = 0.9
    
    # Optimization targets
    TARGET_SHARPE_RATIO: float = 2.0
    TARGET_MAX_DRAWDOWN: float = 0.1
    TARGET_WIN_RATE: float = 0.6

@dataclass
class MetaLearningConfig:
    """Main meta-learning system configuration"""
    MODEL: MetaModelConfig = MetaModelConfig()
    EXPERIENCE: ExperienceConfig = ExperienceConfig()
    ADAPTATION: AdaptationConfig = AdaptationConfig()
    OPTIMIZATION: OptimizationConfig = OptimizationConfig()
    
    # System settings
    LOGGING_LEVEL: str = "INFO"
    DEBUG_MODE: bool = False
    SAVE_INTERVAL: timedelta = timedelta(hours=6)
    
    # Component weights
    COMPONENT_WEIGHTS: Dict[str, float] = {
        'pattern_recognition': 0.3,
        'experience_learning': 0.3,
        'adaptation_generation': 0.4
    }
    
    # Market conditions
    MARKET_CONDITIONS: List[str] = [
        "trending",
        "ranging",
        "volatile",
        "low_volatility",
        "high_correlation",
        "crisis"
    ]
    
    # Integration settings
    MAX_PARALLEL_ADAPTATIONS: int = 3
    SYNC_INTERVAL: timedelta = timedelta(seconds=30)
    STATE_BUFFER_SIZE: int = 1000
    
    # Safety features
    ENABLE_SAFETY_CHECKS: bool = True
    SAFETY_CHECK_INTERVAL: timedelta = timedelta(minutes=1)
    MAX_CONSECUTIVE_FAILURES: int = 3
    SAFETY_THRESHOLDS: Dict[str, float] = {
        'max_drawdown': 0.15,
        'max_volatility': 0.3,
        'min_liquidity': 0.5,
        'max_exposure': 0.8
    }

    def validate(self) -> bool:
        """Validate configuration parameters"""
        try:
            assert self.MODEL.INPUT_FEATURES > 0
            assert self.MODEL.OUTPUT_FEATURES > 0
            assert all(x > 0 for x in self.MODEL.HIDDEN_LAYERS)
            assert 0 < self.MODEL.DROPOUT_RATE < 1
            assert self.MODEL.LEARNING_RATE > 0
            
            assert self.EXPERIENCE.MAX_EXPERIENCES > 0
            assert self.EXPERIENCE.MIN_EXPERIENCES_FOR_TRAINING > 0
            assert self.EXPERIENCE.PATTERN_WINDOW > 0
            
            assert 0 < self.ADAPTATION.ADAPTATION_THRESHOLD < 1
            assert len(self.ADAPTATION.ALLOWED_ADAPTATIONS) > 0
            
            assert self.OPTIMIZATION.MIN_OPTIMIZATION_SAMPLES > 0
            assert self.OPTIMIZATION.PERFORMANCE_BUFFER_SIZE > 0
            
            return True
            
        except AssertionError:
            return False
            
    def get_component_config(self, component: str) -> Dict:
        """Get configuration for specific component"""
        component_configs = {
            'model': self.MODEL,
            'experience': self.EXPERIENCE,
            'adaptation': self.ADAPTATION,
            'optimization': self.OPTIMIZATION
        }
        return component_configs.get(component, {})
