# config/ml_pipeline_config.py

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class MLPipelineConfig:
    """Configuration for ML pipeline"""
    
    # Feature engineering
    FEATURE_WINDOWS: List[int] = [5, 10, 20, 50, 100]
    INTERACTION_DEGREE: int = 2
    MAX_LAG: int = 10
    MIN_FEATURE_IMPORTANCE: float = 0.01
    
    # Model selection
    MODEL_TYPES: List[str] = [
        'lightgbm',
        'catboost',
        'neural_net',
        'random_forest'
    ]
    CROSS_VAL_FOLDS: int = 5
    VALIDATION_SIZE: float = 0.2
    TEST_SIZE: float = 0.1
    
    # Hyperparameter optimization
    N_TRIALS: int = 100
    OPTIMIZATION_TIMEOUT: int = 3600
    OPTIMIZATION_THRESHOLD: float = 0.7
    OPTIMIZATION_METRIC: str = 'f1_score'
    
    # Training
    BATCH_SIZE: int = 64
    MAX_EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 10
    LEARNING_RATE: float = 0.001
    
    # Model registry
    MODEL_REGISTRY_PATH: str = "models/registry"
    MAX_MODELS_RETAINED: int = 5
    MODEL_RETENTION_DAYS: int = 30
    
    # Performance thresholds
    MIN_ACCURACY: float = 0.7
    MIN_F1_SCORE: float = 0.7
    MAX_ACCEPTABLE_LOSS: float = 0.1
    
    # Resource constraints
    MAX_MEMORY_USAGE: float = 0.8
    MAX_GPU_USAGE: float = 0.9
    MAX_TRAINING_TIME: int = 7200  # seconds
