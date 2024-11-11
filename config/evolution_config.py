# File: config/evolution_config.py

from dataclasses import dataclass
from typing import Dict

@dataclass
class EvolutionConfig:
    # Population parameters
    POPULATION_SIZE: int = 100
    ELITE_PERCENTAGE: float = 0.1
    
    # Genetic algorithm parameters
    INITIAL_MUTATION_RATE: float = 0.2
    CROSSOVER_RATE: float = 0.8
    TOURNAMENT_SIZE: int = 5
    
    # Evaluation weights
    PERFORMANCE_WEIGHT: float = 0.5
    ADAPTABILITY_WEIGHT: float = 0.3
    NEURAL_WEIGHT: float = 0.2
    
    # Learning parameters
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    
    # Memory parameters
    MAX_MEMORIES: int = 1000
    MEMORY_PRUNING_THRESHOLD: float = 0.3
    
    # Thresholds
    SUCCESS_THRESHOLD: float = 0.7
    FAILURE_THRESHOLD: float = 0.3
    SIMILARITY_THRESHOLD: float = 0.8
    
    # Strategy features
    STRATEGY_FEATURES: int = 128
    MIN_INDICATORS: int = 3
    MAX_INDICATORS: int = 10
    MAX_CONDITIONS: int = 5
    
    # Adaptation parameters
    ADAPTATION_RATE: float = 0.1
    MIN_MUTATION_RATE: float = 0.05
    MAX_MUTATION_RATE: float = 0.4

class StrategyEvolutionConfig:
    """Configuration for strategy evolution system"""
    
    def __init__(self):
        self.evolution = EvolutionConfig()
        
        # Available components
        self.AVAILABLE_INDICATORS = [
            'SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR',
            'Stochastic', 'ADX', 'OBV', 'VWAP'
        ]
        
        self.CONDITION_TEMPLATES = [
            "CrossAbove({indicator1}, {indicator2})",
            "CrossBelow({indicator1}, {indicator2})",
            "ValueAbove({indicator}, {value})",
            "ValueBelow({indicator}, {value})",
            "IncreasingFor({indicator}, {periods})",
            "DecreasingFor({indicator}, {periods})"
        ]
        
        # Parameter ranges
        self.PARAMETER_RANGES = {
            'fast_period': (5, 50),
            'slow_period': (20, 200),
            'rsi_period': (7, 28),
            'atr_period': (10, 30),
            'stop_loss': (0.01, 0.05),
            'take_profit': (0.02, 0.10)
        }
