# File: src/core/strategy/evolution/strategy_evolution.py

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
from copy import deepcopy

@dataclass
class StrategyGenome:
    indicators: List[str]
    entry_conditions: List[str]
    exit_conditions: List[str]
    parameters: Dict[str, float]
    fitness: float = 0.0
    generation: int = 0
    ancestry: List[str] = None

class StrategyEvolution:
    """Evolves trading strategies using genetic programming and deep learning"""
    
    def __init__(self, config):
        self.config = config
        self.population = []
        self.generation = 0
        self.elite_strategies = []
        self.mutation_rate = config.evolution.INITIAL_MUTATION_RATE
        self.crossover_rate = config.evolution.CROSSOVER_RATE
        self.neural_evaluator = self._build_neural_evaluator()
        self.memory_bank = StrategyMemoryBank(config)
        
    async def evolve_generation(self,
                              performance_data: Dict,
                              market_conditions: Dict) -> List['BaseStrategy']:
        """Evolve a new generation of strategies"""
        try:
            # Evaluate current population
            fitness_scores = await self._evaluate_population(
                performance_data,
                market_conditions
            )
            
            # Update memory bank
            self.memory_bank.update(
                self.population,
                fitness_scores
            )
            
            # Select parents for next generation
            parents = self._select_parents(fitness_scores)
            
            # Create new generation
            new_population = []
            elite_count = int(self.config.evolution.ELITE_PERCENTAGE * len(self.population))
            
            # Keep elite strategies
            new_population.extend(
                self._get_elite_strategies(elite_count)
            )
            
            # Generate new strategies through crossover and mutation
            while len(new_population) < self.config.evolution.POPULATION_SIZE:
                if np.random.random() < self.crossover_rate:
                    # Crossover
                    parent1, parent2 = self._select_parent_pair(parents)
                    child = self._crossover(parent1, parent2)
                else:
                    # Mutation
                    parent = self._select_parent(parents)
                    child = self._mutate(parent)
                
                new_population.append(child)
            
            # Update population
            self.population = new_population
            self.generation += 1
            
            # Adapt rates
            self._adapt_rates(fitness_scores)
            
            return self._convert_to_strategies(new_population)
            
        except Exception as e:
            logging.error(f"Evolution error: {str(e)}")
            raise
    
    async def _evaluate_population(self,
                                 performance_data: Dict,
                                 market_conditions: Dict) -> Dict[str, float]:
        """Evaluate fitness of all strategies"""
        try:
            fitness_scores = {}
            
            for strategy in self.population:
                # Calculate basic performance metrics
                performance_score = self._calculate_performance_score(
                    strategy,
                    performance_data
                )
                
                # Calculate adaptability score
                adaptability_score = self._calculate_adaptability_score(
                    strategy,
                    market_conditions
                )
                
                # Calculate complexity penalty
                complexity_penalty = self._calculate_complexity_penalty(
                    strategy
                )
                
                # Calculate neural evaluation score
                neural_score = await self._neural_evaluate(
                    strategy,
                    market_conditions
                )
                
                # Combine scores
                final_score = (
                    performance_score * self.config.evolution.PERFORMANCE_WEIGHT +
                    adaptability_score * self.config.evolution.ADAPTABILITY_WEIGHT +
                    neural_score * self.config.evolution.NEURAL_WEIGHT -
                    complexity_penalty
                )
                
                fitness_scores[strategy.id] = final_score
            
            return fitness_scores
            
        except Exception as e:
            logging.error(f"Population evaluation error: {str(e)}")
            raise
    
    def _crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> StrategyGenome:
        """Perform crossover between two parent strategies"""
        try:
            child = StrategyGenome(
                indicators=[],
                entry_conditions=[],
                exit_conditions=[],
                parameters={},
                ancestry=[parent1.id, parent2.id]
            )
            
            # Crossover indicators
            child.indicators = self._crossover_indicators(
                parent1.indicators,
                parent2.indicators
            )
            
            # Crossover conditions
            child.entry_conditions = self._crossover_conditions(
                parent1.entry_conditions,
                parent2.entry_conditions
            )
            
            child.exit_conditions = self._crossover_conditions(
                parent1.exit_conditions,
                parent2.exit_conditions
            )
            
            # Crossover parameters
            child.parameters = self._crossover_parameters(
                parent1.parameters,
                parent2.parameters
            )
            
            return child
            
        except Exception as e:
            logging.error(f"Crossover error: {str(e)}")
            raise
    
    def _mutate(self, strategy: StrategyGenome) -> StrategyGenome:
        """Mutate a strategy"""
        try:
            mutated = deepcopy(strategy)
            
            # Mutate indicators
            if np.random.random() < self.mutation_rate:
                mutated.indicators = self._mutate_indicators(
                    mutated.indicators
                )
            
            # Mutate conditions
            if np.random.random() < self.mutation_rate:
                mutated.entry_conditions = self._mutate_conditions(
                    mutated.entry_conditions
                )
                
            if np.random.random() < self.mutation_rate:
                mutated.exit_conditions = self._mutate_conditions(
                    mutated.exit_conditions
                )
            
            # Mutate parameters
            if np.random.random() < self.mutation_rate:
                mutated.parameters = self._mutate_parameters(
                    mutated.parameters
                )
            
            return mutated
            
        except Exception as e:
            logging.error(f"Mutation error: {str(e)}")
            raise
    
    def _build_neural_evaluator(self) -> nn.Module:
        """Build neural network for strategy evaluation"""
        return nn.Sequential(
            nn.Linear(self.config.evolution.STRATEGY_FEATURES, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

class StrategyMemoryBank:
    """Stores and manages successful strategy patterns"""
    
    def __init__(self, config):
        self.config = config
        self.memories = []
        self.success_patterns = {}
        self.failure_patterns = {}
        
    def update(self, population: List[StrategyGenome], fitness_scores: Dict[str, float]):
        """Update memory bank with new information"""
        try:
            # Extract patterns from successful strategies
            successful = [
                s for s in population
                if fitness_scores[s.id] > self.config.evolution.SUCCESS_THRESHOLD
            ]
            
            # Extract patterns from failed strategies
            failed = [
                s for s in population
                if fitness_scores[s.id] < self.config.evolution.FAILURE_THRESHOLD
            ]
            
            # Update pattern databases
            self._update_success_patterns(successful)
            self._update_failure_patterns(failed)
            
            # Prune old memories
            self._prune_memories()
            
        except Exception as e:
            logging.error(f"Memory update error: {str(e)}")
            
    def get_successful_patterns(self, market_conditions: Dict) -> List[Dict]:
        """Get successful patterns similar to current conditions"""
        return self._find_similar_patterns(
            market_conditions,
            self.success_patterns
        )
