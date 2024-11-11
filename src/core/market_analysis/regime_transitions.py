# File: src/core/market_analysis/regime_transitions.py

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass
from .condition_analyzer import MarketCondition

@dataclass
class TransitionCharacteristics:
    speed: float  # How quickly the transition occurred
    smoothness: float  # How smooth vs. abrupt the transition was
    significance: float  # How significant the change in market conditions
    stability: float  # How stable the new regime appears to be

class RegimeTransitionManager:
    """Manages and analyzes regime transitions"""
    def __init__(self, config):
        self.config = config
        self.transition_matrix = pd.DataFrame()
        self.transition_history = []
        
    def _record_transition(self,
                         from_regime: str,
                         to_regime: str,
                         market_conditions: MarketCondition):
        """Record regime transition in history"""
        try:
            transition = {
                'from_regime': from_regime,
                'to_regime': to_regime,
                'conditions': market_conditions,
                'timestamp': datetime.now()
            }
            
            self.transition_history.append(transition)
            self._update_transition_matrix(from_regime, to_regime)
            
        except Exception as e:
            logging.error(f"Error recording transition: {str(e)}")
            
    def _calculate_transition_probability(self,
                                       from_regime: str,
                                       to_regime: str) -> float:
        """Calculate probability of transition between regimes"""
        try:
            if from_regime not in self.transition_matrix.index:
                return 0.0
                
            total_transitions = self.transition_matrix.loc[from_regime].sum()
            if total_transitions == 0:
                return 0.0
                
            probability = (
                self.transition_matrix.loc[from_regime, to_regime] /
                total_transitions
            )
            
            return float(probability)
            
        except Exception as e:
            logging.error(f"Error calculating transition probability: {str(e)}")
            return 0.0
            
    def _analyze_transition_characteristics(self,
                                         from_regime: str,
                                         to_regime: str,
                                         market_conditions: MarketCondition) -> TransitionCharacteristics:
        """Analyze characteristics of regime transition"""
        try:
            # Calculate transition speed
            speed = self._calculate_transition_speed(
                from_regime,
                to_regime
            )
            
            # Calculate smoothness
            smoothness = self._calculate_transition_smoothness(
                market_conditions
            )
            
            # Calculate significance
            significance = self._calculate_transition_significance(
                from_regime,
                to_regime,
                market_conditions
            )
            
            # Calculate stability
            stability = market_conditions.regime_stability
            
            return TransitionCharacteristics(
                speed=speed,
                smoothness=smoothness,
                significance=significance,
                stability=stability
            )
            
        except Exception as e:
            logging.error(f"Error analyzing transition characteristics: {str(e)}")
            return TransitionCharacteristics(0.0, 0.0, 0.0, 0.0)
            
    def _find_similar_transitions(self,
                                from_regime: str,
                                to_regime: str,
                                market_conditions: MarketCondition) -> List[Dict]:
        """Find similar historical transitions"""
        try:
            similar_transitions = []
            
            for transition in self.transition_history:
                if (transition['from_regime'] == from_regime and
                    transition['to_regime'] == to_regime):
                    
                    similarity = self._calculate_condition_similarity(
                        transition['conditions'],
                        market_conditions
                    )
                    
                    if similarity > self.config.SIMILARITY_THRESHOLD:
                        similar_transitions.append({
                            'timestamp': transition['timestamp'],
                            'similarity': similarity,
                            'conditions': transition['conditions']
                        })
            
            return sorted(
                similar_transitions,
                key=lambda x: x['similarity'],
                reverse=True
            )[:5]  # Return top 5 similar transitions
            
        except Exception as e:
            logging.error(f"Error finding similar transitions: {str(e)}")
            return []
            
    def _update_transition_matrix(self, from_regime: str, to_regime: str):
        """Update transition probability matrix"""
        try:
            if from_regime not in self.transition_matrix.index:
                self.transition_matrix.loc[from_regime, :] = 0
            if to_regime not in self.transition_matrix.columns:
                self.transition_matrix.loc[:, to_regime] = 0
                
            self.transition_matrix.loc[from_regime, to_regime] += 1
            
        except Exception as e:
            logging.error(f"Error updating transition matrix: {str(e)}")
            
    def _calculate_transition_speed(self,
                                 from_regime: str,
                                 to_regime: str) -> float:
        """Calculate how quickly the transition occurred"""
        # Implementation specific to your needs
        return 0.5
        
    def _calculate_transition_smoothness(self,
                                      market_conditions: MarketCondition) -> float:
        """Calculate how smooth the transition was"""
        # Implementation specific to your needs
        return 0.5
        
    def _calculate_transition_significance(self,
                                        from_regime: str,
                                        to_regime: str,
                                        market_conditions: MarketCondition) -> float:
        """Calculate significance of the regime change"""
        # Implementation specific to your needs
        return 0.5
        
    def _calculate_condition_similarity(self,
                                     conditions1: MarketCondition,
                                     conditions2: MarketCondition) -> float:
        """Calculate similarity between market conditions"""
        try:
            features1 = self._get_condition_features(conditions1)
            features2 = self._get_condition_features(conditions2)
            
            # Calculate Euclidean distance
            distance = np.sqrt(np.sum((features1 - features2) ** 2))
            
            # Convert distance to similarity score
            similarity = 1 / (1 + distance)
            
            return float(similarity)
            
        except Exception as e:
            logging.error(f"Error calculating condition similarity: {str(e)}")
            return 0.0
            
    def _get_condition_features(self, conditions: MarketCondition) -> np.ndarray:
        """Convert market conditions to feature vector"""
        return np.array([
            conditions.volatility,
            conditions.trend_strength,
            conditions.liquidity,
            conditions.volume_profile,
            conditions.momentum,
            conditions.efficiency_ratio,
            conditions.fractal_dimension,
            conditions.regime_stability
        ])