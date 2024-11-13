import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
from collections import deque

@dataclass
class Experience:
    """Represents a learning experience"""
    state: Dict
    action: Dict
    reward: float
    next_state: Dict
    metadata: Dict
    timestamp: datetime = datetime.now()

class ExperienceMemoryBank:
    """Manages and analyzes learning experiences"""
    
    def __init__(self, config):
        self.config = config
        self.experiences = deque(maxlen=config.MAX_EXPERIENCES)
        self.importance_weights = {}
        self.pattern_cache = {}
        self.logger = logging.getLogger(__name__)
        
    def add_experience(self, experience: Experience) -> None:
        """Add new experience to memory bank"""
        try:
            # Calculate importance weight
            importance = self._calculate_importance(experience)
            
            # Store experience and weight
            self.experiences.append(experience)
            self.importance_weights[id(experience)] = importance
            
            # Update pattern cache if needed
            if len(self.experiences) % self.config.PATTERN_UPDATE_INTERVAL == 0:
                self._update_pattern_cache()
                
        except Exception as e:
            self.logger.error(f"Error adding experience: {str(e)}")
            
    def get_relevant_experiences(self, 
                               current_state: Dict,
                               k: int = 10) -> List[Experience]:
        """Get most relevant experiences for current state"""
        try:
            similarities = []
            
            for exp in self.experiences:
                similarity = self._calculate_similarity(
                    current_state,
                    exp.state
                )
                similarities.append((exp, similarity))
                
            # Sort by similarity and importance
            relevant_experiences = sorted(
                similarities,
                key=lambda x: x[1] * self.importance_weights[id(x[0])],
                reverse=True
            )
            
            return [exp for exp, _ in relevant_experiences[:k]]
            
        except Exception as e:
            self.logger.error(f"Error getting relevant experiences: {str(e)}")
            return []
            
    def _calculate_importance(self, experience: Experience) -> float:
        """Calculate importance weight for experience"""
        try:
            # Factor in reward magnitude
            reward_factor = abs(experience.reward)
            
            # Factor in novelty
            novelty_factor = self._calculate_novelty(experience)
            
            # Factor in learning impact
            impact_factor = self._calculate_learning_impact(experience)
            
            # Combine factors
            importance = (
                self.config.REWARD_WEIGHT * reward_factor +
                self.config.NOVELTY_WEIGHT * novelty_factor +
                self.config.IMPACT_WEIGHT * impact_factor
            )
            
            return float(importance)
            
        except Exception as e:
            self.logger.error(f"Error calculating importance: {str(e)}")
            return 1.0
            
    def _calculate_similarity(self, state1: Dict, state2: Dict) -> float:
        """Calculate similarity between two states"""
        try:
            # Extract feature vectors
            features1 = self._extract_features(state1)
            features2 = self._extract_features(state2)
            
            # Calculate cosine similarity
            similarity = np.dot(features1, features2) / (
                np.linalg.norm(features1) * np.linalg.norm(features2)
            )
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
            
    def _extract_features(self, state: Dict) -> np.ndarray:
        """Extract feature vector from state"""
        try:
            features = []
            
            # Market features
            if 'market_data' in state:
                features.extend(self._extract_market_features(state['market_data']))
                
            # Strategy features
            if 'strategy_state' in state:
                features.extend(self._extract_strategy_features(state['strategy_state']))
                
            # Risk features
            if 'risk_metrics' in state:
                features.extend(self._extract_risk_features(state['risk_metrics']))
                
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return np.zeros(self.config.FEATURE_DIMENSION)
            
    def _calculate_novelty(self, experience: Experience) -> float:
        """Calculate novelty of experience"""
        try:
            if not self.experiences:
                return 1.0
                
            # Calculate average similarity to existing experiences
            similarities = []
            for exp in self.experiences:
                similarity = self._calculate_similarity(
                    experience.state,
                    exp.state
                )
                similarities.append(similarity)
                
            avg_similarity = np.mean(similarities)
            novelty = 1.0 - avg_similarity
            
            return float(novelty)
            
        except Exception as e:
            self.logger.error(f"Error calculating novelty: {str(e)}")
            return 0.0
            
    def _calculate_learning_impact(self, experience: Experience) -> float:
        """Calculate learning impact of experience"""
        try:
            # Calculate state change magnitude
            state_change = self._calculate_state_change(
                experience.state,
                experience.next_state
            )
            
            # Calculate policy change
            policy_change = self._calculate_policy_change(experience)
            
            # Combine impacts
            impact = (state_change + policy_change) / 2
            
            return float(impact)
            
        except Exception as e:
            self.logger.error(f"Error calculating learning impact: {str(e)}")
            return 0.0
            
    def _update_pattern_cache(self):
        """Update cached pattern analysis"""
        try:
            # Extract recent experiences
            recent_experiences = list(self.experiences)[-self.config.PATTERN_WINDOW:]
            
            # Update success patterns
            self.pattern_cache['success'] = self._extract_success_patterns(
                recent_experiences
            )
            
            # Update failure patterns
            self.pattern_cache['failure'] = self._extract_failure_patterns(
                recent_experiences
            )
            
            # Update adaptation patterns
            self.pattern_cache['adaptation'] = self._extract_adaptation_patterns(
                recent_experiences
            )
            
        except Exception as e:
            self.logger.error(f"Error updating pattern cache: {str(e)}")
