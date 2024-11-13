from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import logging
from dataclasses import dataclass

@dataclass
class AdaptationEvent:
    """Represents a system adaptation event"""
    trigger: Dict  # What triggered the adaptation
    action: Dict  # What adaptation was made
    result: Dict  # Result of the adaptation
    timestamp: datetime = datetime.now()

class AdaptationTracker:
    """Tracks and analyzes system adaptations"""
    
    def __init__(self, config):
        self.config = config
        self.adaptation_history = []
        self.success_patterns = {}
        self.failure_patterns = {}
        self.adaptation_metrics = {}
        self.logger = logging.getLogger(__name__)
        
    async def track_adaptation(self, event: AdaptationEvent) -> Dict:
        """Track and analyze an adaptation event"""
        try:
            # Record event
            self.adaptation_history.append(event)
            
            # Analyze adaptation
            analysis = self._analyze_adaptation(event)
            
            # Update metrics
            self._update_metrics(analysis)
            
            # Update patterns
            self._update_patterns(event, analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error tracking adaptation: {str(e)}")
            raise
            
    def _analyze_adaptation(self, event: AdaptationEvent) -> Dict:
        """Analyze adaptation event"""
        try:
            # Calculate effectiveness
            effectiveness = self._calculate_effectiveness(event)
            
            # Calculate adaptation speed
            speed = self._calculate_adaptation_speed(event)
            
            # Calculate stability impact
            stability = self._calculate_stability_impact(event)
            
            # Calculate learning progress
            learning = self._calculate_learning_progress(event)
            
            return {
                'effectiveness': effectiveness,
                'speed': speed,
                'stability': stability,
                'learning': learning,
                'timestamp': event.timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing adaptation: {str(e)}")
            raise
            
    def _calculate_effectiveness(self, event: AdaptationEvent) -> float:
        """Calculate effectiveness of adaptation"""
        try:
            # Get performance before and after
            before_performance = self._get_performance_before(event)
            after_performance = self._get_performance_after(event)
            
            # Calculate improvement
            improvement = (after_performance - before_performance) / max(abs(before_performance), 1e-6)
            
            # Normalize to [0, 1]
            effectiveness = 1 / (1 + np.exp(-improvement))  # Sigmoid
            
            return float(effectiveness)
            
        except Exception as e:
            self.logger.error(f"Error calculating effectiveness: {str(e)}")
            return 0.0
            
    def _calculate_adaptation_speed(self, event: AdaptationEvent) -> float:
        """Calculate how quickly the system adapted"""
        try:
            # Get timestamps
            trigger_time = event.trigger.get('timestamp', event.timestamp)
            completion_time = event.result.get('timestamp', event.timestamp)
            
            # Calculate duration in seconds
            duration = (completion_time - trigger_time).total_seconds()
            
            # Normalize to [0, 1]
            speed = np.exp(-duration / self.config.ADAPTATION_TIMEFRAME)
            
            return float(speed)
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptation speed: {str(e)}")
            return 0.0
            
    def _calculate_stability_impact(self, event: AdaptationEvent) -> float:
        """Calculate impact on system stability"""
        try:
            # Get stability metrics before and after
            before_stability = self._get_stability_before(event)
            after_stability = self._get_stability_after(event)
            
            # Calculate change in stability
            stability_change = after_stability - before_stability
            
            # Normalize to [-1, 1]
            impact = np.tanh(stability_change)
            
            return float(impact)
            
        except Exception as e:
            self.logger.error(f"Error calculating stability impact: {str(e)}")
            return 0.0
            
    def _calculate_learning_progress(self, event: AdaptationEvent) -> float:
        """Calculate learning progress from adaptation"""
        try:
            # Get similar past adaptations
            similar_adaptations = self._find_similar_adaptations(event)
            
            if not similar_adaptations:
                return 1.0  # New adaptation type
                
            # Calculate average effectiveness of similar adaptations
            avg_effectiveness = np.mean([
                self._calculate_effectiveness(adaptation)
                for adaptation in similar_adaptations
            ])
            
            # Calculate current effectiveness
            current_effectiveness = self._calculate_effectiveness(event)
            
            # Calculate progress
            progress = (current_effectiveness - avg_effectiveness) / max(avg_effectiveness, 1e-6)
            
            return float(progress)
            
        except Exception as e:
            self.logger.error(f"Error calculating learning progress: {str(e)}")
            return 0.0
            
    def _update_patterns(self, event: AdaptationEvent, analysis: Dict):
        """Update adaptation patterns"""
        try:
            # Extract pattern features
            features = self._extract_pattern_features(event)
            
            # Update success patterns if effective
            if analysis['effectiveness'] > self.config.SUCCESS_THRESHOLD:
                self._update_success_patterns(features)
            
            # Update failure patterns if ineffective
            if analysis['effectiveness'] < self.config.FAILURE_THRESHOLD:
                self._update_failure_patterns(features)
                
        except Exception as e:
            self.logger.error(f"Error updating patterns: {str(e)}")
            
    def get_adaptation_metrics(self) -> Dict:
        """Get current adaptation metrics"""
        return {
            'success_rate': self._calculate_success_rate(),
            'average_effectiveness': self._calculate_average_effectiveness(),
            'average_speed': self._calculate_average_speed(),
            'stability_score': self._calculate_stability_score(),
            'learning_rate': self._calculate_learning_rate(),
            'pattern_metrics': self._get_pattern_metrics()
        }
