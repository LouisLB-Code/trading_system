import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from scipy.stats import entropy
from sklearn.cluster import DBSCAN

@dataclass
class LearningPattern:
    """Represents a detected learning pattern"""
    pattern_type: str
    features: np.ndarray
    frequency: float
    confidence: float
    impact: float
    context: Dict
    first_seen: datetime
    last_seen: datetime

class PatternExtractor:
    """Extracts and analyzes learning patterns from system behavior"""
    
    def __init__(self, config):
        self.config = config
        self.pattern_database = {}
        self.pattern_clusters = {}
        self.temporal_patterns = []
        self.logger = logging.getLogger(__name__)
        
    async def extract_patterns(self, 
                             performance_history: List[Dict],
                             market_conditions: Dict) -> Dict[str, List[LearningPattern]]:
        """Extract learning patterns from historical data"""
        try:
            # Extract different types of patterns
            strategy_patterns = await self._extract_strategy_patterns(
                performance_history
            )
            
            adaptation_patterns = await self._extract_adaptation_patterns(
                performance_history
            )
            
            market_response_patterns = await self._extract_market_response_patterns(
                performance_history,
                market_conditions
            )
            
            # Combine and analyze patterns
            combined_patterns = self._combine_patterns(
                strategy_patterns,
                adaptation_patterns,
                market_response_patterns
            )
            
            # Update pattern database
            self._update_pattern_database(combined_patterns)
            
            return {
                'strategy_patterns': strategy_patterns,
                'adaptation_patterns': adaptation_patterns,
                'market_response_patterns': market_response_patterns,
                'combined_patterns': combined_patterns
            }
            
        except Exception as e:
            self.logger.error(f"Pattern extraction error: {str(e)}")
            raise
            
    async def _extract_strategy_patterns(self, 
                                       performance_history: List[Dict]) -> List[LearningPattern]:
        """Extract patterns related to strategy behavior"""
        try:
            patterns = []
            
            # Group by strategy
            strategy_groups = self._group_by_strategy(performance_history)
            
            for strategy_name, strategy_data in strategy_groups.items():
                # Extract performance patterns
                performance_pattern = self._extract_performance_pattern(
                    strategy_data
                )
                if performance_pattern:
                    patterns.append(performance_pattern)
                
                # Extract adaptation patterns
                adaptation_pattern = self._extract_adaptation_pattern(
                    strategy_data
                )
                if adaptation_pattern:
                    patterns.append(adaptation_pattern)
                
                # Extract failure patterns
                failure_pattern = self._extract_failure_pattern(
                    strategy_data
                )
                if failure_pattern:
                    patterns.append(failure_pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Strategy pattern extraction error: {str(e)}")
            return []
            
    async def _extract_adaptation_patterns(self,
                                         performance_history: List[Dict]) -> List[LearningPattern]:
        """Extract patterns in system adaptation"""
        try:
            patterns = []
            
            # Sort by timestamp
            sorted_history = sorted(
                performance_history,
                key=lambda x: x['timestamp']
            )
            
            # Analyze adaptation sequences
            adaptation_sequences = self._extract_adaptation_sequences(
                sorted_history
            )
            
            # Cluster similar adaptations
            adaptation_clusters = self._cluster_adaptations(
                adaptation_sequences
            )
            
            # Create patterns from clusters
            for cluster in adaptation_clusters:
                pattern = self._create_adaptation_pattern(cluster)
                if pattern:
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Adaptation pattern extraction error: {str(e)}")
            return []
            
    def _cluster_adaptations(self, 
                           sequences: List[Dict]) -> List[List[Dict]]:
        """Cluster similar adaptation sequences"""
        try:
            # Extract features for clustering
            features = np.array([
                self._extract_adaptation_features(seq)
                for seq in sequences
            ])
            
            # Normalize features
            normalized_features = (features - features.mean(axis=0)) / features.std(axis=0)
            
            # Perform clustering
            clustering = DBSCAN(
                eps=self.config.CLUSTERING_EPS,
                min_samples=self.config.MIN_CLUSTER_SAMPLES
            ).fit(normalized_features)
            
            # Group sequences by cluster
            clusters = {}
            for i, label in enumerate(clustering.labels_):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(sequences[i])
            
            return list(clusters.values())
            
        except Exception as e:
            self.logger.error(f"Adaptation clustering error: {str(e)}")
            return []
            
    def _extract_adaptation_features(self, sequence: Dict) -> np.ndarray:
        """Extract features from adaptation sequence"""
        try:
            features = []
            
            # Performance change
            features.append(
                sequence['end_performance'] - sequence['start_performance']
            )
            
            # Adaptation speed
            features.append(sequence['adaptation_speed'])
            
            # Stability impact
            features.append(sequence['stability_impact'])
            
            # Resource usage
            features.append(sequence['resource_usage'])
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {str(e)}")
            return np.zeros(4)
            
    def _create_adaptation_pattern(self, cluster: List[Dict]) -> Optional[LearningPattern]:
        """Create pattern from adaptation cluster"""
        try:
            if not cluster:
                return None
                
            # Calculate pattern features
            features = np.mean([
                self._extract_adaptation_features(seq)
                for seq in cluster
            ], axis=0)
            
            # Calculate pattern frequency
            frequency = len(cluster) / self.config.PATTERN_WINDOW
            
            # Calculate pattern confidence
            confidence = self._calculate_pattern_confidence(cluster)
            
            # Calculate pattern impact
            impact = self._calculate_pattern_impact(cluster)
            
            # Get context
            context = self._extract_pattern_context(cluster)
            
            # Get timestamps
            first_seen = min(seq['timestamp'] for seq in cluster)
            last_seen = max(seq['timestamp'] for seq in cluster)
            
            return LearningPattern(
                pattern_type='adaptation',
                features=features,
                frequency=frequency,
                confidence=confidence,
                impact=impact,
                context=context,
                first_seen=first_seen,
                last_seen=last_seen
            )
            
        except Exception as e:
            self.logger.error(f"Pattern creation error: {str(e)}")
            return None
            
    def _calculate_pattern_confidence(self, cluster: List[Dict]) -> float:
        """Calculate confidence score for pattern"""
        try:
            # Calculate performance consistency
            performances = [
                seq['end_performance'] - seq['start_performance']
                for seq in cluster
            ]
            performance_std = np.std(performances)
            
            # Calculate timing consistency
            timings = [seq['adaptation_speed'] for seq in cluster]
            timing_std = np.std(timings)
            
            # Calculate stability consistency
            stabilities = [seq['stability_impact'] for seq in cluster]
            stability_std = np.std(stabilities)
            
            # Combine metrics
            confidence = 1.0 / (1.0 + performance_std + timing_std + stability_std)
            
            return float(confidence)
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {str(e)}")
            return 0.0
            
    def _calculate_pattern_impact(self, cluster: List[Dict]) -> float:
        """Calculate impact score for pattern"""
        try:
            impacts = []
            
            for seq in cluster:
                # Performance impact
                perf_impact = seq['end_performance'] - seq['start_performance']
                
                # Stability impact
                stability_impact = seq['stability_impact']
                
                # Resource efficiency
                resource_impact = 1.0 - seq['resource_usage']
                
                # Combine impacts
                total_impact = (
                    self.config.PERFORMANCE_WEIGHT * perf_impact +
                    self.config.STABILITY_WEIGHT * stability_impact +
                    self.config.RESOURCE_WEIGHT * resource_impact
                )
                
                impacts.append(total_impact)
            
            return float(np.mean(impacts))
            
        except Exception as e:
            self.logger.error(f"Impact calculation error: {str(e)}")
            return 0.0
