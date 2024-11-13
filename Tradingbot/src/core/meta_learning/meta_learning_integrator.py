from typing import Dict, Optional, List, Tuple
import logging
from datetime import datetime
import numpy as np
import torch

from .meta_model_optimizer import MetaModelOptimizer
from .experience_bank import ExperienceMemoryBank, Experience
from .adaptation_tracker import AdaptationTracker, AdaptationEvent
from .learning_patterns import PatternExtractor

class MetaLearningIntegrator:
    """Integrates meta-learning system with main trading system"""
    
    def __init__(self, config, trading_system):
        self.config = config
        self.trading_system = trading_system
        
        # Initialize components
        self.meta_optimizer = MetaModelOptimizer(config)
        self.memory_bank = ExperienceMemoryBank(config)
        self.adaptation_tracker = AdaptationTracker(config)
        self.pattern_extractor = PatternExtractor(config)
        
        self.logger = logging.getLogger(__name__)
        self.last_optimization_time = None
        self.adaptation_queue = []
        self.performance_buffer = []
        
    async def process_market_update(self, market_data: Dict) -> Dict:
        """Process market update through meta-learning system"""
        try:
            # Get current system state
            system_state = self.trading_system.get_state()
            
            # Extract patterns
            patterns = await self.pattern_extractor.extract_patterns(
                self.trading_system.performance_history,
                market_data
            )
            
            # Generate adaptation if needed
            adaptation = await self._generate_adaptation(
                patterns,
                system_state,
                market_data
            )
            
            if adaptation:
                # Apply adaptation
                result = await self._apply_adaptation(adaptation)
                
                # Track adaptation
                analysis = await self.adaptation_tracker.track_adaptation(
                    AdaptationEvent(
                        trigger=patterns,
                        action=adaptation,
                        result=result
                    )
                )
                
                # Store experience
                self._store_experience(
                    state=system_state,
                    action=adaptation,
                    result=result,
                    patterns=patterns
                )
                
                # Optimize meta-model if needed
                if self._should_optimize():
                    await self._optimize_model()
                
            # Update performance buffer
            self._update_performance_buffer(system_state, market_data)
            
            return {
                'patterns': patterns,
                'adaptation': adaptation,
                'analysis': analysis if adaptation else None,
                'optimization_status': self._get_optimization_status()
            }
            
        except Exception as e:
            self.logger.error(f"Meta-learning processing error: {str(e)}")
            return {}
            
    async def _generate_adaptation(self,
                                 patterns: Dict,
                                 system_state: Dict,
                                 market_data: Dict) -> Optional[Dict]:
        """Generate system adaptation based on patterns"""
        try:
            # Get relevant experiences
            experiences = self.memory_bank.get_relevant_experiences(
                system_state,
                k=self.config.NUM_EXPERIENCES
            )
            
            # Analyze adaptation need
            adaptation_need = self._analyze_adaptation_need(
                patterns,
                system_state,
                market_data
            )
            
            if not adaptation_need['required']:
                return None
                
            # Generate adaptation strategy
            strategy = await self._generate_adaptation_strategy(
                adaptation_need,
                experiences,
                patterns
            )
            
            # Validate adaptation
            if not self._validate_adaptation(strategy, system_state):
                return None
                
            return strategy
            
        except Exception as e:
            self.logger.error(f"Adaptation generation error: {str(e)}")
            return None
            
    def _analyze_adaptation_need(self,
                               patterns: Dict,
                               system_state: Dict,
                               market_data: Dict) -> Dict:
        """Analyze if system needs adaptation"""
        try:
            need_scores = {
                'performance': self._analyze_performance_need(system_state),
                'market': self._analyze_market_need(market_data, patterns),
                'resource': self._analyze_resource_need(system_state),
                'risk': self._analyze_risk_need(system_state, market_data)
            }
            
            # Calculate overall need
            overall_need = sum(need_scores.values()) / len(need_scores)
            
            return {
                'required': overall_need > self.config.ADAPTATION_THRESHOLD,
                'scores': need_scores,
                'overall_score': overall_need,
                'priority_areas': self._get_priority_areas(need_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Adaptation need analysis error: {str(e)}")
            return {'required': False}
            
    async def _generate_adaptation_strategy(self,
                                          adaptation_need: Dict,
                                          experiences: List[Experience],
                                          patterns: Dict) -> Dict:
        """Generate adaptation strategy based on need and experiences"""
        try:
            strategy = {
                'priority': adaptation_need['priority_areas'],
                'updates': {}
            }
            
            # Generate strategy updates
            if 'strategy' in adaptation_need['priority_areas']:
                strategy['updates']['strategy'] = await self._generate_strategy_updates(
                    patterns,
                    experiences
                )
                
            # Generate risk updates
            if 'risk' in adaptation_need['priority_areas']:
                strategy['updates']['risk'] = await self._generate_risk_updates(
                    patterns,
                    experiences
                )
                
            # Generate execution updates
            if 'execution' in adaptation_need['priority_areas']:
                strategy['updates']['execution'] = await self._generate_execution_updates(
                    patterns,
                    experiences
                )
                
            # Generate resource updates
            if 'resource' in adaptation_need['priority_areas']:
                strategy['updates']['resource'] = await self._generate_resource_updates(
                    patterns,
                    experiences
                )
                
            return strategy
            
        except Exception as e:
            self.logger.error(f"Adaptation strategy generation error: {str(e)}")
            return {}
            
    async def _apply_adaptation(self, adaptation: Dict) -> Dict:
        """Apply adaptation to trading system"""
        try:
            result = {
                'success': True,
                'updates': {},
                'errors': []
            }
            
            for component, updates in adaptation['updates'].items():
                try:
                    if component == 'strategy':
                        result['updates']['strategy'] = await self.trading_system.update_strategies(
                            updates
                        )
                    elif component == 'risk':
                        result['updates']['risk'] = await self.trading_system.update_risk_parameters(
                            updates
                        )
                    elif component == 'execution':
                        result['updates']['execution'] = await self.trading_system.update_execution_parameters(
                            updates
                        )
                    elif component == 'resource':
                        result['updates']['resource'] = await self.trading_system.update_resource_allocation(
                            updates
                        )
                except Exception as e:
                    result['success'] = False
                    result['errors'].append({
                        'component': component,
                        'error': str(e)
                    })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Adaptation application error: {str(e)}")
            return {'success': False, 'error': str(e)}
            
    def _store_experience(self,
                         state: Dict,
                         action: Dict,
                         result: Dict,
                         patterns: Dict):
        """Store learning experience"""
        try:
            # Calculate reward
            reward = self._calculate_reward(result)
            
            # Create experience
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=self.trading_system.get_state(),
                metadata={
                    'patterns': patterns,
                    'timestamp': datetime.now()
                }
            )
            
            # Store experience
            self.memory_bank.add_experience(experience)
            
        except Exception as e:
            self.logger.error(f"Experience storage error: {str(e)}")
            
    def _calculate_reward(self, result: Dict) -> float:
        """Calculate reward for adaptation"""
        try:
            if not result['success']:
                return -1.0
                
            reward = 0.0
            update_weights = {
                'strategy': 0.4,
                'risk': 0.3,
                'execution': 0.2,
                'resource': 0.1
            }
            
            for component, updates in result['updates'].items():
                if component in update_weights:
                    reward += update_weights[component] * self._calculate_component_reward(
                        component,
                        updates
                    )
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Reward calculation error: {str(e)}")
            return 0.0
            
    async def _optimize_model(self):
        """Optimize meta-learning model"""
        try:
            # Get performance data
            performance_data = {
                'metrics': self.trading_system.get_performance_metrics(),
                'history': self.performance_buffer
            }
            
            # Get adaptation data
            adaptation_data = self.adaptation_tracker.get_adaptation_metrics()
            
            # Optimize model
            metrics = await self.meta_optimizer.optimize(
                performance_data,
                adaptation_data
            )
            
            # Update optimization time
            self.last_optimization_time = datetime.now()
            
            # Log optimization results
            self.logger.info(
                f"Meta-model optimization - Loss: {metrics.loss:.4f}, "
                f"Adaptation Score: {metrics.adaptation_score:.4f}, "
                f"Generalization Score: {metrics.generalization_score:.4f}"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model optimization error: {str(e)}")
            
    def _should_optimize(self) -> bool:
        """Check if model optimization is needed"""
        if not self.last_optimization_time:
            return True
            
        time_elapsed = (datetime.now() - self.last_optimization_time).total_seconds()
        return time_elapsed > self.config.OPTIMIZATION_INTERVAL
        
    def _update_performance_buffer(self, system_state: Dict, market_data: Dict):
        """Update performance buffer with recent state"""
        self.performance_buffer.append({
            'state': system_state,
            'market_data': market_data,
            'timestamp': datetime.now()
        })
        
        # Keep buffer size limited
        if len(self.performance_buffer) > self.config.PERFORMANCE_BUFFER_SIZE:
            self.performance_buffer = self.performance_buffer[-self.config.PERFORMANCE_BUFFER_SIZE:]
            
    def _get_optimization_status(self) -> Dict:
        """Get current optimization status"""
        return {
            'last_optimization': self.last_optimization_time,
            'performance_buffer_size': len(self.performance_buffer),
            'experience_count': len(self.memory_bank.experiences),
            'adaptation_count': len(self.adaptation_tracker.adaptation_history)
        }

    def _validate_adaptation(self, strategy: Dict, system_state: Dict) -> bool:
        """Validate adaptation strategy before application"""
        try:
            # Check resource constraints
            if not self._check_resource_constraints(strategy, system_state):
                return False
                
            # Check risk constraints
            if not self._check_risk_constraints(strategy, system_state):
                return False
                
            # Check stability constraints
            if not self._check_stability_constraints(strategy, system_state):
                return False
                
            # Check adaptation rate
            if not self._check_adaptation_rate(strategy):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Adaptation validation error: {str(e)}")
            return False
