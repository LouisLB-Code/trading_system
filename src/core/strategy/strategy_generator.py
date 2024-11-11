# File: src/core/strategy/strategy_generator.py

import numpy as np
from typing import Dict, List
import logging
from dataclasses import dataclass

@dataclass
class StrategyComponent:
    name: str
    type: str
    parameters: Dict
    conditions: List[str]
    weight: float

class StrategyGenerator:
    """Generates trading strategies based on identified patterns"""
    
    def __init__(self, config):
        self.config = config
        self.components_library = self._initialize_components()
        self.strategy_templates = []
        
    def _select_indicators(self, patterns: List['Pattern']) -> List[str]:
        """Select appropriate indicators based on patterns"""
        try:
            indicators = set()
            
            for pattern in patterns:
                # Add pattern-specific indicators
                indicators.update(
                    self._get_pattern_indicators(pattern)
                )
                
                # Add complementary indicators
                indicators.update(
                    self._get_complementary_indicators(pattern)
                )
                
                # Add confirmation indicators
                indicators.update(
                    self._get_confirmation_indicators(pattern)
                )
            
            return list(indicators)
            
        except Exception as e:
            logging.error(f"Indicator selection error: {str(e)}")
            return []
    
    def _generate_entry_conditions(self,
                                 patterns: List['Pattern'],
                                 regime: 'MarketRegime') -> List[str]:
        """Generate entry conditions based on patterns and regime"""
        try:
            conditions = []
            
            # Pattern-based conditions
            for pattern in patterns:
                conditions.extend(
                    self._generate_pattern_conditions(
                        pattern,
                        "entry"
                    )
                )
            
            # Regime-based conditions
            conditions.extend(
                self._generate_regime_conditions(
                    regime,
                    "entry"
                )
            )
            
            # Risk-based conditions
            conditions.extend(
                self._generate_risk_conditions(
                    regime,
                    "entry"
                )
            )
            
            return self._optimize_conditions(conditions)
            
        except Exception as e:
            logging.error(f"Entry condition generation error: {str(e)}")
            return []
    
    def _generate_exit_conditions(self,
                                patterns: List['Pattern'],
                                regime: 'MarketRegime') -> List[str]:
        """Generate exit conditions based on patterns and regime"""
        try:
            conditions = []
            
            # Pattern-based exit conditions
            for pattern in patterns:
                conditions.extend(
                    self._generate_pattern_conditions(
                        pattern,
                        "exit"
                    )
                )
            
            # Regime-based exit conditions
            conditions.extend(
                self._generate_regime_conditions(
                    regime,
                    "exit"
                )
            )
            
            # Protection conditions
            conditions.extend(
                self._generate_protection_conditions(regime)
            )
            
            return self._optimize_conditions(conditions)
            
        except Exception as e:
            logging.error(f"Exit condition generation error: {str(e)}")
            return []
    
    def _generate_strategy_code(self, template: 'StrategyTemplate') -> str:
        """Generate Python code for strategy implementation"""
        try:
            # Generate strategy class
            code = self._generate_class_definition(template)
            
            # Add indicators calculation
            code += self._generate_indicators_code(template.indicators)
            
            # Add entry logic
            code += self._generate_entry_logic(template.entry_conditions)
            
            # Add exit logic
            code += self._generate_exit_logic(template.exit_conditions)
            
            # Add risk management
            code += self._generate_risk_management(template.risk_parameters)
            
            return code
            
        except Exception as e:
            logging.error(f"Strategy code generation error: {str(e)}")
            raise
