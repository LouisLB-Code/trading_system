# tests/unit/test_error_handling.py
import pytest
import asyncio
from datetime import datetime
from src.core.error_handling import (
    AdvancedErrorHandler, 
    ErrorContext,
    ErrorPerformanceTracker,
    ErrorPatternAnalyzer,
    RecoveryStrategyOptimizer
)
from src.exceptions import (
    DataValidationError,
    OrderExecutionError,
    StrategyError,
    SystemError
)

class TestErrorHandling:
    @pytest.fixture
    def setup(self):
        """Setup test environment"""
        self.error_handler = AdvancedErrorHandler()
        self.performance_tracker = ErrorPerformanceTracker()
        self.pattern_analyzer = ErrorPatternAnalyzer()
        self.strategy_optimizer = RecoveryStrategyOptimizer()
        
    @pytest.fixture
    def sample_error_context(self):
        """Create a sample error context"""
        return ErrorContext(
            error_type="ORDER_EXECUTION_ERROR",
            component="trading_system",
            severity="high",
            timestamp=datetime.now(),
            stack_trace="".join(traceback.format_stack()),
            system_state={"status": "error", "active_orders": []},
            recovery_attempts=0
        )

    async def test_error_handling_basic(self, setup):
        """Test basic error handling functionality"""
        # Simulate an order execution error
        error = OrderExecutionError("Failed to execute market order")
        component = "trading_system"
        system_state = {
            "status": "active",
            "active_orders": [],
            "positions": {}
        }
        
        result = await self.error_handler.handle_error(
            error=error,
            component=component,
            system_state=system_state
        )
        
        assert result is not None
        assert 'error' in result
        assert 'analysis' in result
        assert 'recovery' in result
        assert result['error'].error_type == "ORDER_EXECUTION_ERROR"
        
    async def test_error_analysis(self, setup, sample_error_context):
        """Test error analysis functionality"""
        analysis = self.error_handler._analyze_error(sample_error_context)
        
        assert 'pattern_match' in analysis
        assert 'severity' in analysis
        assert 'impact' in analysis
        assert 'cascading_errors' in analysis
        assert 'recommended_actions' in analysis
        
    async def test_recovery_execution(self, setup, sample_error_context):
        """Test recovery strategy execution"""
        strategy = {
            'steps': [
                {
                    'action': 'validate_system_state',
                    'params': {'timeout': 5}
                },
                {
                    'action': 'reset_connection',
                    'params': {'retry_count': 3}
                }
            ]
        }
        
        recovery_result = await self.error_handler._execute_recovery(
            strategy=strategy,
            context=sample_error_context
        )
        
        assert 'success' in recovery_result
        assert 'steps_completed' in recovery_result
        assert 'results' in recovery_result
        assert 'recovery_id' in recovery_result
        
    def test_performance_tracking(self, setup, sample_error_context):
        """Test error handling performance tracking"""
        recovery_result = {
            'success': True,
            'steps_completed': 2,
            'results': [
                {'success': True, 'action': 'validate_system_state'},
                {'success': True, 'action': 'reset_connection'}
            ],
            'recovery_id': 'test_recovery_1'
        }
        
        self.performance_tracker.track_recovery(
            context=sample_error_context,
            recovery_result=recovery_result
        )
        
        # Verify metrics were recorded
        assert sample_error_context.error_type in self.performance_tracker.pattern_effectiveness
        metrics = self.performance_tracker.pattern_effectiveness[sample_error_context.error_type]
        assert metrics['attempts'] > 0
        assert metrics['successes'] > 0
        
    def test_pattern_analysis(self, setup):
        """Test error pattern analysis"""
        # Create sample error history
        error_history = [
            ErrorContext(
                error_type="NETWORK_ERROR",
                component="data_collector",
                severity="medium",
                timestamp=datetime.now(),
                stack_trace="",
                system_state={},
                recovery_attempts=1
            ) for _ in range(3)  # Create 3 similar errors
        ]
        
        analysis = self.pattern_analyzer.analyze_pattern(error_history)
        
        assert 'common_patterns' in analysis
        assert 'sequences' in analysis
        assert 'correlations' in analysis
        assert 'insights' in analysis
        assert 'recommendations' in analysis
        
        # Verify pattern detection
        patterns = analysis['common_patterns']
        assert len(patterns) > 0
        assert ('NETWORK_ERROR', 'data_collector') in patterns
        
    def test_strategy_optimization(self, setup):
        """Test recovery strategy optimization"""
        performance_data = {
            'success_rate': 0.75,
            'avg_recovery_time': 2.5,
            'failure_patterns': ['timeout', 'connection_reset'],
            'recovery_costs': {'attempts': 10, 'system_resources': 'medium'}
        }
        
        optimization = self.strategy_optimizer.optimize_strategy(
            error_type="NETWORK_ERROR",
            performance_data=performance_data
        )
        
        assert 'optimization' in optimization
        assert 'performance_analysis' in optimization
        assert 'expected_improvement' in optimization
        
    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, setup):
        """Test handling multiple errors concurrently"""
        errors = [
            OrderExecutionError("Failed to execute order"),
            DataValidationError("Invalid market data"),
            StrategyError("Strategy execution timeout")
        ]
        
        component = "trading_system"
        system_state = {"status": "active"}
        
        # Handle multiple errors concurrently
        tasks = [
            self.error_handler.handle_error(
                error=error,
                component=component,
                system_state=system_state
            )
            for error in errors
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(errors)
        assert all('error' in result for result in results)
        assert all('recovery' in result for result in results)
        
    @pytest.mark.asyncio
    async def test_error_handler_failure(self, setup):
        """Test error handler's behavior when it fails"""
        # Create an error that will cause the error handler to fail
        class UnhandleableError(Exception):
            def __str__(self):
                raise Exception("Cannot convert error to string")
        
        error = UnhandleableError()
        component = "trading_system"
        system_state = {"status": "active"}
        
        result = await self.error_handler.handle_error(
            error=error,
            component=component,
            system_state=system_state
        )
        
        assert result['error'] is not None
        assert not result.get('success', False)
