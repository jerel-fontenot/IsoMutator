"""
Algorithm Summary:
This test suite validates the UI state-mutation logic for the Strategy Selection dropdown. 
It ensures that the CommandDashboard correctly intercepts ValueChange events and 
dynamically enables or disables the Context Payload input field based on the 
selected attack vector, preventing invalid state combinations.
"""

import pytest
from unittest.mock import MagicMock, patch

# Import the dashboard; the execution guard in app.py prevents the server from starting
from isomutator.ui.app import CommandDashboard

# Mocking a high-level NiceGUI ValueChangeEvent
class MockValueChangeEvent:
    def __init__(self, value):
        self.value = value

@pytest.fixture
def dashboard_ui():
    """Provides a CommandDashboard instance with isolated UI components."""
    with patch('isomutator.ui.app.QueueManager'), \
         patch('isomutator.ui.app.TelemetryService'), \
         patch('isomutator.ui.app.ui'):
        
        dash = CommandDashboard()
        # Mock the specific UI element we are testing state against
        dash.context_input = MagicMock()
        return dash

class TestStrategySelection:

    def test_happy_path_context_strategy_enables_input(self, dashboard_ui):
        """
        Happy Path: If the user selects the 'context' strategy, the application 
        must unlock the context payload input field.
        """
        # Arrange
        event = MockValueChangeEvent(value='context')
        
        # Act
        dashboard_ui._on_strategy_change(event)
        
        # Assert
        dashboard_ui.context_input.enable.assert_called_once()
        dashboard_ui.context_input.disable.assert_not_called()

    def test_edge_case_other_strategies_disable_input(self, dashboard_ui):
        """
        Edge Case: If the user selects any strategy other than 'context' 
        (e.g., 'jailbreak', 'prompt_leaking'), the application must lock 
        the context payload input field to prevent invalid parameters.
        """
        # Arrange
        event = MockValueChangeEvent(value='prompt_leaking')
        
        # Act
        dashboard_ui._on_strategy_change(event)
        
        # Assert
        dashboard_ui.context_input.disable.assert_called_once()
        dashboard_ui.context_input.enable.assert_not_called()

    def test_error_handling_null_event_value(self, dashboard_ui, caplog):
        """
        Error Handling: If the event payload unexpectedly contains a None value,
        the application should default to a safe state (disabled) and not crash.
        """
        # Arrange
        event = MockValueChangeEvent(value=None)
        
        # Act
        dashboard_ui._on_strategy_change(event)
        
        # Assert
        dashboard_ui.context_input.disable.assert_called_once()