"""
Algorithm Summary:
This test suite validates the core initialization sequence of the CommandDashboard.
It replaces the legacy IsoMutatorApp tests. It ensures that when the application 
boots, all critical UI references, queues, and task watchers are successfully 
instantiated before the NiceGUI event loop takes over.

Protocol Adherence:
1. Happy Path: Validates successful instantiation of all layout components.
7. Strict Mocking: Mocks the NiceGUI `ui` module to prevent spinning up an 
   actual web server on port 8080 during unit testing.
"""

import pytest
from unittest.mock import patch, MagicMock

# Import the new, refactored class name
from isomutator.ui.app import CommandDashboard

@pytest.fixture
def dashboard_ui():
    """
    Provides an isolated CommandDashboard instance.
    We mock the UI, QueueManager, and TelemetryService to test 
    pure Python initialization without binding to network ports.
    """
    with patch('isomutator.ui.app.QueueManager'), \
         patch('isomutator.ui.app.TelemetryService'), \
         patch('isomutator.ui.app.ui'):
        
        dash = CommandDashboard()
        return dash

class TestCommandDashboardInitialization:

    def test_happy_path_ui_components_instantiated(self, dashboard_ui):
        """
        Happy Path: When CommandDashboard is initialized, it must successfully 
        build the UI and store references to critical input/output elements.
        """
        # Assert Layout Elements
        assert dashboard_ui.main_layout is not None, "Main CSS Grid layout missing"
        assert dashboard_ui.sidebar_container is not None, "Sidebar container missing"
        assert dashboard_ui.telemetry_container is not None, "Telemetry container missing"
        
        # Assert Inputs
        assert dashboard_ui.target_input is not None, "Target URL input missing"
        assert dashboard_ui.strategy_select is not None, "Strategy dropdown missing"
        assert dashboard_ui.context_input is not None, "Context payload input missing"
        
        # Assert Buttons
        assert dashboard_ui.btn_start is not None, "START button missing"
        assert dashboard_ui.btn_stop is not None, "STOP button missing"
        assert dashboard_ui.btn_reconnect is not None, "RECONNECT button missing"
        
        # Assert System State Managers
        assert hasattr(dashboard_ui, 'workers'), "Workers list not initialized"
        assert dashboard_ui.task_watcher is not None, "TaskWatcher not injected"