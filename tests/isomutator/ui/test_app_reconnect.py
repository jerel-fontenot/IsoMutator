"""
Algorithm Summary:
This test suite validates the state recovery mechanism of the CommandDashboard.
It ensures that the `action_reconnect_wargame` method successfully re-establishes 
the read-only connections to the Redis message broker (QueueManager and TelemetryService) 
and restarts the background polling tasks WITHOUT spawning new worker processes. 
It also verifies safe handling of pre-existing tasks and network connection failures.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from redis.exceptions import ConnectionError

# Assuming the dashboard is imported from your UI module
from isomutator.ui.app import CommandDashboard

@pytest.fixture
def mock_ui_elements():
    """Mocks NiceGUI UI elements to prevent rendering engines from starting during tests."""
    with patch('isomutator.ui.app.ui'), patch('isomutator.ui.app.app'):
        yield

@pytest.fixture
def dashboard(mock_ui_elements):
    """Provides a CommandDashboard instance with isolated UI components."""
    # We mock the build_ui method to avoid complex NiceGUI DOM instantiation in headless tests
    with patch.object(CommandDashboard, 'build_ui'):
        dash = CommandDashboard()
        dash.btn_start = MagicMock()
        dash.btn_stop = MagicMock()
        dash.btn_reconnect = MagicMock()
        dash.target_input = MagicMock()      
        dash.strategy_select = MagicMock()
        dash.system_log = MagicMock()
        dash.telemetry_task = None
        dash.workers = []
        return dash

@pytest.mark.asyncio
class TestCommandDashboardReconnect:

    @patch('isomutator.ui.app.QueueManager')
    @patch('isomutator.ui.app.TelemetryService')
    @patch('isomutator.ui.app.multiprocessing.Process')
    async def test_happy_path_reconnect(self, mock_process, mock_telemetry, mock_qm, dashboard):
        """
        Happy Path: The reconnect action instantiates the broker connections, 
        creates the telemetry listener task, updates UI buttons, but DOES NOT 
        spawn any new multiprocessing workers.
        """
        # Arrange
        dashboard._telemetry_listener = AsyncMock()
        mock_qm.return_value.ping_broker = AsyncMock()

        # Act
        await dashboard.action_reconnect_wargame()

        # Assert
        # UI state should shift to "Active" mode
        dashboard.btn_start.disable.assert_called_once()
        dashboard.btn_reconnect.disable.assert_called_once()
        dashboard.btn_stop.enable.assert_called_once()
        
        # Connections established
        assert mock_qm.call_count >= 1  # Should instantiate telemetry queues
        assert mock_telemetry.call_count == 1
        
        # Telemetry task started
        assert dashboard.telemetry_task is not None
        
        # CRITICAL: No workers should be spawned
        mock_process.assert_not_called()
        assert len(dashboard.workers) == 0
        dashboard.system_log.push.assert_called_with("[SYSTEM] Reconnected to existing wargame telemetry.")

    @patch('isomutator.ui.app.QueueManager')
    @patch('isomutator.ui.app.TelemetryService')
    async def test_edge_case_existing_task_cancellation(self, mock_telemetry, mock_qm, dashboard):
        """
        Edge Case: If the user clicks RECONNECT while a telemetry task is somehow 
        already running (or orphaned in the asyncio loop), it must be safely 
        cancelled before creating a new one to prevent memory leaks and duplicate logs.
        """
        # Arrange
        mock_existing_task = MagicMock(spec=asyncio.Task)
        dashboard.telemetry_task = mock_existing_task
        dashboard._telemetry_listener = AsyncMock()
        mock_qm.return_value.ping_broker = AsyncMock()

        # Act
        await dashboard.action_reconnect_wargame()

        # Assert
        mock_existing_task.cancel.assert_called_once()
        assert dashboard.telemetry_task is not mock_existing_task

    @patch('isomutator.ui.app.QueueManager')
    async def test_error_handling_redis_down_on_reconnect(self, mock_qm, dashboard):
        """
        Error Handling: If Redis is offline when the user attempts to reconnect, 
        the application catches the ConnectionError, pushes a warning to the system log, 
        and resets the UI buttons to allow another attempt.
        """
        # Arrange
        mock_qm.return_value.ping_broker = AsyncMock(side_effect=ConnectionError("Redis is unreachable"))

        # Act
        await dashboard.action_reconnect_wargame()

        # Assert
        dashboard.system_log.push.assert_called_with("[ERROR] Failed to reconnect: Redis broker is offline.")
        dashboard.btn_start.enable.assert_called_once()
        dashboard.btn_reconnect.enable.assert_called_once()
        dashboard.btn_stop.disable.assert_called_once()