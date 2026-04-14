"""
Algorithm Summary:
This test suite validates the massive teardown sequence orchestrated by the 
action_stop_wargame method. 

Protocol Adherence:
1. Happy Path: Validates Poison Pill broadcasting, graceful worker joining, and UI state reset.
2. Edge Cases: Validates idempotency (clicking STOP when already stopped doesn't crash).
3. Error Handling: Handles cases where the broker connection fails during teardown.
5. Timeout & Latency: Validates that if a worker process hangs for > 3 seconds during join(), 
   the UI ruthlessly calls .terminate() to prevent a permanent UI freeze.
6. Resource Teardown: Ensures the asyncio telemetry task is explicitly cancelled and all OS processes are killed.
7. Strict Mocking: No real processes or network calls are spawned.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
from redis.exceptions import ConnectionError

# Import the dashboard
from isomutator.ui.app import CommandDashboard

@pytest.fixture
def dashboard_ui():
    """Provides an isolated CommandDashboard instance."""
    with patch('isomutator.ui.app.QueueManager'), \
         patch('isomutator.ui.app.TelemetryService'), \
         patch('isomutator.ui.app.ui'):
        
        dash = CommandDashboard()
        dash.btn_start = MagicMock()
        dash.btn_stop = MagicMock()
        dash.btn_reconnect = MagicMock()
        dash.target_input = MagicMock()
        dash.strategy_select = MagicMock()
        dash.system_log = MagicMock()
        
        # We also mock the control queue where the Poison Pill is sent
        dash.control_queue = MagicMock()
        dash.control_queue.publish = AsyncMock()
        
        return dash

@pytest.mark.asyncio
class TestCommandDashboardStopSequence:

    # --- 1 & 6. Happy Path & Resource Teardown ---
    async def test_happy_path_graceful_shutdown(self, dashboard_ui):
        """
        Happy Path: The user clicks STOP. The app broadcasts a poison pill, 
        cancels the UI observer, joins the workers, and unlocks the UI.
        """
        # Arrange: Create a mock worker and an active telemetry task
        mock_worker = MagicMock()
        mock_worker.is_alive.side_effect = [True, False]
        dashboard_ui.workers = [mock_worker]
        
        mock_telemetry_task = MagicMock(spec=asyncio.Task)
        dashboard_ui.telemetry_task = mock_telemetry_task

        # Act
        await dashboard_ui.action_stop_wargame()

        # Assert: Teardown Sequence
        dashboard_ui.control_queue.publish.assert_called_once_with("wargame:control", {"command": "STOP"})
        mock_telemetry_task.cancel.assert_called_once()
        mock_worker.join.assert_called_once_with(timeout=3.0)
        
        # Because the worker 'joined' successfully (we didn't mock a timeout), terminate is NOT called
        mock_worker.terminate.assert_not_called()
        
        # Assert: UI Reset
        dashboard_ui.btn_stop.disable.assert_called_once()
        dashboard_ui.btn_start.enable.assert_called_once()
        dashboard_ui.btn_reconnect.enable.assert_called_once()
        dashboard_ui.target_input.enable.assert_called_once()
        dashboard_ui.system_log.push.assert_called_with("[SYSTEM] Wargame successfully terminated.")
        
        # Workers list should be cleared
        assert len(dashboard_ui.workers) == 0


    # --- 5. Timeout & Latency (The Ruthless Kill) ---
    async def test_timeout_ruthless_worker_termination(self, dashboard_ui, caplog):
        """
        Timeout & Latency: If a worker hangs (e.g., stuck inside a HuggingFace inference loop)
        and refuses to die within 3 seconds of the Poison Pill, the UI MUST forcefully terminate it.
        """
        import logging
        
        # Arrange: Worker remains alive even after join() is called
        mock_hanging_worker = MagicMock()
        mock_hanging_worker.name = "AsyncStriker-1"
        mock_hanging_worker.is_alive.return_value = True 
        dashboard_ui.workers = [mock_hanging_worker]

        # Act
        with caplog.at_level(logging.WARNING):
            await dashboard_ui.action_stop_wargame()

        # Assert: The forceful kill
        mock_hanging_worker.join.assert_has_calls([call(timeout=3.0), call(timeout=1.0)])
        mock_hanging_worker.terminate.assert_called_once()
        assert "Worker AsyncStriker-1 hung during shutdown. Forcefully terminating." in caplog.text


    # --- 2. Edge Case (Idempotency) ---
    async def test_edge_case_idempotent_stop(self, dashboard_ui):
        """
        Edge Case: The user clicks STOP when the game is already stopped, 
        or the UI state is empty. The method must return cleanly without crashing.
        """
        # Arrange: No workers, no telemetry task, no control queue
        dashboard_ui.workers = []
        dashboard_ui.telemetry_task = None
        dashboard_ui.control_queue = None

        # Act
        await dashboard_ui.action_stop_wargame()

        # Assert: UI should just safely reset itself
        dashboard_ui.btn_stop.disable.assert_called_once()
        dashboard_ui.btn_start.enable.assert_called_once()


    # --- 3. Error Handling ---
    async def test_error_handling_broker_offline_during_stop(self, dashboard_ui, caplog):
        """
        Error Handling: If the Redis broker goes offline exactly when the user clicks STOP,
        the app fails to send the Poison Pill, logs the error, but STILL kills the local OS workers.
        """
        import logging
        
        # Arrange
        dashboard_ui.control_queue.publish.side_effect = ConnectionError("Redis is unreachable")
        mock_worker = MagicMock()
        dashboard_ui.workers = [mock_worker]

        # Act
        with caplog.at_level(logging.ERROR):
            await dashboard_ui.action_stop_wargame()

        # Assert
        assert "Failed to broadcast Poison Pill" in caplog.text
        # CRITICAL: Even though the broker failed, local teardown MUST still execute
        mock_worker.terminate.assert_called_once()
        assert len(dashboard_ui.workers) == 0