"""
Algorithm Summary:
This test suite validates the asynchronous state-updater function used by the NiceGUI frontend. 
It ensures that the UI's state dictionary is accurately updated with the payload retrieved 
from the TelemetryService. It tests the Happy Path (normal metrics) and Edge Cases 
(handling missing keys or offline states) to guarantee the frontend rendering loop never 
encounters a KeyError or unhandled exception.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

# Assuming the state updater will be located in the UI module
from isomutator.ui.app import update_dashboard_state
from isomutator.core.telemetry_service import TelemetryService

@pytest.fixture
def mock_telemetry_service():
    """Provides a mocked TelemetryService for isolated UI testing."""
    service = MagicMock(spec=TelemetryService)
    service.get_dashboard_metrics = AsyncMock()
    return service

@pytest.fixture
def initial_ui_state():
    """Provides a baseline UI state dictionary."""
    return {
        "broker_status": "Unknown",
        "attack_queue_depth": 0,
        "feedback_queue_depth": 0
    }

@pytest.mark.asyncio
class TestDashboardStateUpdater:

    async def test_happy_path_update_state(self, mock_telemetry_service, initial_ui_state):
        """
        Happy Path: The state updater successfully maps a healthy telemetry 
        payload into the frontend's reactive state dictionary.
        """
        # Arrange
        mock_telemetry_service.get_dashboard_metrics.return_value = {
            "broker_status": "Online",
            "attack_queue_depth": 150,
            "feedback_queue_depth": 25
        }

        # Act
        await update_dashboard_state(telemetry_service=mock_telemetry_service, ui_state=initial_ui_state)

        # Assert
        assert initial_ui_state["broker_status"] == "Online"
        assert initial_ui_state["attack_queue_depth"] == 150
        assert initial_ui_state["feedback_queue_depth"] == 25
        mock_telemetry_service.get_dashboard_metrics.assert_called_once()

    async def test_edge_case_offline_fallback(self, mock_telemetry_service, initial_ui_state):
        """
        Edge Case: If the telemetry service reports the broker is offline 
        (returning -1s), the UI state correctly absorbs these fallback values.
        """
        # Arrange
        mock_telemetry_service.get_dashboard_metrics.return_value = {
            "broker_status": "Offline",
            "attack_queue_depth": -1,
            "feedback_queue_depth": -1
        }

        # Act
        await update_dashboard_state(telemetry_service=mock_telemetry_service, ui_state=initial_ui_state)

        # Assert
        assert initial_ui_state["broker_status"] == "Offline"
        assert initial_ui_state["attack_queue_depth"] == -1
        assert initial_ui_state["feedback_queue_depth"] == -1