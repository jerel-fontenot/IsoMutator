"""
Algorithm Summary:
This test suite validates the asynchronous TelemetryService, which is responsible 
for safely polling the Redis-backed QueueManager to retrieve real-time pipeline 
metrics (queue depths, worker health) for the NiceGUI dashboard. It ensures 
the service successfully parses normal metrics, handles empty states correctly, 
and fails gracefully without crashing the UI if the underlying message broker 
disconnects.
"""

import pytest
import logging
from unittest.mock import AsyncMock, MagicMock
from redis.exceptions import ConnectionError

# Assuming the implementation will be in src/isomutator/core/telemetry_service.py
from isomutator.core.telemetry_service import TelemetryService
from isomutator.core.queue_manager import QueueManager

@pytest.fixture
def mock_queue_manager():
    """Provides a mocked QueueManager for isolated testing."""
    qm = MagicMock(spec=QueueManager)
    # Mock the asynchronous methods we expect the telemetry service to call
    qm.get_queue_depth = AsyncMock()
    qm.ping_broker = AsyncMock()
    return qm

@pytest.fixture
def telemetry_service(mock_queue_manager):
    """Provides a TelemetryService instance injected with the mocked QueueManager."""
    return TelemetryService(queue_manager=mock_queue_manager)


@pytest.mark.asyncio
class TestTelemetryService:
    
    async def test_happy_path_fetch_metrics(self, telemetry_service, mock_queue_manager):
        """
        Happy Path: The service successfully polls Redis and returns 
        the expected metrics dictionary for a healthy, active pipeline.
        """
        # Arrange
        mock_queue_manager.ping_broker.return_value = True
        # Setup side_effect to return different depths based on the queue name
        async def mock_get_depth(queue_name):
            if queue_name == "attack":
                return 42
            elif queue_name == "feedback":
                return 17
            return 0
        mock_queue_manager.get_queue_depth.side_effect = mock_get_depth

        # Act
        metrics = await telemetry_service.get_dashboard_metrics()

        # Assert
        assert metrics["broker_status"] == "Online"
        assert metrics["attack_queue_depth"] == 42
        assert metrics["feedback_queue_depth"] == 17
        mock_queue_manager.ping_broker.assert_called_once()
        assert mock_queue_manager.get_queue_depth.call_count == 2

    async def test_edge_cases_empty_queues(self, telemetry_service, mock_queue_manager):
        """
        Edge Cases: The service handles a completely empty pipeline gracefully,
        returning integer 0s rather than Nulls or raising errors.
        """
        # Arrange
        mock_queue_manager.ping_broker.return_value = True
        mock_queue_manager.get_queue_depth.return_value = 0

        # Act
        metrics = await telemetry_service.get_dashboard_metrics()

        # Assert
        assert metrics["broker_status"] == "Online"
        assert metrics["attack_queue_depth"] == 0
        assert metrics["feedback_queue_depth"] == 0

    async def test_error_handling_redis_disconnection(self, telemetry_service, mock_queue_manager, caplog):
        """
        Error Handling: If the Redis broker goes offline unexpectedly, 
        the service must fail gracefully, return safe fallback values, 
        and log the failure without crashing the async event loop.
        """
        # Arrange
        mock_queue_manager.ping_broker.side_effect = ConnectionError("Redis is unreachable")
        
        # We capture logs to ensure the TRACE/ERROR fallback logging fires correctly
        with caplog.at_level(logging.ERROR):
            # Act
            metrics = await telemetry_service.get_dashboard_metrics()

        # Assert
        assert metrics["broker_status"] == "Offline"
        assert metrics["attack_queue_depth"] == -1  # -1 visually indicates an error state to the UI
        assert metrics["feedback_queue_depth"] == -1
        assert "Failed to fetch telemetry metrics" in caplog.text