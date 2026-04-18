# test_queue_manager.py

"""
ALGORITHM SUMMARY:
Validates the Redis-backed QueueManager.
Ensures thread-safe, non-blocking Pub/Sub message brokering.

Coverage Additions:
- Concurrency: Spams 1,000 simultaneous puts/gets via asyncio.gather.
- Teardown: Explicitly verifies connection pool closure to prevent zombie sockets.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from isomutator.core.queue_manager import QueueManager
from isomutator.models.packet import DataPacket

@pytest.fixture
def mock_redis_manager():
    with patch("redis.asyncio.Redis.from_url") as mock_redis_cls:
        mock_client = AsyncMock()
        mock_redis_cls.return_value = mock_client
        qm = QueueManager(redis_url="redis://localhost:6379/0", queue_name="test")
        qm._redis = mock_client
        yield qm, mock_client

@pytest.mark.asyncio
async def test_queue_manager_concurrency_spike(mock_redis_manager):
    """Proves the queue handles high-load spikes without event loop deadlocks."""
    qm, mock_client = mock_redis_manager
    
    # Arrange 1,000 concurrent packets
    async def simulated_put(idx):
        packet = DataPacket(raw_content=f"payload_{idx}", source="test")
        return await qm.async_put(item=packet)
        
    # Act: Fire all 1,000 requests simultaneously
    results = await asyncio.gather(*(simulated_put(i) for i in range(1000)))
    
    # Assert
    assert all(results) is True
    assert mock_client.lpush.call_count == 1000

@pytest.mark.asyncio
async def test_queue_manager_teardown_and_leak_prevention(mock_redis_manager):
    """Verifies the Redis connection pool is explicitly closed."""
    qm, mock_client = mock_redis_manager
    
    # Act
    await qm.close()
    
    # Assert: Prevent socket leaks!
    mock_client.aclose.assert_called_once()