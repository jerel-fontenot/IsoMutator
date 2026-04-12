"""
ALGORITHM SUMMARY:
Validates the Redis-backed QueueManager.
This replaces the local multiprocessing pipes with a distributed Pub/Sub 
and List-based message broker architecture.

TECHNOLOGY QUIRKS:
- Redis Mocking: We use `unittest.mock.AsyncMock` to completely isolate 
  the unit tests from requiring a live, running Redis server on the host 
  machine, preventing CI/CD pipeline failures.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# We will import the refactored QueueManager once implemented
from isomutator.core.queue_manager import QueueManager
from isomutator.models.packet import DataPacket

@pytest.fixture
def mock_redis_manager():
    """Provides a QueueManager with a mocked internal Redis client."""
    with patch("redis.asyncio.Redis.from_url") as mock_redis_cls:
        mock_client = AsyncMock()
        mock_redis_cls.return_value = mock_client
        
        # Instantiate pointing to a dummy URL
        qm = QueueManager(redis_url="redis://localhost:6379/0", queue_name="test_queue")
        qm._redis = mock_client
        yield qm, mock_client


@pytest.mark.asyncio
async def test_redis_async_put(mock_redis_manager):
    qm, mock_client = mock_redis_manager
    packet = DataPacket(raw_content="test", source="test")
    
    # 1. Action
    success = await qm.async_put(packet)
    
    # 2. Assertions
    assert success is True
    mock_client.lpush.assert_called_once()
    
    # Verify it pushed a serialized JSON string to the correct queue key
    call_args = mock_client.lpush.call_args[0]
    assert call_args[0] == "isomutator:queue:test_queue"
    assert "test" in call_args[1] # The JSON string payload


@pytest.mark.asyncio
async def test_redis_get_batch(mock_redis_manager):
    qm, mock_client = mock_redis_manager
    packet = DataPacket(raw_content="test", source="test")
    
    # Mock BRPOP (blocking pop) to return our packet
    mock_client.brpop.return_value = ("isomutator:queue:test_queue", packet.to_json())
    # Mock LPOP (sweep pop) to return nothing (empty queue after first item)
    mock_client.lpop.return_value = None
    
    batch = await qm.get_batch(target_size=5, max_wait=1.0)
    
    assert len(batch) == 1
    assert isinstance(batch[0], DataPacket)
    assert batch[0].raw_content == "test"
    mock_client.brpop.assert_called_once()


@pytest.mark.asyncio
async def test_redis_telemetry_broadcast(mock_redis_manager):
    qm, mock_client = mock_redis_manager
    
    await qm.broadcast_telemetry("wiretap", {"turn": 1, "attacker": "foo", "target": "bar"})
    
    mock_client.publish.assert_called_once()
    call_args = mock_client.publish.call_args[0]
    assert call_args[0] == "isomutator:telemetry:wiretap"