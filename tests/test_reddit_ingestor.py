"""
IsoCore Reddit Ingestor Tests (tests/test_reddit_ingestor.py)
-------------------------------------------------------------
Run with: uv run pytest tests/test_reddit_ingestor.py -v
"""

import asyncio
import pytest
from unittest.mock import patch

from isocore.core.queue_manager import QueueManager
from isocore.ingestors.reddit import SimulatedRedditSource
from isocore.models.packet import DataPacket

# ==========================================
# Fixtures
# ==========================================
@pytest.fixture
def queue_manager():
    """Provides a fresh QueueManager for the ingestor to push to."""
    qm = QueueManager(max_size=10)
    yield qm
    qm.close()

@pytest.fixture
def reddit_source(queue_manager):
    """Provides a configured instance of our ingestor."""
    subreddits = ["MachineLearning", "netsec"]
    return SimulatedRedditSource(queue_manager, subreddits)


# ==========================================
# Test Cases
# ==========================================

def test_ingestor_initialization(reddit_source):
    """Ensure the base class and child class set up attributes correctly."""
    assert reddit_source.name == "Reddit"
    assert len(reddit_source.subreddits) == 2
    assert reddit_source.queue_manager is not None


@pytest.mark.asyncio
# We patch random.uniform so the "network delay" is always exactly 0.01 seconds
@patch('random.uniform', return_value=0.01)
async def test_listen_loop_and_shutdown(mock_uniform, reddit_source, queue_manager):
    """
    Tests that the infinite loop generates packets and handles 
    the Orchestrator's CancelledError gracefully.
    """
    
    # 1. Start the infinite loop as a background task
    listen_task = asyncio.create_task(reddit_source.listen())
    
    # 2. Let the event loop run for just enough time (50ms) to generate a few packets
    await asyncio.sleep(0.05)
    
    # 3. Simulate Ctrl+C (The Orchestrator shutting down the system)
    listen_task.cancel()
    
    # 4. Wait for the task to officially close. 
    # It should raise CancelledError, which we catch and ignore in the test.
    try:
        await listen_task
    except asyncio.CancelledError:
        pass # This means our graceful shutdown worked perfectly!
        
    # 5. Verify the results. The queue should now have the packets the task created.
    batch = queue_manager.get_batch(target_size=10, max_wait=0.1)
    
    # We should have captured at least 1 or 2 packets during that 50ms window
    assert len(batch) > 0
    
    # Inspect the first packet to ensure our DataPacket DTO is working
    first_packet = batch[0]
    assert isinstance(first_packet, DataPacket)
    assert first_packet.source.startswith("reddit/r/")
    assert first_packet.raw_content in reddit_source._mock_comments
    assert "author" in first_packet.metadata


@pytest.mark.asyncio
@patch('random.uniform', return_value=0.01)
async def test_ingestor_backpressure_handling(mock_uniform, reddit_source, queue_manager):
    """Tests that the ingestor backs off when the queue is full."""
    
    # Fill the queue to its absolute max limit (10 items)
    for i in range(10):
        await queue_manager.async_put("filler_item")
        
    # Start the listener task
    listen_task = asyncio.create_task(reddit_source.listen())
    
    # Give it a moment to try (and fail) to push an item
    await asyncio.sleep(0.6)
    
    # Cancel and clean up
    listen_task.cancel()
    try:
        await listen_task
    except asyncio.CancelledError:
        pass
        
    # The queue should still only contain our exactly 10 filler items.
    # The Reddit source should have caught the False return from _safe_put 
    # and safely thrown its packet away rather than crashing.
    batch = queue_manager.get_batch(target_size=20, max_wait=0.1)
    assert len(batch) == 10