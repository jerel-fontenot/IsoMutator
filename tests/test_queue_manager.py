"""
IsoCore Queue Manager Tests (tests/test_queue_manager.py)
---------------------------------------------------------
Run with: uv run pytest tests/test_queue_manager.py -v
"""

import asyncio
import pytest
from isocore.core.queue_manager import QueueManager

# ==========================================
# Fixtures
# ==========================================
@pytest.fixture
def queue_manager():
    """
    Provides a fresh QueueManager for each test.
    The 'yield' acts as a teardown, ensuring background threads
    are closed even if the test fails.
    """
    qm = QueueManager(max_size=5) # Small max_size to test backpressure easily
    yield qm
    qm.close()

# ==========================================
# Test Cases
# ==========================================

@pytest.mark.asyncio
async def test_async_put_and_get_exact_batch(queue_manager):
    """Test standard flow: Put 3 items, ask for exactly 3 items."""
    # 1. Producer pushes data
    assert await queue_manager.async_put("item_1") is True
    assert await queue_manager.async_put("item_2") is True
    assert await queue_manager.async_put("item_3") is True

    # 2. Consumer pulls data (runs synchronously)
    batch = queue_manager.get_batch(target_size=3, max_wait=0.5)
    
    assert len(batch) == 3
    assert batch == ["item_1", "item_2", "item_3"]


@pytest.mark.asyncio
async def test_get_batch_partial_sweep(queue_manager):
    """Test the sweep logic: Put 2 items, ask for 5. It should return 2 quickly."""
    await queue_manager.async_put("packet_A")
    await queue_manager.async_put("packet_B")

    # Ask for 5, but the queue only has 2. 
    # It shouldn't wait forever; it should grab 2 and return.
    batch = queue_manager.get_batch(target_size=5, max_wait=0.5)
    
    assert len(batch) == 2
    assert batch == ["packet_A", "packet_B"]


def test_get_batch_empty_queue(queue_manager):
    """Test consumer behavior when the internet is completely quiet."""
    # Queue is empty. We expect it to wait for 'max_wait' (0.2s) and return an empty list.
    batch = queue_manager.get_batch(target_size=10, max_wait=0.2)
    
    assert isinstance(batch, list)
    assert len(batch) == 0


@pytest.mark.asyncio
async def test_queue_backpressure_rejection(queue_manager):
    """Test that the queue safely rejects items when full, protecting system RAM."""
    # The fixture sets max_size=5. Let's fill it up.
    for i in range(5):
        success = await queue_manager.async_put(f"fill_{i}")
        assert success is True
        
    # The queue is now exactly full. The next put should fail and return False.
    # We set a tiny timeout so the test runs fast.
    overflow_success = await queue_manager.async_put("overflow_packet", timeout=0.1)
    
    assert overflow_success is False


@pytest.mark.asyncio
async def test_poison_pill_delivery(queue_manager):
    """Test the graceful shutdown mechanism."""
    queue_manager.send_poison_pill()
    
    # The consumer should receive the exact POISON_PILL string
    batch = queue_manager.get_batch(target_size=1, max_wait=0.5)
    
    assert len(batch) == 1
    assert batch[0] == "POISON_PILL"