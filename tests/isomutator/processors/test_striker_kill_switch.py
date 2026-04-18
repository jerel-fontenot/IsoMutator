"""
ALGORITHM SUMMARY:
Validates the Out-of-Band (OOB) Kill Switch integration in the AsyncStriker.
Ensures that the worker immediately breaks its continuous network loop when 
the `shutdown_event` is flipped, entirely bypassing the queue's Poison Pill.
"""

import pytest
import asyncio
import multiprocessing
from unittest.mock import AsyncMock, MagicMock
from isomutator.processors.striker import AsyncStriker

@pytest.mark.asyncio
async def test_striker_respects_oob_kill_switch():
    """
    Happy Path (OOB Kill Switch): Verifies that the AsyncStriker immediately 
    breaks its processing loop when the multiprocessing.Event is set.
    """
    mock_attack_q = MagicMock()
    # Simulate a hanging queue (TimeoutError) to prove it doesn't block the exit
    mock_attack_q.get_batch = AsyncMock(side_effect=asyncio.TimeoutError) 
    
    shutdown_event = multiprocessing.Event()
    
    striker = AsyncStriker(
        attack_queue=mock_attack_q,
        eval_queue=MagicMock(),
        log_queue=MagicMock(),
        target_url="http://mock",
        shutdown_event=shutdown_event  # Inject the Kill Switch
    )
    
    # Inject a mock logger since we are bypassing run() ---
    striker.logger = MagicMock()
    
    # ACT: Flip the switch BEFORE entering the loop
    shutdown_event.set()
    
    # If the kill switch works, this will return immediately instead of looping forever
    await striker._strike_loop()
    
    # ASSERT: Loop exited cleanly
    assert striker.shutdown_event.is_set() is True