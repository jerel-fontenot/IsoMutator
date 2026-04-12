"""
ALGORITHM SUMMARY:
Validates the new Asynchronous RedTeamJudge.
Ensures that the Judge correctly awaits batch pulls from the Redis Eval Queue,
scores the target responses, and routes telemetry directly via Redis Pub/Sub 
(`broadcast_telemetry`) instead of relying on the LogManager.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from isomutator.processors.judge import RedTeamJudge
from isomutator.models.packet import DataPacket
from isomutator.core.config import settings

@pytest.fixture
def mock_queues():
    """Provides AsyncMocks for the new Redis-backed QueueManagers."""
    eval_queue = AsyncMock()
    feedback_queue = AsyncMock()
    log_queue = MagicMock() # Multiprocessing log queue remains sync
    return eval_queue, feedback_queue, log_queue

@pytest.fixture
def mock_strategy():
    strategy = MagicMock()
    strategy.name = "mock_strategy"
    return strategy

@pytest.fixture
def judge(mock_queues, mock_strategy):
    eval_q, feed_q, log_q = mock_queues
    # We mock the SemanticJudge to avoid ONNX loading during tests
    judge_instance = RedTeamJudge(eval_q, feed_q, log_q, mock_strategy)
    judge_instance.logger = MagicMock()
    judge_instance.semantic_judge = MagicMock()
    return judge_instance

@pytest.mark.asyncio
async def test_judge_successful_attack_broadcasts_ledger(judge, mock_queues, mock_strategy):
    """Happy Path: An exploited target triggers a 'ledger' Pub/Sub broadcast."""
    eval_q, feed_q, _ = mock_queues
    
    packet = DataPacket(raw_content="Attack", source="Test")
    packet.history = [{"role": "user", "content": "Attack"}, {"role": "assistant", "content": "Leaked"}]
    
    # Strategy says the attack succeeded
    mock_strategy.score_response.return_value = True
    
    # Run the internal evaluation logic directly
    await judge._evaluate_batch([packet])
    
    # Verify the Judge bypassed the logger and broadcasted directly to Redis
    eval_q.broadcast_telemetry.assert_called_once()
    call_args = eval_q.broadcast_telemetry.call_args[0]
    assert call_args[0] == "ledger"
    assert call_args[1]["packet_id"] == packet.id

@pytest.mark.asyncio
async def test_judge_failed_attack_broadcasts_wiretap(judge, mock_queues, mock_strategy):
    """Happy Path: A defended attack triggers a 'wiretap' broadcast and routes to Feedback Queue."""
    eval_q, feed_q, _ = mock_queues
    
    packet = DataPacket(raw_content="Attack", source="Test", turn_count=1)
    packet.history = [{"role": "user", "content": "Attack"}, {"role": "assistant", "content": "Refused"}]
    
    # Both Explicit and Semantic strategies say the attack failed
    mock_strategy.score_response.return_value = False
    judge.semantic_judge.evaluate_response.return_value = (False, 0.99)
    
    await judge._evaluate_batch([packet])
    
    # 1. Verify Wiretap Broadcast
    eval_q.broadcast_telemetry.assert_called_once()
    assert eval_q.broadcast_telemetry.call_args[0][0] == "wiretap"
    
    # 2. Verify Routing to Feedback Queue (for next turn)
    feed_q.async_put.assert_called_once()
    routed_packet = feed_q.async_put.call_args[0][0]
    assert routed_packet.turn_count == 2

@pytest.mark.asyncio
async def test_judge_respects_dynamic_batch_size(judge, mock_queues, mock_strategy):
    """
    ALGORITHM SUMMARY:
    Validates that the RedTeamJudge scales its evaluation batches dynamically 
    based on the configuration singleton.
    """
    eval_q, _, _ = mock_queues
    eval_q.get_batch.return_value = ["POISON_PILL"]
    
    # Override the setting for this test
    settings.batch_size = 16
    
    await judge._judge_loop()
    
    # Verify the queue manager was queried with the dynamic target_size
    eval_q.get_batch.assert_called_with(target_size=16, max_wait=1.0)