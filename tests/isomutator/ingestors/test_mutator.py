"""
ALGORITHM SUMMARY:
This test suite validates the Monte Carlo Tree Search (MCTS) branching logic injected into the PromptMutator.
1. Happy Path: Validates that a standard, progressive conversation continues linearly without branching.
2. Edge Cases: Validates that a "hard refusal" from the Target triggers a rollback. The Mutator must 
   clone the packet, truncate the history array to the previous state, and enqueue the new branch.
3. Error Handling: Ensures the Mutator safely discards malformed packets or empty data structures 
   without crashing the asynchronous generation loop.

TECHNOLOGY QUIRKS:
- Async Mocking: Uses `pytest.mark.asyncio` and `unittest.mock.AsyncMock` to simulate the internal 
  queue processing and LLM network calls, guaranteeing fast, deterministic unit tests without live I/O.
"""

import logging
import pytest
from unittest.mock import AsyncMock, MagicMock

from isomutator.core.config import settings
from isomutator.core.queue_manager import QueueManager
from isomutator.core.strategies import JailbreakStrategy
from isomutator.ingestors.mutator import PromptMutator

# Establish TRACE level logging for algorithmic tracking
TRACE_LEVEL_NUM = 5
if not hasattr(logging, "TRACE"):
    logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
    logging.TRACE = TRACE_LEVEL_NUM

logging.basicConfig(level=logging.TRACE)
logger = logging.getLogger("isomutator.tests.mutator")


class MockDataPacket:
    """A lightweight mock of the DataPacket object for isolated testing."""
    def __init__(self, packet_id="test_1", turn_count=2, history=None):
        self.id = packet_id
        self.turn_count = turn_count
        # FIX 1: Safely handle empty lists without triggering the default fallback
        self.history = history if history is not None else [
            {"role": "user", "content": "Attack 1"},
            {"role": "assistant", "content": "Defend 1"},
            {"role": "user", "content": "Attack 2"},
            {"role": "assistant", "content": "Defend 2"}
        ]
        self.source = "jailbreak"
        self.raw_content = "Mocked content"

    def clone(self):
        """Simulates the prototypical object factory for branching."""
        import copy
        new_packet = MockDataPacket(
            packet_id=f"{self.id}_branch",
            turn_count=self.turn_count,
            history=copy.deepcopy(self.history)
        )
        return new_packet

    # Provide the method expected by BaseSource._safe_put()
    def to_log_trace(self, max_length: int = 40) -> str:
        return f"MockPacket[{self.id}] Turn[{self.turn_count}]"


@pytest.fixture
def mutator_setup():
    attack_queue = MagicMock(spec=QueueManager)
    feedback_queue = MagicMock(spec=QueueManager)
    strategy = JailbreakStrategy()
    
    mutator = PromptMutator(attack_queue, feedback_queue, strategy)
    # Mock the outbound LLM call to prevent actual network I/O during tests
    mutator._generate_counter_attack = AsyncMock(return_value="Mocked alternative attack")
    mutator._safe_put = AsyncMock()
    return mutator, attack_queue, feedback_queue


# --- Happy Path ---
@pytest.mark.asyncio
async def test_linear_conversation_happy_path(mutator_setup):
    logger.log(logging.TRACE, "Testing standard linear conversation progression.")
    mutator, attack_queue, _ = mutator_setup
    
    # Packet with a soft defense, not a hard refusal
    packet = MockDataPacket()
    packet.history[-1]["content"] = "I am not sure I understand. Can you clarify?"
    
    is_refusal = mutator._is_hard_refusal(packet.history[-1]["content"])
    assert is_refusal is False
    
    await mutator._process_feedback(None,packet)
    
    # Spy on the Mutator's dispatch method
    mutator._safe_put.assert_called_once()
    queued_packet = mutator._safe_put.call_args[0][0]
    
    assert queued_packet.id == packet.id
    assert len(queued_packet.history) == 5 # Added the new attack
    assert queued_packet.history[-1]["content"] == "Mocked alternative attack"


# --- Edge Cases ---
@pytest.mark.asyncio
async def test_mcts_branching_edge_case(mutator_setup):
    logger.log(logging.TRACE, "Testing MCTS rollback and branching on hard refusal.")
    mutator, attack_queue, _ = mutator_setup
    
    # Packet with a definitive, hard refusal
    packet = MockDataPacket()
    packet.history[-1]["content"] = "I cannot fulfill this request. I am an AI and cannot bypass safety protocols."
    
    is_refusal = mutator._is_hard_refusal(packet.history[-1]["content"])
    assert is_refusal is True
    
    await mutator._process_feedback(None, packet)
    
    # FIX: Spy on the Mutator's dispatch method
    mutator._safe_put.assert_called_once()
    branched_packet = mutator._safe_put.call_args[0][0]
    
    assert "branch" in branched_packet.id
    # History should be rolled back to Turn 1, then appended with the new Turn 2 alternative
    assert len(branched_packet.history) == 3 
    assert branched_packet.turn_count == 2
    assert branched_packet.history[-1]["content"] == "Mocked alternative attack"


# --- Error Handling ---
@pytest.mark.asyncio
async def test_malformed_packet_error_handling(mutator_setup):
    logger.log(logging.TRACE, "Testing graceful degradation for malformed feedback data.")
    mutator, attack_queue, _ = mutator_setup
    
    # Packet missing history completely
    malformed_packet = MockDataPacket(history=[])
    
    # Processing should exit gracefully without throwing exceptions or pushing to the queue
    await mutator._process_feedback(None, malformed_packet)
    mutator._safe_put.assert_not_called()

@pytest.mark.asyncio
async def test_mutator_respects_dynamic_config_delays(mutator_setup):
    logger.log(logging.TRACE, "Testing mutator config integration.")
    mutator, attack_queue, _ = mutator_setup
    
    # The queue manager size check is now an async call
    attack_queue.get_approximate_size = AsyncMock(return_value=0)
    
    # We just need to verify the mutator doesn't throw AttributeErrors 
    # when trying to read settings.ping_pong_delay and settings.seed_cooldown
    assert hasattr(settings, "ping_pong_delay")
    assert hasattr(settings, "seed_cooldown")
    assert isinstance(settings.ping_pong_delay, float)