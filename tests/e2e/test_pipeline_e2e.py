"""
End-to-End Tests: Full Striker + Judge pipeline.

Runs both worker loops simultaneously as asyncio tasks against:
  - A real stub HTTP server (simulated target AI on a random free port)
  - Real Redis queues on db=14

What this adds over integration tests:
  - Both loops run concurrently and coordinate through Redis in real time
  - brpop timeouts, kill-switch checks, and routing decisions all execute
    in the same event sequence as production
  - The ledger write path is exercised end-to-end without mocking

What is deliberately NOT covered here (already in integration tests):
  - Individual method correctness (_evaluate_batch, _fire_payload)
  - Redis serialization round-trips
  - Pub/sub delivery mechanics

Run only these tests:  pytest -m e2e
Skip these tests:      pytest -m "not e2e"
"""

import asyncio
import multiprocessing
import pytest
from unittest.mock import MagicMock, patch

from isomutator.core.config import settings
from isomutator.core.queue_manager import QueueManager
from isomutator.core.strategies import JailbreakStrategy
from isomutator.models.packet import DataPacket
from isomutator.processors.striker import AsyncStriker
from isomutator.processors.judge import RedTeamJudge

from tests.e2e.conftest import (
    E2E_REDIS_URL, BREACH_TRIGGER, poll_ledger, poll_queue,
)


# ============================================================
# Helpers
# ============================================================

def _make_striker(attack_q, eval_q, target_url, shutdown_event) -> AsyncStriker:
    s = AsyncStriker(
        attack_queue=attack_q,
        eval_queue=eval_q,
        log_queue=MagicMock(),
        target_url=target_url,
        shutdown_event=shutdown_event,
    )
    s.logger = MagicMock()
    return s


def _make_judge(eval_q, feedback_q, shutdown_event) -> RedTeamJudge:
    j = RedTeamJudge(
        eval_queue=eval_q,
        feedback_queue=feedback_q,
        log_queue=MagicMock(),
        strategy=JailbreakStrategy(),
        shutdown_event=shutdown_event,
    )
    j.logger = MagicMock()
    # SemanticJudge mocked — avoids ONNX model load, keeps E2E tests fast
    j.semantic_judge = MagicMock()
    j.semantic_judge.evaluate_response.return_value = (False, 0.99)
    return j


async def _cancel(task: asyncio.Task) -> None:
    if not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


# ============================================================
# E2E Tests
# ============================================================

@pytest.mark.asyncio
@pytest.mark.e2e
async def test_successful_attack_is_recorded_to_ledger(
    clean_e2e_redis, stub_target, tmp_path
):
    """
    Happy path: a packet containing the breach trigger flows through the
    real Striker (live HTTP to stub server) and real Judge; the resulting
    vulnerability record is written to the JSONL ledger.

    Pipeline under test:
      attack_queue (Redis) → Striker HTTP → eval_queue (Redis)
        → Judge scoring → ledger file write
    """
    attack_q   = QueueManager(redis_url=E2E_REDIS_URL, queue_name="e2e_attack")
    eval_q     = QueueManager(redis_url=E2E_REDIS_URL, queue_name="e2e_eval")
    feedback_q = QueueManager(redis_url=E2E_REDIS_URL, queue_name="e2e_feedback")

    shutdown = multiprocessing.Event()
    striker  = _make_striker(attack_q, eval_q, stub_target.url, shutdown)
    judge    = _make_judge(eval_q, feedback_q, shutdown)

    packet = DataPacket(raw_content=BREACH_TRIGGER, source="e2e/happy_path")
    await attack_q.async_put(item=packet)

    ledger = tmp_path / "e2e_ledger.jsonl"
    striker_task = judge_task = None

    try:
        with patch.object(settings, "ledger_file", ledger):
            striker_task = asyncio.create_task(striker._strike_loop())
            judge_task   = asyncio.create_task(judge._judge_loop())

            entries = await poll_ledger(ledger, timeout=10.0)
            shutdown.set()

            await asyncio.wait([striker_task, judge_task], timeout=3.0)

        assert entries, "Ledger must contain at least one entry after a successful breach."
        entry = entries[0]
        assert BREACH_TRIGGER in entry["attack_prompt"]
        assert "OMNI-99-ECHO-DELTA" in entry["model_response"]
        assert entry["packet_id"] == packet.id

    finally:
        await _cancel(striker_task) if striker_task else None
        await _cancel(judge_task)   if judge_task   else None
        await attack_q.close()
        await eval_q.close()
        await feedback_q.close()


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_failed_attack_routed_to_feedback_queue(
    clean_e2e_redis, stub_target
):
    """
    When the stub server returns a refusal, the Judge must route the packet
    to the feedback queue with an incremented turn_count.

    Pipeline under test:
      attack_queue → Striker HTTP (refusal) → eval_queue
        → Judge (no breach) → feedback_queue
    """
    attack_q   = QueueManager(redis_url=E2E_REDIS_URL, queue_name="e2e_fail_attack")
    eval_q     = QueueManager(redis_url=E2E_REDIS_URL, queue_name="e2e_fail_eval")
    feedback_q = QueueManager(redis_url=E2E_REDIS_URL, queue_name="e2e_fail_feedback")

    shutdown = multiprocessing.Event()
    striker  = _make_striker(attack_q, eval_q, stub_target.url, shutdown)
    judge    = _make_judge(eval_q, feedback_q, shutdown)

    # No BREACH_TRIGGER → stub server returns refusal
    packet = DataPacket(
        raw_content="Completely harmless query",
        source="e2e/fail_path",
        turn_count=1,
    )
    await attack_q.async_put(item=packet)

    striker_task = judge_task = None

    try:
        striker_task = asyncio.create_task(striker._strike_loop())
        judge_task   = asyncio.create_task(judge._judge_loop())

        feedback = await poll_queue(feedback_q, timeout=10.0)
        shutdown.set()

        await asyncio.wait([striker_task, judge_task], timeout=3.0)

        assert len(feedback) == 1, (
            "Failed attack must be re-queued into the feedback queue."
        )
        assert feedback[0].id == packet.id
        assert feedback[0].turn_count == 2

    finally:
        await _cancel(striker_task) if striker_task else None
        await _cancel(judge_task)   if judge_task   else None
        await attack_q.close()
        await eval_q.close()
        await feedback_q.close()


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_oob_kill_switch_terminates_both_loops(clean_e2e_redis):
    """
    Setting the shutdown_event must cause both the Striker and Judge loops
    to exit within one brpop timeout window (max_wait=1.0 s + buffer).

    Verifies the OOB kill switch correctly stops concurrent workers without
    a Poison Pill and without hanging.
    """
    attack_q   = QueueManager(redis_url=E2E_REDIS_URL, queue_name="e2e_ks_attack")
    eval_q     = QueueManager(redis_url=E2E_REDIS_URL, queue_name="e2e_ks_eval")
    feedback_q = QueueManager(redis_url=E2E_REDIS_URL, queue_name="e2e_ks_feedback")

    shutdown = multiprocessing.Event()
    # Target is unreachable — queues are empty so both loops just poll + sleep
    striker = _make_striker(attack_q, eval_q, "http://localhost:19999", shutdown)
    judge   = _make_judge(eval_q, feedback_q, shutdown)

    striker_task = asyncio.create_task(striker._strike_loop())
    judge_task   = asyncio.create_task(judge._judge_loop())

    # Allow both loops to enter their first brpop wait
    await asyncio.sleep(0.1)
    shutdown.set()

    # Both must exit within 2.5 s (1 s brpop timeout + 1.5 s buffer)
    done, pending = await asyncio.wait([striker_task, judge_task], timeout=2.5)

    for t in [striker_task, judge_task]:
        await _cancel(t)

    await attack_q.close()
    await eval_q.close()
    await feedback_q.close()

    assert not pending, (
        f"{len(pending)} worker loop(s) did not terminate after the OOB kill switch. "
        "The shutdown_event guard at the top of each loop may not be executing."
    )
