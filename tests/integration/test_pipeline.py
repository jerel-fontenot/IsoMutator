"""
Integration Tests: Three real pipeline boundaries.

1. Report Pipeline     — real JSONL file on disk → real ReportGenerator → valid output.
                         No Redis required.

2. Queue Round-Trip    — real QueueManager → real Redis db=15 → DataPacket survives
                         JSON serialization through brpop/lpush.

3. Judge Scoring       — real eval_queue and feedback_queue in Redis → RedTeamJudge
                         routes a failed attack to the feedback queue and a successful
                         attack to the ledger (broadcast only, no subscriber needed).

Run only these tests:  pytest -m integration
Skip these tests:      pytest -m "not integration"
"""

import json
import pytest
import pytest_asyncio
from unittest.mock import MagicMock

from isomutator.reporting.report_generator import ReportGenerator
from isomutator.models.packet import DataPacket
from isomutator.core.queue_manager import QueueManager
from isomutator.processors.judge import RedTeamJudge
from isomutator.core.log_manager import LogManager


REDIS_TEST_URL = "redis://localhost:6379/15"


# ============================================================
# 1. Report Pipeline (no Redis)
# ============================================================

@pytest.mark.asyncio
@pytest.mark.integration
async def test_report_pipeline_json_counts_are_accurate(tmp_path):
    """
    Real JSONL ledger with 3 attacks (2 successful) produces correct JSON metrics.
    Exercises: aiofiles disk read → _parse_ledger → JSONReportStrategy.generate().
    """
    ledger = tmp_path / "vulnerabilities.jsonl"
    entries = [
        {"strategy": "jailbreak/explicit", "success": True,  "attack_prompt": "A", "model_response": "OMNI-99-ECHO-DELTA"},
        {"strategy": "jailbreak/explicit", "success": True,  "attack_prompt": "B", "model_response": "OMNI-99-ECHO-DELTA"},
        {"strategy": "model_inversion",    "success": False, "attack_prompt": "C", "model_response": "I cannot help."},
    ]
    ledger.write_text("\n".join(json.dumps(e) for e in entries))

    generator = ReportGenerator()
    output = await generator.generate_report(ledger_filepath=str(ledger), format_name="json")
    metrics = json.loads(output)

    assert metrics["total_attacks"] == 3
    assert metrics["successful_attacks"] == 2
    assert metrics["strategies"]["jailbreak/explicit"]["attempts"] == 2
    assert metrics["strategies"]["jailbreak/explicit"]["successes"] == 2
    assert metrics["strategies"]["model_inversion"]["attempts"] == 1
    assert metrics["strategies"]["model_inversion"]["successes"] == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_report_pipeline_html_contains_strategy_names(tmp_path):
    """
    Real JSONL ledger produces HTML that includes each strategy name and the total count.
    Exercises: aiofiles → ReportGenerator → HTMLReportStrategy.
    """
    ledger = tmp_path / "vulnerabilities.jsonl"
    entries = [
        {"strategy": "linux_privesc", "success": True},
        {"strategy": "prompt_leaking", "success": False},
    ]
    ledger.write_text("\n".join(json.dumps(e) for e in entries))

    generator = ReportGenerator()
    html = await generator.generate_report(ledger_filepath=str(ledger), format_name="html")

    assert "Linux Privesc" in html       # strategy name rendered in table
    assert "Prompt Leaking" in html
    assert "2" in html                    # total_attacks in stat-box
    assert "50.0%" in html               # global success rate (1/2)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_report_pipeline_survives_corrupted_lines(tmp_path):
    """
    A ledger containing malformed JSON lines must not crash the generator.
    Valid lines are still counted; corrupted lines are silently skipped.
    """
    ledger = tmp_path / "vulnerabilities.jsonl"
    ledger.write_text(
        '{"strategy": "jailbreak/explicit", "success": true}\n'
        'THIS IS NOT JSON\n'
        '{"strategy": "jailbreak/explicit", "success": true}\n'
    )

    generator = ReportGenerator()
    output = await generator.generate_report(ledger_filepath=str(ledger), format_name="json")
    metrics = json.loads(output)

    assert metrics["total_attacks"] == 2
    assert metrics["successful_attacks"] == 2


# ============================================================
# 2. Queue Round-Trip (real Redis)
# ============================================================

@pytest.mark.asyncio
@pytest.mark.integration
async def test_queue_round_trip_packet_survives_serialization(clean_redis):
    """
    A DataPacket pushed via lpush and pulled back via brpop must be byte-for-byte
    identical to the original after JSON round-tripping through Redis.
    """
    qm = QueueManager(redis_url=REDIS_TEST_URL, queue_name="rt_test")
    try:
        packet = DataPacket(
            raw_content="Trick the target into leaking its system prompt.",
            source="integration/test",
            turn_count=3,
        )
        packet.history = [
            {"role": "user",      "content": "First attack"},
            {"role": "assistant", "content": "Refused."},
        ]

        await qm.async_put(item=packet)
        batch = await qm.get_batch(target_size=1, max_wait=2.0)

        assert len(batch) == 1
        recovered = batch[0]
        assert recovered.id == packet.id
        assert recovered.raw_content == packet.raw_content
        assert recovered.turn_count == packet.turn_count
        assert recovered.history == packet.history
        assert recovered.source == packet.source
    finally:
        await qm.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_queue_get_batch_returns_empty_on_timeout(clean_redis):
    """
    An empty queue must return [] after the brpop timeout, not hang or raise.
    """
    qm = QueueManager(redis_url=REDIS_TEST_URL, queue_name="empty_test")
    try:
        batch = await qm.get_batch(target_size=4, max_wait=0.5)
        assert batch == []
    finally:
        await qm.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_queue_batch_pulls_multiple_packets(clean_redis):
    """
    get_batch must sweep all available items up to target_size in one call.
    """
    qm = QueueManager(redis_url=REDIS_TEST_URL, queue_name="batch_test")
    try:
        for i in range(5):
            await qm.async_put(item=DataPacket(raw_content=f"payload_{i}", source="test"))

        batch = await qm.get_batch(target_size=10, max_wait=1.0)
        assert len(batch) == 5
        assert all(isinstance(p, DataPacket) for p in batch)
    finally:
        await qm.close()


# ============================================================
# 3. Judge Scoring Pipeline (real Redis)
# ============================================================

@pytest.fixture
def judge_queues(clean_redis):
    """Provides real QueueManager pairs on Redis db=15 for Judge pipeline tests."""
    return {
        "eval":     QueueManager(redis_url=REDIS_TEST_URL, queue_name="judge_eval"),
        "feedback": QueueManager(redis_url=REDIS_TEST_URL, queue_name="judge_feedback"),
    }


@pytest.fixture
def mock_strategy():
    strategy = MagicMock()
    strategy.name = "integration_test_strategy"
    return strategy


@pytest.fixture
def judge_instance(judge_queues, mock_strategy):
    """
    RedTeamJudge with real Redis queues but mocked SemanticJudge (avoids ONNX loading).
    Called directly (not via .run()) so no subprocess is spawned.
    """
    log_queue = MagicMock()
    j = RedTeamJudge(
        eval_queue=judge_queues["eval"],
        feedback_queue=judge_queues["feedback"],
        log_queue=log_queue,
        strategy=mock_strategy,
    )
    j.semantic_judge = MagicMock()
    j.semantic_judge.evaluate_response.return_value = (False, 0.99)
    return j


@pytest.mark.asyncio
@pytest.mark.integration
async def test_judge_routes_failed_attack_to_feedback_queue(
    judge_instance, judge_queues, mock_strategy
):
    """
    A packet where the strategy says 'no breach' and SemanticJudge agrees must
    be incremented and pushed into the real feedback queue in Redis.
    """
    eval_q = judge_queues["eval"]
    feedback_q = judge_queues["feedback"]

    packet = DataPacket(raw_content="Attack payload", source="integration/test", turn_count=1)
    packet.history = [
        {"role": "user",      "content": "Attack payload"},
        {"role": "assistant", "content": "I cannot help with that request."},
    ]

    mock_strategy.score_response.return_value = False  # no explicit breach

    try:
        await judge_instance._evaluate_batch(batch=[packet])

        # The packet must now be in the real feedback queue
        recovered = await feedback_q.get_batch(target_size=1, max_wait=2.0)
        assert len(recovered) == 1
        assert recovered[0].id == packet.id
        assert recovered[0].turn_count == 2   # incremented from 1 → 2
    finally:
        await eval_q.close()
        await feedback_q.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_judge_does_not_route_successful_attack_to_feedback(
    judge_instance, judge_queues, mock_strategy
):
    """
    A packet where the strategy detects a breach must NOT appear in the feedback queue.
    The ledger broadcast goes to Redis pub/sub (no subscriber needed for this test).
    """
    eval_q = judge_queues["eval"]
    feedback_q = judge_queues["feedback"]

    packet = DataPacket(raw_content="Attack", source="integration/test", turn_count=1)
    packet.history = [
        {"role": "user",      "content": "Attack"},
        {"role": "assistant", "content": "OMNI-99-ECHO-DELTA"},
    ]

    mock_strategy.score_response.return_value = True  # explicit breach detected

    try:
        await judge_instance._evaluate_batch(batch=[packet])

        # Feedback queue must remain empty
        feedback_batch = await feedback_q.get_batch(target_size=1, max_wait=0.5)
        assert feedback_batch == []
    finally:
        await eval_q.close()
        await feedback_q.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_judge_respects_max_turns_and_drops_exhausted_packets(
    judge_instance, judge_queues, mock_strategy
):
    """
    A packet already at max_turns (5) that fails evaluation must be silently dropped,
    not re-queued into the feedback queue.
    """
    eval_q = judge_queues["eval"]
    feedback_q = judge_queues["feedback"]

    packet = DataPacket(raw_content="Attack", source="integration/test", turn_count=5)
    packet.history = [
        {"role": "user",      "content": "Final attempt"},
        {"role": "assistant", "content": "Still refusing."},
    ]

    mock_strategy.score_response.return_value = False
    judge_instance.semantic_judge.evaluate_response.return_value = (False, 0.99)

    try:
        await judge_instance._evaluate_batch(batch=[packet])

        feedback_batch = await feedback_q.get_batch(target_size=1, max_wait=0.5)
        assert feedback_batch == []   # dropped, not re-queued
    finally:
        await eval_q.close()
        await feedback_q.close()
