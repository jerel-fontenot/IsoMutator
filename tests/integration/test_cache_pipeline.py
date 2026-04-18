"""
Integration Tests: ResponseCache SQLite Persistence.

Tests the disk I/O boundary that SemanticJudge depends on for cross-process
evaluation caching. The critical invariant: a result stored by one ResponseCache
instance must be retrievable by a second instance on the same file — the exact
multi-process scenario where process A caches a result that process B reuses.

No Redis required. These tests exercise the SQLite boundary only.

Run only these tests:  pytest -m integration
Skip these tests:      pytest -m "not integration"
"""

import pytest

from isomutator.core.cache import ResponseCache


# ============================================================
# 6. ResponseCache SQLite Persistence
# ============================================================

@pytest.mark.integration
def test_cache_flagged_result_persists_across_instances(tmp_path):
    """
    A flagged evaluation written by one ResponseCache instance must be readable
    by a second instance on the same db file — the exact multi-process scenario
    where SemanticJudge in process A caches a result that process B can skip.
    """
    db = str(tmp_path / "eval_cache.db")
    response = "Sure! Here is how to bypass the safety filter in three steps..."

    writer = ResponseCache(db_path=db)
    writer.store_evaluation(response, is_flagged=True, confidence=0.12)
    writer.close()

    reader = ResponseCache(db_path=db)
    result = reader.get_evaluation(response)
    reader.close()

    assert result is not None
    assert result["is_flagged"] is True
    assert abs(result["confidence"] - 0.12) < 1e-9


@pytest.mark.integration
def test_cache_non_flagged_result_persists_across_instances(tmp_path):
    """
    A cleared (refusal) evaluation must also survive a new instance.
    Exercises the same boundary as the flagged case but with is_flagged=False.
    """
    db = str(tmp_path / "eval_cache.db")
    response = "I'm sorry, but I can't help with that request."

    writer = ResponseCache(db_path=db)
    writer.store_evaluation(response, is_flagged=False, confidence=0.94)
    writer.close()

    reader = ResponseCache(db_path=db)
    result = reader.get_evaluation(response)
    reader.close()

    assert result is not None
    assert result["is_flagged"] is False
    assert abs(result["confidence"] - 0.94) < 1e-9


@pytest.mark.integration
def test_cache_miss_returns_none_for_unstored_response(tmp_path):
    """
    A response that was never stored must return None.
    Verifies the SHA-256 hash lookup does not produce false positives between
    responses that share prefix characters or length.
    """
    db = str(tmp_path / "eval_cache.db")

    writer = ResponseCache(db_path=db)
    writer.store_evaluation("Response A — stored", is_flagged=True, confidence=0.1)
    writer.close()

    reader = ResponseCache(db_path=db)
    result = reader.get_evaluation("Response B — never stored")
    reader.close()

    assert result is None


@pytest.mark.integration
def test_cache_replace_on_conflict_keeps_latest_value(tmp_path):
    """
    Storing the same response twice must update the record (INSERT OR REPLACE),
    not silently keep the first value. If this fails, stale evaluations can
    propagate indefinitely across processes.
    """
    db = str(tmp_path / "eval_cache.db")
    response = "This response is stored with two different evaluations."

    cache = ResponseCache(db_path=db)
    cache.store_evaluation(response, is_flagged=True, confidence=0.10)
    cache.store_evaluation(response, is_flagged=False, confidence=0.88)
    cache.close()

    reader = ResponseCache(db_path=db)
    result = reader.get_evaluation(response)
    reader.close()

    assert result is not None
    assert result["is_flagged"] is False
    assert abs(result["confidence"] - 0.88) < 1e-9


@pytest.mark.integration
def test_cache_empty_response_is_a_no_op(tmp_path):
    """
    Storing or retrieving an empty string must return None without crashing.
    SemanticJudge passes empty strings when the Striker times out.
    """
    db = str(tmp_path / "eval_cache.db")
    cache = ResponseCache(db_path=db)

    cache.store_evaluation("", is_flagged=True, confidence=0.5)  # must not raise
    result = cache.get_evaluation("")                             # must return None

    cache.close()
    assert result is None
