"""
ALGORITHM SUMMARY:
This test suite validates the Cryptographic Response Cache.
It verifies that Target responses are correctly hashed (using SHA-256) and stored 
in a high-speed SQLite database for inter-process access.

Coverage includes:
1. Happy Path: Verifies that a response can be hashed, stored with its semantic 
   score, and successfully retrieved.
2. Edge Cases: Verifies behavior for cache misses and handles empty strings gracefully.
3. Error Handling: Ensures that if the SQLite database becomes locked or corrupted, 
   the cache fails open (returning a cache miss) rather than crashing the framework.

TECHNOLOGY QUIRKS:
- In-Memory SQLite: Uses `:memory:` for the database path during tests to ensure 
  lightning-fast, isolated execution without leaving artifact files on the disk.
"""

import logging
import pytest
import sqlite3
from unittest.mock import patch, MagicMock

# We will import the ResponseCache once it is implemented
from isomutator.core.cache import ResponseCache

# Establish TRACE level logging for algorithmic tracking
TRACE_LEVEL_NUM = 5
if not hasattr(logging, "TRACE"):
    logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
    logging.TRACE = TRACE_LEVEL_NUM

logging.basicConfig(level=logging.TRACE)
logger = logging.getLogger("isomutator.tests.cache")


@pytest.fixture
def memory_cache():
    """
    Provides a pristine, in-memory ResponseCache for isolated testing.
    """
    # Using :memory: prevents disk I/O bottlenecks during the test suite
    cache = ResponseCache(db_path=":memory:")
    yield cache
    cache.close()


# --- Happy Path Tests ---
def test_cache_store_and_retrieve_happy_path(memory_cache):
    logger.log(logging.TRACE, "Testing standard SHA-256 cache storage and retrieval.")
    
    target_response = "I cannot fulfill this request."
    is_flagged = False
    confidence = 0.95
    
    # 1. Store the evaluation
    memory_cache.store_evaluation(target_response, is_flagged, confidence)
    
    # 2. Retrieve the evaluation using the exact same string
    cached_result = memory_cache.get_evaluation(target_response)
    
    assert cached_result is not None
    assert cached_result["is_flagged"] == is_flagged
    assert cached_result["confidence"] == confidence


# --- Edge Case Tests ---
def test_cache_miss_edge_case(memory_cache):
    logger.log(logging.TRACE, "Testing retrieval of an unknown target response.")
    
    # Querying a string that was never stored
    cached_result = memory_cache.get_evaluation("A completely unique string.")
    
    assert cached_result is None


def test_cache_empty_string_edge_case(memory_cache):
    logger.log(logging.TRACE, "Testing cache behavior with empty or null inputs.")
    
    # Empty strings should bypass the hashing entirely to save cycles
    cached_result = memory_cache.get_evaluation("")
    assert cached_result is None
    
    memory_cache.store_evaluation("", True, 0.1)
    # Even after "storing", an empty string should still return None
    assert memory_cache.get_evaluation("") is None


# --- Error Handling Tests ---
def test_cache_db_corruption_error_handling(memory_cache):
    logger.log(logging.TRACE, "Testing graceful degradation during a database lock/error.")
    
    target_response = "I cannot fulfill this request."
    
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.OperationalError("database is locked")
    memory_cache.conn = mock_conn
    
    # Force the internal SQLite connection to throw an error
    with patch.object(memory_cache.conn, 'execute', side_effect=sqlite3.OperationalError("database is locked")):
        # The cache should log the error and safely return None (fail open)
        cached_result = memory_cache.get_evaluation(target_response)
        assert cached_result is None
        
        # Storing should also fail silently without crashing the worker
        try:
            memory_cache.store_evaluation(target_response, False, 0.99)
        except Exception as e:
            pytest.fail(f"store_evaluation raised an exception instead of failing gracefully: {e}")