"""
Integration test infrastructure.

Redis tests use db=15 exclusively so they can never touch production data.
The clean_redis fixture flushes that database before and after every test.
"""

import pytest
import redis as redis_sync


REDIS_TEST_URL = "redis://localhost:6379/15"


@pytest.fixture(scope="session")
def redis_session():
    """
    Opens one synchronous Redis connection to db=15 for the entire session.
    Skips every test in the module if Redis is unreachable.
    """
    try:
        client = redis_sync.Redis(
            host="localhost", port=6379, db=15,
            decode_responses=True, socket_connect_timeout=2,
        )
        client.ping()
        yield client
        client.flushdb()
        client.close()
    except (redis_sync.exceptions.ConnectionError, redis_sync.exceptions.TimeoutError):
        pytest.skip("Redis not reachable — skipping integration tests")


@pytest.fixture
def clean_redis(redis_session):
    """Flushes db=15 before and after each test for full isolation."""
    redis_session.flushdb()
    yield redis_session
    redis_session.flushdb()
