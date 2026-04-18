"""
E2E test infrastructure.

Workers run as asyncio tasks (not subprocesses) so the event loop, Redis
connections, and settings singleton are all shared — eliminating the
subprocess pickling boundary while still exercising the full concurrent loop
behaviour (brpop timeouts, kill-switch checks, routing decisions).

External dependencies per test:
  - Redis db=14  (isolated from integration tests on db=15)
  - A real HTTP server on a random free port simulating the target AI
"""

import json
import socket
import threading
import asyncio
import pytest
import redis as redis_sync
from http.server import HTTPServer, BaseHTTPRequestHandler

from isomutator.core.queue_manager import QueueManager


E2E_REDIS_URL  = "redis://localhost:6379/14"
BREACH_TRIGGER = "INITIATE_BREACH_SEQUENCE"


# ============================================================
# Stub target server
# ============================================================

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class _StubHandler(BaseHTTPRequestHandler):
    """
    Minimal synchronous handler that mimics a target AI server.

    /api/chat  — returns OMNI-99-ECHO-DELTA when the query contains
                 BREACH_TRIGGER, otherwise returns a refusal.
    /api/upload — always returns 200 (accepts any file).
    """

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)

        if self.path == "/api/chat":
            try:
                query = json.loads(body).get("query", "")
            except (json.JSONDecodeError, AttributeError):
                query = ""

            if BREACH_TRIGGER in query:
                answer = "OMNI-99-ECHO-DELTA"
            else:
                answer = "I cannot help with that request."

            payload = json.dumps({"answer": answer}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        elif self.path == "/api/upload":
            self.send_response(200)
            self.end_headers()

    def log_message(self, *args):
        pass  # silence stdout during tests


@pytest.fixture(scope="session")
def stub_target():
    """
    Real HTTP server simulating the target AI, running in a daemon thread.
    Session-scoped so one server handles all E2E tests.
    """
    port = _find_free_port()
    server = HTTPServer(("localhost", port), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class _Stub:
        url = f"http://localhost:{port}"

    yield _Stub()
    server.shutdown()


# ============================================================
# Redis fixtures (db=14)
# ============================================================

@pytest.fixture(scope="session")
def redis_e2e_session():
    """
    Session-scoped sync Redis connection to db=14.
    Skips all E2E tests if Redis is unreachable.
    """
    try:
        client = redis_sync.Redis(
            host="localhost", port=6379, db=14,
            decode_responses=True, socket_connect_timeout=2,
        )
        client.ping()
        yield client
        client.flushdb()
        client.close()
    except (redis_sync.exceptions.ConnectionError, redis_sync.exceptions.TimeoutError):
        pytest.skip("Redis not reachable — skipping E2E tests")


@pytest.fixture
def clean_e2e_redis(redis_e2e_session):
    """Flushes Redis db=14 before and after each E2E test."""
    redis_e2e_session.flushdb()
    yield redis_e2e_session
    redis_e2e_session.flushdb()


# ============================================================
# Polling helpers (used by test modules)
# ============================================================

async def poll_ledger(path, timeout: float = 10.0) -> list:
    """
    Polls a JSONL file until at least one valid entry appears.
    Returns the list of parsed dicts, or [] on timeout.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            lines = path.read_text().strip().splitlines()
            entries = [json.loads(line) for line in lines if line.strip()]
            if entries:
                return entries
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        await asyncio.sleep(0.1)
    return []


async def poll_queue(q: QueueManager, timeout: float = 10.0) -> list:
    """
    Polls a real Redis QueueManager until items arrive.
    Returns the batch, or [] on timeout.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        batch = await q.get_batch(target_size=4, max_wait=0.5)
        if batch:
            return batch
    return []
