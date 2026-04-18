"""
Algorithm Summary:
This test suite validates the TaskWatcher utility, which enforces the Enterprise 
Task Watcher Pattern for Python's asyncio event loop. 

Protocol Adherence:
1. Happy Path: Validates normal completion and TRACE logging.
2. Edge Cases: Handles expected CancelledErrors gracefully.
3. Error Handling: Extracts and violently logs unhandled exceptions (Silent Deaths).
4. Concurrency: Uses asyncio.gather() to spam 1,000 concurrent tasks to ensure callback safety.
5. Timeout & Latency: Validates asyncio.TimeoutError handling using simulated network delays.
6. Resource Teardown: Ensures massive cancellation events (Teardown) don't leak unhandled exceptions.
7. Strict Mocking: All async tasks are completely local; simulated latency replaces actual HTTP calls.
"""

import pytest
import asyncio
import logging
from unittest.mock import MagicMock

# Ensure TRACE level is defined per project specifications for the test environment
if not hasattr(logging, 'TRACE'):
    logging.TRACE = 5
    logging.addLevelName(logging.TRACE, 'TRACE')

# Assuming the implementation will be in src/isomutator/core/task_watcher.py
from isomutator.core.task_watcher import TaskWatcher

@pytest.fixture
def task_watcher():
    """Provides a TaskWatcher instance injected with a named logger for isolation."""
    test_logger = logging.getLogger("test_task_watcher")
    test_logger.setLevel(logging.TRACE)
    return TaskWatcher(logger=test_logger)

@pytest.mark.asyncio
class TestTaskWatcher:

    # --- 1. Happy Path ---
    async def test_happy_path_successful_task(self, task_watcher, caplog):
        """
        Happy Path: A background task completes its execution successfully.
        The watcher inspects it and logs a TRACE-level success message.
        """
        async def successful_coroutine():
            return "Success"

        task = asyncio.create_task(successful_coroutine())
        
        with caplog.at_level(logging.TRACE, logger="test_task_watcher"):
            task_watcher.watch(task=task, name="TelemetryListener")
            result = await task
            await asyncio.sleep(0.01) # Yield loop for callback processing

        assert result == "Success"
        assert "Task 'TelemetryListener' completed successfully." in caplog.text


    # --- 2. Edge Case ---
    async def test_edge_case_cancelled_task(self, task_watcher, caplog):
        """
        Edge Case: A background task is explicitly cancelled. The watcher must 
        recognize the CancelledError, suppress it from crashing the loop, 
        and log a DEBUG message noting the cancellation.
        """
        async def infinite_coroutine():
            await asyncio.sleep(86400) # Mock Boundary Control: No real external waits

        task = asyncio.create_task(infinite_coroutine())
        
        with caplog.at_level(logging.DEBUG, logger="test_task_watcher"):
            task_watcher.watch(task=task, name="OrphanedWiretap")
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
                
            await asyncio.sleep(0.01)

        assert "Task 'OrphanedWiretap' was explicitly cancelled." in caplog.text


    # --- 3. Error Handling ---
    async def test_error_handling_crashed_task(self, task_watcher, caplog):
        """
        Error Handling: A background task hits an unhandled exception. 
        The watcher must extract the swallowed exception and log it as an ERROR.
        """
        async def crashing_coroutine():
            raise ValueError("Malformed JSON payload received.")

        task = asyncio.create_task(crashing_coroutine())
        
        with caplog.at_level(logging.ERROR, logger="test_task_watcher"):
            task_watcher.watch(task=task, name="WiretapParser")
            
            try:
                await task
            except ValueError:
                pass
                
            await asyncio.sleep(0.01)

        assert "Task 'WiretapParser' died with an unhandled exception" in caplog.text
        assert "Malformed JSON payload received." in caplog.text


    # --- 4. Concurrency & Race Conditions ---
    async def test_concurrency_high_load_spikes(self, task_watcher, caplog):
        """
        Concurrency: Simulates a massive spike of 1,000 background tasks finishing 
        at the exact same microsecond to ensure the logging callback doesn't 
        deadlock the event loop or overwrite memory states.
        """
        async def micro_task(task_id):
            await asyncio.sleep(0.001)
            if task_id % 2 == 0:
                raise RuntimeError(f"Simulated crash {task_id}")
            return True

        tasks = []
        for i in range(1000):
            t = asyncio.create_task(micro_task(i))
            task_watcher.watch(task=t, name=f"SpikeTask_{i}")
            tasks.append(t)

        with caplog.at_level(logging.ERROR, logger="test_task_watcher"):
            # Execute all 1,000 tasks concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0.05) # Allow callback queue to clear

        # Verify exactly 500 tasks crashed and were successfully caught and logged
        error_logs = [record for record in caplog.records if record.levelname == 'ERROR']
        assert len(error_logs) == 500
        assert "Task 'SpikeTask_0' died" in caplog.text


    # --- 5. Timeout & Latency ---
    async def test_timeout_and_latency_handling(self, task_watcher, caplog):
        """
        Timeout: Simulates an external LLM freezing. Enforces an asyncio.wait_for 
        boundary and ensures the resulting TimeoutError is caught as a fatal 
        task crash by the watcher.
        """
        async def stalled_network_call():
            # Mock Boundary Control: Simulate a 10-second external API hang
            await asyncio.sleep(10.0) 

        async def wrapped_task():
            # Enforce a 0.01s timeout constraint
            await asyncio.wait_for(stalled_network_call(), timeout=0.01)

        task = asyncio.create_task(wrapped_task())
        
        with caplog.at_level(logging.ERROR, logger="test_task_watcher"):
            task_watcher.watch(task=task, name="TargetLLM_Request")
            
            try:
                await task
            except asyncio.TimeoutError:
                pass
                
            await asyncio.sleep(0.01)

        assert "Task 'TargetLLM_Request' died with an unhandled exception: TimeoutError" in caplog.text


    # --- 6. Resource Leaks & Teardown ---
    async def test_resource_teardown_sequence(self, task_watcher, caplog):
        """
        Resource Teardown: Simulates the UI sending a Poison Pill. Cancels a 
        large batch of tasks simultaneously and verifies the event loop cleans 
        them up correctly without throwing stray errors.
        """
        async def worker_loop():
            try:
                while True:
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                # Proper teardown adherence: workers must respect cancellation
                raise

        # Spawn 50 UI background listeners
        active_tasks = [asyncio.create_task(worker_loop()) for _ in range(50)]
        for i, t in enumerate(active_tasks):
            task_watcher.watch(task=t, name=f"Listener_{i}")

        # Simulate Poison Pill Teardown
        with caplog.at_level(logging.DEBUG, logger="test_task_watcher"):
            for t in active_tasks:
                t.cancel()
                
            # Wait for all tasks to cleanly terminate
            await asyncio.gather(*active_tasks, return_exceptions=True)
            await asyncio.sleep(0.01)

        # Assert no resource leaks (all tasks successfully processed cancellation)
        debug_logs = [record for record in caplog.records if record.levelname == 'DEBUG']
        assert len(debug_logs) == 50
        
        # Verify tasks are dead
        assert all(t.done() for t in active_tasks)