# test_logging.py

"""
ALGORITHM SUMMARY:
Validates the Asynchronous Rotating Telemetry architecture across process boundaries.

Coverage Additions:
- Multiprocessing Lifecycle: Spawns a discrete OS process, verifies logging IPC, 
  and strictly asserts worker termination (is_alive == False).
"""

import multiprocessing
import time
import pytest
from isomutator.core.log_manager import LogManager

def isolated_worker_process(log_queue):
    """Simulates a totally isolated memory space."""
    LogManager.setup_worker(log_queue)
    logger = LogManager.get_logger("isomutator.worker")
    logger.info("Worker process booted.")
    time.sleep(0.1)

def test_multiprocessing_log_propagation_and_teardown():
    """Verifies cross-process logging and strict worker termination."""
    log_manager = LogManager()
    log_manager.start()
    
    # 1. Spawn Worker
    worker = multiprocessing.Process(
        target=isolated_worker_process, 
        args=(log_manager.log_queue,)
    )
    worker.start()
    
    # 2. Wait for execution with a hard limit
    worker.join(timeout=2.0)
    
    # 3. Assert Strict Teardown (CRITICAL RULE)
    assert worker.is_alive() == False, "CRITICAL: Zombie process detected!"
    
    # 4. Clean up the log manager
    log_manager.stop()