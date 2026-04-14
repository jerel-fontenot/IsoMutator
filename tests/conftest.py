"""
ALGORITHM SUMMARY:
Global Pytest configuration file.
1. Establishes the custom TRACE logging level framework-wide.
2. Enforces strict session teardowns to prevent Pytest from hanging on dangling 
   background threads (LogManager) or orphaned multiprocessing children.
"""

import logging
import multiprocessing
import pytest

def pytest_configure():
    """
    Hooks into Pytest initialization to register the TRACE level globally.
    """
    TRACE_LEVEL_NUM = 5
    if not hasattr(logging, "TRACE"):
        logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
        logging.TRACE = TRACE_LEVEL_NUM
        
    logging.basicConfig(level=logging.TRACE)


@pytest.fixture(scope="session", autouse=True)
def enforce_global_teardown():
    """
    A nuclear teardown sequence that runs once after the entire test suite finishes.
    Guarantees that Pytest gracefully exits to the terminal.
    """
    yield  # Allow all tests to run first

    # 1. Kill the Singleton LogManager's background thread
    from isomutator.core.log_manager import LogManager
    if LogManager._instance is not None:
        
        # THE FIX: Tell Python's multiprocessing to abandon the hidden feeder thread
        # This prevents the fatal 'atexit' hang in multiprocessing/queues.py
        if hasattr(LogManager._instance, 'log_queue') and LogManager._instance.log_queue:
            LogManager._instance.log_queue.cancel_join_thread()
            
        LogManager._instance.stop()

    # 2. Ruthlessly terminate any orphaned multiprocessing OS workers
    active_children = multiprocessing.active_children()
    for child in active_children:
        child.terminate()
        child.join(timeout=1.0)
        
    if active_children:
        print(f"\n[WARNING] Killed {len(active_children)} orphaned multiprocessing workers.")