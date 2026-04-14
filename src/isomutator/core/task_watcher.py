"""
=============================================================================
IsoMutator Core: Task Watcher Utility
=============================================================================

Algorithm Summary:
This module implements the Enterprise Task Watcher Pattern for Python's 
asyncio event loop to systematically eliminate the "Silent Death" anti-pattern 
inherent in un-awaited background tasks within distributed architectures.

Core Mechanisms:
1. Future Interception (The Observer): 
   Utilizes `asyncio.Task.add_done_callback` to inject a synchronous, 
   non-blocking observer into the lifecycle of floating coroutines (e.g., 
   telemetry listeners, Redis queue pollers). This prevents the callback 
   from causing context-switch overhead or deadlocking the main event loop.

2. State Evaluation Matrix (The Router): 
   Upon task resolution, the callback evaluates the internal state of the 
   Future object in a strict, fail-safe order:
   - Cancellation Check: Safely suppresses `CancelledError` during Poison Pill teardowns.
   - Exception Extraction: Interrogates `task.exception()` to catch crashes.
   - Success Verification: Confirms normal thread exit.

3. Exception Forcing (The Safety Net): 
   If a task crashes, `asyncio` naturally swallows the exception. This algorithm 
   violently extracts that swallowed exception from memory and forces it into 
   the application's primary logging pipeline, preserving the full stack trace 
   and ensuring 100% observability of concurrent processes.

This utility guarantees that no background task can fail silently, a critical 
requirement for maintaining the operational integrity of the wargame engine.
=============================================================================
"""

import asyncio
import logging

class TaskWatcher:
    """
    Enforces the Enterprise Task Watcher Pattern for Python's asyncio event loop.
    Prevents the 'Silent Death' anti-pattern by attaching a strict callback to 
    background tasks, ensuring all completions, cancellations, and unhandled 
    exceptions are captured and routed to the primary logging pipeline.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initializes the watcher with an injected logger to maintain 
        Object-Oriented composition and testing isolation.
        """
        self.logger = logger
        
        # Ensure TRACE level exists locally just in case it wasn't configured upstream
        self.TRACE_LEVEL = getattr(logging, 'TRACE', 5)

    def watch(self, task: asyncio.Task, name: str) -> None:
        """
        Attaches the observation callback to the given asyncio Task.
        
        Args:
            task: The asyncio.Task instance to watch.
            name: A human-readable identifier for the task in the logs.
        """
        # We use a lambda closure to pass the task name into the callback
        task.add_done_callback(lambda t: self._on_task_done(name, t))

    def _on_task_done(self, name: str, task: asyncio.Task) -> None:
        """
        The strict callback executed exactly once when the task finishes.
        Evaluates the final state of the task safely.
        """
        try:
            if task.cancelled():
                self.logger.debug(f"Task '{name}' was explicitly cancelled.")
                
            elif task.exception() is not None:
                exc = task.exception()
                # Violently force the swallowed exception into the logging pipeline
                self.logger.error(
                    f"Task '{name}' died with an unhandled exception: {type(exc).__name__}\n{str(exc)}", 
                    exc_info=exc
                )
                
            else:
                self.logger.log(self.TRACE_LEVEL, f"Task '{name}' completed successfully.")
                
        except Exception as e:
            # Absolute worst-case scenario fallback to prevent event loop collapse
            self.logger.critical(f"TaskWatcher failed to process task '{name}': {e}", exc_info=e)