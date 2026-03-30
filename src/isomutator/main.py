"""
ALGORITHM SUMMARY:
This is the IsoMutator Orchestrator. It manages the stateful, three-stage AI red-teaming pipeline.
1. It initializes the Queue network: Attack Queue (outgoing payloads), Eval Queue (incoming Target responses), 
   and the Feedback Queue (Target refusals routed back for counter-arguments).
2. It boots the multiprocessing workers (RedTeamJudge and AsyncStriker).
3. It engages the PromptMutator in the main event loop to drive the autonomous generation cycle.
4. It catches OS-level interrupt signals to execute a safe, ordered teardown of all threads and queues.

TECHNOLOGY QUIRKS:
- Asyncio Task Management: The Mutator runs indefinitely in the main event loop. When Ctrl+C 
  is pressed, the `handle_shutdown` algorithm cancels this task, raising a CancelledError which 
  we catch cleanly to prevent ugly stack traces.
- Multiprocessing Queue Teardown Fix: Previously, the global queue references were lost. We now 
  track all instantiated QueueManagers in `_active_queues` to ensure their underlying semaphores 
  are cleanly released during shutdown, preventing memory leaks.
- Start Method Enforcement: We force the `spawn` context for multiprocessing to prevent the 
  notorious async/fork deadlocks that occur natively on Linux environments.
"""

import asyncio
import signal
import sys
import multiprocessing

from isomutator.core.queue_manager import QueueManager
from isomutator.core.log_manager import LogManager
from isomutator.core.config import settings
from isomutator.ingestors.mutator import PromptMutator
from isomutator.processors.striker import AsyncStriker
from isomutator.processors.judge import RedTeamJudge

# Global references for the shutdown handler
_active_queues = [] # Fixed: Now properly tracks queues for teardown
_log_manager = None
_inference_workers = []
_system_logger = None
_shutdown_event = multiprocessing.Event()


def handle_shutdown(sig, frame):
    """
    The Graceful Shutdown Algorithm. 
    Intercepts SIGINT (Ctrl+C) and unwinds the architecture in a strict, safe order.
    """
    print("\n[Orchestrator] Shutdown signal received (Ctrl+C).")
    if _system_logger:
        _system_logger.info("Commencing safe teardown...")

    # 1. Stop the Async Ingestors (Mutator)
    try:
        loop = asyncio.get_running_loop()
        for task in asyncio.all_tasks(loop):
            if task is not asyncio.current_task(loop):
                task.cancel()
    except RuntimeError:
        pass # Loop might already be closed

    # 2. Engage the Emergency Stop for Workers
    if _system_logger:
        _system_logger.info("Engaging Emergency Stop Event. Instructing workers to bypass queues...")
    _shutdown_event.set()

    # 3. Drain and Terminate the Multiprocessing Fleet
    if _inference_workers:
        if _system_logger:
            _system_logger.info(f"Waiting for {len(_inference_workers)} workers to finish their current payload batch...")
        for worker in _inference_workers:
            if worker.is_alive():
                worker.join(timeout=settings.shutdown_timeout)
                if worker.is_alive():
                    if _system_logger:
                        _system_logger.warning(f"Worker {worker.name} is deadlocked. Terminating forcefully.")
                    worker.terminate()

    # 4. Safely Close the Inter-Process Queues
    if _active_queues:
        if _system_logger:
            _system_logger.info("Releasing queue semaphores...")
        for q in _active_queues:
            q.close()

    # 5. Flush the Logging Buffers
    if _log_manager:
        if _system_logger:
            _system_logger.info("Flushing remaining logs to disk...")
        _log_manager.stop()

    print("--- IsoMutator Shutdown Complete ---")
    sys.exit(0)


async def boot_sequence():
    """
    The Boot Sequence Algorithm.
    Wires the queues to the workers, starts the sub-processes, and locks the main thread 
    into the Mutator's generation loop.
    """
    global _system_logger, _inference_workers, _active_queues

    _system_logger.info("IsoMutator Boot Sequence Initiated. Constructing pipeline...")
    
    # 1. Boot the Three-Stage Queues
    _system_logger.trace("Initializing Attack, Evaluation, and Feedback queues (max_size=1000)...")
    _attack_queue = QueueManager(max_size=1000)
    _eval_queue = QueueManager(max_size=1000)
    _feedback_queue = QueueManager(max_size=1000)
    
    # Register queues for safe teardown
    _active_queues.extend([_attack_queue, _eval_queue, _feedback_queue])

    # 2. Boot the Red Team Judge (The Scorer & Router)
    _system_logger.trace("Spawning RedTeamJudge worker process...")
    judge = RedTeamJudge(
        eval_queue=_eval_queue,
        feedback_queue=_feedback_queue,
        log_queue=_log_manager.log_queue
    )
    judge.start()
    _inference_workers.append(judge)

    # 3. Boot the Async Striker (The Outbound Cannon)
    _system_logger.trace("Spawning AsyncStriker worker process...")
    striker = AsyncStriker(
        attack_queue=_attack_queue,
        eval_queue=_eval_queue,
        log_queue=_log_manager.log_queue,
        # Pointing to the remote Ollama daemon's chat endpoint
        target_url="http://192.9.159.125:11434/api/chat" 
    )
    striker.start()
    _inference_workers.append(striker) 

    # 4. Boot the Payload Generator (The Brain)
    _system_logger.info("Starting Stateful Asynchronous Prompt Mutator...")
    mutator = PromptMutator(
        attack_queue=_attack_queue, 
        feedback_queue=_feedback_queue
    )
    
    try:
        # Lock the main thread into the generation loop
        await mutator.listen()
    except asyncio.CancelledError:
        _system_logger.trace("Main event loop caught CancelledError. Generator shut down cleanly.")


def main():
    """
    Entry point for IsoMutator.
    Handles low-level OS configuration before passing control to the async boot sequence.
    """
    print("--- Starting IsoMutator ---")
    print("Press Ctrl+C to stop.")

    # Force "spawn" to prevent async/fork deadlocks on Linux
    multiprocessing.set_start_method("spawn", force=True)
    
    # Boot the logging bridge outside the async loop to guarantee early availability
    global _log_manager, _system_logger
    try:
        _log_manager = LogManager()
        _log_manager.start()
        _system_logger = LogManager.get_logger("isomutator.system")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to boot LogManager. Check your JSON path. Details: {e}")
        sys.exit(1)

    # Start the async environment and bind the OS signal handler
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.add_signal_handler(signal.SIGINT, lambda: handle_shutdown(signal.SIGINT, None))
    
    try:
        loop.run_until_complete(boot_sequence())
    finally:
        loop.close()


if __name__ == "__main__":
    main()