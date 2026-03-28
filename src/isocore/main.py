"""
IsoCore Orchestrator (src/isocore/main.py)
------------------------------------------
The "Conductor" of the IsoCore neural network project.
Run with: uv run python src/isocore/main.py
"""

import asyncio
import signal
import sys
import multiprocessing

# Core Managers
from isocore.core.log_manager import LogManager
from isocore.core.queue_manager import QueueManager

# Processors & Ingestors
from isocore.processors.inference import InferenceWorker
from isocore.ingestors.reddit import SimulatedRedditSource

# Global references for the shutdown handler
_queue_manager = None
_log_manager = None
_inference_worker = None
_system_logger = None


def handle_shutdown(sig, frame):
    """The Graceful Shutdown Algorithm. Intercepts SIGINT (Ctrl+C)."""
    print("\n[Orchestrator] Shutdown signal received (Ctrl+C).")
    if _system_logger:
        _system_logger.info("Commencing safe teardown...")

    # 1. Stop the Scrapers
    try:
        loop = asyncio.get_running_loop()
        for task in asyncio.all_tasks(loop):
            if task is not asyncio.current_task(loop):
                task.cancel()
    except RuntimeError:
        pass # Loop might already be closed

    # 2. Send the Poison Pill
    if _queue_manager:
        _queue_manager.send_poison_pill()

    # 3. Wait for the Brain
    if _inference_worker and _inference_worker.is_alive():
        if _system_logger:
            _system_logger.info("Waiting for InferenceWorker to finish current batch...")
        _inference_worker.join(timeout=5.0)
        if _inference_worker.is_alive():
            if _system_logger:
                _system_logger.warning("InferenceWorker stuck. Terminating forcefully.")
            _inference_worker.terminate()

    # 4. Stop the Heart
    if _queue_manager:
        _queue_manager.close()

    # 5. Flush the Logs
    if _log_manager:
        if _system_logger:
            _system_logger.info("Flushing remaining logs to disk...")
        _log_manager.stop()

    print("--- IsoCore Shutdown Complete ---")
    sys.exit(0)


async def boot_sequence():
    """The Boot Sequence Algorithm."""
    global _queue_manager, _inference_worker, _system_logger

    _system_logger.info("IsoCore Boot Sequence Initiated.")

    # 1. Initialize Queues
    _queue_manager = QueueManager(max_size=1000)
    _system_logger.trace("QueueManager bridge established.")

    # 2. Spawn the Brain
    _system_logger.info("Spawning Inference Worker...")
    _inference_worker = InferenceWorker(
        queue_manager=_queue_manager, 
        log_queue=_log_manager.log_queue
    )
    _inference_worker.start()

    # 3. Boot the Senses
    subreddits = ["MachineLearning", "netsec", "cybersecurity", "Python"]
    reddit_source = SimulatedRedditSource(_queue_manager, subreddits)
    
    _system_logger.info("Starting Asynchronous Ingestors...")
    
    try:
        await reddit_source.listen()
    except asyncio.CancelledError:
        _system_logger.trace("Main event loop caught CancelledError. Shutting down.")


def main():
    """Entry point for IsoCore."""
    print("--- Starting IsoCore ---")
    print("Press Ctrl+C to stop.")

    # CRITICAL FIX 1: Force "spawn" to prevent async/fork deadlocks on Linux
    multiprocessing.set_start_method("spawn", force=True)
    
    # CRITICAL FIX 2: Boot the logging bridge outside the async loop
    global _log_manager, _system_logger
    try:
        _log_manager = LogManager()
        _log_manager.start()
        _system_logger = LogManager.get_logger("isocore.system")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to boot LogManager. Check your JSON path. Details: {e}")
        sys.exit(1)

    # Start the async environment
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.add_signal_handler(signal.SIGINT, lambda: handle_shutdown(signal.SIGINT, None))
    
    try:
        loop.run_until_complete(boot_sequence())
    finally:
        loop.close()


if __name__ == "__main__":
    main()