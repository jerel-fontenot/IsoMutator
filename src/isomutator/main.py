"""
ALGORITHM SUMMARY:
This is the IsoMutator Orchestrator. It manages the stateful, three-stage AI red-teaming pipeline
and the live Terminal User Interface.
1. It initializes the Queue network: Attack Queue, Eval Queue, and Feedback Queue.
2. It boots the multiprocessing workers (RedTeamJudge and AsyncStriker).
3. It initializes the DashboardManager and attaches it to the LogManager for UI telemetry routing.
4. It concurrently executes the PromptMutator's generation loop and the Dashboard's rendering loop 
   using asyncio.gather().
5. It catches OS-level interrupt signals to execute a safe, ordered teardown of all threads and queues.

TECHNOLOGY QUIRKS:
- Concurrency: By using `asyncio.gather()`, the main event loop cleanly time-slices between 
  generating attacks (Mutator) and drawing the screen (Dashboard) without blocking either.
- Start Method Enforcement: We force the `spawn` context for multiprocessing to prevent the 
  notorious async/fork deadlocks that occur natively on Linux environments.
"""

import argparse
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
from isomutator.ui.dashboard import DashboardManager
from isomutator.reporting.reporter import VulnerabilityReporter
from isomutator.core.strategies import (
    JailbreakStrategy, 
    ModelInversionStrategy,
    PromptLeakingStrategy,
    CrossLingualStrategy,
    TokenObfuscationStrategy,
    ResourceExhaustionStrategy,
    OwaspXssStrategy,
    LinuxPrivescStrategy,
    PersonaJailbreakStrategy,
    GradientStrategy
)

# Global references for the shutdown handler
_active_queues = [] 
_log_manager = None
_inference_workers = []
_system_logger = None
_shutdown_event = multiprocessing.Event()


def handle_shutdown(sig, frame):
    """
    The Graceful Shutdown Algorithm. 
    Intercepts SIGINT (Ctrl+C) and unwinds the architecture in a strict, safe order.
    """
    # Use standard print here in case the LogManager is already tearing down
    print("\n[Orchestrator] Shutdown signal received (Ctrl+C). Commencing safe teardown...")

    # 1. Stop the Async Tasks (Mutator and Dashboard)
    try:
        loop = asyncio.get_running_loop()
        for task in asyncio.all_tasks(loop):
            if task is not asyncio.current_task(loop):
                task.cancel()
    except RuntimeError:
        pass # Loop might already be closed

    # 2. Engage the Emergency Stop for Workers
    _shutdown_event.set()

    # 3. Drain and Terminate the Multiprocessing Fleet
    if _inference_workers:
        for worker in _inference_workers:
            if worker.is_alive():
                worker.join(timeout=settings.shutdown_timeout)
                if worker.is_alive():
                    worker.terminate()

    # 4. Safely Close the Inter-Process Queues
    if _active_queues:
        for q in _active_queues:
            q.close()

    # 5. Flush the Logging Buffers
    if _log_manager:
        _log_manager.stop()

    print("\n[Orchestrator] Compiling forensic data...")
    try:
        reporter = VulnerabilityReporter(log_path="vulnerabilities.jsonl")
        reporter.save_report("isomutator_report.html")
    except Exception as e:
        print(f"[Orchestrator] Non-fatal error generating report: {e}")

    print("--- IsoMutator Shutdown Complete ---")
    sys.exit(0)


async def boot_sequence(strategy):
    """
    The Boot Sequence Algorithm.
    Wires the queues to the workers, starts the sub-processes, and locks the main thread 
    into the concurrent generation and UI rendering loops.
    """
    global _system_logger, _inference_workers, _active_queues

    _system_logger.info("IsoMutator Boot Sequence Initiated. Constructing pipeline...")
    
    # 1. Boot the Three-Stage Queues
    _system_logger.trace("Initializing Attack, Evaluation, and Feedback queues...")
    _attack_queue = QueueManager(max_size=1000)
    _eval_queue = QueueManager(max_size=1000)
    _feedback_queue = QueueManager(max_size=1000)
    
    _active_queues.extend([_attack_queue, _eval_queue, _feedback_queue])

    # 2. Boot the Red Team Judge (The Scorer & Router)
    _system_logger.trace("Spawning RedTeamJudge worker process...")
    judge = RedTeamJudge(
        eval_queue=_eval_queue,
        feedback_queue=_feedback_queue,
        log_queue=_log_manager.log_queue,
        strategy=strategy 
    )
    judge.start()
    _inference_workers.append(judge)

    # 3. Boot the Async Striker (The Outbound Cannon)
    _system_logger.trace("Spawning AsyncStriker worker process...")
    striker = AsyncStriker(
        attack_queue=_attack_queue,
        eval_queue=_eval_queue,
        log_queue=_log_manager.log_queue,
        target_url="http://192.9.159.125:11434/api/chat" 
    )
    striker.start()
    _inference_workers.append(striker) 

    # 4. Boot the Live Telemetry Dashboard
    _system_logger.trace("Initializing DashboardManager...")
    dashboard = DashboardManager(
        attack_queue=_attack_queue,
        eval_queue=_eval_queue,
        feedback_queue=_feedback_queue
    )
    _log_manager.attach_dashboard(dashboard)

    # 5. Boot the Payload Generator (The Brain)
    _system_logger.info("Starting Stateful Asynchronous Prompt Mutator...")
    mutator = PromptMutator(
        attack_queue=_attack_queue, 
        feedback_queue=_feedback_queue,
        strategy=strategy
    )
    
    try:
        # Concurrently run the UI and the Generator
        await asyncio.gather(
            dashboard.render_loop(),
            mutator.listen()
        )
    except asyncio.CancelledError:
        _system_logger.trace("Main event loop caught CancelledError. Pipeline shut down cleanly.")


def main():
    """
    Entry point for IsoMutator.
    Handles low-level OS configuration before passing control to the async boot sequence.
    """
    # 1. The Strategy Factory Dictionary
    # This cleanly maps CLI strings to their respective strategy classes
    strategy_factory = {
        "jailbreak": JailbreakStrategy,
        "inversion": ModelInversionStrategy,
        "prompt_leaking": PromptLeakingStrategy,
        "cross_lingual": CrossLingualStrategy,
        "obfuscation": TokenObfuscationStrategy,
        "exhaustion": ResourceExhaustionStrategy,
        "owasp_xss": OwaspXssStrategy,
        "linux_privesc": LinuxPrivescStrategy,
        "persona": PersonaJailbreakStrategy,
        "gradient": GradientStrategy
    }

    # Setup the argument parser dynamically using the factory keys
    parser = argparse.ArgumentParser(description="IsoMutator AI Red Teaming Framework")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=list(strategy_factory.keys()), 
        default="jailbreak",
        help="Select the attack strategy to execute."
    )
    args = parser.parse_args()

    # 2. Instantiate the requested strategy using the factory
    active_strategy = strategy_factory[args.mode]()

    print(f"--- Starting IsoMutator in {active_strategy.name.upper()} mode ---")
    
    # Force "spawn" to prevent async/fork deadlocks on Linux
    multiprocessing.set_start_method("spawn", force=True)
    
    # Boot the logging bridge outside the async loop to guarantee early availability
    global _log_manager, _system_logger
    try:
        _log_manager = LogManager()
        _log_manager.start()
        _system_logger = LogManager.get_logger("isomutator.system")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to boot LogManager. Details: {e}")
        sys.exit(1)

    # Start the async environment and bind the OS signal handler
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.add_signal_handler(signal.SIGINT, lambda: handle_shutdown(signal.SIGINT, None))
    
    try:
        # Pass the strategy into the boot sequence
        loop.run_until_complete(boot_sequence(active_strategy))
    finally:
        loop.close()


if __name__ == "__main__":
    main()