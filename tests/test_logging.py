"""
Test Script: test_logging.py
Run with: uv run python test_logging.py
"""
import multiprocessing
import time
import sys
from pathlib import Path

# Ensure the src directory is in the path so we can import our IsoCore module
sys.path.insert(0, str(Path(__file__).parent / "src"))

from isocore.core.log_manager import LogManager


def fake_inference_worker(log_queue):
    """This function simulates the InferenceWorker running in a totally separate OS process."""
    # 1. The Handshake: Connect this isolated process to the Main Process's log queue
    LogManager.setup_worker(log_queue)
    
    # 2. Get the specific logger for the brain
    # (In logging.json, we set 'isocore.brain' to propagate TRACE messages)
    logger = LogManager.get_logger("isocore.brain")
    
    # 3. Prove it works
    logger.info("Worker process booted successfully.")
    logger.trace("Loading Neural Network weights... (simulated)")
    time.sleep(0.5)
    logger.trace("Batch 1 processed. Matrix shape: (32, 512). Inference time: 42ms.")
    logger.debug("Worker memory usage normal.")
    logger.info("Worker process shutting down.")


def main():
    print("--- Starting IsoCore Logging Test ---")
    
    # 1. Initialize the nervous system (loads configs/logging.json)
    log_manager = LogManager()
    log_manager.start()
    
    # 2. Get the system logger for the main process
    # (In logging.json, this will hit the root level, meaning TRACE goes to file, INFO to console)
    sys_logger = LogManager.get_logger("isocore.system")
    sys_logger.info("Orchestrator boot sequence initiated.")
    sys_logger.trace("Checking Fedora system resources...") # Should only appear in the file!

    # 3. Spawn the worker
    sys_logger.info("Spawning Inference Worker...")
    worker_process = multiprocessing.Process(
        target=fake_inference_worker, 
        args=(log_manager.log_queue,),
        name="Worker-GPU-1" # We name it so it shows up nicely in the log format
    )
    worker_process.start()
    
    # 4. Wait for the worker to finish
    worker_process.join()
    sys_logger.info("Worker finished. Commencing Orchestrator shutdown.")
    
    # 5. Flush and stop the background thread
    log_manager.stop()
    print("--- Test Complete ---")
    
    # 6. Read the file to prove the TRACE logs were captured
    print("\n--- Contents of logs/isocore.log ---")
    try:
        with open("logs/isocore.log", "r") as f:
            print(f.read())
    except FileNotFoundError:
        print("ERROR: Log file not found!")

if __name__ == "__main__":
    main()