"""
IsoCore Queue Manager (src/isocore/core/queue_manager.py)
---------------------------------------------------------
Bridges the asyncio event loop (Ingestors) with the 
multiprocessing workers (Inference) safely and efficiently.
"""

import asyncio
import queue
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# We import the LogManager to grab our pre-configured system logger
from isocore.core.log_manager import LogManager


class QueueManager:
    def __init__(self, max_size: int = 10000):
        """
        Initializes the communication bridge.
        max_size prevents RAM exhaustion if the scrapers outpace the Neural Network.
        """
        self.logger = LogManager.get_logger("isocore.system")
        
        # The underlying OS-level pipe
        self._queue = multiprocessing.Queue(maxsize=max_size)
        
        # A dedicated thread pool just for putting items into the queue.
        # This prevents the async loop from freezing during OS-level memory locks.
        self._put_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="QueuePusher")
        
        self.logger.trace(f"QueueManager initialized with max_size={max_size}")

    # ==========================================
    # Serialization Rules (For multiprocessing "spawn")
    # ==========================================
    def __getstate__(self):
        """
        Dictates what gets Pickled when crossing the OS process boundary.
        We must strip out the ThreadPoolExecutor because threads cannot be serialized.
        """
        state = self.__dict__.copy()
        
        # The InferenceWorker only consumes data, so it doesn't need the put_executor.
        # We safely delete it from the payload before it crosses the bridge.
        if '_put_executor' in state:
            del state['_put_executor']
            
        return state

    def __setstate__(self, state):
        """
        Rebuilds the object on the other side of the process boundary.
        """
        self.__dict__.update(state)
        
        # Set it to None on the worker side so we don't hit an AttributeError
        # (even though the InferenceWorker will never call async_put anyway).
        self._put_executor = None

    # ==========================================
    # Producer Side (Asyncio Web Scrapers)
    # ==========================================
    async def async_put(self, item, timeout: float = 0.5) -> bool:
        """
        Safely puts an item into the multiprocessing queue without blocking the event loop.
        Returns True if successful, False if the queue is full.
        """
        loop = asyncio.get_running_loop()
        
        # We wrap the blocking queue.put call in a standard function
        def _blocking_put():
            try:
                self._queue.put(item, block=True, timeout=timeout)
                return True
            except queue.Full:
                return False

        # We tell the async loop to run that blocking function in a background thread
        success = await loop.run_in_executor(self._put_executor, _blocking_put)
        
        if not success:
            self.logger.warning("Queue is FULL! Scrapers are outpacing the Inference Worker. Dropping packet.")
        
        return success

    # ==========================================
    # Consumer Side (Multiprocessing Brain)
    # ==========================================
    def get_batch(self, target_size: int = 32, max_wait: float = 1.0) -> list:
        """
        Pulls a batch of items for the Neural Network. 
        Blocks until at least ONE item is available, then sweeps the rest instantly.
        """
        batch = []
        
        try:
            # 1. The Blocking Wait: Wait for the first item (sleeps the CPU if empty)
            first_item = self._queue.get(block=True, timeout=max_wait)
            batch.append(first_item)
            
            # 2. The Sweep: Grab up to (target_size - 1) more items instantly
            while len(batch) < target_size:
                try:
                    # Give the OS pipe a tiny 10ms grace period to flush the next item
                    next_item = self._queue.get(block=True, timeout=0.01)
                    batch.append(next_item)
                except queue.Empty:
                    # Queue is empty, stop sweeping and return what we have
                    break
                    
        except queue.Empty:
            # The max_wait timeout was hit before even a single item arrived.
            # This is totally fine; it just means the internet is quiet right now.
            pass
            
        return batch

    # ==========================================
    # Lifecycle & Telemetry
    # ==========================================
    def send_poison_pill(self):
        """Drops the shutdown signal into the queue."""
        self.logger.info("Sending Poison Pill to Inference Worker...")
        try:
            self._queue.put("POISON_PILL", block=True, timeout=2.0)
        except queue.Full:
            self.logger.error("Could not send Poison Pill! Queue is deadlocked.")

    def get_approximate_size(self) -> int:
        """
        Returns the rough size of the queue. 
        (Note: qsize() is notoriously unreliable on macOS, but works well on Fedora/Linux).
        """
        try:
            return self._queue.qsize()
        except NotImplementedError:
            return -1
            
    def close(self):
        """Cleans up the background threads."""
        self._put_executor.shutdown(wait=False)
        self._queue.close()
        self.logger.trace("QueueManager pipes closed.")