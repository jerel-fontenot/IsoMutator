"""
ALGORITHM SUMMARY:
The QueueManager bridges the asyncio event loops (Mutator) with the 
multiprocessing workers (Striker, Judge) safely and efficiently.
It provides both asynchronous and synchronous methods for pushing and pulling 
data across OS-level process boundaries without deadlocking the CPU.
"""

import asyncio
import queue
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from isomutator.core.log_manager import LogManager


class QueueManager:
    def __init__(self, max_size: int = 10000):
        """
        Initializes the communication bridge.
        max_size prevents RAM exhaustion if generators outpace the workers.
        """
        self.logger = LogManager.get_logger("isomutator.system")
        
        # The underlying OS-level pipe
        self._queue = multiprocessing.Queue(maxsize=max_size)
        
        # A dedicated thread pool just for putting items into the queue asynchronously.
        # This prevents the async loop from freezing during OS-level memory locks.
        self._put_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="QueuePusher")
        
        self.logger.trace(f"QueueManager initialized with max_size={max_size}")

    def __getstate__(self):
        """
        Dictates what gets Pickled when crossing the OS process boundary.
        We must strip out the ThreadPoolExecutor because threads cannot be serialized.
        """
        state = self.__dict__.copy()
        
        if '_put_executor' in state:
            del state['_put_executor']
            
        return state

    def __setstate__(self, state):
        """
        Rebuilds the object on the other side of the process boundary.
        """
        self.__dict__.update(state)
        self._put_executor = None

    # ==========================================
    # Asynchronous Methods (For the Mutator)
    # ==========================================
    async def async_put(self, item, timeout: float = 0.5) -> bool:
        """
        Safely puts an item into the multiprocessing queue without blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        
        def _blocking_put():
            try:
                self._queue.put(item, block=True, timeout=timeout)
                return True
            except queue.Full:
                return False

        success = await loop.run_in_executor(self._put_executor, _blocking_put)
        
        if not success:
            self.logger.warning("Queue is FULL! Dropping packet to prevent deadlock.")
        
        return success

    # ==========================================
    # Synchronous Methods (For the Judge & Striker)
    # ==========================================
    def put(self, item, timeout: float = 2.0) -> bool:
        """
        Synchronously puts an item into the queue.
        Used by multiprocessing workers (like the Judge) to route feedback data.
        """
        try:
            self._queue.put(item, block=True, timeout=timeout)
            return True
        except queue.Full:
            self.logger.warning("Queue is FULL! Failed to route packet.")
            return False

    def get_batch(self, target_size: int = 32, max_wait: float = 1.0) -> list:
        """
        Pulls a batch of items. Blocks until at least ONE item is available, 
        then sweeps the rest instantly up to the target_size.
        """
        batch = []
        
        try:
            # 1. Wait for the first item (sleeps the CPU if empty)
            first_item = self._queue.get(block=True, timeout=max_wait)
            batch.append(first_item)
            
            # 2. Sweep: Grab up to (target_size - 1) more items instantly
            while len(batch) < target_size:
                try:
                    next_item = self._queue.get(block=True, timeout=0.01)
                    batch.append(next_item)
                except queue.Empty:
                    break
                    
        except queue.Empty:
            pass
            
        return batch

    # ==========================================
    # Lifecycle & Telemetry
    # ==========================================
    def send_poison_pill(self):
        """Drops the shutdown signal into the queue."""
        self.logger.info("Sending Poison Pill through queue...")
        try:
            self._queue.put("POISON_PILL", block=True, timeout=2.0)
        except queue.Full:
            self.logger.error("Could not send Poison Pill! Queue is deadlocked.")

    def get_approximate_size(self) -> int:
        """Returns the rough size of the queue."""
        try:
            return self._queue.qsize()
        except NotImplementedError:
            return -1
            
    def close(self):
        """Cleans up the background threads."""
        if self._put_executor:
            self._put_executor.shutdown(wait=False)
        self._queue.close()
        self.logger.trace("QueueManager pipes closed.")