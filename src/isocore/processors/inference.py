"""
IsoCore Inference Worker (src/isocore/processors/inference.py)
--------------------------------------------------------------
The heavy-lifting process. Runs completely isolated from the async 
event loop, consuming batches of data and simulating neural network math.
"""

import multiprocessing
import time
import random
import signal
from typing import List

from isocore.core.queue_manager import QueueManager
from isocore.core.log_manager import LogManager
from isocore.models.packet import DataPacket, ResultPacket

class InferenceWorker(multiprocessing.Process):
    def __init__(self, queue_manager: QueueManager, log_queue: multiprocessing.Queue):
        # Name the process so it looks clean in our rolling log file
        super().__init__(name="Worker-GPU-0")
        self.queue_manager = queue_manager
        self.log_queue = log_queue
        
        # We declare these here, but we DO NOT initialize them yet.
        # They must only be initialized inside the run() method!
        self.logger = None
        self._model = None

    def _load_model(self):
        """Simulates loading a massive PyTorch/TensorFlow model into memory."""
        self.logger.info("Allocating VRAM and loading Neural Network weights...")
        time.sleep(2.0) # Simulate a 2-second load time
        self._model = "IsoCore-DistilBERT-v1"
        self.logger.info(f"Model '{self._model}' loaded successfully. Ready for inference.")

    def run(self):
        """
        This is the entry point for the new OS Process. 
        Everything in here happens in an isolated memory space.
        """
        # 1. The Handshake: Connect this process's root logger back to the Main Process
        LogManager.setup_worker(self.log_queue)
        self.logger = LogManager.get_logger("isocore.brain")

        # Tell this isolated process to ignore Ctrl+C from the terminal.
        # It will only shut down when it receives the POISON_PILL from the queue.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.logger.trace("Worker process booted. Environment isolated.")

        # 2. Load the Brain
        self._load_model()

        # 3. The Infinite Inference Loop
        try:
            while True:
                # Block until we have at least 1 item, then sweep up to 32
                batch = self.queue_manager.get_batch(target_size=32, max_wait=1.0)
                
                if not batch:
                    # The queue was empty for a full second. Just loop and wait again.
                    continue

                # Check for the Graceful Shutdown signal
                if "POISON_PILL" in batch:
                    self.logger.info("Poison Pill swallowed. Commencing safe worker shutdown.")
                    break

                # We have a valid batch of DataPackets! Process them.
                self._process_batch(batch)

        except Exception as e:
            self.logger.error(f"Fatal error in InferenceWorker: {e}", exc_info=True)
        finally:
            # Clean up VRAM/RAM before the process truly dies
            self.logger.trace("Clearing model from memory...")
            self._model = None
            self.logger.info("InferenceWorker has exited cleanly.")

    def _process_batch(self, batch: List[DataPacket]):
        """Simulates the matrix multiplication for a batch of text."""
        start_time = time.time()
        
        # In a real app, you would tokenize all text here and pass a single large tensor to the model.
        # We will simulate the math taking slightly longer for larger batches (approx 5ms per item).
        compute_time = len(batch) * 0.005
        time.sleep(compute_time)
        
        results = []
        for packet in batch:
            # Simulate generating a probability distribution
            sentiment = {
                "positive": round(random.uniform(0.1, 0.9), 2),
                "negative": round(random.uniform(0.1, 0.9), 2)
            }
            
            # Normalize so they sum to 1.0 roughly
            total = sentiment["positive"] + sentiment["negative"]
            sentiment["positive"] = round(sentiment["positive"] / total, 2)
            sentiment["negative"] = round(sentiment["negative"] / total, 2)

            result = ResultPacket(
                original_packet_id=packet.id,
                source=packet.source,
                sentiment_scores=sentiment,
                end_to_end_latency_ms=round((time.time() - packet.timestamp) * 1000, 2)
            )
            results.append(result)
            
            # Use our custom TRACE level to watch the algorithmic math happen
            self.logger.trace(f"Inference complete for {packet.id[:8]} -> {sentiment}")

        end_time = time.time()
        self.logger.debug(
            f"Processed batch of {len(batch)} items in {round((end_time - start_time) * 1000, 2)}ms. "
            f"Avg latency: {round(sum(r.end_to_end_latency_ms for r in results) / len(results), 2)}ms"
        )