"""
ALGORITHM SUMMARY:
The Async Striker is the outbound network engine for the red teaming pipeline.
It is specifically optimized for CPU-bound environments (Sequential Processing).
1. It continuously polls the Attack Queue for mutated payloads (Strictly 1 at a time).
2. Evaluates payload strategy (Single-Stage Conversational vs. Dual-Stage Context Injection).
3. If Dual-Stage, uploads the poisoned staged file to the target via multipart/form-data.
4. Fires the prompt at the designated target AI server over HTTP.
5. Appends the target's response to the packet's conversational history array.
6. Passes the updated packet into the Eval Queue for the AI Judge to score.

TECHNOLOGY QUIRKS:
- Connection Pooling (aiohttp): Instantiates a single `aiohttp.ClientSession()` that stays 
  open for the life of the worker, drastically reducing network overhead.
- Sequential CPU Optimization: Bypasses `asyncio.gather` concurrency by restricting 
  the batch size to 1, preventing CPU context-switching thrashing for SLMs.
- aiofiles: Offloads physical disk reads of staged payloads to a background thread.
"""

import os
import asyncio
import aiohttp
import aiofiles
import multiprocessing
import signal

from isomutator.core.queue_manager import QueueManager
from isomutator.core.log_manager import LogManager
from isomutator.core.config import settings
from isomutator.models.packet import DataPacket

class AsyncStriker(multiprocessing.Process):
    """
    Isolated OS Process that runs an asynchronous event loop to fire 
    sequential network attacks against a target API.
    """
    def __init__(self, attack_queue: QueueManager, eval_queue: QueueManager, log_queue: multiprocessing.Queue, target_url: str):
        super().__init__(name="Worker-Striker")
        self.attack_queue = attack_queue
        self.eval_queue = eval_queue
        self.log_queue = log_queue
        self.target_url = target_url.rstrip("/")
        self.staging_dir = os.path.join("tmp", "isomutator_staging")
        self.logger = None

    def run(self):
        """The entry point for the isolated OS process."""
        # Shield this child process from Ctrl+C to enforce graceful teardown
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        LogManager.setup_worker(self.log_queue)
        self.logger = LogManager.get_logger("isomutator.striker")
        
        try:
            asyncio.run(self._strike_loop())
        except asyncio.CancelledError:
            self.logger.trace("Async loop cancelled safely during shutdown.")

    async def _strike_loop(self):
        self.logger.info(f"Striker online. Cannons aimed at: {self.target_url}")
        
        async with aiohttp.ClientSession() as session:
            while True:
                # 1. Pull batch from the queue to ensure sequential CPU processing
                batch = await self.attack_queue.get_batch(
                    target_size=settings.batch_size, 
                    max_wait=1.0
                )
                if not batch:
                    continue
                    
                # 2. Emergency stop check
                if any(p == "POISON_PILL" for p in batch):
                    self.logger.info("Poison Pill swallowed. Shutting down cannons cleanly.")
                    break

                self.logger.trace("Firing payload sequentially...")

                # 3. Fire the payload and wait for the response
                for packet in batch:
                    updated_packet = await self._fire_payload(session, packet)

                    # 4. Forward the surviving packet to the AI Judge
                    if updated_packet:
                        await self.eval_queue.async_put(updated_packet)
                        self.logger.trace(f"Strike {packet.id[:8]} completed and forwarded to Judge.")

    async def _fire_payload(self, session: aiohttp.ClientSession, packet: DataPacket) -> DataPacket | None:
        """
        Executes a single HTTP strike against the target API.
        Handles both standard requests and Context Injection uploads.
        """
        try:
            self.logger.trace(f"Sending payload {packet.id[:8]} to {self.target_url}...")

            # ==========================================
            # STAGE 1: Context Injection (If Required)
            # ==========================================
            # Safely check if the packet requires file staging (defaults to False)
            requires_staging = getattr(packet, 'requires_staging', False)
            
            if requires_staging:
                staged_filename = getattr(packet, 'staged_filename', None)
                self.logger.trace(f"Payload {packet.id[:8]} requires staging. Initiating Upload for {staged_filename}.")
                
                if not staged_filename:
                    self.logger.error("Packet requires staging but no filename was provided.")
                    return None
                    
                filepath = os.path.join(self.staging_dir, staged_filename)
                
                if not os.path.exists(filepath):
                    self.logger.error(f"Error: Staged payload file not found at {filepath}")
                    return None

                async with aiofiles.open(filepath, 'rb') as f:
                    file_data = await f.read()

                form_data = aiohttp.FormData()
                form_data.add_field(
                    'file', 
                    file_data, 
                    filename=staged_filename, 
                    content_type='text/plain'
                )
                
                upload_url = f"{self.target_url}/upload"
                async with session.post(upload_url, data=form_data, timeout=settings.network_timeout) as upload_resp:
                    if upload_resp.status != 200:
                        self.logger.error(f"Stage 1 Upload failed. Target returned HTTP {upload_resp.status}")
                        return None
                        
                self.logger.trace(f"Stage 1 complete. File {staged_filename} ingested by target.")

            # ==========================================
            # STAGE 2: Conversational Trigger
            # ==========================================
            chat_url = f"{self.target_url}/api/chat"
            
            # Use the new API contract: {"query": "prompt text"}
            payload = {
                "query": packet.raw_content
            }
            
            async with session.post(chat_url, json=payload, timeout=settings.network_timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"Target server rejected strike {packet.id[:8]}: {response.status} - {error_text}")
                    return None

                result_json = await response.json()
                
                # Extract response based on the new CorpRAG-Target API contract
                target_response = result_json.get("answer", "")

                # Record the full exchange to the packet's history
                packet.history.append({"role": "user", "content": packet.raw_content})
                packet.history.append({"role": "assistant", "content": target_response})
                
                self.logger.trace(f"Strike {packet.id[:8]} returned {len(target_response)} characters.")
                
                # Return the exact same packet to preserve ID, Turn Count, and History
                return packet
                
        except asyncio.TimeoutError:
            self.logger.warning(f"Strike {packet.id[:8]} timed out. Remote server took too long.")
            return None
        except Exception as e:
            self.logger.error(f"Strike failed for packet {packet.id[:8]}: {e}")
            return None