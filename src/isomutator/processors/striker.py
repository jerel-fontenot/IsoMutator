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
  the batch size to 1 using `get_batch(target_size=1)`, preventing CPU context-switching 
  thrashing for SLMs.
- Out-of-Band (OOB) Kill Switch: Checks a multiprocessing.Event on every loop iteration 
  to instantly abandon network requests and tear down safely during UI interrupts.
- aiofiles: Offloads physical disk reads of staged payloads to a background thread.
"""

import os
import asyncio
import aiohttp
import aiofiles
import multiprocessing
import multiprocessing.synchronize

from isomutator.core.queue_manager import QueueManager
from isomutator.core.log_manager import LogManager
from isomutator.core.config import settings
from isomutator.models.packet import DataPacket


class AsyncStriker(multiprocessing.Process):
    """
    Isolated OS Process that runs an asynchronous event loop to handle outbound HTTP strikes.
    """
    def __init__(self, *, attack_queue: QueueManager, eval_queue: QueueManager, log_queue, target_url: str, shutdown_event: multiprocessing.synchronize.Event | None = None):
        super().__init__(name="AsyncStriker")
        # Store parent references (these will be safely overwritten in the local memory space during run)
        self.attack_queue = attack_queue
        self.eval_queue = eval_queue
        self.log_queue = log_queue
        self.target_url = target_url
        
        # Store the Out-of-Band Kill Switch
        self.shutdown_event = shutdown_event

    def run(self):
        """Process entry point. Must force a new event loop for this memory space."""
        LogManager.setup_worker(self.log_queue)
        self.logger = LogManager.get_logger("isomutator.striker")

        attack_name = getattr(self.attack_queue, 'queue_name', 'attack')
        eval_name = getattr(self.eval_queue, 'queue_name', 'eval')

        try:
            self.attack_queue = QueueManager(queue_name=attack_name)
            self.eval_queue = QueueManager(queue_name=eval_name)
            asyncio.run(self._run_with_cleanup())
        except KeyboardInterrupt:
            self.logger.info("Striker caught KeyboardInterrupt. Shutting down.")
        except Exception as e:
            self.logger.error(f"[CRITICAL] Striker {self.name} Crashed: {e}")
        finally:
            print(f"[SYSTEM] Striker {self.name} releasing resources...")
            self.log_queue.cancel_join_thread()
            self.log_queue.close()

    async def _run_with_cleanup(self):
        loop_task = asyncio.create_task(self._strike_loop())

        async def _shutdown_watcher():
            # Polls the multiprocessing.Event every 200ms and cancels the
            # loop task the moment it is set, interrupting any in-flight
            # HTTP request immediately rather than waiting for its timeout.
            while not (self.shutdown_event and self.shutdown_event.is_set()):
                await asyncio.sleep(0.2)
            loop_task.cancel()

        watcher = asyncio.create_task(_shutdown_watcher())
        try:
            await loop_task
        except asyncio.CancelledError:
            pass
        finally:
            watcher.cancel()
            await self.attack_queue.close()
            await self.eval_queue.close()

    async def _strike_loop(self):
        self.logger.trace("Entering _strike_loop algorithm.")
        
        # Open the single ClientSession for the life of the worker
        async with aiohttp.ClientSession() as session:
            try:
                while True:
                    # 1. Out-of-Band Kill Switch Check
                    if self.shutdown_event and self.shutdown_event.is_set():
                        self.logger.info("OOB Kill Switch activated. Striker abandoning queue.")
                        break

                    # 2. Pull the next packet sequentially using the correct QueueManager method
                    # Utilizing get_batch with target_size=1 prevents CPU thrashing and respects the queue contract.
                    batch = await self.attack_queue.get_batch(target_size=1, max_wait=1.0)
                    
                    if not batch:
                        continue # Queue is empty, loop around to check the kill switch again
                        
                    packet = batch[0]

                    if not packet or packet == "POISON_PILL":
                        self.logger.info("Poison Pill received. Striker shutting down.")
                        break

                    self.logger.trace("Firing payload sequentially...")
                    result_packet = await self._fire_payload(session=session, packet=packet)

                    if result_packet:
                        # 4. Offload to Evaluation Queue
                        await self.eval_queue.async_put(item=result_packet)
                        self.logger.trace(f"Strike {packet.id[:8]} completed and forwarded to Judge.")
                    
            except asyncio.CancelledError:
                self.logger.info("Striker loop cancelled by orchestrator.")
                raise

    async def _fire_payload(self, *, session: aiohttp.ClientSession, packet: DataPacket) -> DataPacket | None:
        """
        Executes the network request. Evaluates if the packet requires Context Staging 
        (file upload) or standard conversational interaction.
        """
        try:
            chat_url = f"{self.target_url.rstrip('/')}/api/chat"
            
            # --- Dual-Stage Evaluation (Context Injection) ---
            # Only require the filename, as the file is read directly from disk
            if getattr(packet, 'staged_filename', None):
                upload_url = f"{self.target_url.rstrip('/')}/api/upload"
                self.logger.trace(f"Dual-Stage execution required for packet {packet.id[:8]}. Uploading file...")
                
                filepath = str(settings.staging_dir / packet.staged_filename)
                
                # Respect the file system verification contract
                if not os.path.exists(filepath):
                    self.logger.error(f"Staged file {filepath} went missing before upload.")
                    return None
                
                try:
                    async with aiofiles.open(filepath, mode='rb') as f:
                        file_data = await f.read()
                except FileNotFoundError:
                    self.logger.error(f"Staged file {filepath} went missing before upload.")
                    return None
                    
                form_data = aiohttp.FormData()
                form_data.add_field('file', file_data, filename=packet.staged_filename, content_type='text/plain')
                
                async with session.post(upload_url, data=form_data, timeout=aiohttp.ClientTimeout(total=settings.network_timeout)) as upload_res:
                    if upload_res.status != 200:
                        self.logger.error(f"File upload rejected by target for {packet.id[:8]}: HTTP {upload_res.status}")
                        return None
                    self.logger.trace("File upload successful. Proceeding to trigger phase.")

            # --- Primary Inference Strike ---
            self.logger.trace(f"Sending payload {packet.id[:8]} to {chat_url}...")
            
            # Formatting to standard CorpRAG Target schema: {"query": "prompt text"}
            payload = {
                "query": packet.raw_content
            }
            
            async with session.post(chat_url, json=payload, timeout=aiohttp.ClientTimeout(total=settings.network_timeout)) as response:
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