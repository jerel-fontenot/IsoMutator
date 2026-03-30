"""
ALGORITHM SUMMARY:
The Base Ingestor is the abstract blueprint for all asynchronous data sources.
It standardizes how child ingestors (like the PromptMutator) connect to the 
QueueManager, establish their specific logging channels, and safely push 
DataPackets across the asynchronous boundary.
"""

import abc
import asyncio
from isomutator.core.queue_manager import QueueManager
from isomutator.core.log_manager import LogManager

class BaseSource(abc.ABC):
    def __init__(self, queue_manager: QueueManager, name: str):
        self.queue_manager = queue_manager
        self.name = name
        # Grab a strictly formatted logger specific to this source (e.g., 'isomutator.ingest.promptmutator')
        self.logger = LogManager.get_logger(f"isomutator.ingest.{name.lower()}")

    @abc.abstractmethod
    async def listen(self):
        """
        The main asynchronous loop for this ingestor. 
        Must be implemented by any child class to drive the generation cycle.
        """
        pass
        
    async def _safe_put(self, packet) -> bool:
        """
        A helper method to standardize how subclasses push data into the pipeline.
        Logs the trace automatically before attempting the asynchronous put.
        """
        self.logger.trace(packet.to_log_trace())
        return await self.queue_manager.async_put(packet)