"""
IsoCore Reddit Ingestor (src/isocore/ingestors/reddit.py)
---------------------------------------------------------
Simulates pulling live comments from Reddit subreddits.
"""

import asyncio
import random
from isocore.ingestors.base import BaseSource
from isocore.models.packet import DataPacket

class SimulatedRedditSource(BaseSource):
    def __init__(self, queue_manager, subreddits: list[str]):
        # Call the parent __init__ to set up the logger and queue
        super().__init__(queue_manager, name="Reddit")
        self.subreddits = subreddits
        
        # Some fake data to simulate the AI Security / Tech space
        self._mock_comments = [
            "New zero-day just dropped on GitHub.",
            "I think this LLM is susceptible to prompt injection.",
            "Can someone explain Hilbert Spaces to me like I'm 5?",
            "Just deployed my neural network to Fedora!",
            "Is anyone else seeing high latency on the API?"
        ]

    async def listen(self):
        self.logger.info(f"Starting Reddit listener for: {self.subreddits}")
        
        try:
            while True:
                # 1. Simulate Network Latency (The non-blocking wait)
                # This 'await' releases the event loop so other ingestors can run!
                network_delay = random.uniform(0.1, 1.5)
                await asyncio.sleep(network_delay)
                
                # 2. "Download" the data
                raw_text = random.choice(self._mock_comments)
                subreddit = random.choice(self.subreddits)
                
                # 3. Package it into our standardized DTO
                packet = DataPacket(
                    raw_content=raw_text,
                    source=f"reddit/r/{subreddit}",
                    metadata={"author": "anon_user", "score": random.randint(1, 100)}
                )
                
                # 4. Push it across the bridge
                success = await self._safe_put(packet)
                
                if not success:
                    # If the queue is full, we back off to give the Brain time to catch up
                    self.logger.debug("Queue full. Reddit scraper backing off for 2 seconds.")
                    await asyncio.sleep(2.0)
                    
        except asyncio.CancelledError:
            # The Orchestrator sent the shutdown signal!
            self.logger.info("Reddit listener task cancelled. Shutting down cleanly.")
            raise  # Re-raise so the event loop knows this task closed properly