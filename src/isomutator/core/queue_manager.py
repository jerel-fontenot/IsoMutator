"""
ALGORITHM SUMMARY:
The QueueManager is the Redis-backed nervous system of the IsoMutator framework.
It bridges independent microservices (Mutators, Strikers, Judges, and the GUI) 
via a decoupled Event-Driven Architecture.

1. Work Queues: Uses Redis Lists (`LPUSH` and `BRPOP`) to distribute DataPackets safely.
2. UI Telemetry: Uses Redis Pub/Sub (`PUBLISH`) to broadcast non-blocking state updates 
   (like Wiretap and Ledger events) to any connected GUI consumers.
"""

import json
from redis.asyncio import Redis

from isomutator.core.log_manager import LogManager
from isomutator.models.packet import DataPacket

class QueueManager:
    def __init__(self, redis_url: str = "redis://localhost:6379/0", queue_name: str = "default"):
        self.logger = LogManager.get_logger("isomutator.system")
        self.queue_name = queue_name
        self.queue_key = f"isomutator:queue:{queue_name}"
        
        # Asynchronous Redis connection pool
        self._redis = Redis.from_url(redis_url, decode_responses=True)
        self.logger.trace(f"QueueManager connected to Redis at {redis_url} (Queue: {self.queue_key})")

    async def async_put(self, item: DataPacket, timeout: float = 0.5) -> bool:
        """Pushes a serialized DataPacket onto the left side of the Redis List."""
        try:
            await self._redis.lpush(self.queue_key, item.to_json())
            return True
        except Exception as e:
            self.logger.error(f"Redis LPUSH failed on {self.queue_key}: {e}")
            return False

    async def get_batch(self, target_size: int = 32, max_wait: float = 1.0) -> list:
        """
        Pulls a batch of packets. Blocks until at least ONE item is available, 
        then sweeps the rest instantly up to the target_size.
        """
        batch = []
        try:
            # 1. Blocking Pop (Waits up to max_wait for the first item)
            # BRPOP returns a tuple: (queue_name, popped_value)
            result = await self._redis.brpop(self.queue_key, timeout=max_wait)
            
            if not result:
                return batch # Queue is empty, timeout reached
                
            _, first_item_json = result
            batch.append(DataPacket.from_json(first_item_json))
            
            # 2. Sweep (Grab remaining items instantly)
            while len(batch) < target_size:
                next_item_json = await self._redis.lpop(self.queue_key)
                if not next_item_json:
                    break
                batch.append(DataPacket.from_json(next_item_json))
                
        except Exception as e:
            self.logger.error(f"Redis BRPOP/LPOP failed on {self.queue_key}: {e}")
            
        return batch

    async def broadcast_telemetry(self, event_type: str, data: dict):
        """Publishes transient state updates for the decoupled GUI to consume."""
        channel = f"isomutator:telemetry:{event_type}"
        try:
            await self._redis.publish(channel, json.dumps(data))
        except Exception as e:
            self.logger.error(f"Redis PUBLISH failed on {channel}: {e}")

    async def get_approximate_size(self) -> int:
        """Returns the length of the Redis List."""
        try:
            return await self._redis.llen(self.queue_key)
        except Exception:
            return -1

    async def send_poison_pill(self):
        """Broadcasts a global shutdown event."""
        self.logger.info("Broadcasting Poison Pill to all microservices...")
        await self.broadcast_telemetry("system", {"command": "POISON_PILL"})

    async def close(self):
        """Closes the Redis connection pool cleanly."""
        await self._redis.aclose()
        self.logger.trace(f"Redis connection pool closed for {self.queue_name}.")

    def __getstate__(self):
        """
        Prepares the object to cross the multiprocessing boundary.
        We must drop the live Redis connection because sockets and locks cannot be pickled.
        """
        state = self.__dict__.copy()
        if '_redis' in state:
            del state['_redis']
        return state

    def __setstate__(self, state):
        """
        Reconstructs the object after it arrives in the new OS child process.
        We establish a brand new, isolated Redis connection for this specific worker.
        """
        self.__dict__.update(state)
        
        # Re-import and re-establish the connection. 
        # (Make sure these connection parameters match what you have in your __init__ method!)
        import redis.asyncio as redis
        self._redis = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)