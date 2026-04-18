"""
Algorithm Summary:
The TelemetryService acts as an asynchronous read-only layer over the message broker.
It systematically polls the Redis-backed QueueManager to retrieve the current depth 
of critical operational queues (attack and feedback) and assesses the overall health 
of the broker connection. If the broker is unreachable, it catches the network exception, 
logs the failure, and returns a safe fallback state to prevent UI event loop crashes.
"""

from redis.exceptions import ConnectionError

from isomutator.core.queue_manager import QueueManager
from isomutator.core.log_manager import LogManager

logger = LogManager.get_logger(__name__)

class TelemetryService:
    """
    Service responsible for safely polling pipeline metrics for the NiceGUI dashboard.
    Demonstrates composition by wrapping the QueueManager instance rather than inheriting.
    """

    def __init__(self, queue_manager: QueueManager):
        """
        Initializes the TelemetryService with a required QueueManager instance.
        
        Args:
            queue_manager (QueueManager): The initialized broker management interface.
        """
        self._queue_manager = queue_manager
        logger.debug("TelemetryService initialized via composition.")

    async def get_dashboard_metrics(self) -> dict:
        """
        Asynchronously fetches real-time queue depths and broker status.
        
        Returns:
            dict: A mapping of telemetry data points to populate the UI dashboard.
        """
        logger.trace("Entering get_dashboard_metrics algorithm.")
        
        # Define the baseline fallback state
        metrics = {
            "broker_status": "Unknown",
            "attack_queue_depth": -1,
            "feedback_queue_depth": -1
        }

        try:
            logger.trace("Pinging broker to verify connection health.")
            is_online = await self._queue_manager.ping_broker()
            
            if is_online:
                metrics["broker_status"] = "Online"
                logger.trace("Broker state confirmed Online. Proceeding with depth polls.")
                
                logger.trace("Polling attack queue depth.")
                metrics["attack_queue_depth"] = await self._queue_manager.get_queue_depth("attack")
                
                logger.trace("Polling feedback queue depth.")
                metrics["feedback_queue_depth"] = await self._queue_manager.get_queue_depth("feedback")
                
        except ConnectionError as e:
            # Error Handling: Gracefully log the failure and maintain the -1 fallback values
            logger.error(f"Failed to fetch telemetry metrics: Redis is unreachable. {e}")
            metrics["broker_status"] = "Offline"
            
        logger.trace(f"Exiting get_dashboard_metrics with payload: {metrics}")
        return metrics