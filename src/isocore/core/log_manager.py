"""
IsoCore Log Manager (src/isocore/core/log_manager.py)
-----------------------------------------------------
Handles the centralized, process-safe logging architecture.
Reads configuration from JSON and manages the QueueListener.
"""

import json
import logging
import logging.config
import logging.handlers
import multiprocessing
from pathlib import Path

# ==========================================
# 1. The TRACE Injection
# ==========================================
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)

logging.Logger.trace = trace


# ==========================================
# 2. The LogManager Singleton
# ==========================================
class LogManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str = "configs/logging.json"):
        if self._initialized:
            return

        # Ensure the logs directory exists before dictConfig tries to use it
        Path("logs").mkdir(parents=True, exist_ok=True)

        self.log_queue = multiprocessing.Queue()
        self.listener = None
        
        self._setup_from_config(config_path)
        self._initialized = True

    def _setup_from_config(self, config_path: str):
        """Loads JSON config and wires the QueueListener."""
        # Load the raw JSON dictionary
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # 1. Apply the configuration
        # This creates the physical files and binds handlers to the root logger
        logging.config.dictConfig(config_dict)

        root_logger = logging.getLogger()

        # 2. The Extraction Phase
        # We grab the physical handlers (Console and File) that dictConfig just made
        physical_handlers = root_logger.handlers.copy()

        # 3. Disconnect them from the root logger
        # We do NOT want the Main Process writing directly to the file
        root_logger.handlers.clear()

        # 4. Give the physical handlers to the background listener
        self.listener = logging.handlers.QueueListener(
            self.log_queue, 
            *physical_handlers, 
            respect_handler_level=True
        )

        # 5. Connect the root logger back to the Queue
        # Now, anything logged in the Main Process goes into the queue
        main_queue_handler = logging.handlers.QueueHandler(self.log_queue)
        root_logger.addHandler(main_queue_handler)

    def start(self):
        """Starts the background thread that writes logs to disk."""
        if self.listener:
            self.listener.start()
            logging.getLogger("isocore.system").trace("LogManager QueueListener started via JSON config.")

    def stop(self):
        """Flushes the queue and stops the background thread safely."""
        if self.listener:
            logging.getLogger("isocore.system").trace("LogManager QueueListener stopping...")
            self.listener.stop()

    @staticmethod
    def setup_worker(log_queue: multiprocessing.Queue, level: int = TRACE_LEVEL_NUM):
        """Wires an isolated process's root logger back to the shared queue."""
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        queue_handler = logging.handlers.QueueHandler(log_queue)
        root_logger.addHandler(queue_handler)
        
        logging.getLogger("isocore.worker").trace("Worker logger attached to shared queue.")

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)