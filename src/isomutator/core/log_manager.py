"""
ALGORITHM SUMMARY:
The LogManager establishes a centralized, asynchronous, and process-safe logging architecture.
1. It defines a custom TRACE level (Level 5) for granular algorithm tracking.
2. Initialization: It reads the physical handler definitions (Console, Rotating File) 
   from a user-defined `logging.json` file via `dictConfig`. If the config is missing 
   or malformed, it gracefully degrades to a default console stream.
3. Handler Hijacking: It extracts these physical handlers from the root logger, 
   programmatically appends the custom `UIDispatchHandler`, and assigns them all 
   to a background `QueueListener`.
4. UI Routing (Observer Pattern): The `UIDispatchHandler` intercepts LogRecords tagged 
   with specific `ui_event` attributes and routes them to the `DashboardManager`.

TECHNOLOGY QUIRKS:
- Singleton Pattern: Enforced via `__new__` to guarantee that only one underlying 
  `multiprocessing.Queue` and `QueueListener` are ever instantiated, preventing OS-level pipe leaks.
- Programmatic Injection: The `UIDispatchHandler` is injected into the handler list 
  after the JSON configuration is applied, avoiding complex JSON parser rules.
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
if not hasattr(logging, "TRACE"):
    logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
    logging.TRACE = TRACE_LEVEL_NUM

def trace(self, message, *args, **kws):
    """Allows logger.trace('message') calls across the codebase."""
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)

logging.Logger.trace = trace

# ==========================================
# 2. UI Telemetry Router
# ==========================================
class UIDispatchHandler(logging.Handler):
    """
    Custom logging handler that intercepts structured UI events and 
    routes them to the DashboardManager.
    """
    def __init__(self):
        super().__init__()
        self.dashboard = None

    def attach_dashboard(self, dashboard):
        self.dashboard = dashboard

    def emit(self, record):
        if not self.dashboard:
            return

        # Check if this LogRecord contains UI routing data
        ui_event = getattr(record, "ui_event", None)
        if ui_event == "wiretap":
            self.dashboard.add_wiretap_event(
                turn=getattr(record, "turn", 0),
                attacker_text=getattr(record, "attacker", ""),
                target_text=getattr(record, "target", "")
            )
        elif ui_event == "ledger":
            self.dashboard.add_vulnerability(
                turn=getattr(record, "turn", 0),
                strategy=getattr(record, "strategy", "unknown"),
                packet_id=getattr(record, "packet_id", "unknown")
            )

# ==========================================
# 3. The LogManager Singleton
# ==========================================
class LogManager:
    """
    Centralized manager for multi-process logging and UI event routing.
    """
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
        self.ui_handler = UIDispatchHandler()
        
        self._setup_from_config(config_path)
        self._initialized = True

    def _setup_from_config(self, config_path: str):
        """Loads JSON config, injects the UI handler, and wires the QueueListener.
        Includes a graceful fallback if the JSON file is missing or corrupted.
        """
        root_logger = logging.getLogger()
        physical_handlers = []

        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)

            # 1. Apply the configuration
            logging.config.dictConfig(config_dict)
            
            # 2. Extract the physical handlers dictConfig just created
            physical_handlers = root_logger.handlers.copy()
            
        except Exception as e:
            # --- FALLBACK MECHANISM ---
            # If the JSON is missing or malformed, default to a safe StreamHandler
            fallback_handler = logging.StreamHandler()
            fallback_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("[%(levelname)-5s] FALLBACK: %(message)s")
            fallback_handler.setFormatter(formatter)
            physical_handlers = [fallback_handler]
            print(f"[LogManager] Warning: Failed to load {config_path} ({type(e).__name__}). Using fallback StreamHandler.")

        # Append the custom UI handler directly to the physical handler array
        physical_handlers.append(self.ui_handler)

        # 3. Disconnect them from the root logger
        root_logger.handlers.clear()

        # 4. Give the physical handlers to the background listener
        self.listener = logging.handlers.QueueListener(
            self.log_queue, 
            *physical_handlers, 
            respect_handler_level=True
        )

        # 5. Connect the root logger back to the Queue
        main_queue_handler = logging.handlers.QueueHandler(self.log_queue)
        root_logger.addHandler(main_queue_handler)

    def attach_dashboard(self, dashboard):
        """Links the UI Dispatcher to the live DashboardManager."""
        self.ui_handler.attach_dashboard(dashboard)
        logging.getLogger("isomutator.system").trace("DashboardManager successfully attached to UI Dispatcher.")

    def start(self):
        """Starts the background thread that writes logs to disk."""
        if self.listener:
            self.listener.start()
            logging.getLogger("isomutator.system").trace("LogManager QueueListener started.")

    def stop(self):
        """Flushes the queue and stops the background thread safely."""
        if self.listener:
            logging.getLogger("isomutator.system").trace("LogManager QueueListener stopping...")
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
        
        logging.getLogger("isomutator.worker").trace("Worker logger attached to shared queue.")

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)