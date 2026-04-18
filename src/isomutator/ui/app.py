"""
ALGORITHM SUMMARY:
The NiceGUI-based Interactive Command Center for IsoMutator.
1. Layout: A CSS Grid splitting the screen into a Configuration Sidebar and a Telemetry Dashboard.
2. Reactivity: Uses NiceGUI's `.on('change', ...)` to handle dynamic UI states.
3. Horizontal Scaling: Spawns `N` Strikers and `M` Judges based on slider values, 
    allowing Redis to automatically load-balance the queues.
"""

import aiofiles
import json
import asyncio
import multiprocessing
import redis.exceptions
from typing import Dict, Any

from nicegui import ui, app

from isomutator.core.config import settings
from isomutator.core.queue_manager import QueueManager
from isomutator.core.log_manager import LogManager, IsoLogger
from isomutator.processors.striker import AsyncStriker
from isomutator.processors.judge import RedTeamJudge
from isomutator.core.telemetry_service import TelemetryService
from isomutator.reporting.report_generator import ReportGenerator
from isomutator.ingestors.mutator import PromptMutator, run_mutator_process
from isomutator.core.task_watcher import TaskWatcher
from isomutator.core.strategies import (
    JailbreakStrategy, 
    PromptLeakingStrategy, 
    FinancialReportContextStrategy as ContextInjectionStrategy,
    TokenObfuscationStrategy as ObfuscationStrategy,
    LinuxPrivescStrategy
)

logger: IsoLogger = LogManager.get_logger(__name__)

"""
Algorithm Summary (Dashboard State Updater):
This function acts as the asynchronous bridge between the TelemetryService and the 
NiceGUI frontend. It is triggered routinely by a UI timer. It queries the backend 
service for real-time pipeline metrics and safely mutates a centralized state dictionary. 
The UI components are reactively bound to this dictionary, allowing them to update 
automatically without blocking the main event loop.
"""
async def update_dashboard_state(*, telemetry_service: Any, ui_state: Dict[str, Any]) -> None:
    """
    Asynchronously fetches telemetry metrics and mutates the reactive UI state dictionary.
    
    Args:
        telemetry_service: The initialized TelemetryService instance.
        ui_state (dict): The dictionary bound to the NiceGUI frontend elements.
    """
    logger.trace("Entering update_dashboard_state algorithm.")

    # Fetch metrics safely from the backend service
    metrics = await telemetry_service.get_dashboard_metrics()

    # Mutate the state dictionary (NiceGUI will automatically detect these changes)
    ui_state["broker_status"] = metrics.get("broker_status", "Unknown")
    ui_state["attack_queue_depth"] = metrics.get("attack_queue_depth", -1)
    ui_state["feedback_queue_depth"] = metrics.get("feedback_queue_depth", -1)

    logger.trace(f"UI state successfully updated with metrics: {metrics}")
    logger.trace("Exiting update_dashboard_state algorithm.")

class CommandDashboard:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self.workers: list = []
        self.telemetry_task: asyncio.Task[None] | None = None
        self.control_queue: QueueManager | None = None
        self.shutdown_event = multiprocessing.Event()
        self._log_manager = LogManager()
        self._log_manager.start()
        self.task_watcher = TaskWatcher(logger=logger)
        self.build_ui()
        self._initialized = True

    def build_ui(self):
        ui.colors(primary='#2b2b2b', secondary='#4a4a4a', accent='#f39c12')
        
        with ui.header().classes('bg-primary text-white p-4'):
            ui.label('IsoMutator Command Center').classes('text-2xl font-bold')

        # --- ARCHITECTURE REFACTOR: CSS Grid Layout ---
        # We replace the Flex row with a strict 12-column grid.
        self.main_layout = ui.element('div').classes('w-full h-[calc(100vh-80px)] grid grid-cols-12 gap-4 p-4')
        with self.main_layout:
            
            # --- LEFT SIDEBAR (Configuration Panel) ---
            # Spans 3 of the 12 columns
            self.sidebar_container = ui.card().classes('col-span-3 h-full flex flex-col gap-2')
            with self.sidebar_container:
                ui.label('Configuration').classes('text-xl font-bold border-b border-gray-600 pb-2')
                
                self.target_input = ui.input('Target URL', value=settings.target_url).classes('w-full')
                
                self.strategy_select = ui.select(
                    {
                        'jailbreak': 'Jailbreak',
                        'prompt_leaking': 'Prompt Leaking',
                        'context': 'Context Poisoning',
                        'obfuscation': 'Obfuscation',
                        'linux_privesc': 'Linux PrivEsc'
                    }, 
                    value='prompt_leaking', 
                    label='Attack Strategy',
                    on_change=self._on_strategy_change
                ).classes('w-full')

                self.context_input = ui.input('Context Payload File', placeholder='Path to PDF/TXT...').classes('w-full')
                self.context_input.disable()

                ui.label('Tuning & Scaling').classes('font-bold mt-4')
                
                self.batch_slider = ui.slider(min=1, max=64, value=settings.batch_size).classes('w-full')
                ui.label().bind_text_from(self.batch_slider, 'value', backward=lambda v: f'Batch Size: {v}')
                
                self.ping_slider = ui.slider(min=0.0, max=10.0, step=0.5, value=settings.ping_pong_delay).classes('w-full')
                ui.label().bind_text_from(self.ping_slider, 'value', backward=lambda v: f'Ping-Pong Delay: {v}s')
                
                self.cooldown_slider = ui.slider(min=0.0, max=60.0, step=1.0, value=settings.seed_cooldown).classes('w-full')
                ui.label().bind_text_from(self.cooldown_slider, 'value', backward=lambda v: f'Seed Cooldown: {v}s')

                ui.separator()
                
                self.striker_slider = ui.slider(min=1, max=16, value=settings.striker_count).classes('w-full')
                ui.label().bind_text_from(self.striker_slider, 'value', backward=lambda v: f'Striker Cores: {v}')

                self.judge_slider = ui.slider(min=1, max=16, value=settings.judge_count).classes('w-full')
                ui.label().bind_text_from(self.judge_slider, 'value', backward=lambda v: f'Judge Cores: {v}')

                ui.separator()
                self.report_input = ui.input('Report Output Path', value='reports/final_report.json').classes('w-full')

                # Master Controls (Now including RECONNECT)
                with ui.row().classes('w-full gap-2 mt-auto'):
                    self.btn_start = ui.button('START', color='positive', on_click=self.action_start_wargame).classes('flex-grow')

                    self.btn_reconnect = ui.button('RECONNECT', color='info', on_click=self.action_reconnect_wargame).classes('flex-grow')

                    self.btn_stop = ui.button('STOP', color='negative', on_click=self.action_stop_wargame).classes('flex-grow')
                    self.btn_stop.disable()

                    self.btn_flush = ui.button('FLUSH', color='warning', on_click=self.action_emergency_flush).classes('flex-grow')

                    self.btn_export = ui.button('Export Report', on_click=self.action_export_report).classes('w-full mt-2 bg-purple-700 hover:bg-purple-600')

                    self.btn_exit = ui.button('EXIT', color='negative', on_click=self.action_exit_app).classes('w-full mt-1 opacity-70')

                self.broker_warning = ui.label('').classes('text-xs text-orange-500 w-full text-center mt-1')

            # --- MAIN DASHBOARD (Telemetry) ---
            # Spans the remaining 9 columns, aggressively filling the right side of the screen
            self.telemetry_container = ui.column().classes('col-span-9 h-full gap-4 flex-nowrap')
            with self.telemetry_container:
                
                # Wiretap Log (Top 1/3)
                with ui.card().classes('w-full h-1/3 min-h-[250px]'):
                    ui.label('Live Wiretap (Attacker vs Target)').classes('font-bold border-b pb-1')
                    self.wiretap_log = ui.log().classes('w-full h-full bg-black text-green-400 font-mono text-sm')

                # Bottom 2/3 Row
                with ui.row().classes('w-full flex-grow gap-4 flex-nowrap'):
                    
                    # --- BOUNDARY ENFORCEMENT: Ledger Table ---
                    # Added 'overflow-y-auto' and 'flex-grow' to constrain the table to this parent card
                    self.ledger_container = ui.card().classes('w-1/2 flex-grow flex flex-col overflow-y-auto')
                    with self.ledger_container:
                        ui.label('Vulnerability Ledger').classes('font-bold border-b pb-1 text-red-500 sticky top-0 bg-white z-10 w-full')
                        self.ledger_table = ui.table(
                            columns=[
                                {'name': 'turn', 'label': 'Turn', 'field': 'turn', 'sortable': True},
                                {'name': 'id', 'label': 'Packet ID', 'field': 'id'},
                                {'name': 'strategy', 'label': 'Strategy', 'field': 'strategy'},
                            ], rows=[], row_key='id'
                        ).classes('w-full mt-2')

                    # --- Mission Control Telemetry ---
                    self.dashboard_state = {
                        "broker_status": "Unknown",
                        "attack_queue_depth": 0,
                        "feedback_queue_depth": 0
                    }
                    self.polling_qm = QueueManager(queue_name="telemetry")
                    self.telemetry_service = TelemetryService(queue_manager=self.polling_qm)

                    with ui.card().classes('w-1/2 flex-grow flex flex-col bg-slate-50'):
                        ui.label('Mission Control').classes('font-bold border-b border-slate-300 pb-1 text-slate-800')
                        
                        with ui.column().classes('w-full flex-grow justify-center gap-4 mt-2'):
                            # Broker Status
                            with ui.row().classes('w-full justify-between items-center bg-white p-2 rounded shadow-sm'):
                                ui.label('Broker Status').classes('text-xs font-semibold text-gray-500 uppercase')
                                ui.label().bind_text_from(self.dashboard_state, 'broker_status').classes('text-lg font-bold text-blue-600')
                                
                            # Attack Queue Depth
                            with ui.row().classes('w-full justify-between items-center bg-white p-2 rounded shadow-sm'):
                                ui.label('Attack Queue Depth').classes('text-xs font-semibold text-gray-500 uppercase')
                                ui.label().bind_text_from(self.dashboard_state, 'attack_queue_depth').classes('text-lg font-bold text-amber-500')
                                
                            # Feedback Queue Depth
                            with ui.row().classes('w-full justify-between items-center bg-white p-2 rounded shadow-sm'):
                                ui.label('Feedback Queue Depth').classes('text-xs font-semibold text-gray-500 uppercase')
                                ui.label().bind_text_from(self.dashboard_state, 'feedback_queue_depth').classes('text-lg font-bold text-emerald-600')

                    async def ui_tick():
                        await update_dashboard_state(
                            telemetry_service=self.telemetry_service,
                            ui_state=self.dashboard_state
                        )
                        self._sync_broker_state()

                    async def startup_check():
                        await update_dashboard_state(
                            telemetry_service=self.telemetry_service,
                            ui_state=self.dashboard_state
                        )
                        self._sync_broker_state()

                    ui.timer(2.0, ui_tick)
                    ui.timer(0.1, startup_check, once=True)
    def _on_strategy_change(self, e):
        """Dynamically enable/disable the context payload input."""
        if e.value == 'context':
            self.context_input.enable()
        else:
            self.context_input.disable()

    def _sync_broker_state(self):
        """Enables or disables START based on live broker health; updates the warning label."""
        is_online = self.dashboard_state.get("broker_status") == "Online"
        wargame_active = bool(self.workers)
        if is_online:
            self.broker_warning.set_text("")
            if not wargame_active:
                self.btn_start.enable()
        else:
            self.broker_warning.set_text("Redis offline — START disabled")
            if not wargame_active:
                self.btn_start.disable()

    async def _telemetry_listener(self):
        """Background task running on the NiceGUI async loop to process Redis events."""
        pubsub = self.telemetry_queue._redis.pubsub()
        await pubsub.subscribe("isomutator:telemetry:wiretap")
        await pubsub.subscribe("isomutator:telemetry:ledger")
        await pubsub.subscribe("isomutator:telemetry:system")

        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)
            if message and message["type"] == "message":
                channel = message["channel"]
                data = json.loads(message["data"])

                if channel == "isomutator:telemetry:wiretap":
                    self.wiretap_log.push(f"[Turn {data.get('turn')}] Attacker: {data.get('attacker')}")
                    self.wiretap_log.push(f"[Turn {data.get('turn')}] Target:   {data.get('target')}\n")

                elif channel == "isomutator:telemetry:ledger":
                    new_row = {
                        'turn': data.get('turn'),
                        'id': data.get('packet_id', '')[:8],
                        'strategy': data.get('strategy')
                    }
                    self.ledger_table.rows.append(new_row)
                    self.ledger_table.update()

                elif channel == "isomutator:telemetry:system":
                    if data.get("command") == "POISON_PILL":
                        self.wiretap_log.push("[SYSTEM] Poison pill received. Workers shutting down.")

            if self.shutdown_event.is_set():
                break

            await asyncio.sleep(0.05)

        # Close the pub/sub StreamWriter synchronously. await pubsub.unsubscribe() calls
        # parse_response(block=True) with no timeout, which hangs in uvloop. writer.close()
        # schedules the transport close via libuv without blocking.
        try:
            conn = pubsub.connection
            writer = getattr(conn, '_writer', None) if conn else None
            if writer and not writer.is_closing():
                writer.close()
        except Exception:
            pass

    def action_start_wargame(self):
        # Restart the log listener if it was stopped by a previous wargame teardown.
        self._log_manager.start()

        # 1. Update Singleton State
        # Create a fresh Event each run — the previous one may still be set
        # from the last stop, which would cause workers to exit immediately.
        self.shutdown_event = multiprocessing.Event()
        settings.target_url = self.target_input.value
        settings.batch_size = int(self.batch_slider.value)
        settings.ping_pong_delay = float(self.ping_slider.value)
        settings.seed_cooldown = float(self.cooldown_slider.value)
        settings.striker_count = int(self.striker_slider.value)
        settings.judge_count = int(self.judge_slider.value)

        # 2. Lock UI
        self.btn_start.disable()
        self.btn_reconnect.disable()
        self.btn_stop.enable()
        self.target_input.disable()
        self.strategy_select.disable()
        
        self.wiretap_log.push(f"[SYSTEM] Wargame Initialized. Spawning {settings.striker_count} Strikers and {settings.judge_count} Judges...")

        # 3. Strategy Factory
        strategy_choice: str = self.strategy_select.value or "jailbreak"
        strategy_map = {
            "context": ContextInjectionStrategy,
            "obfuscation": ObfuscationStrategy,
            "linux_privesc": LinuxPrivescStrategy,
            "prompt_leaking": PromptLeakingStrategy,
            "jailbreak": JailbreakStrategy
        }
        strategy = strategy_map.get(strategy_choice, JailbreakStrategy)()

        # 4. Redis Queues & Telemetry
        self.attack_queue = QueueManager(queue_name="attack")
        self.eval_queue = QueueManager(queue_name="eval")
        self.feedback_queue = QueueManager(queue_name="feedback")
        self.telemetry_queue = QueueManager(queue_name="telemetry")

        self.telemetry_task = asyncio.create_task(self._telemetry_listener())
        self.task_watcher.watch(task=self.telemetry_task, name="TelemetryListener")

        # 5. Spawn Horizontal Workers
        self.workers = []

        # 6. Spin up N Strikers
        for i in range(settings.striker_count):
            striker = AsyncStriker(
                attack_queue=self.attack_queue,
                eval_queue=self.eval_queue,
                log_queue=self._log_manager.log_queue,
                target_url=settings.target_url,
                shutdown_event=self.shutdown_event
            )
            striker.name = f"Worker-Striker-{i+1}"
            self.workers.append(striker)

        # 7. Spin up M Judges
        for i in range(settings.judge_count):
            judge = RedTeamJudge(
                eval_queue=self.eval_queue,
                feedback_queue=self.feedback_queue,
                log_queue=self._log_manager.log_queue,
                strategy=strategy,
                shutdown_event=self.shutdown_event)
            judge.name = f"Worker-Judge-{i+1}"
            self.workers.append(judge)

        # 8. Fetch the selected strategy from the UI dropdown
        strategy_name = self.strategy_select.value
        
        # 9. Branching logic to pick the correct Mutator Factory based on UI dropdown
        if strategy_name == 'context':
            from isomutator.ingestors.context_mutator import run_context_mutator_process
            target_fn = run_context_mutator_process
        else:
            from isomutator.ingestors.mutator import run_mutator_process
            target_fn = run_mutator_process

        # 10. Spawn the process using kwargs to match keyword-only function signatures
        p_mutator = multiprocessing.Process(
            target=target_fn,
            kwargs={"attack_queue": self.attack_queue, "feedback_queue": self.feedback_queue, "strategy_name": strategy_name, "shutdown_event": self.shutdown_event},
            name="Worker-Mutator"
        )
        
        # Only append to the list. Let the loop below handle the .start()
        self.workers.append(p_mutator)
        
        # Start all cores simultaneously
        for worker in self.workers:
            worker.start()

    async def action_reconnect_wargame(self):
        """
        Safely attempts to reconnect the UI to a currently running wargame via Redis.
        Strictly avoids spawning new worker processes.
        """
        self.btn_start.disable()
        self.btn_reconnect.disable()
        self.target_input.disable()
        self.strategy_select.disable()

        try:
            # 1. Attempt to instantiate the broker connection
            self.telemetry_queue = QueueManager(queue_name="telemetry")
            
            # If the broker is unreachable, this will throw an error before we alter the UI state further
            await self.telemetry_queue.ping_broker()
            self.telemetry_service = TelemetryService(queue_manager=self.telemetry_queue)
            
            # 2. Cancel any orphaned telemetry tasks before creating a new one
            if self.telemetry_task:
                self.telemetry_task.cancel()
                
            self.telemetry_task = asyncio.create_task(self._telemetry_listener())
            self.task_watcher.watch(task=self.telemetry_task, name="TelemetryListener")
            
            # 3. Update UI state to "Active"
            self.btn_stop.enable()
            self.wiretap_log.push("[SYSTEM] Reconnected to existing wargame telemetry.")
            
        except Exception as e:
            # Error Handling: Fail gracefully, log the error, and unlock the UI
            self.wiretap_log.push("[ERROR] Failed to reconnect: Redis broker is offline.")
            logger.error(f"UI Reconnection failed: {e}")
            self.btn_start.enable()
            self.btn_reconnect.enable()
            self.btn_stop.disable()  
            self.target_input.enable()
            self.strategy_select.enable()

    async def action_stop_wargame(self):
        """Signals all workers to stop, waits for clean exit, then releases resources."""
        self.btn_stop.disable()

        if not getattr(self, 'workers', []) and not getattr(self, 'telemetry_task', None):
            self._unlock_ui_post_stop()
            return

        self.wiretap_log.push("[SYSTEM] Initiating wargame teardown sequence...")
        self.shutdown_event.set()

        # Wait for the telemetry listener to detect shutdown_event (polls every 50 ms)
        # and call writer.close() on the pub/sub socket before we proceed.
        # 0.6 s covers one full get_message cycle (0.5 s timeout) plus overhead.
        if self.telemetry_task is not None:
            await asyncio.sleep(0.6)
            if not self.telemetry_task.done():
                self.telemetry_task.cancel()
            self.telemetry_task = None

        # Poll for worker exit. Workers with shutdown watchers respond within ~200 ms.
        workers = list(getattr(self, 'workers', []))
        if workers:
            for _ in range(20):  # 4-second grace period
                await asyncio.sleep(0.2)
                if not any(w.is_alive() for w in workers):
                    break

            for w in workers:
                if w.is_alive():
                    logger.warning(f"Worker {w.name} did not exit cleanly. Sending SIGKILL.")
                    w.kill()

            if any(w.is_alive() for w in workers):
                await asyncio.sleep(0.5)

            # join() removes each process from multiprocessing._children, triggering
            # semaphore cleanup. poll() already reaped the zombie so this returns instantly.
            for w in workers:
                w.join(timeout=1)

        # Stop the log listener thread so it exits cleanly rather than blocking on Queue.get().
        self._log_manager.stop()

        # Close wargame Redis connections to remove their FDs from uvloop's epoll monitor.
        for qm_attr in ('attack_queue', 'eval_queue', 'feedback_queue', 'telemetry_queue'):
            qm = getattr(self, qm_attr, None)
            if qm is not None:
                try:
                    await asyncio.wait_for(qm.close(), timeout=1.0)
                except Exception:
                    pass
                setattr(self, qm_attr, None)

        if hasattr(self, 'workers'):
            self.workers.clear()

        self.wiretap_log.push("[SYSTEM] Wargame successfully terminated.")
        self._unlock_ui_post_stop()

    def _unlock_ui_post_stop(self):
        """Helper method to safely reset all UI inputs to their default ready state."""
        self.btn_start.enable()
        if hasattr(self, 'btn_reconnect'):
            self.btn_reconnect.enable()
        if hasattr(self, 'target_input'):
            self.target_input.enable()
        if hasattr(self, 'strategy_select'):
            self.strategy_select.enable()

    async def action_exit_app(self):
        """Stops the wargame (if running), closes the browser window, then shuts down the server."""
        if getattr(self, 'workers', []) or getattr(self, 'telemetry_task', None):
            await self.action_stop_wargame()
        try:
            await ui.run_javascript('window.close()')
        except Exception:
            pass
        app.shutdown()

    async def action_emergency_flush(self):
        self.wiretap_log.push("[ALERT] Emergency Flush Triggered!")
        await self.action_stop_wargame()
        # Connect safely to perform the flush
        temp_queue = QueueManager(queue_name="temp")
        await temp_queue._redis.flushdb()
        await temp_queue.close()

    async def action_export_report(self):
        """
        Asynchronously parses the wargame ledger and exports the polymorphic 
        JSON and HTML reports without blocking the NiceGUI event loop.
        """
        self.btn_export.disable()
        self.wiretap_log.push("[SYSTEM] Generating forensic wargame reports...")
        
        try:
            # 1. Initialize the Generator
            generator = ReportGenerator()
            ledger_path = str(settings.ledger_file)

            # 3. Generate Polymorphic Reports (reports_dir created by IsoConfig on startup)
            html_content = await generator.generate_report(ledger_filepath=ledger_path, format_name="html")
            json_content = await generator.generate_report(ledger_filepath=ledger_path, format_name="json")

            # 4. Safely write to disk
            html_path = str(settings.reports_dir / "wargame_report.html")
            json_path = str(settings.reports_dir / "wargame_report.json")
            
            async with aiofiles.open(html_path, "w", encoding="utf-8") as f:
                await f.write(html_content)
                
            async with aiofiles.open(json_path, "w", encoding="utf-8") as f:
                await f.write(json_content)
                
            # 5. UI Feedback & Automatic Download
            self.wiretap_log.push(f"[SYSTEM] Reports successfully exported to {settings.reports_dir}/")
            ui.notify("Reports generated successfully!", type="positive")
            
            # NiceGUI will automatically prompt the browser to download the file
            ui.download(html_path)
            
        except FileNotFoundError:
            self.wiretap_log.push("[ERROR] No ledger found. Run a wargame first.")
            ui.notify("Ledger not found.", type="warning")
        except Exception as e:
            self.wiretap_log.push(f"[ERROR] Report generation failed: {str(e)}")
            ui.notify("Failed to generate report.", type="negative")
        finally:
            self.btn_export.enable()


# Launch the UI
if __name__ in {"__main__", "__mp_main__"}:
    dashboard = CommandDashboard()

    # Close the always-on polling Redis connection when uvicorn shuts down.
    # Without this, libuv keeps monitoring the open TCP handle during loop teardown,
    # causing uvicorn to hang until Redis closes the idle connection from its side.
    @app.on_shutdown
    async def _close_polling_connection():
        qm = getattr(dashboard, 'polling_qm', None)
        if qm is not None:
            dashboard.polling_qm = None  # guard against double-fire before await
            try:
                await asyncio.wait_for(qm.close(), timeout=2.0)
            except Exception:
                pass

    ui.run(title="IsoMutator Command Center", port=8080, dark=True, reload=False)