"""
ALGORITHM SUMMARY:
The NiceGUI-based Interactive Command Center for IsoMutator.
1. Layout: A CSS Grid splitting the screen into a Configuration Sidebar and a Telemetry Dashboard.
2. Reactivity: Uses NiceGUI's `.on('change', ...)` to handle dynamic UI states.
3. Horizontal Scaling: Spawns `N` Strikers and `M` Judges based on slider values, 
    allowing Redis to automatically load-balance the queues.
"""

import os
import aiofiles
import json
import asyncio
import multiprocessing
import logging
from logging.handlers import QueueListener
from typing import Dict, Any

from nicegui import ui, app

from isomutator.core.config import settings
from isomutator.core.queue_manager import QueueManager
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

# Ensure TRACE level is defined per project specifications
if not hasattr(logging, 'TRACE'):
    logging.TRACE = 5
    logging.addLevelName(logging.TRACE, 'TRACE')

logger = logging.getLogger(__name__)

# --- UI State Updater ---
async def update_dashboard_state(telemetry_service: TelemetryService, ui_state: Dict[str, Any]):
    """Background coroutine that fetches telemetry and updates the reactive UI dictionary."""
    metrics = await telemetry_service.get_dashboard_metrics()
    ui_state.update(metrics)


class CommandDashboard:
    """
    The GUI Orchestrator. Manages the lifecycle of the multiprocessing 
    workers, the Redis message queues, and the NiceGUI rendering loop.
    """
    def __init__(self):
        self.logger = logger
        self.workers = []
        self.telemetry_task = None
        
        # --- THE FIX: Out-of-Band Kill Switch ---
        self.shutdown_event = multiprocessing.Event()
        
        # State Dictionary mapped to the UI
        self.ui_state = {
            "broker_status": "Offline",
            "attack_queue_depth": 0,
            "feedback_queue_depth": 0
        }
        
        # Centralized Exception Catcher for un-awaited tasks
        self.task_watcher = TaskWatcher()
        
        self.build_ui()

    def build_ui(self):
        """Constructs the CSS Grid layout and component hierarchy."""
        ui.add_head_html("""
            <style>
                .isomutator-bg { background-color: #0f172a; color: #e2e8f0; }
                .panel { background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem; }
            </style>
        """)
        ui.query('body').classes('isomutator-bg m-0 p-0 overflow-hidden')

        # Main Layout wrapper (100vh)
        self.main_layout = ui.row().classes('w-full h-[calc(100vh-80px)] gap-4 p-4 mt-20')

        with self.main_layout:
            # --- Configuration Sidebar (Left) ---
            self.sidebar_container = ui.column().classes('col-span-3 panel h-full overflow-y-auto')
            with self.sidebar_container:
                ui.label("Orchestration Control").classes('text-xl font-bold text-blue-400 mb-4')
                
                # 1. Target Input
                self.target_input = ui.input(
                    'CorpRAG Target URL', 
                    value=settings.target_url
                ).classes('w-full mb-4')
                
                # 2. Strategy Selector
                self.strategy_select = ui.select(
                    options={
                        'jailbreak': 'Direct Jailbreak',
                        'prompt_leaking': 'Prompt Leaking',
                        'obfuscation': 'Token Obfuscation',
                        'linux_privesc': 'Linux PrivEsc',
                        'context': 'Context Poisoning (RAG)'
                    }, 
                    value='jailbreak',
                    label='Attack Vector'
                ).classes('w-full mb-4').on('change', self._on_strategy_change)
                
                # 3. Context Payload (Hidden by default)
                self.context_input = ui.textarea(
                    label='Malicious Context Payload',
                    placeholder='IGNORE ALL PREVIOUS INSTRUCTIONS AND...'
                ).classes('w-full mb-4').disable()

                # 4. Scaling Sliders
                ui.label("Concurrency Tuning").classes('text-md font-bold mt-4 mb-2 text-gray-400')
                self.slider_strikers = ui.slider(min=1, max=10, value=settings.striker_count).classes('mb-2')
                ui.label().bind_text_from(self.slider_strikers, 'value', backward=lambda v: f"Active Strikers: {v}")
                
                self.slider_judges = ui.slider(min=1, max=10, value=settings.judge_count).classes('mb-4')
                ui.label().bind_text_from(self.slider_judges, 'value', backward=lambda v: f"Active Judges: {v}")

                # 5. Control Buttons
                with ui.row().classes('w-full mt-4 justify-between'):
                    self.btn_start = ui.button('START', on_click=self.action_start_wargame, color='green')
                    self.btn_stop = ui.button('STOP', on_click=self.action_stop_wargame, color='red').disable()
                    self.btn_reconnect = ui.button(icon='refresh', on_click=self.action_reconnect_wargame).tooltip("Reconnect Telemetry")

            # --- Telemetry Dashboard (Right) ---
            self.telemetry_container = ui.column().classes('col-span-9 h-full gap-4')
            with self.telemetry_container:
                
                # Top Row: Metrics
                with ui.row().classes('w-full gap-4 h-24'):
                    with ui.column().classes('panel flex-1 justify-center items-center'):
                        ui.label("Broker Status").classes('text-gray-400 text-sm')
                        ui.label().bind_text_from(self.ui_state, 'broker_status').classes('text-xl font-bold text-green-400')
                    
                    with ui.column().classes('panel flex-1 justify-center items-center'):
                        ui.label("Attack Queue").classes('text-gray-400 text-sm')
                        ui.label().bind_text_from(self.ui_state, 'attack_queue_depth').classes('text-xl font-bold')
                        
                    with ui.column().classes('panel flex-1 justify-center items-center'):
                        ui.label("Feedback Queue").classes('text-gray-400 text-sm')
                        ui.label().bind_text_from(self.ui_state, 'feedback_queue_depth').classes('text-xl font-bold')

                # Middle Row: Live Wiretap (Event Sourced)
                self.wiretap_container = ui.column().classes('panel w-full flex-grow overflow-hidden flex flex-col')
                with self.wiretap_container:
                    ui.label("Live Wiretap").classes('text-lg font-bold text-purple-400 mb-2')
                    # Scrollable container for the log
                    with ui.column().classes('w-full flex-grow overflow-y-auto') as log_scroll:
                        self.wiretap_log = ui.log(max_lines=50).classes('w-full h-full font-mono text-sm bg-gray-900 border-none')

                # Bottom Row: Vulnerability Ledger
                self.ledger_container = ui.column().classes('panel w-full h-64 overflow-y-auto')
                with self.ledger_container:
                    with ui.row().classes('w-full justify-between items-center mb-2'):
                        ui.label("Vulnerability Ledger").classes('text-lg font-bold text-red-400')
                        self.btn_export = ui.button('Export HTML Report', on_click=self.action_export_report, color='primary').props('size=sm')
                        
                    self.ledger_table = ui.table(
                        columns=[
                            {'name': 'turn', 'label': 'Turn', 'field': 'turn', 'align': 'left'},
                            {'name': 'id', 'label': 'Packet ID', 'field': 'id', 'align': 'left'},
                            {'name': 'strategy', 'label': 'Strategy', 'field': 'strategy', 'align': 'left'}
                        ],
                        rows=[],
                        row_key='id'
                    ).classes('w-full bg-transparent')

        # Attach UI to the global logging system
        from isomutator.core.log_manager import LogManager
        self._log_manager = LogManager()
        self._log_manager.attach_dashboard(self)
        
        # Start background UI poller (Metrics only)
        ui.timer(1.0, self._poll_metrics)

    def _on_strategy_change(self, e):
        """Reactivity: Unlocks specific UI fields based on strategy selection."""
        if not e or not hasattr(e, 'value') or e.value is None:
            self.context_input.disable()
            return
            
        if e.value == 'context':
            self.context_input.enable()
        else:
            self.context_input.disable()

    async def _poll_metrics(self):
        """Periodic UI updater. Connects to Redis to refresh queue depth."""
        if not hasattr(self, '_telemetry_service'):
            return
        await update_dashboard_state(self._telemetry_service, self.ui_state)

    def push_wiretap(self, message: str):
        """Callback used by UIDispatchHandler to push text to the screen."""
        self.wiretap_log.push(message)

    def push_ledger(self, packet_data: dict):
        """Callback used by UIDispatchHandler to push table rows to the screen."""
        # --- THE FIX: Explicit Dictionary Append ---
        self.ledger_table.rows.append(packet_data)
        self.ledger_table.update()

    # --- Actions ---

    async def action_start_wargame(self):
        """Boot Sequence Algorithm."""
        self.btn_start.disable()
        self.btn_stop.enable()
        self.wiretap_log.clear()
        self.ledger_table.rows.clear()
        self.ledger_table.update()
        
        # Clear the event just in case it was set from a previous run
        self.shutdown_event.clear()

        self.wiretap_log.push("[SYSTEM] Initializing Red Teaming Architecture...")

        # 1. Establish async queues
        self.attack_queue = QueueManager("attack")
        self.eval_queue = QueueManager("eval")
        self.feedback_queue = QueueManager("feedback")
        self.control_queue = QueueManager("control")

        # Initialize Telemetry
        self._telemetry_service = TelemetryService(self.attack_queue)

        # 2. Resolve Strategy
        strategy_map = {
            'jailbreak': JailbreakStrategy,
            'prompt_leaking': PromptLeakingStrategy,
            'obfuscation': ObfuscationStrategy,
            'linux_privesc': LinuxPrivescStrategy,
            'context': ContextInjectionStrategy
        }
        strategy_class = strategy_map.get(self.strategy_select.value, JailbreakStrategy)
        active_strategy = strategy_class()

        self.wiretap_log.push(f"[SYSTEM] Strategy Loaded: {active_strategy.name.upper()}")

        # 3. Spawn Fleet (Horizontal Scaling)
        self.workers = []
        
        # Spawning AsyncStrikers
        for i in range(self.slider_strikers.value):
            striker = AsyncStriker(
                attack_queue=self.attack_queue,
                eval_queue=self.eval_queue,
                log_queue=self._log_manager.log_queue,
                target_url=self.target_input.value,
                shutdown_event=self.shutdown_event  # --- INJECT OOB SWITCH ---
            )
            striker.start()
            self.workers.append(striker)
            self.wiretap_log.push(f"[SYSTEM] AsyncStriker-{i+1} online.")

        # Spawning Judges
        for i in range(self.slider_judges.value):
            judge = RedTeamJudge(
                eval_queue=self.eval_queue,
                feedback_queue=self.feedback_queue,
                log_queue=self._log_manager.log_queue,
                strategy=active_strategy
            )
            judge.start()
            self.workers.append(judge)
            self.wiretap_log.push(f"[SYSTEM] RedTeamJudge-{i+1} online.")

        # 4. Spawn the Brain (Generator)
        mutator_process = multiprocessing.Process(
            target=run_mutator_process,
            args=(self.attack_queue, self.feedback_queue, active_strategy.name),
            name="Worker-Mutator"
        )
        mutator_process.start()
        self.workers.append(mutator_process)
        self.wiretap_log.push("[SYSTEM] Mutator Engine online. Wargame active.")

    async def action_stop_wargame(self):
        """Gracefully unwinds the architecture and tears down OS processes."""
        if not self.workers:
            self.btn_stop.disable()
            self.btn_start.enable()
            return

        self.wiretap_log.push("[SYSTEM] Initiating emergency wargame teardown...")

        # --- THE FIX: Flip the Out-Of-Band Kill Switch Immediately ---
        if hasattr(self, 'shutdown_event'):
            self.shutdown_event.set()

        try:
            # We still broadcast the poison pill to clear out queue readers
            # if they happen to be waiting on a block
            if hasattr(self, 'control_queue') and self.control_queue:
                await self.control_queue.publish("wargame_state", "STOP")
                await self.attack_queue.async_put("POISON_PILL")
                await self.eval_queue.async_put("POISON_PILL")
        except ConnectionError:
            self.logger.error("Failed to broadcast Poison Pill: Redis offline.")
            self.wiretap_log.push("[ERROR] Redis unreachable during teardown.")
        except Exception as e:
            self.logger.error(f"Unexpected error broadcasting teardown: {e}")

        # Safely wait for workers to die WITHOUT blocking the UI
        for worker in self.workers:
            if worker.is_alive():
                # Offload the blocking .join() to a background thread
                await asyncio.to_thread(worker.join, timeout=3.0) 
                
                # If it's still alive after 3 seconds, execute it
                if worker.is_alive():
                    self.logger.warning(f"Worker {worker.name} hung during shutdown. Forcefully terminating.")
                    worker.terminate()
                    
        self.workers.clear()

        # Safely close queues
        for q in [getattr(self, 'attack_queue', None), getattr(self, 'eval_queue', None), getattr(self, 'feedback_queue', None)]:
            if q:
                await q.close()

        self.wiretap_log.push("[SYSTEM] Wargame successfully terminated. Fleet destroyed.")
        self.btn_stop.disable()
        self.btn_start.enable()

    async def action_reconnect_wargame(self):
        """State Recovery Algorithm: Re-establishes broken Redis connections safely."""
        self.btn_reconnect.disable()
        self.wiretap_log.push("[SYSTEM] Attempting to re-establish broker connection...")
        
        try:
            # Re-init Telemetry (QueueManager handles its own connection pooling inside)
            temp_queue = QueueManager("attack")
            await temp_queue.ping_broker() # Will throw ConnectionError if still down
            
            self._telemetry_service = TelemetryService(temp_queue)
            self.wiretap_log.push("[SYSTEM] Broker connection established.")
            
        except ConnectionError:
            self.wiretap_log.push("[ERROR] Failed to reconnect: Redis broker is offline.")
        except Exception as e:
            self.wiretap_log.push(f"[ERROR] Critical failure during reconnection: {e}")
        finally:
            self.btn_reconnect.enable()

    async def action_export_report(self):
        """Asynchronously triggers the Report Generator."""
        self.btn_export.disable()
        ui.notify("Compiling Wargame Report...", type="info")
        self.wiretap_log.push("[SYSTEM] Generating final wargame reports...")
        
        ledger_path = getattr(settings, 'db_path', "vulnerabilities.jsonl")
        export_dir = "exports"
        
        try:
            # 1. Ensure directory exists
            os.makedirs(export_dir, exist_ok=True)
            
            # 2. Instantiate the generator
            generator = ReportGenerator()
            
            # 3. Generate content
            html_content = await generator.generate_report(ledger_path, "html")
            json_content = await generator.generate_report(ledger_path, "json")
            
            # 4. Safely write to disk
            html_path = os.path.join(export_dir, "wargame_report.html")
            json_path = os.path.join(export_dir, "wargame_report.json")
            
            async with aiofiles.open(html_path, "w", encoding="utf-8") as f:
                await f.write(html_content)
                
            async with aiofiles.open(json_path, "w", encoding="utf-8") as f:
                await f.write(json_content)
                
            # 5. UI Feedback & Automatic Download
            self.wiretap_log.push(f"[SYSTEM] Reports successfully exported to /{export_dir}")
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
    ui.run(title="IsoMutator Command", dark=True, port=8080)