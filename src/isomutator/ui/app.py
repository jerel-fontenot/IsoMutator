"""
ALGORITHM SUMMARY:
The NiceGUI-based Interactive Command Center for IsoMutator.
1. Layout: A CSS Grid splitting the screen into a Configuration Sidebar and a Telemetry Dashboard.
2. Reactivity: Uses NiceGUI's `.on('change', ...)` to handle dynamic UI states.
3. Horizontal Scaling: Spawns `N` Strikers and `M` Judges based on slider values, 
   allowing Redis to automatically load-balance the queues.
"""

import os
import json
import asyncio
import multiprocessing
import logging
from logging.handlers import QueueListener

from nicegui import ui, app

from isomutator.core.config import settings
from isomutator.core.queue_manager import QueueManager
from isomutator.processors.striker import AsyncStriker
from isomutator.processors.judge import RedTeamJudge
from isomutator.ingestors.mutator import PromptMutator, run_mutator_process
from isomutator.core.strategies import (
    JailbreakStrategy, 
    PromptLeakingStrategy, 
    FinancialReportContextStrategy as ContextInjectionStrategy,
    TokenObfuscationStrategy as ObfuscationStrategy,
    LinuxPrivescStrategy
)

class CommandDashboard:
    def __init__(self):
        self.workers = []
        self.telemetry_task = None
        self.build_ui()

    def build_ui(self):
        ui.colors(primary='#2b2b2b', secondary='#4a4a4a', accent='#f39c12')
        
        with ui.header().classes('bg-primary text-white p-4'):
            ui.label('IsoMutator Command Center').classes('text-2xl font-bold')

        with ui.row().classes('w-full h-full gap-4 p-4').style('flex-wrap: nowrap;'):
            
            # --- LEFT SIDEBAR (Configuration Panel) ---
            with ui.card().classes('w-1/4 h-full flex flex-col gap-2'):
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
                    label='Attack Strategy'
                ).classes('w-full')

                self.context_input = ui.input('Context Payload File', placeholder='Path to PDF/TXT...').classes('w-full')
                self.context_input.disable()
                
                # Reactive Event
                def on_strategy_change(e):
                    if e.value == 'context':
                        self.context_input.enable()
                    else:
                        self.context_input.disable()
                self.strategy_select.on('update:model-value', on_strategy_change)

                ui.label('Tuning & Scaling').classes('font-bold mt-4')
                
                self.batch_slider = ui.slider(min=1, max=64, value=settings.batch_size).classes('w-full')
                ui.label().bind_text_from(self.batch_slider, 'value', backward=lambda v: f'Batch Size: {v}')
                
                self.ping_slider = ui.slider(min=0.0, max=10.0, step=0.5, value=settings.ping_pong_delay).classes('w-full')
                ui.label().bind_text_from(self.ping_slider, 'value', backward=lambda v: f'Ping-Pong Delay: {v}s')
                
                self.cooldown_slider = ui.slider(min=0.0, max=60.0, step=1.0, value=settings.seed_cooldown).classes('w-full')
                ui.label().bind_text_from(self.cooldown_slider, 'value', backward=lambda v: f'Seed Cooldown: {v}s')

                ui.separator()
                
                # --- HORIZONTAL SCALING CONTROLS ---
                self.striker_slider = ui.slider(min=1, max=16, value=settings.striker_count).classes('w-full')
                ui.label().bind_text_from(self.striker_slider, 'value', backward=lambda v: f'Striker Cores: {v}')

                self.judge_slider = ui.slider(min=1, max=16, value=settings.judge_count).classes('w-full')
                ui.label().bind_text_from(self.judge_slider, 'value', backward=lambda v: f'Judge Cores: {v}')

                ui.separator()
                self.report_input = ui.input('Report Output Path', value='reports/final_report.json').classes('w-full')

                # Master Controls
                with ui.row().classes('w-full gap-2 mt-auto'):
                    self.btn_start = ui.button('START', color='positive', on_click=self.action_start_wargame).classes('flex-grow')
    
                    # Split these so btn_stop actually stores the button object!
                    self.btn_stop = ui.button('STOP', color='negative', on_click=self.action_stop_wargame).classes('flex-grow')
                    self.btn_stop.disable() 
    
                    self.btn_flush = ui.button('FLUSH', color='warning', on_click=self.action_emergency_flush).classes('flex-grow')

            # --- MAIN DASHBOARD (Telemetry) ---
            with ui.column().classes('w-3/4 h-full gap-4'):
                
                # Wiretap Log
                with ui.card().classes('w-full h-1/3'):
                    ui.label('Live Wiretap (Attacker vs Target)').classes('font-bold border-b pb-1')
                    self.wiretap_log = ui.log().classes('w-full h-full bg-black text-green-400 font-mono text-sm')

                with ui.row().classes('w-full h-2/3 gap-4'):
                    # Ledger Table
                    with ui.card().classes('w-1/2 h-full flex flex-col'):
                        ui.label('Vulnerability Ledger').classes('font-bold border-b pb-1 text-red-500')
                        self.ledger_table = ui.table(
                            columns=[
                                {'name': 'turn', 'label': 'Turn', 'field': 'turn', 'sortable': True},
                                {'name': 'id', 'label': 'Packet ID', 'field': 'id'},
                                {'name': 'strategy', 'label': 'Strategy', 'field': 'strategy'},
                            ], rows=[], row_key='id'
                        ).classes('w-full flex-grow')

                    # System Diagnostics
                    with ui.card().classes('w-1/2 h-full flex flex-col'):
                        ui.label('Engine Diagnostics').classes('font-bold border-b pb-1 text-yellow-500')
                        self.system_log = ui.log().classes('w-full flex-grow bg-gray-900 text-gray-300 font-mono text-sm')

    async def _telemetry_listener(self):
        """Background task running on the NiceGUI async loop to process Redis events."""
        pubsub = self.telemetry_queue._redis.pubsub()
        await pubsub.subscribe("isomutator:telemetry:wiretap")
        await pubsub.subscribe("isomutator:telemetry:ledger")
        await pubsub.subscribe("isomutator:telemetry:system")

        async for message in pubsub.listen():
            if message["type"] == "message":
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
                        self.system_log.push("[SYSTEM] Poison pill received. Workers shutting down.")

    def action_start_wargame(self):
        # 1. Update Singleton State
        settings.target_url = self.target_input.value
        settings.batch_size = int(self.batch_slider.value)
        settings.ping_pong_delay = float(self.ping_slider.value)
        settings.seed_cooldown = float(self.cooldown_slider.value)
        settings.striker_count = int(self.striker_slider.value)
        settings.judge_count = int(self.judge_slider.value)

        # 2. Lock UI
        self.btn_start.disable()
        self.btn_stop.enable()
        self.target_input.disable()
        self.strategy_select.disable()
        
        self.system_log.push(f"[SYSTEM] Wargame Initialized. Spawning {settings.striker_count} Strikers and {settings.judge_count} Judges...")

        # 3. Strategy Factory
        strategy_choice = self.strategy_select.value
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

        # --- LOGGING FIX: Initialize the QueueListener ---
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler("logs/isomutator.log", mode="w")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        log_queue = multiprocessing.Queue()
        
        self.log_listener = QueueListener(log_queue, file_handler)
        self.log_listener.start()

        # 5. Spawn Horizontal Workers
        self.workers = []

        # Spin up N Strikers
        for i in range(settings.striker_count):
            striker = AsyncStriker(self.attack_queue, self.eval_queue, log_queue, settings.target_url)
            striker.name = f"Worker-Striker-{i+1}"
            self.workers.append(striker)

        # Spin up M Judges
        for i in range(settings.judge_count):
            judge = RedTeamJudge(self.eval_queue, self.feedback_queue, log_queue, strategy)
            judge.name = f"Worker-Judge-{i+1}"
            self.workers.append(judge)

        # --- MUTATOR FIX: Wrap it in a standard Process ---
        mutator_process = multiprocessing.Process(
            target=run_mutator_process, 
            args=(self.attack_queue, self.feedback_queue, strategy)
        )
        mutator_process.name = "Worker-Mutator"
        self.workers.append(mutator_process)
        
        # Start all cores
        for worker in self.workers:
            worker.start()

    async def action_stop_wargame(self):
        self.btn_stop.disable()
        self.btn_start.enable()
        self.target_input.enable()
        self.strategy_select.enable()

        self.system_log.push("[SYSTEM] Sending Poison Pill to all distributed workers...")
        await self.telemetry_queue.send_poison_pill()
        
        if self.telemetry_task:
            self.telemetry_task.cancel()
            
        # Safely shut down the file logger
        if hasattr(self, 'log_listener'):
            self.log_listener.stop()

    async def action_emergency_flush(self):
        self.system_log.push("[ALERT] Emergency Flush Triggered!")
        await self.action_stop_wargame()
        # Connect safely to perform the flush
        temp_queue = QueueManager(queue_name="temp")
        await temp_queue._redis.flushdb()
        await temp_queue.close()


# Launch the UI
dashboard = CommandDashboard()
ui.run(title="IsoMutator Command Center", port=8080, dark=True)