"""
Algorithm Summary:
Tests for action_start_wargame() in CommandDashboard.

Protocol Adherence:
1. Happy Path: Settings sync, UI lock, correct worker counts spawned and started.
2. Context Strategy: Branches to run_context_mutator_process factory.
3. Worker Naming: Strikers and Judges receive sequential names.
4. Scaling: Arbitrary N strikers + M judges = N+M+1 total workers started.
5. Telemetry: asyncio task is created and registered with the TaskWatcher.
6. Strategy Fallback: None dropdown value defaults to JailbreakStrategy.
7. Strict Mocking: No real processes, event loops, or network connections.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, call

from isomutator.ui.app import CommandDashboard
from isomutator.core.strategies import (
    JailbreakStrategy,
    PromptLeakingStrategy,
    FinancialReportContextStrategy,
)


@pytest.fixture
def dashboard_ui():
    """Provides an isolated CommandDashboard with all UI elements replaced by mocks."""
    with patch('isomutator.ui.app.QueueManager'), \
         patch('isomutator.ui.app.TelemetryService'), \
         patch('isomutator.ui.app.ui'):
        dash = CommandDashboard()

    # Sliders
    dash.target_input = MagicMock()
    dash.target_input.value = "http://target:8080"
    dash.strategy_select = MagicMock()
    dash.strategy_select.value = "jailbreak"
    dash.batch_slider = MagicMock()
    dash.batch_slider.value = 4
    dash.ping_slider = MagicMock()
    dash.ping_slider.value = 2.0
    dash.cooldown_slider = MagicMock()
    dash.cooldown_slider.value = 15.0
    dash.striker_slider = MagicMock()
    dash.striker_slider.value = 1
    dash.judge_slider = MagicMock()
    dash.judge_slider.value = 1

    # Controls
    dash.btn_start = MagicMock()
    dash.btn_stop = MagicMock()
    dash.btn_reconnect = MagicMock()
    dash.wiretap_log = MagicMock()

    # Internal services
    dash._log_manager = MagicMock()
    dash._log_manager.log_queue = MagicMock()
    dash.task_watcher = MagicMock()
    # Prevent unawaited-coroutine warning: create_task receives a plain mock, not a real coroutine
    dash._telemetry_listener = MagicMock()

    return dash


class TestActionStartWargame:

    # --- 1. Happy Path ---
    @patch('asyncio.create_task')
    @patch('isomutator.ui.app.multiprocessing.Process')
    @patch('isomutator.ui.app.RedTeamJudge')
    @patch('isomutator.ui.app.AsyncStriker')
    def test_happy_path_spawns_correct_worker_counts(
        self, MockStriker, MockJudge, MockProcess, mock_create_task, dashboard_ui
    ):
        """1 Striker + 1 Judge + 1 Mutator process created with default slider values."""
        mock_create_task.return_value = MagicMock(spec=asyncio.Task)

        dashboard_ui.action_start_wargame()

        assert MockStriker.call_count == 1
        assert MockJudge.call_count == 1
        MockProcess.assert_called_once()
        assert len(dashboard_ui.workers) == 3

    # --- 2. UI Lock ---
    @patch('asyncio.create_task')
    @patch('isomutator.ui.app.multiprocessing.Process')
    @patch('isomutator.ui.app.RedTeamJudge')
    @patch('isomutator.ui.app.AsyncStriker')
    def test_ui_is_locked_on_start(
        self, MockStriker, MockJudge, MockProcess, mock_create_task, dashboard_ui
    ):
        """START disabled, STOP enabled, inputs locked when wargame launches."""
        mock_create_task.return_value = MagicMock(spec=asyncio.Task)

        dashboard_ui.action_start_wargame()

        dashboard_ui.btn_start.disable.assert_called_once()
        dashboard_ui.btn_stop.enable.assert_called_once()
        dashboard_ui.btn_reconnect.disable.assert_called_once()
        dashboard_ui.target_input.disable.assert_called_once()
        dashboard_ui.strategy_select.disable.assert_called_once()

    # --- 3. Settings Sync ---
    @patch('asyncio.create_task')
    @patch('isomutator.ui.app.multiprocessing.Process')
    @patch('isomutator.ui.app.RedTeamJudge')
    @patch('isomutator.ui.app.AsyncStriker')
    def test_settings_singleton_updated_from_ui_controls(
        self, MockStriker, MockJudge, MockProcess, mock_create_task, dashboard_ui
    ):
        """Settings singleton reflects slider values before workers are spawned."""
        from isomutator.core.config import settings
        mock_create_task.return_value = MagicMock(spec=asyncio.Task)
        dashboard_ui.batch_slider.value = 16
        dashboard_ui.ping_slider.value = 3.5
        dashboard_ui.cooldown_slider.value = 30.0
        dashboard_ui.striker_slider.value = 4
        dashboard_ui.judge_slider.value = 2

        dashboard_ui.action_start_wargame()

        assert settings.batch_size == 16
        assert settings.ping_pong_delay == 3.5
        assert settings.seed_cooldown == 30.0
        assert settings.striker_count == 4
        assert settings.judge_count == 2

    # --- 4. All Workers Started ---
    @patch('asyncio.create_task')
    @patch('isomutator.ui.app.multiprocessing.Process')
    @patch('isomutator.ui.app.RedTeamJudge')
    @patch('isomutator.ui.app.AsyncStriker')
    def test_all_workers_receive_start_call(
        self, MockStriker, MockJudge, MockProcess, mock_create_task, dashboard_ui
    ):
        """Every spawned worker must have .start() called exactly once."""
        mock_create_task.return_value = MagicMock(spec=asyncio.Task)
        dashboard_ui.striker_slider.value = 2
        dashboard_ui.judge_slider.value = 2

        striker_mocks = [MagicMock() for _ in range(2)]
        judge_mocks = [MagicMock() for _ in range(2)]
        mutator_mock = MagicMock()
        MockStriker.side_effect = striker_mocks
        MockJudge.side_effect = judge_mocks
        MockProcess.return_value = mutator_mock

        dashboard_ui.action_start_wargame()

        for w in striker_mocks + judge_mocks + [mutator_mock]:
            w.start.assert_called_once()

    # --- 5. Scaling ---
    @patch('asyncio.create_task')
    @patch('isomutator.ui.app.multiprocessing.Process')
    @patch('isomutator.ui.app.RedTeamJudge')
    @patch('isomutator.ui.app.AsyncStriker')
    def test_scaling_four_strikers_three_judges(
        self, MockStriker, MockJudge, MockProcess, mock_create_task, dashboard_ui
    ):
        """4 strikers + 3 judges + 1 mutator = 8 total workers."""
        mock_create_task.return_value = MagicMock(spec=asyncio.Task)
        dashboard_ui.striker_slider.value = 4
        dashboard_ui.judge_slider.value = 3

        dashboard_ui.action_start_wargame()

        assert MockStriker.call_count == 4
        assert MockJudge.call_count == 3
        assert len(dashboard_ui.workers) == 8

    # --- 6. Worker Naming ---
    @patch('asyncio.create_task')
    @patch('isomutator.ui.app.multiprocessing.Process')
    @patch('isomutator.ui.app.RedTeamJudge')
    @patch('isomutator.ui.app.AsyncStriker')
    def test_workers_receive_sequential_names(
        self, MockStriker, MockJudge, MockProcess, mock_create_task, dashboard_ui
    ):
        """Strikers named Worker-Striker-N and Judges named Worker-Judge-N."""
        mock_create_task.return_value = MagicMock(spec=asyncio.Task)
        dashboard_ui.striker_slider.value = 2
        dashboard_ui.judge_slider.value = 2

        striker_mocks = [MagicMock() for _ in range(2)]
        judge_mocks = [MagicMock() for _ in range(2)]
        MockStriker.side_effect = striker_mocks
        MockJudge.side_effect = judge_mocks

        dashboard_ui.action_start_wargame()

        assert striker_mocks[0].name == "Worker-Striker-1"
        assert striker_mocks[1].name == "Worker-Striker-2"
        assert judge_mocks[0].name == "Worker-Judge-1"
        assert judge_mocks[1].name == "Worker-Judge-2"

    # --- 7. Context Strategy Branching ---
    @patch('asyncio.create_task')
    @patch('isomutator.ui.app.multiprocessing.Process')
    @patch('isomutator.ui.app.RedTeamJudge')
    @patch('isomutator.ui.app.AsyncStriker')
    def test_context_strategy_uses_context_mutator_factory(
        self, MockStriker, MockJudge, MockProcess, mock_create_task, dashboard_ui
    ):
        """'context' strategy must pass run_context_mutator_process as the Process target."""
        from isomutator.ingestors.context_mutator import run_context_mutator_process
        mock_create_task.return_value = MagicMock(spec=asyncio.Task)
        dashboard_ui.strategy_select.value = "context"

        dashboard_ui.action_start_wargame()

        target_used = MockProcess.call_args.kwargs['target']
        assert target_used is run_context_mutator_process

    @patch('asyncio.create_task')
    @patch('isomutator.ui.app.multiprocessing.Process')
    @patch('isomutator.ui.app.RedTeamJudge')
    @patch('isomutator.ui.app.AsyncStriker')
    def test_non_context_strategy_uses_prompt_mutator_factory(
        self, MockStriker, MockJudge, MockProcess, mock_create_task, dashboard_ui
    ):
        """Any non-context strategy must pass run_mutator_process as the Process target."""
        from isomutator.ingestors.mutator import run_mutator_process
        mock_create_task.return_value = MagicMock(spec=asyncio.Task)
        dashboard_ui.strategy_select.value = "linux_privesc"

        dashboard_ui.action_start_wargame()

        target_used = MockProcess.call_args.kwargs['target']
        assert target_used is run_mutator_process

    # --- 8. Telemetry Task ---
    @patch('asyncio.create_task')
    @patch('isomutator.ui.app.multiprocessing.Process')
    @patch('isomutator.ui.app.RedTeamJudge')
    @patch('isomutator.ui.app.AsyncStriker')
    def test_telemetry_task_created_and_registered(
        self, MockStriker, MockJudge, MockProcess, mock_create_task, dashboard_ui
    ):
        """A telemetry asyncio task is created and immediately registered with TaskWatcher."""
        mock_task = MagicMock(spec=asyncio.Task)
        mock_create_task.return_value = mock_task

        dashboard_ui.action_start_wargame()

        mock_create_task.assert_called_once()
        dashboard_ui.task_watcher.watch.assert_called_once_with(
            task=mock_task, name="TelemetryListener"
        )
        assert dashboard_ui.telemetry_task is mock_task

    # --- 9. Strategy Fallback ---
    @patch('asyncio.create_task')
    @patch('isomutator.ui.app.multiprocessing.Process')
    @patch('isomutator.ui.app.RedTeamJudge')
    @patch('isomutator.ui.app.AsyncStriker')
    def test_none_strategy_value_falls_back_to_jailbreak(
        self, MockStriker, MockJudge, MockProcess, mock_create_task, dashboard_ui
    ):
        """A None strategy_select value must not crash; JailbreakStrategy is used."""
        mock_create_task.return_value = MagicMock(spec=asyncio.Task)
        dashboard_ui.strategy_select.value = None

        # Must not raise
        dashboard_ui.action_start_wargame()

        # Judge must have been constructed with a JailbreakStrategy instance
        strategy_arg = MockJudge.call_args.kwargs['strategy']
        assert isinstance(strategy_arg, JailbreakStrategy)

    # --- 10. Mutator process kwargs ---
    @patch('asyncio.create_task')
    @patch('isomutator.ui.app.multiprocessing.Process')
    @patch('isomutator.ui.app.RedTeamJudge')
    @patch('isomutator.ui.app.AsyncStriker')
    def test_mutator_process_receives_correct_kwargs(
        self, MockStriker, MockJudge, MockProcess, mock_create_task, dashboard_ui
    ):
        """The Mutator process must receive attack_queue, feedback_queue, and strategy_name."""
        mock_create_task.return_value = MagicMock(spec=asyncio.Task)
        dashboard_ui.strategy_select.value = "prompt_leaking"

        dashboard_ui.action_start_wargame()

        kwargs_passed = MockProcess.call_args.kwargs['kwargs']
        assert 'attack_queue' in kwargs_passed
        assert 'feedback_queue' in kwargs_passed
        assert kwargs_passed['strategy_name'] == "prompt_leaking"
        assert MockProcess.call_args.kwargs['name'] == "Worker-Mutator"
