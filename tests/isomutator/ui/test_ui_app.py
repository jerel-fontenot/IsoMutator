import pytest
from unittest.mock import patch

from textual.widgets import Select, Input
from isomutator.ui.app import IsoMutatorApp
from isomutator.core.config import settings

@pytest.mark.asyncio
async def test_ui_initializes_with_validators_and_paths():
    """Ensures the GUI inputs and new path inputs reflect backend settings."""
    app = IsoMutatorApp()
    
    async with app.run_test(size=(120, 40)) as pilot:
        # Check Target URL
        assert app.query_one("#target_url_input", Input).value == settings.target_url
        
        # Check validated inputs instead of sliders
        batch_input = app.query_one("#batch_size_input", Input)
        assert batch_input.value == str(settings.batch_size)
        
        ping_input = app.query_one("#ping_pong_input", Input)
        assert ping_input.value == str(settings.ping_pong_delay)
        
        # Check new paths
        assert app.query_one("#report_path_input", Input).value == "reports/final_report.json"
        
        # Ensure the strategy dropdown has all our backend options
        strategy_select = app.query_one("#strategy_select", Select)
        options = [opt[1] for opt in strategy_select._options]
        assert "obfuscation" in options
        assert "linux_privesc" in options
        assert "context" in options

@pytest.mark.asyncio
async def test_ui_reactive_context_file_input():
    """Ensures the Context Payload File input only unlocks for Context strategies."""
    app = IsoMutatorApp()
    
    async with app.run_test(size=(120, 40)) as pilot:
        strategy_select = app.query_one("#strategy_select", Select)
        context_file_input = app.query_one("#context_file_input", Input)
        
        # By default (prompt_leaking), it should be disabled
        assert context_file_input.disabled is True
        
        # Simulate selecting "context"
        strategy_select.value = "context"
        await pilot.pause() # Allow the reactive UI to update
        
        # It should now be unlocked for user input
        assert context_file_input.disabled is False
        
        # Simulate reverting to a non-context strategy
        strategy_select.value = "jailbreak"
        await pilot.pause()
        
        # It should lock again
        assert context_file_input.disabled is True

@pytest.mark.asyncio
async def test_ui_start_button_locks_configuration():
    """Ensures clicking Start disables the entire configuration panel."""
    app = IsoMutatorApp()
    
    with patch("isomutator.ui.app.QueueManager"):
        with patch("isomutator.ui.app.multiprocessing.Process.start"):
            
            async with app.run_test(size=(120, 40)) as pilot:
                start_button = app.query_one("#start_button")
                batch_input = app.query_one("#batch_size_input", Input)
                report_input = app.query_one("#report_path_input", Input)
                
                assert batch_input.disabled is False
                assert report_input.disabled is False
                
                start_button.press()
                await pilot.pause()
                
                # Verify all new inputs are locked
                assert batch_input.disabled is True
                assert report_input.disabled is True
                assert start_button.disabled is True