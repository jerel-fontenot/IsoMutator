"""
Algorithm Summary:
This test suite validates the structural layout and responsive boundaries 
of the CommandDashboard UI. It ensures that the primary application wrapper 
utilizes CSS Grid for deterministic screen-space allocation (eliminating dead space) 
and verifies that the Vulnerability Ledger container enforces strict vertical 
boundaries (overflow-y-auto) to prevent layout shifting during data population.
"""

import pytest
from unittest.mock import patch, MagicMock

# Import the dashboard; the __main__ guard we added earlier prevents the server from starting
from isomutator.ui.app import CommandDashboard

@pytest.fixture
def dashboard_ui():
    """
    Provides a CommandDashboard instance with the backend Redis/Telemetry 
    components safely mocked to isolate the UI rendering logic.
    """
    with patch('isomutator.ui.app.QueueManager'), \
         patch('isomutator.ui.app.TelemetryService'), \
         patch('isomutator.ui.app.ui.timer'):
        
        dash = CommandDashboard()
        return dash

class TestCommandDashboardLayout:

    def test_main_layout_uses_css_grid(self, dashboard_ui):
        """
        Happy Path: The main wrapper utilizes a 12-column CSS Grid to ensure 
        the left sidebar and right telemetry panels consume 100% of the viewport 
        without leaving dead space.
        """
        # Arrange & Act: Handled by fixture initialization
        
        # Assert: Inspect the Tailwind classes applied to the main layout container
        layout_classes = dashboard_ui.main_layout.classes
        
        assert 'grid' in layout_classes, "Main layout must use CSS Grid."
        assert 'grid-cols-12' in layout_classes, "Main layout must be divided into 12 columns."
        assert 'w-full' in layout_classes, "Main layout must consume full width."
        assert 'h-[calc(100vh-80px)]' in layout_classes, "Main layout must consume exact viewport height minus the header."

    def test_ledger_enforces_vertical_boundaries(self, dashboard_ui):
        """
        Edge Case: To prevent an infinitely expanding table from pushing the 
        Mission Control panel off the screen, the ledger's parent container 
        must enforce an internal scrollbar via 'overflow-y-auto'.
        """
        # Arrange & Act: Handled by fixture initialization
        
        # Assert: Inspect the Tailwind classes applied to the ledger container
        ledger_classes = dashboard_ui.ledger_container.classes
        
        assert 'overflow-y-auto' in ledger_classes, "Ledger container must allow internal vertical scrolling."
        assert 'flex-grow' in ledger_classes or any(c.startswith('max-h-') for c in ledger_classes), \
            "Ledger container must have constrained height dimensions."

    def test_sidebar_and_dashboard_grid_spans(self, dashboard_ui):
        """
        Happy Path: The Configuration sidebar spans 3 columns, while the 
        Telemetry dashboard heavily favors the right side by spanning 9 columns.
        """
        sidebar_classes = dashboard_ui.sidebar_container.classes
        telemetry_classes = dashboard_ui.telemetry_container.classes
        
        assert 'col-span-3' in sidebar_classes, "Sidebar should span 3 columns."
        assert 'col-span-9' in telemetry_classes, "Telemetry Dashboard should span 9 columns."