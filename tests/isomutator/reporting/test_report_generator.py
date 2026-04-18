"""
Algorithm Summary:
This test suite validates the ReportGenerator orchestrator. It ensures the class 
can asynchronously ingest JSONL ledger files, calculate accurate success metrics, 
gracefully handle corrupted data, and output polymorphic reports via the Strategy Pattern.

Protocols Covered:
1. Happy Path
2. Edge Cases (Empty files)
3. Error Handling (Corrupted JSON, Missing Files)
4. Concurrency & Memory Isolation
5. Timeout & Latency
6. Resource Teardown
7. Strict Mocking (aiofiles)
"""

import pytest
import pytest_asyncio
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from isomutator.reporting.report_generator import ReportGenerator
from isomutator.core.exceptions import ReportingError

# ---------------------------------------------------------------------------
# Fixtures & Mocks
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def report_generator():
    """Provides an isolated ReportGenerator instance."""
    generator = ReportGenerator(timeout_seconds=2.0)
    
    # Inject a mock strategy for testing
    mock_strategy = MagicMock()
    mock_strategy.generate.return_value = "MOCK_REPORT_OUTPUT"
    generator.register_strategy(name="mock_json", strategy=mock_strategy)
    
    yield generator

def create_mock_aiofile(lines):
    """Helper to generate a properly mocked aiofiles instance."""
    mock_file = AsyncMock()
    mock_file.__aiter__.return_value = lines
    mock_file.read = AsyncMock(return_value="".join(lines))
    return mock_file

# ---------------------------------------------------------------------------
# The 7-Protocol Test Suite
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReportGenerator:

    # Protocol 1 & 7: Happy Path & Strict Mocking
    @patch("aiofiles.open")
    async def test_happy_path_report_generation(self, mock_aiofiles, report_generator):
        valid_jsonl = [
            '{"strategy": "jailbreak", "success": true, "latency": 1.2}\n',
            '{"strategy": "jailbreak", "success": false, "latency": 0.8}\n',
            '{"strategy": "context", "success": true, "latency": 2.1}\n'
        ]
        mock_aiofiles.return_value.__aenter__.return_value = create_mock_aiofile(valid_jsonl)
        
        report = await report_generator.generate_report(ledger_filepath="dummy_ledger.jsonl", format_name="mock_json")
        
        # Verify file was opened
        mock_aiofiles.assert_called_once_with("dummy_ledger.jsonl", mode='r')
        
        # Verify metrics were calculated correctly inside the generator
        metrics = report_generator.last_metrics
        assert metrics["total_attacks"] == 3
        assert metrics["successful_attacks"] == 2
        assert metrics["strategies"]["jailbreak"]["attempts"] == 2
        assert metrics["strategies"]["jailbreak"]["successes"] == 1
        
        # Verify polymorphic output
        assert report == "MOCK_REPORT_OUTPUT"

    # Protocol 2: Edge Cases (Empty File)
    @patch("aiofiles.open")
    async def test_edge_case_empty_ledger(self, mock_aiofiles, report_generator):
        mock_aiofiles.return_value.__aenter__.return_value = create_mock_aiofile([])
        
        report = await report_generator.generate_report(ledger_filepath="empty.jsonl", format_name="mock_json")
        
        metrics = report_generator.last_metrics
        assert metrics["total_attacks"] == 0
        assert report == "MOCK_REPORT_OUTPUT"

    # Protocol 3: Error Handling (Corrupted JSON)
    @patch("aiofiles.open")
    async def test_error_handling_corrupted_json_lines(self, mock_aiofiles, report_generator, caplog):
        import logging
        mixed_jsonl = [
            '{"strategy": "jailbreak", "success": true}\n',
            '{corrupted_line_missing_quotes: true}\n', # Bad JSON
            '{"strategy": "context", "success": false}\n'
        ]
        mock_aiofiles.return_value.__aenter__.return_value = create_mock_aiofile(mixed_jsonl)
        
        with caplog.at_level(logging.WARNING):
            await report_generator.generate_report(ledger_filepath="mixed.jsonl", format_name="mock_json")
        
        # Generator should skip the bad line but process the valid ones
        metrics = report_generator.last_metrics
        assert metrics["total_attacks"] == 2
        assert "Failed to parse JSON on line 2" in caplog.text

    # Protocol 3: Error Handling (Missing File)
    @patch("aiofiles.open")
    async def test_error_handling_missing_file(self, mock_aiofiles, report_generator):
        mock_aiofiles.side_effect = FileNotFoundError("Ledger not found")
        
        with pytest.raises(ReportingError, match="Failed to locate ledger file"):
            await report_generator.generate_report(ledger_filepath="ghost.jsonl", format_name="mock_json")

    # Protocol 4: Concurrency & Race Conditions
    @patch("aiofiles.open")
    async def test_concurrency_report_isolation(self, mock_aiofiles):
        # We need to instantiate multiple generators to ensure no class-level state leakage
        gen1 = ReportGenerator()
        gen2 = ReportGenerator()
        
        gen1.register_strategy(name="mock_json", strategy=MagicMock(return_value="R1"))
        gen2.register_strategy(name="mock_json", strategy=MagicMock(return_value="R2"))
        
        mock_file1 = create_mock_aiofile(['{"strategy": "s1", "success": true}\n'])
        mock_file2 = create_mock_aiofile(['{"strategy": "s2", "success": false}\n' for _ in range(5)])
        
        # AsyncMock returns the files in the order they are called
        mock_aiofiles.return_value.__aenter__.side_effect = [mock_file1, mock_file2]
        
        task1 = gen1.generate_report(ledger_filepath="f1.jsonl", format_name="mock_json")
        task2 = gen2.generate_report(ledger_filepath="f2.jsonl", format_name="mock_json")
        
        await asyncio.gather(task1, task2)
        
        assert gen1.last_metrics["total_attacks"] == 1
        assert gen2.last_metrics["total_attacks"] == 5

    # Protocol 5: Timeout & Latency
    @patch("aiofiles.open")
    async def test_timeout_latency_handling(self, mock_aiofiles, report_generator, caplog):
        import logging
        
        # Create a mock file that sleeps forever when iterated
        class SlowMockFile:
            async def __aiter__(self):
                await asyncio.sleep(10.0)
                yield ""
                
        mock_aiofiles.return_value.__aenter__.return_value = SlowMockFile()
        report_generator.timeout_seconds = 0.1
        
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ReportingError, match="Report generation timed out"):
                await report_generator.generate_report(ledger_filepath="slow.jsonl", format_name="mock_json")
        
        assert "Timeout exceeded while parsing ledger" in caplog.text

    # Protocol 6: Resource Teardown
    @patch("aiofiles.open")
    async def test_resource_teardown_on_exception(self, mock_aiofiles, report_generator):
        mock_file = AsyncMock()
        # Simulate a crash during iteration
        mock_file.__aiter__.side_effect = Exception("Sudden Disk Failure")
        mock_aiofiles.return_value.__aenter__.return_value = mock_file
        
        with pytest.raises(ReportingError):
            await report_generator.generate_report(ledger_filepath="crash.jsonl", format_name="mock_json")
        
        # Ensure that despite the crash inside the loop, aiofiles context manager exited cleanly
        mock_aiofiles.return_value.__aexit__.assert_called_once()