"""
The Data Layer orchestrator for parsing telemetry ledgers and generating reports.
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional

import aiofiles

from isomutator.core.exceptions import ReportingError
from isomutator.reporting.strategies import ReportStrategy, JSONReportStrategy, HTMLReportStrategy

class ReportGenerator:
    """
    Asynchronously ingests JSONL wargame ledgers and generates structured reports
    using the Strategy Pattern for polymorphic output formats.
    """

    def __init__(self, timeout_seconds: float = 30.0):
        self.logger = logging.getLogger(__name__)
        self.timeout_seconds = timeout_seconds
        
        # State tracking for testability
        self.last_metrics: Dict[str, Any] = {}
        
        # Strategy Registry
        self._strategies: Dict[str, ReportStrategy] = {
            "json": JSONReportStrategy(),
            "html": HTMLReportStrategy()
        }

    def register_strategy(self, *, name: str, strategy: ReportStrategy) -> None:
        """Dynamically registers a new reporting strategy format."""
        self._strategies[name] = strategy

    async def generate_report(self, *, ledger_filepath: str, format_name: str = "json") -> str:
        """
        Main execution pipeline. Reads the file, aggregates data, and formats the output.
        """
        strategy = self._strategies.get(format_name)
        if not strategy:
            raise ReportingError(f"Reporting format '{format_name}' is not registered.")

        try:
            # Enforce latency boundaries via wait_for
            metrics = await asyncio.wait_for(
                self._parse_ledger(filepath=ledger_filepath),
                timeout=self.timeout_seconds
            )
            self.last_metrics = metrics
            
            # Delegate to the polymorphic strategy
            return strategy.generate(metrics)

        except asyncio.TimeoutError as e:
            self.logger.error(f"Timeout exceeded while parsing ledger: {ledger_filepath}")
            raise ReportingError("Report generation timed out") from e

    async def _parse_ledger(self, *, filepath: str) -> Dict[str, Any]:
        """
        Asynchronously reads the JSONL file and aggregates success metrics.
        Defensively handles corrupted lines without crashing.
        """
        metrics = {
            "total_attacks": 0,
            "successful_attacks": 0,
            "strategies": {}
        }

        try:
            async with aiofiles.open(filepath, mode='r') as f:
                line_number = 0
                async for line in f:
                    line_number += 1
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        entry = json.loads(line)
                        self._update_metrics(metrics=metrics, entry=entry)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse JSON on line {line_number} of {filepath}. Skipping.")
                        continue
                        
        except FileNotFoundError as e:
            self.logger.error(f"Failed to locate ledger file: {filepath}")
            raise ReportingError(f"Failed to locate ledger file: {filepath}") from e
        except Exception as e:
            # Protocol 6: aiofiles context manager guarantees safe teardown even on fatal errors
            self.logger.error(f"Unexpected error parsing ledger: {str(e)}")
            raise ReportingError(f"Failed to parse ledger: {str(e)}") from e

        return metrics

    def _update_metrics(self, *, metrics: Dict[str, Any], entry: Dict[str, Any]) -> None:
        """Helper to mutate the metrics dictionary based on a single ledger entry."""
        strategy_name = entry.get("strategy", "unknown")
        success = entry.get("success", False)

        metrics["total_attacks"] += 1
        if success:
            metrics["successful_attacks"] += 1

        if strategy_name not in metrics["strategies"]:
            metrics["strategies"][strategy_name] = {
                "attempts": 0,
                "successes": 0
            }

        metrics["strategies"][strategy_name]["attempts"] += 1
        if success:
            metrics["strategies"][strategy_name]["successes"] += 1