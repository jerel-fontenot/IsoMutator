"""
Polymorphic Output Strategies for the Reporting Engine.
"""
from abc import ABC, abstractmethod
import json

class ReportStrategy(ABC):
    @abstractmethod
    def generate(self, metrics: dict) -> str:
        """Transforms the aggregated metrics into a specific output format."""
        pass

class JSONReportStrategy(ReportStrategy):
    def generate(self, metrics: dict) -> str:
        """Outputs the report as a formatted JSON string."""
        return json.dumps(metrics, indent=4)

class HTMLReportStrategy(ReportStrategy):
    def generate(self, metrics: dict) -> str:
        """Outputs the report as a styled HTML dashboard."""
        
        # Calculate global success rate
        total = metrics.get("total_attacks", 0)
        successes = metrics.get("successful_attacks", 0)
        global_rate = (successes / total * 100) if total > 0 else 0.0

        # Build Strategy Rows
        strategy_rows = ""
        for strat_name, stats in metrics.get("strategies", {}).items():
            s_attempts = stats.get("attempts", 0)
            s_successes = stats.get("successes", 0)
            s_rate = (s_successes / s_attempts * 100) if s_attempts > 0 else 0.0
            
            strategy_rows += f"""
            <tr>
                <td>{strat_name.replace('_', ' ').title()}</td>
                <td>{s_attempts}</td>
                <td>{s_successes}</td>
                <td>{s_rate:.1f}%</td>
            </tr>
            """

        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>IsoMutator Wargame Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0d1117; color: #c9d1d9; padding: 2rem; }}
                h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }}
                .summary {{ display: flex; gap: 2rem; margin-bottom: 2rem; }}
                .stat-box {{ background-color: #161b22; border: 1px solid #30363d; padding: 1.5rem; border-radius: 8px; text-align: center; min-width: 150px; }}
                .stat-value {{ font-size: 2rem; font-weight: bold; color: #79c0ff; margin-bottom: 0.5rem; }}
                .stat-label {{ font-size: 0.9rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }}
                table {{ width: 100%; border-collapse: collapse; background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
                th, td {{ padding: 1rem; text-align: left; border-bottom: 1px solid #30363d; }}
                th {{ background-color: #21262d; color: #c9d1d9; font-weight: 600; }}
                tr:last-child td {{ border-bottom: none; }}
            </style>
        </head>
        <body>
            <h1>IsoMutator: Red Team Wargame Report</h1>
            
            <div class="summary">
                <div class="stat-box">
                    <div class="stat-value">{total}</div>
                    <div class="stat-label">Total Attacks</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{successes}</div>
                    <div class="stat-label">Successful Bypasses</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{global_rate:.1f}%</div>
                    <div class="stat-label">Global Success Rate</div>
                </div>
            </div>

            <h2>Strategy Breakdown</h2>
            <table>
                <thead>
                    <tr>
                        <th>Strategy Vector</th>
                        <th>Attempts</th>
                        <th>Successes</th>
                        <th>Success Rate</th>
                    </tr>
                </thead>
                <tbody>
                    {strategy_rows}
                </tbody>
            </table>
        </body>
        </html>
        """
        return html_template