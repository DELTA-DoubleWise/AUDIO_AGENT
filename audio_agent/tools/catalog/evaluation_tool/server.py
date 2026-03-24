#!/usr/bin/env python3
"""
Evaluation Tool MCP Server

Provides RPS calculation and tool evaluation capabilities via MCP.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


# Try to import from audio_agent, fallback to standalone implementation
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from audio_agent.evaluation.core.rps import RPSCalculator, RPSRegistry
except ImportError:
    # Standalone mode - implement minimal RPS functionality
    from dataclasses import dataclass, field
    from datetime import datetime
    
    @dataclass
    class SOTARecord:
        dataset: str
        score: float
        metric: str
        higher_is_better: bool = False
    
    @dataclass 
    class ToolPerformance:
        tool_name: str
        dataset: str
        score: float
        metric: str
        rps: float | None = None
        metadata: dict = field(default_factory=dict)
        timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    class RPSCalculator:
        DEFAULT_SOTA = {
            "aishell1": SOTARecord("aishell1", 0.80, "cer", False),
            "aishell5": SOTARecord("aishell5", 24.74, "cer", False),
            "cs_dialogue": SOTARecord("cs_dialogue", 7.00, "mer", False),
            "kespeech": SOTARecord("kespeech", 3.81, "cer", False),
            "voxpopuli_en": SOTARecord("voxpopuli_en", 6.72, "wer", False),
            "contextasr_en": SOTARecord("contextasr_en", 3.47, "wer", False),
            "contextasr_zh": SOTARecord("contextasr_zh", 2.50, "cer", False),
            "librispeech_clean": SOTARecord("librispeech_clean", 1.70, "wer", False),
            "librispeech_gr": SOTARecord("librispeech_gr", 92.02, "accuracy", True),
            "covost2_en2zh": SOTARecord("covost2_en2zh", 46.25, "bleu", True),
            "covost2_zh2en": SOTARecord("covost2_zh2en", 60.14, "bleu", True),
            "iemocap": SOTARecord("iemocap", 69.38, "accuracy", True),
            "mmsu_reason": SOTARecord("mmsu_reason", 89.07, "accuracy", True),
        }
        
        def __init__(self):
            self._sota = dict(self.DEFAULT_SOTA)
        
        def calculate(self, dataset: str, score: float) -> float | None:
            sota = self._sota.get(dataset)
            if not sota:
                return None
            if sota.higher_is_better:
                return score / sota.score if sota.score > 0 else None
            else:
                return sota.score / score if score > 0 else None
        
        def get_sota(self, dataset: str) -> SOTARecord | None:
            return self._sota.get(dataset)
        
        def list_datasets(self) -> list[str]:
            return list(self._sota.keys())
    
    class RPSRegistry:
        def __init__(self, db_path: Path | None = None):
            self._db_path = db_path or Path("evaluation_results.json")
            self._calculator = RPSCalculator()
            self._performances: dict = {}
            self._load()
        
        def record(self, tool_name: str, dataset: str, score: float, metric: str = "auto"):
            if metric == "auto":
                sota = self._calculator.get_sota(dataset)
                metric = sota.metric if sota else "unknown"
            rps = self._calculator.calculate(dataset, score)
            record = ToolPerformance(tool_name, dataset, score, metric, rps)
            key = f"{tool_name}:{dataset}"
            if key not in self._performances:
                self._performances[key] = []
            self._performances[key].append(record)
            self._save()
            return record
        
        def get_best_tool(self, dataset: str) -> str | None:
            best_tool = None
            best_rps = 0.0
            for key, performances in self._performances.items():
                tool_name = key.split(":")[0]
                for perf in performances:
                    if perf.dataset == dataset and perf.rps is not None:
                        if perf.rps > best_rps:
                            best_rps = perf.rps
                            best_tool = tool_name
            return best_tool
        
        def get_tool_rps(self, tool_name: str, dataset: str) -> float | None:
            key = f"{tool_name}:{dataset}"
            performances = self._performances.get(key, [])
            if performances:
                return performances[-1].rps
            return None
        
        def _load(self):
            if not self._db_path.exists():
                return
            try:
                import json
                with open(self._db_path, "r") as f:
                    data = json.load(f)
                for key, records in data.items():
                    self._performances[key] = [ToolPerformance(**r) for r in records]
            except Exception:
                pass
        
        def _save(self):
            try:
                import json
                data = {
                    key: [
                        {
                            "tool_name": p.tool_name,
                            "dataset": p.dataset,
                            "score": p.score,
                            "metric": p.metric,
                            "rps": p.rps,
                            "metadata": p.metadata,
                            "timestamp": p.timestamp,
                        }
                        for p in performances
                    ]
                    for key, performances in self._performances.items()
                }
                with open(self._db_path, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass


class EvaluationToolServer:
    """MCP Server for evaluation tools."""
    
    def __init__(self):
        self._initialized = False
        self._calculator = RPSCalculator()
        
        # Initialize registry
        db_path = os.environ.get("RESULTS_DB", "./evaluation_results.json")
        self._registry = RPSRegistry(Path(db_path))
        
        # Tool definitions
        self._tools = [
            {
                "name": "calculate_rps",
                "description": "Calculate Relative Performance Score (RPS) for a tool's performance on a dataset. RPS = 1.0 matches SOTA, RPS > 1.0 beats SOTA, RPS < 1.0 is below SOTA.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset": {
                            "type": "string",
                            "description": "Dataset name (e.g., 'aishell1', 'librispeech_clean')"
                        },
                        "score": {
                            "type": "number",
                            "description": "Tool's performance score"
                        },
                        "metric": {
                            "type": "string",
                            "description": "Metric type (cer, wer, accuracy, bleu). Auto-detected if not provided.",
                            "default": "auto"
                        }
                    },
                    "required": ["dataset", "score"]
                }
            },
            {
                "name": "record_performance",
                "description": "Record a tool's performance on a dataset and calculate/store RPS",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the tool"
                        },
                        "dataset": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "score": {
                            "type": "number",
                            "description": "Performance score"
                        },
                        "metric": {
                            "type": "string",
                            "description": "Metric type (auto-detected from dataset if not provided)",
                            "default": "auto"
                        }
                    },
                    "required": ["tool_name", "dataset", "score"]
                }
            },
            {
                "name": "recommend_tool",
                "description": "Recommend the best tool for a specific dataset based on RPS scores",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "task": {
                            "type": "string",
                            "description": "Task type filter (e.g., 'ASR', 'SER')",
                            "default": ""
                        }
                    },
                    "required": ["dataset"]
                }
            },
            {
                "name": "list_datasets",
                "description": "List all available benchmark datasets with SOTA information",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_sota",
                "description": "Get SOTA information for a specific dataset",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset": {
                            "type": "string",
                            "description": "Dataset name"
                        }
                    },
                    "required": ["dataset"]
                }
            }
        ]
    
    def run(self) -> None:
        """Run the server."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                request = json.loads(line)
                response = self._handle_request(request)
                if response:
                    self._send_response(response)
            except json.JSONDecodeError as e:
                self._send_error(None, -32700, f"Parse error: {e}")
            except Exception as e:
                request_id = request.get("id") if isinstance(request, dict) else None
                self._send_error(request_id, -32603, f"Internal error: {e}")
    
    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle JSON-RPC request."""
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params", {})
        
        if method == "initialize":
            return self._handle_initialize(request_id, params)
        elif method == "notifications/initialized":
            return None
        elif method == "tools/list":
            return self._handle_tools_list(request_id)
        elif method == "tools/call":
            return self._handle_tools_call(request_id, params)
        elif method == "shutdown":
            return self._handle_shutdown(request_id)
        else:
            return self._error_response(request_id, -32601, f"Method not found: {method}")
    
    def _handle_initialize(self, request_id: Any, params: dict) -> dict[str, Any]:
        """Handle initialize request."""
        self._initialized = True
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "evaluation-tool-server",
                    "version": "1.0.0"
                }
            }
        }
    
    def _handle_tools_list(self, request_id: Any) -> dict[str, Any]:
        """Handle tools/list request."""
        if not self._initialized:
            return self._error_response(request_id, -32001, "Server not initialized")
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": self._tools}
        }
    
    def _handle_tools_call(self, request_id: Any, params: dict) -> dict[str, Any]:
        """Handle tools/call request."""
        if not self._initialized:
            return self._error_response(request_id, -32001, "Server not initialized")
        
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            result = self._execute_tool(tool_name, arguments)
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as e:
            error_msg = str(e)
            print(f"Tool execution error: {error_msg}", file=sys.stderr)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {error_msg}"}],
                    "isError": True,
                    "error": error_msg
                }
            }
    
    def _execute_tool(self, tool_name: str, arguments: dict) -> dict[str, Any]:
        """Execute a tool."""
        if tool_name == "calculate_rps":
            return self._calculate_rps(arguments)
        elif tool_name == "record_performance":
            return self._record_performance(arguments)
        elif tool_name == "recommend_tool":
            return self._recommend_tool(arguments)
        elif tool_name == "list_datasets":
            return self._list_datasets(arguments)
        elif tool_name == "get_sota":
            return self._get_sota(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _calculate_rps(self, arguments: dict) -> dict[str, Any]:
        """Calculate RPS for a score."""
        dataset = arguments.get("dataset")
        score = arguments.get("score")
        metric = arguments.get("metric", "auto")
        
        if not dataset or score is None:
            raise ValueError("dataset and score are required")
        
        rps = self._calculator.calculate(dataset, score)
        sota = self._calculator.get_sota(dataset)
        
        if rps is None:
            text = f"No SOTA found for dataset: {dataset}"
        else:
            interpretation = ""
            if rps >= 1.0:
                interpretation = "matches or beats SOTA"
            else:
                interpretation = "below SOTA"
            
            text = (
                f"Dataset: {dataset}\n"
                f"Your score: {score:.4f} {sota.metric if sota else metric}\n"
                f"SOTA: {sota.score:.4f} {sota.metric if sota else metric}\n"
                f"RPS: {rps:.4f} ({interpretation})\n\n"
                f"RPS interpretation:\n"
                f"  RPS = 1.0: Matches SOTA\n"
                f"  RPS > 1.0: Beats SOTA\n"
                f"  RPS < 1.0: Below SOTA"
            )
        
        return {
            "content": [{"type": "text", "text": text}],
            "isError": False,
            "rps": rps,
            "dataset": dataset,
            "score": score
        }
    
    def _record_performance(self, arguments: dict) -> dict[str, Any]:
        """Record tool performance."""
        tool_name = arguments.get("tool_name")
        dataset = arguments.get("dataset")
        score = arguments.get("score")
        metric = arguments.get("metric", "auto")
        
        if not all([tool_name, dataset, score is not None]):
            raise ValueError("tool_name, dataset, and score are required")
        
        # Auto-detect metric if needed
        if metric == "auto":
            sota = self._calculator.get_sota(dataset)
            metric = sota.metric if sota else "unknown"
        
        # Record in registry
        record = self._registry.record(tool_name, dataset, score, metric)
        
        text = (
            f"Recorded performance for {tool_name} on {dataset}:\n"
            f"  Score: {score:.4f} {metric}\n"
            f"  RPS: {record.rps:.4f if record.rps else 'N/A'}"
        )
        
        return {
            "content": [{"type": "text", "text": text}],
            "isError": False,
            "rps": record.rps,
            "tool_name": tool_name,
            "dataset": dataset
        }
    
    def _recommend_tool(self, arguments: dict) -> dict[str, Any]:
        """Recommend best tool for a dataset."""
        dataset = arguments.get("dataset")
        task = arguments.get("task", "")
        
        if not dataset:
            raise ValueError("dataset is required")
        
        best_tool = self._registry.get_best_tool(dataset)
        
        if best_tool:
            rps = self._registry.get_tool_rps(best_tool, dataset)
            text = (
                f"Recommended tool for {dataset}:\n"
                f"  Tool: {best_tool}\n"
                f"  RPS: {rps:.4f if rps else 'N/A'}\n\n"
                f"This tool has the highest RPS score on this dataset."
            )
        else:
            # Fallback to SOTA info
            sota = self._calculator.get_sota(dataset)
            if sota:
                text = (
                    f"No performance records found for {dataset}.\n\n"
                    f"SOTA baseline:\n"
                    f"  Score: {sota.score:.4f} {sota.metric}\n\n"
                    f"Evaluate your tool and record performance to get recommendations."
                )
            else:
                text = f"Unknown dataset: {dataset}"
        
        return {
            "content": [{"type": "text", "text": text}],
            "isError": False,
            "recommended_tool": best_tool
        }
    
    def _list_datasets(self, arguments: dict) -> dict[str, Any]:
        """List all available datasets."""
        datasets = self._calculator.list_datasets()
        
        lines = ["Available benchmark datasets:", ""]
        for dataset in sorted(datasets):
            sota = self._calculator.get_sota(dataset)
            if sota:
                lines.append(f"  {dataset}: SOTA = {sota.score:.4f} {sota.metric}")
        
        text = "\n".join(lines)
        
        return {
            "content": [{"type": "text", "text": text}],
            "isError": False,
            "datasets": datasets
        }
    
    def _get_sota(self, arguments: dict) -> dict[str, Any]:
        """Get SOTA info for a dataset."""
        dataset = arguments.get("dataset")
        
        if not dataset:
            raise ValueError("dataset is required")
        
        sota = self._calculator.get_sota(dataset)
        
        if sota:
            text = (
                f"SOTA for {dataset}:\n"
                f"  Score: {sota.score:.4f} {sota.metric}\n"
                f"  Higher is better: {sota.higher_is_better}"
            )
        else:
            text = f"Unknown dataset: {dataset}"
        
        return {
            "content": [{"type": "text", "text": text}],
            "isError": False,
            "sota": {
                "score": sota.score if sota else None,
                "metric": sota.metric if sota else None,
                "higher_is_better": sota.higher_is_better if sota else None
            } if sota else None
        }
    
    def _handle_shutdown(self, request_id: Any) -> dict[str, Any]:
        """Handle shutdown request."""
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    
    def _error_response(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message}
        }
    
    def _send_response(self, response: dict[str, Any]) -> None:
        """Send response to stdout."""
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    
    def _send_error(self, request_id: Any, code: int, message: str) -> None:
        """Send error response."""
        self._send_response(self._error_response(request_id, code, message))


def main():
    """Main entry point."""
    # Ensure unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    server = EvaluationToolServer()
    server.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
