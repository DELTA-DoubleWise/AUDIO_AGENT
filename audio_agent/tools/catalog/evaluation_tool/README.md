# Evaluation Tool

Tool evaluation and RPS-based tool selection for AUDIO_AGENT.

## Features

- **RPS Calculation**: Calculate Relative Performance Score vs SOTA
- **Performance Recording**: Record and persist tool performance metrics
- **Tool Recommendation**: Recommend best tool for specific datasets
- **Dataset Information**: Query SOTA baselines for benchmark datasets

## Available Tools

### calculate_rps
Calculate RPS for a tool's performance on a dataset.

```python
{
    "dataset": "aishell1",
    "score": 0.85,
    "metric": "cer"  # optional, auto-detected
}
```

RPS interpretation:
- RPS = 1.0: Matches SOTA
- RPS > 1.0: Beats SOTA
- RPS < 1.0: Below SOTA

### record_performance
Record a tool's performance and calculate/store RPS.

```python
{
    "tool_name": "my-asr-model",
    "dataset": "aishell1",
    "score": 0.85,
    "metric": "cer"  # optional
}
```

### recommend_tool
Recommend the best tool for a dataset based on RPS scores.

```python
{
    "dataset": "aishell1",
    "task": "ASR"  # optional filter
}
```

### list_datasets
List all available benchmark datasets with SOTA information.

### get_sota
Get SOTA information for a specific dataset.

```python
{
    "dataset": "aishell1"
}
```

## Supported Datasets

| Dataset | Task | SOTA | Metric |
|---------|------|------|--------|
| aishell1 | ASR | 0.80 | CER |
| aishell5 | ASR | 24.74 | CER |
| cs_dialogue | ASR | 7.00 | MER |
| kespeech | ASR | 3.81 | CER |
| voxpopuli_en | ASR | 6.72 | WER |
| contextasr_en | ASR | 3.47 | WER |
| contextasr_zh | ASR | 2.50 | CER |
| librispeech_clean | ASR | 1.70 | WER |
| librispeech_gr | GR | 92.02 | Accuracy |
| covost2_en2zh | S2TT | 46.25 | BLEU |
| covost2_zh2en | S2TT | 60.14 | BLEU |
| iemocap | SER | 69.38 | Accuracy |
| mmsu_reason | SLU | 89.07 | Accuracy |

## Usage

The tool is automatically loaded when using `register_all_mcp_tools()`:

```python
from audio_agent.tools.catalog import register_all_mcp_tools
from audio_agent.tools.registry import ToolRegistry

registry = ToolRegistry()
await register_all_mcp_tools(registry)
```

Or manually:

```python
from audio_agent.tools.catalog.loader import load_mcp_server_config
from audio_agent.tools.mcp import MCPServerManager, MCPToolAdapter

server_manager = MCPServerManager()
config = load_mcp_server_config("evaluation_tool")
server_manager.register_config("evaluation_tool", config)
```
