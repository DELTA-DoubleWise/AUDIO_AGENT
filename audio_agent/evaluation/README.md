# Audio Agent Evaluation Module

集成自 `sure-eval` 的评估系统，为 AUDIO_AGENT 提供工具评估和 RPS-based 选择能力。

## 概述

本模块提供以下功能：

1. **RPS 计算** - 相对于 SOTA 的性能评分
2. **工具评估** - 在基准数据集上评估工具
3. **工具推荐** - 基于 RPS 选择最佳工具
4. **适配器** - 与 audio_agent.tools 无缝集成

## 快速开始

```python
from audio_agent.evaluation import (
    RPSCalculator,
    RPSRegistry, 
    ToolEvaluator,
    adapt_tool_for_evaluation,
)

# 计算 RPS
calc = RPSCalculator()
rps = calc.calculate("aishell1", score=0.85)
print(f"RPS: {rps:.3f}")  # RPS: 0.941

# 记录工具性能
registry = RPSRegistry()
registry.record("my-tool", "aishell1", score=0.85)

# 推荐最佳工具
best_tool = registry.get_best_tool("aishell1")
print(f"Best tool: {best_tool}")
```

## 与 audio_agent.tools 集成

```python
from audio_agent.tools.dummy_tools import DummyASRTool
from audio_agent.evaluation import adapt_tool_for_evaluation, ToolEvaluator

# 包装工具以进行评估
tool = DummyASRTool()
evaluated = adapt_tool_for_evaluation(tool)

# 使用 ToolEvaluator
evaluator = ToolEvaluator()
results = evaluator.evaluate_dataset(evaluated, "aishell1", max_samples=10)
```

## 项目结构

```
audio_agent/evaluation/
├── __init__.py          # 导出接口
├── README.md            # 本文档
├── demo.py              # 演示脚本
├── adapter.py           # 工具适配器
└── core/
    ├── __init__.py
    ├── rps.py           # RPS 计算和注册表
    └── evaluator.py     # 工具评估器
```

## 支持的基准数据集

| 数据集 | 任务 | 语言 | SOTA |
|--------|------|------|------|
| aishell1 | ASR | zh | 0.80 CER |
| aishell5 | ASR | zh | 24.74 CER |
| cs_dialogue | ASR | cs | 7.00 MER |
| kespeech | ASR | zh | 3.81 CER |
| voxpopuli_en | ASR | en | 6.72 WER |
| contextasr_en | ASR | en | 3.47 WER |
| contextasr_zh | ASR | zh | 2.50 CER |
| librispeech_clean | ASR | en | 1.70 WER |
| librispeech_gr | GR | en | 92.02 Acc |
| covost2_en2zh | S2TT | zh | 46.25 BLEU |
| covost2_zh2en | S2TT | en | 60.14 BLEU |
| iemocap | SER | en | 69.38 Acc |
| mmsu_reason | SLU | en | 89.07 Acc |

## RPS 计算说明

```
误差指标 (CER, WER): RPS = SOTA / score
准确率指标 (Acc, BLEU): RPS = score / SOTA
```

- **RPS = 1.0**: 与 SOTA 持平
- **RPS > 1.0**: 超越 SOTA
- **RPS < 1.0**: 低于 SOTA

## 运行演示

```bash
cd /cpfs/user/jingpeng/workspace/AUDIO_AGENT
python audio_agent/evaluation/demo.py
```

## 代码风格

遵循 AUDIO_AGENT 规范：
- Python 3.11+ 类型注解
- Pydantic v2 进行数据验证
- 结构化日志记录
- 清晰的错误处理
