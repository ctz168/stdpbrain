# 训练、优化和测评流程说明

## 概述

本文档描述了类人脑 AI 架构的完整训练、优化和测评流程。由于当前使用简化模型 (SimpleLanguageModel) 进行测试，实际部署时应替换为真实 Qwen3.5-0.8B 模型。

## 环境准备

```bash
# 激活 conda 环境
conda activate stdpbrain

# 确认环境
python --version  # Python3.11.10
python-c "import torch; print(torch.__version__)"  # PyTorch 2.5.1
```

## 训练流程

### 1. 预训练适配器训练

**目的**: 将基础模型适配到特定任务域

**代码示例**:
```python
from training.pretrain_adapter import PretrainAdapter
from core.interfaces_working import SimpleLanguageModel
from configs.arch_config import default_config

model = SimpleLanguageModel()
adapter= PretrainAdapter(model, config=default_config)

train_data = {
    'samples': [
        {'input': '你好', 'output': '你好！我是类人脑 AI 助手'},
        {'input': '介绍你自己', 'output': '我基于 Qwen3.5-0.8B，具有海马体记忆系统'},
        # ... 更多样本
    ]
}

metrics = adapter.train(
    datasets=train_data,
    learning_rate=1e-5,
    batch_size=2,
    epochs=3
)
```

**预期输出**:
- 训练损失曲线
- 准确率指标
- 模型检查点保存

### 2. 在线学习优化 (STDP)

**目的**: 通过 STDP 机制实现实时学习

**代码示例**:
```python
from core.stdp_engine import STDPEngine
from hippocampus.hippocampus_system import HippocampusSystem
import torch
import time

stdp = STDPEngine(config=default_config, device='cpu')
hippo = HippocampusSystem(config=default_config, device='cpu')

# 模拟交互会话
for i in range(10):
   # 编码新记忆
    features = torch.randn(1, 768)
    hippo.encode(
        features=features,
        token_id=i,
        timestamp=int(time.time() * 1000),
        context=[{'user_input': f'输入{i}'}]
    )
    
    # STDP 权重更新
    stdp.step(
       model_components={'attention': None, 'ffn': None},
        inputs={'context_tokens': [i], 'current_token': i, 'features': features},
        outputs={'attention_output': torch.randn(1, 768), 'ffn_output': torch.randn(1, 768)},
        timestamp=time.time() * 1000
    )

print(f"STDP 周期数：{stdp.get_stats()['cycle_count']}")
print(f"记忆数量：{hippo.get_stats()['num_memories']}")
```

**预期效果**:
- 突触权重根据时序调整
- 重要记忆被强化
- 无关连接被修剪

### 3. 离线记忆巩固

**目的**: 在空闲时段巩固重要记忆

**代码示例**:
```python
from training.offline_consolidation import OfflineConsolidation

consolidation = OfflineConsolidation(
   model=model,
    hippocampus=hippo,
    stdp_engine=stdp,
    config=default_config
)

result = consolidation.consolidate()
print(f"巩固记忆数：{result['num_memories_consolidated']}")
print(f"修剪弱记忆：{result['num_weak_memories_pruned']}")
```

**触发条件**:
- CPU 使用率低于阈值 (默认<20%)
- 距离上次巩固超过时间间隔 (默认>1 小时)

## 评估流程

### 1. 基础能力评估

```python
from evaluation.base_capability_eval import BaseCapabilityEvaluator

evaluator = BaseCapabilityEvaluator(ai_interface=None)
score = evaluator.evaluate()
detailed = evaluator.evaluate_detailed()

print(f"基础能力得分：{score:.3f}")
# 详细维度:
# - conversation: 通用对话
# - instruction_follow: 指令遵循
# - semantic_understanding: 语义理解
# - chinese_capability: 中文处理
```

### 2. 推理能力评估

```python
from evaluation.reasoning_eval import ReasoningEvaluator

evaluator = ReasoningEvaluator(ai_interface=None)
score = evaluator.evaluate()
detailed = evaluator.evaluate_detailed()

print(f"推理能力得分：{score:.3f}")
# 详细维度:
# - logical_reasoning: 逻辑推理
# - mathematical: 数学计算
# - causal_inference: 因果推断
# - analogical: 类比推理
```

### 3. 海马体能力评估

```python
from evaluation.hippocampus_eval import HippocampusEvaluator

evaluator = HippocampusEvaluator(ai_interface=None)
score = evaluator.evaluate()
detailed = evaluator.evaluate_detailed()

print(f"海马体能力得分：{score:.3f}")
# 详细维度:
# - episodic_recall: 情景记忆召回 (≥95%)
# - pattern_separation: 模式分离 (≤3% 混淆率)
# - long_sequence_retention: 长时序保持 (≥90%)
# - pattern_completion: 模式补全 (≥85%)
# - anti_forgetting: 抗遗忘 (≥95%)
# - cross_session_learning: 跨会话学习 (≥90%)
```

### 4. 自闭环优化评估

```python
from evaluation.self_loop_eval import SelfLoopEvaluator

evaluator = SelfLoopEvaluator(ai_interface=None)
score = evaluator.evaluate()
detailed = evaluator.evaluate_detailed()

print(f"自闭环优化得分：{score:.3f}")
# 详细维度:
# - mode1_self_combine: 自组合质量
# - mode2_self_game: 自博弈提升
# - mode3_self_eval: 自评估准确率
# - mode_switching: 模式切换正确性
```

## 综合评分标准

| 维度 | 权重 | 合格线 | 优秀线 |
|------|------|--------|--------|
| 基础能力 | 25% | 0.95 | 0.98 |
| 推理能力 | 25% | 0.60 | 0.75 |
| 海马体能力 | 25% | 0.90 | 0.95 |
| 自闭环优化 | 25% | 0.90 | 0.95 |

**综合评分** = Σ(各维度得分 × 权重)

## 运行完整流程

创建测试脚本 `run_full_pipeline.py`:

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/hilbert/Desktop/stdpbrian')

# 1. 准备
import torch
from configs.arch_config import default_config
print("Environment ready")

# 2. 训练
from training.pretrain_adapter import PretrainAdapter
from core.interfaces_working import SimpleLanguageModel
model = SimpleLanguageModel()
adapter= PretrainAdapter(model, config=default_config)
# adapter.train(...)

# 3. 在线学习
from core.stdp_engine import STDPEngine
from hippocampus.hippocampus_system import HippocampusSystem
stdp = STDPEngine(config=default_config, device='cpu')
hippo = HippocampusSystem(config=default_config, device='cpu')

# 4. 评估
from evaluation.base_capability_eval import BaseCapabilityEvaluator
from evaluation.hippocampus_eval import HippocampusEvaluator
base_eval = BaseCapabilityEvaluator(ai_interface=None)
hippo_eval = HippocampusEvaluator(ai_interface=None)

print(f"Base Score: {base_eval.evaluate():.3f}")
print(f"Hippocampus Score: {hippo_eval.evaluate():.3f}")
```

运行:
```bash
/opt/anaconda3/envs/stdpbrain/bin/python run_full_pipeline.py
```

## 输出文件

训练和评估结果保存在以下位置:

- `outputs/training/` - 训练相关输出
  - `pretrain_result.json` - 预训练结果
  - `online_learning_result.json` - 在线学习统计
  - `offline_consolidation_result.json` - 记忆巩固结果

- `outputs/evaluation/` - 评估相关输出
  - `comprehensive_evaluation.json` - 综合评估详情
  - `reasoning_eval_result.json` - 推理能力评估
  - `hippocampus_eval_result.json` - 海马体能力评估

- `outputs/final_report.json` - 最终综合报告
- `outputs/final_report.txt` - 文本格式报告

## 性能基准

基于 Qwen3.5-0.8B 的预期性能:

| 指标 | 目标值 | 实测值 |
|------|--------|--------|
| 显存占用 | ≤420MB | - |
| 首 token 延迟 | ≤10ms | - |
| 基础能力保持率 | ≥95% | - |
| 推理能力提升 | ≥60% | - |
| 海马体召回率 | ≥95% | - |
| STDP 更新频率 | 100Hz | - |

## 故障排查

### 常见问题

1. **导入错误**
   ```
   ModuleNotFoundError: No module named 'xxx'
   ```
   解决：确保已激活正确的 conda 环境 (`conda activate stdpbrain`)

2. **CUDA 不可用**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决：使用 CPU 模式或减小 batch size

3. **特征维度不匹配**
   ```
   RuntimeError: mat1 and mat2 shapes cannot be multiplied
   ```
   解决：统一特征维度为 768 或模型配置的 hidden_size

## 下一步优化建议

1. **数据增强**: 扩大训练数据集，增加多样性
2. **超参数调优**: 学习率、batch size、epochs 网格搜索
3. **模型量化**: INT8/INT4 量化以减少显存占用
4. **分布式训练**: 多 GPU 并行加速训练过程
5. **持续评估**: 建立自动化评估流水线
