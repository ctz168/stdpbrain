# 类人脑双系统全闭环 AI架构 - 文档索引

欢迎使用类人脑双系统全闭环AI架构！本文档索引帮助您快速找到所需信息。

---

## 📖 新手入门

### 第一次使用？

1. **[QUICKSTART.md](QUICKSTART.md)** - 5 分钟快速开始指南
   - 安装依赖
   - 下载模型
   - 运行第一个示例

2. **[README.md](README.md)** - 项目说明文档
   - 核心特性介绍
   - 使用示例
   - API 接口说明

3. **[requirements.txt](requirements.txt)** - Python 依赖包列表

---

## 🏗️ 架构设计

### 想了解系统架构？

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - 详细架构设计文档 ⭐ **推荐**
   - 核心架构原理
   - 7 大模块详细设计
   - 生物脑对应关系
   - 技术实现细节
   - 内存与计算开销分析

2. **[configs/arch_config.py](configs/arch_config.py)** - 配置详解
   - HardConstraints (刚性红线)
   - STDPConfig (STDP超参数)
   - HippocampusConfig (海马体配置)
   - SelfLoopConfig (自闭环配置)
   - TrainingConfig (训练配置)
   - EvaluationConfig (测评配置)
   - DeploymentConfig (部署配置)

---

## 💻 代码实现

### 想查看具体实现？

#### 核心模块 (`core/`)
- **[dual_weight_layers.py](core/dual_weight_layers.py)** - 双权重层实现 (模块 1)
- **[stdp_engine.py](core/stdp_engine.py)** - STDP 权重更新引擎 (模块 3)
- **[refresh_engine.py](core/refresh_engine.py)** - 100Hz 高刷新推理引擎 (模块 2)
- **[interfaces.py](core/interfaces.py)** - 统一 API 接口

#### 海马体系统 (`hippocampus/`)
- **[ec_encoder.py](hippocampus/ec_encoder.py)** - EC 内嗅皮层特征编码
- **[dg_separator.py](hippocampus/dg_separator.py)** - DG 齿状回模式分离
- **[ca3_memory.py](hippocampus/ca3_memory.py)** - CA3 情景记忆库
- **[ca1_gate.py](hippocampus/ca1_gate.py)** - CA1 注意力门控
- **[swr_consolidation.py](hippocampus/swr_consolidation.py)** - SWR 离线回放
- **[hippocampus_system.py](hippocampus/hippocampus_system.py)** - 完整系统集成

#### 自闭环系统 (`self_loop/`)
- **[self_loop_optimizer.py](self_loop/self_loop_optimizer.py)** - 三模式优化器

#### 训练模块 (`training/`)
- **[trainer.py](training/trainer.py)** - 主训练器
- **[pretrain_adapter.py](training/pretrain_adapter.py)** - 预适配微调
- **[online_learner.py](training/online_learner.py)** - 在线学习
- **[offline_consolidation.py](training/offline_consolidation.py)** - 离线巩固

#### 测评体系 (`evaluation/`)
- **[evaluator.py](evaluation/evaluator.py)** - 综合评估器
- **[hippocampus_eval.py](evaluation/hippocampus_eval.py)** - 海马体评估
- **[base_capability_eval.py](evaluation/base_capability_eval.py)** - 基础能力评估

#### Telegram Bot 
- **[telegram_bot/bot.py](telegram_bot/bot.py)** - Bot 主程序 (~350 行)
- **[telegram_bot/stream_handler.py](telegram_bot/stream_handler.py)** - 流式输出处理器 (~200 行)
- **[telegram_bot/run.py](telegram_bot/run.py)** - 启动脚本 (~180 行)
- **[telegram_bot/config.example.py](telegram_bot/config.example.py)** - 配置示例
- **[telegram_bot/test_bot.py](telegram_bot/test_bot.py)** - 测试脚本

---

## 📊 项目状态

### 想了解项目进度？

1. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - 项目交付总结 ⭐ **推荐**
   - 交付清单
   - 核心特性实现状态
   - 预期性能指标
   - 注意事项

2. **[COMPLETION_REPORT.md](COMPLETION_REPORT.md)** - 项目完成报告
   - 执行摘要
   - 技术实现详情
   - 代码质量分析
   - 待完成工作

3. **[本文件 (INDEX.md)](INDEX.md)** - 文档索引

---

## 🚀 部署与测试

### 想部署到端侧设备？

1. **[deployment/README.md](deployment/README.md)** - 端侧部署指南
   - 安卓手机部署 (MNN)
   - 树莓派部署
   - 模型量化指南
   - 性能优化建议

2. **[scripts/verify_installation.py](scripts/verify_installation.py)** - 安装验证脚本
   ```bash
   python scripts/verify_installation.py
   ```

3. **[tests/test_core.py](tests/test_core.py)** - 核心模块测试
   ```bash
   pytest tests/ -v
   ```

---

## 🤖 Telegram Bot (新增)

### 想通过 Telegram 进行交互？

1. **[telegram_bot/README.md](telegram_bot/README.md)** - Bot 使用指南 ⭐ **推荐**
   - 快速开始
   - 配置选项
   - 流式输出设置
   - 故障排查

2. **[TELEGRAM_BOT_SUMMARY.md](TELEGRAM_BOT_SUMMARY.md)** - 功能总结
   - 核心功能介绍
   - 架构图
   - 性能指标
   - 使用示例

3. **[INSTALL_TELEGRAM.md](INSTALL_TELEGRAM.md)** - 安装说明
   - 依赖安装
   - 验证方法
   - 常见问题

4. **[telegram_bot/config.example.py](telegram_bot/config.example.py)** - 配置示例
   - Bot Token 配置
   - 流式参数设置
   - 对话管理选项

5. **[telegram_bot/test_bot.py](telegram_bot/test_bot.py)** - 测试脚本
   ```bash
   python telegram_bot/test_bot.py
   ```

---

## 🔧 使用帮助

### 命令行工具

```bash
# 对话模式
python main.py --mode chat

# 生成模式
python main.py --mode generate --input "问题"

# 综合评测
python main.py --mode eval

# 查看统计
python main.py --mode stats

# Telegram Bot 模式 (新增)
python main.py --mode telegram
# 或
python telegram_bot/run.py --token YOUR_BOT_TOKEN
```

### Telegram Bot (新增)

**快速启动:**
```bash
# 安装依赖
pip install python-telegram-bot aiohttp

# 启动 Bot
python main.py --mode telegram
```

**可用命令:**
- `/start` - 开始对话
- `/help` - 显示帮助
- `/clear` - 清除历史
- `/stats` - 查看系统统计

**详细文档:**
- **[telegram_bot/README.md](telegram_bot/README.md)** - Bot 使用指南
- **[TELEGRAM_BOT_SUMMARY.md](TELEGRAM_BOT_SUMMARY.md)** - 功能总结
- **[INSTALL_TELEGRAM.md](INSTALL_TELEGRAM.md)** - 安装说明

### Python API

```python
from core.interfaces import create_brain_ai

# 创建 AI 实例
ai = create_brain_ai(model_path="./models/Qwen3.5-0.8B-Base")

# 对话
response = ai.chat("你好")

# 生成
output = ai.generate("写一篇短文", max_tokens=200)

# 获取统计
stats = ai.get_stats()

# 保存检查点
ai.save_checkpoint("./checkpoints/latest.pt")
```

---

## 📋 快速参考表

### 核心概念

| 概念 | 说明 | 位置 |
|------|------|------|
| 双权重架构 | 90% 静态 +10% 动态 | [ARCHITECTURE.md](ARCHITECTURE.md#一核心架构原理) |
| 100Hz 刷新 | 10ms 周期执行 | [ARCHITECTURE.md](ARCHITECTURE.md#模块 2100hz 人脑级高刷新单周期推理引擎) |
| STDP 学习 | 时序依赖可塑性 | [ARCHITECTURE.md](ARCHITECTURE.md#模块 3 全链路 stdp 时序可塑性权重刷新系统) |
| 海马体五单元 | EC-DG-CA3-CA1-SWR | [ARCHITECTURE.md](ARCHITECTURE.md#模块 5 海马体记忆系统全模块) |
| 自闭环三模式 | 自组合/自博弈/自评判 | [ARCHITECTURE.md](ARCHITECTURE.md#模块 4 单智体自生成 - 自博弈 - 自评判闭环系统) |

### 关键指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 显存占用 | ≤420MB | INT4 量化后 |
| 推理延迟 | ≤10ms | 单 token |
| 刷新周期 | 10ms | 100Hz |
| 记忆召回 | ≥95% | 线索召回准确率 |
| 抗混淆率 | ≤3% | 模式分离混淆率 |

### 文件速查

| 需求 | 文件 |
|------|------|
| 快速开始 | [QUICKSTART.md](QUICKSTART.md) |
| 架构设计 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| API 使用 | [README.md](README.md#api-接口) |
| 配置说明 | [configs/arch_config.py](configs/arch_config.py) |
| 部署指南 | [deployment/README.md](deployment/README.md) |
| 故障排查 | [QUICKSTART.md](QUICKSTART.md#故障排查) |
| **Telegram Bot** | **[telegram_bot/README.md](telegram_bot/README.md)** |
| **Bot 安装** | **[INSTALL_TELEGRAM.md](INSTALL_TELEGRAM.md)** |
| **Bot 总结** | **[TELEGRAM_BOT_SUMMARY.md](TELEGRAM_BOT_SUMMARY.md)** |

---

## 🎯 推荐阅读路径

### 开发者路径
1. [README.md](README.md) - 了解项目
2. [QUICKSTART.md](QUICKSTART.md) - 快速上手
3. [ARCHITECTURE.md](ARCHITECTURE.md) - 深入理解
4. [core/interfaces.py](core/interfaces.py) - 学习 API
5. [tests/test_core.py](tests/test_core.py) - 运行测试

### 研究者路径
1. [ARCHITECTURE.md](ARCHITECTURE.md) - 架构原理
2. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - 技术细节
3. [COMPLETION_REPORT.md](COMPLETION_REPORT.md) - 实现分析
4. 各模块源码 - 深入学习

### 部署工程师路径
1. [deployment/README.md](deployment/README.md) - 部署指南
2. [requirements.txt](requirements.txt) - 依赖安装
3. [scripts/verify_installation.py](scripts/verify_installation.py) - 环境验证
4. [main.py](main.py) - 运行测试

### Telegram Bot 用户路径 (新增)
1. [INSTALL_TELEGRAM.md](INSTALL_TELEGRAM.md) - 安装依赖
2. [telegram_bot/README.md](telegram_bot/README.md) - Bot 使用
3. [TELEGRAM_BOT_SUMMARY.md](TELEGRAM_BOT_SUMMARY.md) - 功能了解
4. [telegram_bot/test_bot.py](telegram_bot/test_bot.py) - 运行测试
5. 在 Telegram 中联系 Bot - 开始对话

---

## 📞 获取帮助

遇到问题？

1. 查看 [QUICKSTART.md](QUICKSTART.md#故障排查) - 故障排查
2. 运行 `python scripts/verify_installation.py` - 环境验证
3. 查看 [README.md](README.md) - 常见问题
4. 检查项目结构是否完整

### Telegram Bot 问题？

1. 查看 [INSTALL_TELEGRAM.md](INSTALL_TELEGRAM.md) - 安装说明
2. 查看 [telegram_bot/README.md](telegram_bot/README.md#故障排查) - Bot 故障排查
3. 运行 `python telegram_bot/test_bot.py` - 测试 Bot 功能
4. 检查 Bot Token 是否正确配置

---

*最后更新：2026-03-09*  
*项目版本：v1.0 (包含 Telegram Bot)*
