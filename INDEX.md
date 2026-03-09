# 类人脑双系统全闭环 AI 架构 - 文档索引

欢迎使用类人脑双系统全闭环 AI 架构！本文档索引帮助您快速找到所需信息。

---

## 🚀 新手入门（重要）

### 第一次使用？从这里开始 ⭐

1. **[QUICKSTART.md](QUICKSTART.md)** - 5 分钟快速开始指南 ⭐⭐⭐ **必读**
   - ✅ Python3.11 环境配置
   - ✅ Conda 一键安装脚本
   - ✅ 依赖包安装
   - ✅ 模型下载
   - ✅ 运行第一个示例
   - ✅ Telegram Bot 配置

2. **[requirements.txt](requirements.txt)** - Python 依赖包清单 ⭐
   - 核心依赖（必需）
   - 可选依赖
   - 一键安装命令

3. **[check_env.py](check_env.py)** - 环境验证脚本 ⭐
   ```bash
  python check_env.py  # 验证安装是否成功
   ```

4. **[README.md](README.md)** - 项目说明
   - 核心特性介绍
   - 使用示例
   - 性能指标

---

## 🏗️ 架构设计

### 想了解系统架构？

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - 详细架构设计文档 ⭐⭐⭐
   - 核心架构原理
   - 7 大模块详细设计
   - 生物脑对应关系
   - 技术实现细节

2. **[configs/arch_config.py](configs/arch_config.py)** - 配置详解
   - HardConstraints (刚性红线)
   - STDPConfig (STDP 超参数)
   - HippocampusConfig (海马体配置)
   - SelfLoopConfig (自闭环配置)

---

## 💻 代码实现

### 想查看具体实现？

#### 核心模块 (`core/`)
- **[qwen_interface.py](core/qwen_interface.py)** ⭐ **Qwen 真实模型接口** (新增)
- **[dual_weight_layers.py](core/dual_weight_layers.py)** - 双权重层实现
- **[stdp_engine.py](core/stdp_engine.py)** - STDP 权重更新引擎
- **[refresh_engine.py](core/refresh_engine.py)** -100Hz 高刷新推理引擎
- **[interfaces.py](core/interfaces.py)** - 统一 API 接口

#### 海马体系统 (`hippocampus/`)
- **[ec_encoder.py](hippocampus/ec_encoder.py)** - EC 内嗅皮层特征编码
- **[dg_separator.py](hippocampus/dg_separator.py)** - DG 齿状回模式分离
- **[ca3_memory.py](hippocampus/ca3_memory.py)** - CA3 情景记忆库
- **[ca1_gate.py](hippocampus/ca1_gate.py)** - CA1 注意力门控
- **[swr_consolidation.py](hippocampus/swr_consolidation.py)** - SWR 离线回放
- **[hippocampus_system.py](hippocampus/hippocampus_system.py)** - 完整系统集成

#### 其他模块
- **[self_loop/self_loop_optimizer.py](self_loop/self_loop_optimizer.py)** - 自闭环优化器
- **[telegram_bot/bot.py](telegram_bot/bot.py)** - Telegram Bot 主程序
- **[main.py](main.py)** - 项目主入口

---

## 🧪 测试与验证

### 想测试功能？

1. **[check_env.py](check_env.py)** - 环境验证 ⭐ **首先运行**
   ```bash
  python check_env.py
   ```

2. **[simple_test.py](simple_test.py)** - 简单测试 ⭐ **推荐首次运行**
   ```bash
  python simple_test.py
   ```

3. **[final_test.py](final_test.py)** - 完整功能测试
   ```bash
  python final_test.py
   ```

4. **[TEST_REPORT.md](TEST_REPORT.md)** - 测试报告 ⭐
   - 测试结果详情
   - 性能数据
   - 已知问题

---

## 📊 项目状态

### 想了解项目进度？

1. **[PROJECT_COMPLETION.md](PROJECT_COMPLETION.md)** - 项目完成总结 ⭐⭐⭐ **新增**
   - ✅ 已完成功能清单
   - ✅ 测试结果统计
   - ✅ 文件清单
   - ✅ 使用方法总结

2. **[RUN_GUIDE.md](RUN_GUIDE.md)** - 运行指南 ⭐⭐⭐ **新增**
   - 详细使用方法
   - 代码示例
   - 故障排查
   - 性能优化

3. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - 快速参考卡 ⭐ **新增**
   - 一键启动命令
   - 常用操作速查
   - 性能指标
   - 文档导航

---

## 🤖 Telegram Bot

### 想通过 Telegram 交互？

1. **[telegram_bot/README.md](telegram_bot/README.md)** - Bot 使用指南 ⭐
   - 快速开始
   - 配置选项
   - 流式输出设置

2. **[TELEGRAM_BOT_SUMMARY.md](TELEGRAM_BOT_SUMMARY.md)** - 功能总结
   - 核心功能介绍
   - 架构图
   - 性能指标

3. **[INSTALL_TELEGRAM.md](INSTALL_TELEGRAM.md)** - 安装说明
   - 依赖安装
   - 验证方法
   - 常见问题

4. **[telegram_bot/test_bot.py](telegram_bot/test_bot.py)** - 测试脚本
   ```bash
  python telegram_bot/test_bot.py
   ```

---

## 🔧 部署与优化

### 想部署到生产环境？

1. **[RUN_GUIDE.md](RUN_GUIDE.md)** - 运行指南 ⭐
   - CPU/GPU模式切换
   - INT4 量化配置
   - 性能优化建议

2. **[SETUP_ENVIRONMENT.md](SETUP_ENVIRONMENT.md)** - 环境配置说明
   - Python 版本要求
   - Conda 环境配置
   - Pyenv 替代方案

3. **[setup_conda_env.sh](setup_conda_env.sh)** - Conda 环境脚本 ⭐
   ```bash
   chmod +x setup_conda_env.sh
   ./setup_conda_env.sh
   ```

---

## 📋 快速参考表

### 核心概念

| 概念 | 说明 | 位置 |
|------|------|------|
| 双权重架构 | 90% 静态 +10% 动态 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 100Hz 刷新 | 10ms 周期执行 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| STDP 学习 | 时序依赖可塑性 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 海马体五单元 | EC-DG-CA3-CA1-SWR | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Qwen 真实模型 | 752M 参数，官方 tokenizer | [RUN_GUIDE.md](RUN_GUIDE.md) |

### 关键指标

| 指标 | CPU 模式 | GPU 模式 (预期) |
|------|---------|---------------|
| 生成速度 | 3-4 tokens/s | 30-50 tokens/s |
| 内存占用 | ~3GB | ~1.5GB |
| 启动时间 | ~10s | ~5s |

### 文件速查

| 需求 | 文件 |
|------|------|
| 🚀 快速开始 | [QUICKSTART.md](QUICKSTART.md) ⭐ |
| 📖 运行指南 | [RUN_GUIDE.md](RUN_GUIDE.md) ⭐ |
| 🧪 测试报告 | [TEST_REPORT.md](TEST_REPORT.md) ⭐ |
| 📊 完成总结 | [PROJECT_COMPLETION.md](PROJECT_COMPLETION.md) ⭐ |
| 🔍 快速参考 | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) ⭐ |
| 🏗️ 架构设计 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 📦 依赖清单 | [requirements.txt](requirements.txt) ⭐ |
| ✅ 环境验证 | [check_env.py](check_env.py) ⭐ |
| 🤖 Telegram Bot | [telegram_bot/README.md](telegram_bot/README.md) |

---

## 🎯 推荐阅读路径

### 新手路径 ⭐
1. [QUICKSTART.md](QUICKSTART.md) - 5 分钟快速上手
2. [check_env.py](check_env.py) - 验证环境
3. [simple_test.py](simple_test.py) - 简单测试
4. [RUN_GUIDE.md](RUN_GUIDE.md) - 深入学习

### 开发者路径
1. [README.md](README.md) - 了解项目
2. [ARCHITECTURE.md](ARCHITECTURE.md) - 深入理解
3. [core/qwen_interface.py](core/qwen_interface.py) - 学习 API
4. [tests/test_core.py](tests/test_core.py) - 运行测试

### 研究者路径
1. [ARCHITECTURE.md](ARCHITECTURE.md) - 架构原理
2. [PROJECT_COMPLETION.md](PROJECT_COMPLETION.md) - 技术细节
3. 各模块源码 - 深入学习

### 部署工程师路径
1. [QUICKSTART.md](QUICKSTART.md) - 快速上手
2. [requirements.txt](requirements.txt) - 依赖安装
3. [check_env.py](check_env.py) - 环境验证
4. [RUN_GUIDE.md](RUN_GUIDE.md) - 部署优化

---

## 🔧 使用帮助

### 命令行工具

```bash
# 环境验证
python check_env.py

# 简单测试
python simple_test.py

# 完整测试
python final_test.py

# 对话模式
python main.py --mode chat

# 生成模式
python main.py --mode generate --input "问题"

# 查看统计
python main.py --mode stats

# Telegram Bot 模式
python main.py --mode telegram
```

### Python API

```python
from core.qwen_interface import create_qwen_ai

# 创建 AI 实例
ai = create_qwen_ai(model_path="./models/Qwen3.5-0.8B-Base")

# 对话
response = ai.chat("你好")

# 生成
output = ai.generate("写一篇短文", max_new_tokens=200)

# 获取统计
stats = ai.get_stats()
```

---

## 📞 获取帮助

遇到问题？

1. ✅ 运行 `python check_env.py` - 验证环境
2. 📖 查看 [QUICKSTART.md](QUICKSTART.md) - 故障排查章节
3. 📊 查看 [TEST_REPORT.md](TEST_REPORT.md) - 已知问题
4. 💬 查看 [RUN_GUIDE.md](RUN_GUIDE.md) - 常见问题

### Telegram Bot 问题？

1. 查看 [INSTALL_TELEGRAM.md](INSTALL_TELEGRAM.md) - 安装说明
2. 查看 [telegram_bot/README.md](telegram_bot/README.md) - 故障排查
3. 运行 `python telegram_bot/test_bot.py` - 测试功能

---

## 📁 完整文件清单

### 核心代码 (~4,800 行)
- `core/qwen_interface.py` - Qwen 模型接口 ⭐
- `core/dual_weight_layers.py` - 双权重层
- `core/stdp_engine.py` - STDP 引擎
- `core/refresh_engine.py` - 刷新引擎
- `hippocampus/*.py` - 海马体五子系统
- `self_loop/*.py` - 自闭环优化
- `telegram_bot/*.py` - Telegram Bot

### 测试文件
- `check_env.py` - 环境验证 ⭐
- `simple_test.py` - 简单测试 ⭐
- `final_test.py` - 完整测试 ⭐
- `test_full_system.py` - 全系统测试

### 文档
- `QUICKSTART.md` - 快速开始 ⭐⭐⭐
- `RUN_GUIDE.md` - 运行指南 ⭐⭐⭐
- `TEST_REPORT.md` - 测试报告 ⭐⭐⭐
- `PROJECT_COMPLETION.md` - 完成总结 ⭐⭐⭐
- `QUICK_REFERENCE.md` - 快速参考 ⭐
- `README.md` - 项目说明
- `ARCHITECTURE.md` - 架构详解
- `INDEX.md` - 本文档

### 脚本
- `setup_conda_env.sh` - Conda 环境脚本 ⭐
- `requirements.txt` - 依赖清单 ⭐

---

*最后更新：2026-03-09*  
*项目版本：v1.0*  
*Python: 3.11+ | PyTorch: 2.5+*
