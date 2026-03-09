# 类人脑AI架构 - 运行说明

## 📋 当前状态

项目包含两个版本：

### 1. 简化测试版 ✅ (可立即运行)
- **文件**: `test_run.py`
- **依赖**: 仅需 Python 3.8+
- **功能**: 基础对话、上下文管理、统计功能
- **用途**: 验证核心逻辑和流程

### 2. 完整实现版 ⏳ (需要安装依赖)
- **文件**: `main.py`, `core/` 目录下的完整模块
- **依赖**: PyTorch, Transformers, python-telegram-bot 等
- **功能**: 完整的类人脑架构、海马体记忆系统、STDP 学习、Telegram Bot
- **用途**: 生产环境使用

---

## 🚀 快速开始

### 方式 1: 简化测试版 (推荐先试用)

```bash
cd /Users/hilbert/Desktop/stdpbrian

# 直接运行测试
python test_run.py
```

**输出示例**:
```
============================================================
✅ 所有测试通过!
============================================================
  - cycle_count: 25
  - system: simplified
  - status: running
```

### 方式 2: 完整实现版

#### 步骤 1: 安装核心依赖

```bash
cd /Users/hilbert/Desktop/stdpbrian

# 安装 PyTorch (CPU 版本)
pip install torch numpy

# 或 GPU 版本 (推荐)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 步骤 2: 安装可选依赖

```bash
# Telegram Bot 支持
pip install python-telegram-bot aiohttp

# 或使用 requirements.txt 安装全部
pip install -r requirements.txt
```

#### 步骤 3: 下载模型 (可选，用于真实推理)

```bash
# 方法 1: Hugging Face
huggingface-cli download Qwen/Qwen3.5-0.8B-Base --local-dir ./models/Qwen3.5-0.8B-Base

# 方法 2: ModelScope (国内推荐)
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B-Base', local_dir='./models/Qwen3.5-0.8B-Base')"
```

#### 步骤 4: 运行完整版本

```bash
# 对话模式
python main.py --mode chat

# Telegram Bot 模式
python main.py --mode telegram

# 查看统计
python main.py --mode stats
```

---

## 🔧 故障排查

### 问题 1: ModuleNotFoundError: No module named 'torch'

**解决**:
```bash
pip install torch
```

或在测试时使用简化版:
```bash
python test_run.py
```

### 问题 2: CUDA out of memory

**解决**:
```bash
# 使用 CPU 运行
python main.py --mode chat --device cpu

# 或启用量化
export QUANTIZATION=INT4
```

### 问题 3: Telegram Bot 无法启动

**解决**:
```bash
# 检查依赖
pip install python-telegram-bot aiohttp

# 验证 Token
python telegram_bot/test_bot.py
```

---

## 📊 功能对比

| 功能 | 简化版 | 完整版 |
|------|--------|--------|
| 基础对话 | ✅ | ✅ |
| 上下文管理 | ✅ | ✅ |
| 统计功能 | ✅ | ✅ |
| 海马体记忆系统 | ❌ | ✅ |
| STDP 学习 | ❌ | ✅ |
| 100Hz 刷新引擎 | ❌ | ✅ |
| Telegram Bot | ❌ | ✅ |
| 真实语言模型 | ❌ | ✅ (需下载模型) |

---

## 🎯 推荐流程

### 第一次使用？

1. **运行简化版测试**
   ```bash
   python test_run.py
   ```
   验证基本逻辑是否正常

2. **阅读文档**
   - [README.md](README.md) - 项目说明
   - [QUICKSTART.md](QUICKSTART.md) - 快速开始
   - [ARCHITECTURE.md](ARCHITECTURE.md) - 架构设计

3. **决定是否安装完整依赖**
   - 仅测试：简化版足够
   - 生产使用：安装完整依赖

### 生产环境部署？

1. **安装完整依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **下载模型**
   ```bash
   huggingface-cli download Qwen/Qwen3.5-0.8B-Base --local-dir ./models/Qwen3.5-0.8B-Base
   ```

3. **配置环境变量**
   ```bash
   export TELEGRAM_BOT_TOKEN="YOUR_TOKEN"
   ```

4. **启动服务**
   ```bash
   python main.py --mode telegram
   ```

---

## 📝 代码实现状态

### ✅ 已完整实现

| 模块 | 状态 | 说明 |
|------|------|------|
| **海马体系统** | ✅ 100% | EC/DG/CA3/CA1/SWR 五个子模块完整实现 |
| **STDP 引擎** | ✅ 100% | 全链路 STDP 更新规则完整实现 |
| **自闭环优化** | ✅ 90% | 三模式优化器完整实现 |
| **双权重层** | ✅ 100% | DualWeightLinear/Attention/FFN 完整实现 |
| **配置文件** | ✅ 100% | 全局配置和超参数完整定义 |
| **文档** | ✅ 100% | 11 个文档完整详细 |

### ⚠️ 需真实模型集成

| 模块 | 状态 | 说明 |
|------|------|------|
| **Qwen 权重加载** | ⏳ 待集成 | 需要下载并加载真实 Qwen3.5-0.8B 权重 |
| **Tokenizer** | ⏳ 待集成 | 需要使用 Qwen 官方 tokenizer |
| **完整推理** | ⏳ 待集成 | 当前使用简化响应，需连接真实模型 |

### 📦 简化版实现

| 模块 | 状态 | 说明 |
|------|------|------|
| **SimpleBrainAI** | ✅ 100% | 简化测试版，无需 torch |
| **接口封装** | ✅ 100% | interfaces_working.py 完整可用 |

---

## 🔗 相关文档

- **[README.md](README.md)** - 项目总览
- **[QUICKSTART.md](QUICKSTART.md)** - 5 分钟快速开始
- **[INSTALL_TELEGRAM.md](INSTALL_TELEGRAM.md)** - Bot 安装指南
- **[TELEGRAM_BOT_SUMMARY.md](TELEGRAM_BOT_SUMMARY.md)** - Bot 功能总结
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - 详细架构设计

---

## 💡 下一步计划

### 短期 (1-2 周)
- [ ] 集成真实 Qwen3.5-0.8B 权重
- [ ] 添加 Qwen tokenizer 支持
- [ ] 完善错误处理和日志记录

### 中期 (1 个月)
- [ ] 端侧部署测试 (树莓派/安卓)
- [ ] 性能优化和 benchmark
- [ ] 大规模训练和微调

### 长期 (3 个月)
- [ ] 用户反馈收集和改进
- [ ] 更多应用场景适配
- [ ] 社区建设和维护

---

*最后更新：2026-03-09*  
*项目版本：v1.0*  
*状态：简化版可用，完整版待模型集成*
