# Telegram Bot 集成总结

## 📦 新增内容

已成功为类人脑AI架构添加 **Telegram Bot** 功能，支持流式输出和实时交互。

---

## 🗂️ 新增文件

### 核心模块 (4 个文件)

1. **`telegram_bot/__init__.py`** - 模块初始化
2. **`telegram_bot/bot.py`** - Bot 主程序 (~350 行)
   - `BrainAIBot` 类：核心 Bot 逻辑
   - 支持命令：`/start`, `/help`, `/clear`, `/stats`
   - 多用户对话历史管理
   - 流式消息处理

3. **`telegram_bot/stream_handler.py`** - 流式输出处理器 (~200 行)
   - `StreamHandler` 类：Token 级流式生成
   - `TypingSimulator` 类：打字状态模拟
   - 异步生成器支持

4. **`telegram_bot/run.py`** - 启动脚本 (~180 行)
   - 命令行参数解析
   - AI 模型集成
   - 同步/异步模式支持

### 配置与文档 (4 个文件)

5. **`telegram_bot/config.example.py`** - 配置示例 (~60 行)
6. **`telegram_bot/README.md`** - 使用指南 (~300 行)
7. **`telegram_bot/test_bot.py`** - 测试脚本 (~120 行)
8. **`TELEGRAM_BOT_SUMMARY.md`** - 本文档

### 修改文件

9. **`requirements.txt`** - 添加依赖
   - `python-telegram-bot>=20.0`
   - `aiohttp>=3.8.0`

10. **`main.py`** - 添加 telegram 模式
    - 新增 `--mode telegram` 选项
    - 新增 `--telegram-token` 参数
    - 新增 `--async-mode` 参数

---

## 🎯 核心功能

### 1. 流式输出 (Streaming)

**特点**:
- Token 级别的实时输出
- 可配置的块大小和延迟
- 打字状态显示

**效果**:
```
用户：介绍一下类人脑架构

Bot: [typing...]
Bot: 我🤔 思...考...中...
Bot: 我是基于海马体 - 新皮层双系统架构的类人脑AI...
```

### 2. 多轮对话上下文

**特点**:
- 自动保留最近 10 轮对话
- 每个用户独立的对话历史
- 支持上下文关联的连贯对话

**示例**:
```
用户：今天天气怎么样？
Bot: 我不了解您所在位置的实际天气...

用户：那北京呢？  # Bot 理解"那"指的是天气查询
Bot: 北京的天气通常是...
```

### 3. 系统统计

发送 `/stats` 命令查看:
- 🧠 海马体记忆数量
- 💾 内存使用情况 (MB)
- ⚡ STDP 更新周期数
- 🔄 推理周期总数
- ⏱️ 平均延迟 (ms)

### 4. 命令支持

| 命令 | 说明 |
|------|------|
| `/start` | 重新开始，显示欢迎信息 |
| `/help` | 显示帮助和使用说明 |
| `/clear` | 清除对话历史 |
| `/stats` | 查看系统统计 |

---

## 🚀 快速开始

### 方式 1: 使用 main.py

```bash
python main.py --mode telegram
```

Bot 会自动使用默认 Token: `8533918353:AAG6Pxr0A4C4kJpCVjYzbtwtFzN4KZCcRag`

### 方式 2: 使用专用脚本

```bash
python telegram_bot/run.py \
    --token 8533918353:AAG6Pxr0A4C4kJpCVjYzbtwtFzN4KZCcRag \
    --model-path ./models/Qwen3.5-0.8B-Base \
    --chunk-size 1 \
    --delay-ms 50
```

### 方式 3: 环境变量

```bash
export TELEGRAM_BOT_TOKEN="8533918353:AAG6Pxr0A4C4kJpCVjYzbtwtFzN4KZCcRag"
python telegram_bot/run.py
```

---

## ⚙️ 配置选项

### 流式输出配置

```python
# 在 config.py 中设置
STREAM_CHUNK_SIZE = 1      # 每次输出 1 个 token
STREAM_DELAY_MS = 50       # 延迟 50ms
MAX_TOKENS = 200           # 最大生成 200 tokens
```

### 对话配置

```python
MAX_CONTEXT_LENGTH = 10    # 保留 10 轮对话
SESSION_TIMEOUT = 3600     # 会话超时 1 小时
```

### 性能优化

**更快的输出:**
```bash
python telegram_bot/run.py --chunk-size 3 --delay-ms 20
```

**更流畅的输出:**
```bash
python telegram_bot/run.py --chunk-size 1 --delay-ms 100
```

---

## 📊 架构图

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Telegram  │────▶│  BrainAIBot  │────▶│ StreamHandler│
│    Users    │◀────│   (Bot Core) │◀────│ (Streaming)  │
└─────────────┘     └──────────────┘     └─────────────┘
                           │                     │
                           ▼                     ▼
                    ┌──────────────┐     ┌─────────────┐
                    │ User History │     │  BrainAI    │
                    │  Manager     │     │  Interface  │
                    └──────────────┘     └─────────────┘
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │ Hippocampus  │
                                         │  System      │
                                         └──────────────┘
```

---

## 🧪 测试

### 运行测试套件

```bash
python telegram_bot/test_bot.py
```

测试内容:
- ✓ Bot 初始化
- ✓ 流式处理器
- ✓ 打字模拟器

### 手动测试

1. 启动 Bot
2. 在 Telegram 中搜索 Bot 用户名
3. 发送 `/start` 开始对话
4. 发送任意消息测试响应
5. 发送 `/stats` 查看统计
6. 发送 `/clear` 清除历史

---

## 🔧 高级功能

### 异步模式

适合高并发场景:

```bash
python telegram_bot/run.py --async-mode
```

### 自定义 Bot

继承 `BrainAIBot` 类:

```python
from telegram_bot.bot import BrainAIBot

class MyCustomBot(BrainAIBot):
    async def handle_message(self, update, context):
        # 自定义处理逻辑
        pass

bot = MyCustomBot(token="YOUR_TOKEN")
bot.run()
```

### 管理员权限

在 `config.py` 中设置:

```python
ADMIN_USER_IDS = [你的 Telegram ID]
```

---

## 📈 性能指标

### 预期性能

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 首字延迟 | <500ms | 从发送到看到第一个字 |
| 输出速度 | 20-50 字/秒 | 取决于 chunk_size 和 delay_ms |
| 并发用户 | 100+ | 异步模式下 |
| 内存占用 | +50MB | 相比命令行模式 |

### 优化建议

1. **启用 GPU**: `--device cuda`
2. **异步模式**: 适合高并发
3. **调整块大小**: 平衡速度和流畅度
4. **定期清理历史**: 使用 `/clear`

---

## 🔒 安全提示

### 保护 Bot Token

- ❌ 不要提交到 Git
- ✅ 使用环境变量
- ✅ 添加 `config.py` 到 `.gitignore`

### .gitignore 示例

```gitignore
# Telegram Bot 配置
telegram_bot/config.py
*.env
```

---

## 📞 故障排查

### Bot 无法启动

**错误**: `Unauthorized`
- **原因**: Token 无效
- **解决**: 从 @BotFather 重新获取

**错误**: `Network error`
- **原因**: 网络问题
- **解决**: 检查连接或使用代理

### 响应慢

1. 检查模型是否加载
2. 增大 `--chunk-size`
3. 减小 `--delay-ms`
4. 启用 GPU 加速

### 内存溢出

```bash
# 使用 INT4 量化
export QUANTIZATION=INT4

# 减少上下文长度
# 编辑 config.py: MAX_CONTEXT_LENGTH = 5
```

---

## 📝 使用示例

### 基本对话

```
User: /start
Bot: 🤖 欢迎使用类人脑AI 助手！...

User: 你好
Bot: 你好！我是类人脑AI 助手...

User: 介绍一下你自己
Bot: 我是基于海马体 - 新皮层双系统架构...
```

### 查看统计

```
User: /stats
Bot: 📊 系统统计

🧠 海马体记忆数：15
💾 内存使用：1.23MB
⚡ STDP 周期：42
🔄 推理周期：128
⏱️ 平均延迟：8.56ms
```

---

## 🎯 下一步

1. **下载模型**: 确保已下载 Qwen3.5-0.8B
2. **安装依赖**: `pip install -r requirements.txt`
3. **测试 Bot**: `python telegram_bot/test_bot.py`
4. **启动 Bot**: `python telegram_bot/run.py`
5. **在 Telegram 中联系**: 搜索 Bot 用户名

---

## 📚 相关文档

- [telegram_bot/README.md](telegram_bot/README.md) - 详细使用指南
- [README.md](README.md) - 项目主文档
- [QUICKSTART.md](QUICKSTART.md) - 快速开始

---

*添加日期：2026-03-09*  
*版本：v1.0*  
*状态：✅ 完成并可用*
