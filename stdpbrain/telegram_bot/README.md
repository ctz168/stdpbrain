# Telegram Bot 使用指南

## 🤖 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

确保安装了 `python-telegram-bot>=20.0` 和 `aiohttp`。

### 2. 启动 Bot

**简单启动:**
```bash
python telegram_bot/run.py
```

**使用 main.py 启动:**
```bash
python main.py --mode telegram
```

**自定义配置:**
```bash
python telegram_bot/run.py \
    --token YOUR_BOT_TOKEN \
    --model-path ./models/Qwen3.5-0.8B \
    --chunk-size 2 \
    --delay-ms 100 \
    --async-mode
```

### 3. 在 Telegram 中联系 Bot

1. 打开 Telegram
2. 搜索你的 Bot 用户名 (例如：@BrainAIBot)
3. 点击 "Start" 或发送 `/start`
4. 开始对话！

---

## ⚙️ 配置选项

### 环境变量方式

```bash
export TELEGRAM_BOT_TOKEN="7983263905:AAFsMuGRdZzWv7KfUaAkJocu0l7LsHrScuc"
python telegram_bot/run.py
```

### 配置文件方式

复制配置示例:
```bash
cd telegram_bot
cp config.example.py config.py
```

编辑 `config.py`:
```python
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
STREAM_CHUNK_SIZE = 1      # 每次输出 1 个 token
STREAM_DELAY_MS = 50       # 延迟 50ms
MAX_CONTEXT_LENGTH = 10    # 保留 10 轮对话
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--token` | Bot Token | 无 (必须提供) |
| `--model-path` | 模型路径 | `./models/Qwen3.5-0.8B` |
| `--device` | 运行设备 | 自动选择 |
| `--chunk-size` | 流式块大小 | 1 |
| `--delay-ms` | 流式延迟 (毫秒) | 50 |
| `--async-mode` | 启用异步模式 | 关闭 |

---

## 💬 可用命令

Bot 支持以下命令:

| 命令 | 说明 |
|------|------|
| `/start` | 重新开始对话，显示欢迎信息 |
| `/help` | 显示帮助信息 |
| `/clear` | 清除对话历史 |
| `/stats` | 查看系统统计 (内存、延迟等) |

---

## ✨ 流式输出特性

### 实时打字状态

Bot 会在思考时显示"typing..."状态，让用户知道正在处理。

### Token 级流式输出

响应会逐个 token 实时显示，而不是等待完整生成后才显示。

**效果示例:**
```
用户：介绍一下类人脑架构

Bot: 我🤔 思...考...中...

Bot: 我是基于海马体 - 新皮层双系统架构的类人脑AI...
```

### 调整流式参数

**更快的输出:**
```bash
python telegram_bot/run.py --chunk-size 3 --delay-ms 20
```

**更流畅的输出:**
```bash
python telegram_bot/run.py --chunk-size 1 --delay-ms 100
```

---

## 🔧 高级功能

### 多用户支持

Bot 天然支持多用户并发访问，每个用户的对话历史独立管理。

### 上下文管理

Bot 会自动保留最近 10 轮对话作为上下文，实现连贯的多轮对话。

示例:
```
用户：今天天气怎么样？
Bot: 我不了解您所在位置的实际天气情况...

用户：那北京呢？  # Bot 知道"那"指的是天气
Bot: 北京的天气通常是...
```

### 查看系统统计

发送 `/stats` 命令查看:
- 🧠 海马体记忆数量
- 💾 内存使用情况
- ⚡ STDP 更新周期数
- 🔄 推理周期数
- ⏱️ 平均延迟

---

## 🐛 故障排查

### Bot 无法启动

**错误**: `Unauthorized`
- **原因**: Token 无效或已过期
- **解决**: 从 [@BotFather](https://t.me/BotFather) 重新获取 Token

**错误**: `Network error`
- **原因**: 网络连接问题
- **解决**: 检查网络连接，可能需要代理

### Bot 响应慢

**可能原因**:
1. 模型未正确加载 → 检查 `--model-path`
2. 设备性能不足 → 使用 `--device cpu` 或启用量化
3. 流式参数过小 → 增大 `--chunk-size` 或 `--delay-ms`

### 内存溢出

**解决方法**:
```bash
# 使用 INT4 量化
export QUANTIZATION=INT4

# 减少上下文长度
# 编辑 config.py, 设置 MAX_CONTEXT_LENGTH = 5
```

---

## 📊 性能优化建议

### 1. 启用 GPU 加速

```bash
python telegram_bot/run.py --device cuda
```

### 2. 调整批处理大小

在 `config.py` 中:
```python
STREAM_CHUNK_SIZE = 2  # 每次输出 2 个 token
STREAM_DELAY_MS = 30   # 降低延迟
```

### 3. 使用异步模式

```bash
python telegram_bot/run.py --async-mode
```

适合高并发场景。

---

## 🔒 安全建议

### 1. 保护 Bot Token

- ❌ 不要将 Token 提交到 Git
- ✅ 使用环境变量或配置文件
- ✅ 添加 `config.py` 到 `.gitignore`

### 2. 限制管理员访问

在 `config.py` 中设置:
```python
ADMIN_USER_IDS = [你的 Telegram ID]
```

### 3. 启用日志

```python
LOG_LEVEL = "INFO"  # 或 "DEBUG"
LOG_FILE = "./logs/telegram_bot.log"
```

---

## 📝 示例代码

### 集成到自己的应用

```python
from telegram_bot.bot import BrainAIBot
from core.interfaces import create_brain_ai

# 创建 AI
ai = create_brain_ai()

# 创建 Bot
bot = BrainAIBot(
    token="YOUR_TOKEN",
    ai_interface=ai,
    stream_chunk_size=1,
    stream_delay_ms=50
)

# 运行
bot.run()
```

### 自定义消息处理

```python
from telegram_bot.bot import BrainAIBot

class MyBot(BrainAIBot):
    async def handle_message(self, update, context):
        # 自定义处理逻辑
        user_message = update.message.text
        
        if "你好" in user_message:
            await update.message.reply_text("你好！有什么可以帮助你的？")
        else:
            # 调用父类处理
            await super().handle_message(update, context)

bot = MyBot(token="YOUR_TOKEN")
bot.run()
```

---

## 🎯 最佳实践

1. **使用异步模式**: 适合生产环境
2. **启用日志**: 便于问题排查
3. **合理设置上下文长度**: 平衡性能和连贯性
4. **定期清理历史**: 使用 `/clear` 命令
5. **监控资源使用**: 使用 `/stats` 命令

---

## 📞 获取帮助

遇到问题？

1. 查看日志文件: `logs/telegram_bot.log`
2. 运行诊断：`python scripts/verify_installation.py`
3. 检查配置：确认 Token 正确

---

*最后更新：2026-03-09*  
*项目版本：v1.0*
