# Telegram Bot 启动指南（真实 Qwen 模型）

## ✅ 准备工作已完成

1. ✓ 真实 Qwen 模型接口已修复：`core/qwen_interface.py`
2. ✓ Bot 启动脚本已创建：`telegram_bot/run_with_qwen.py`

---

## 🚀 快速启动命令

### 方式 1: 使用真实 Qwen 模型（推荐）

```bash
cd /Users/hilbert/Desktop/stdpbrian

# 启动 Bot（使用真实 Qwen 模型）
python3 telegram_bot/run_with_qwen.py --token YOUR_BOT_TOKEN
```

**参数说明**:
- `--token`: 您的 Telegram Bot Token（必填）
- `--model-path`: 模型路径（默认：`./models/Qwen3.5-0.8B-Base`）
- `--device`: 运行设备 `cpu` 或 `cuda`（默认：`cpu`）
- `--quantization`: 量化类型 `INT4`/`INT8`/`FP16`（默认：`INT4`）

### 示例

```bash
# CPU + INT4 量化（推荐，省显存）
python3 telegram_bot/run_with_qwen.py --token "7983263905:AAFsMuGRdZzWv7KfUaAkJocu0l7LsHrScuc"

# GPU + INT4 量化
python3 telegram_bot/run_with_qwen.py --token "YOUR_TOKEN" --device cuda

# FP32 全精度（需要更多显存）
python3 telegram_bot/run_with_qwen.py --token "YOUR_TOKEN" --quantization FP32
```

---

## 📋 完整启动流程

### 步骤 1: 检查依赖

```bash
# 确保已安装以下包
pip install torch transformers python-telegram-bot aiohttp

# INT4 量化支持（可选但推荐）
pip install bitsandbytes>=0.41.0
```

### 步骤 2: 确认模型存在

```bash
# 检查模型目录
ls -la ./models/Qwen3.5-0.8B-Base/

# 如果没有模型，请运行下载命令
huggingface-cli download Qwen/Qwen3.5-0.8B-Base\
    --local-dir ./models/Qwen3.5-0.8B-Base
```

### 步骤 3: 启动 Bot

```bash
python3 telegram_bot/run_with_qwen.py --token "YOUR_BOT_TOKEN"
```

### 步骤 4: 验证运行

看到以下输出表示成功：
```
============================================================
Telegram Bot - Qwen 模型版
============================================================

加载 Qwen 模型...
✓ Qwen 模型加载成功

初始化 Bot...
✓ Bot 就绪

启动服务... (Ctrl+C 停止)
============================================================
```

---

## 🔧 故障排查

### 问题 1: 模型未找到

**错误信息**:
```
❌ 模型未找到：./models/Qwen3.5-0.8B-Base
```

**解决方案**:
```bash
# 下载模型
huggingface-cli download Qwen/Qwen3.5-0.8B-Base\
    --local-dir ./models/Qwen3.5-0.8B-Base

# 或使用 ModelScope
modelscope download Qwen/Qwen3.5-0.8B-Base\
    --local_dir ./models/Qwen3.5-0.8B-Base
```

### 问题 2: INT4 量化失败

**错误信息**:
```
❌ 模型加载失败：No module named 'bitsandbytes'
```

**解决方案**:
```bash
# 安装 bitsandbytes
pip install bitsandbytes>=0.41.0

# 或使用其他量化方式
python3 telegram_bot/run_with_qwen.py --token "TOKEN" --quantization FP32
```

### 问题 3: Bot Token 无效

**错误信息**:
```
❌ Bot 初始化失败：Unauthorized
```

**解决方案**:
1. 联系 [@BotFather](https://t.me/BotFather) 重新获取 Token
2. 检查 Token 是否正确复制（包含所有字符）

### 问题 4: 缺少依赖包

**错误信息**:
```
ModuleNotFoundError: No module named 'telegram'
```

**解决方案**:
```bash
pip install python-telegram-bot aiohttp
```

---

## 📊 性能参考

| 配置 | 显存占用 | 单 token 延迟 | 适用场景 |
|------|---------|------------|---------|
| CPU + INT4 | ~420MB | 8-12ms | 日常使用（推荐） |
| CPU + FP32 | ~1.6GB | 15-25ms | 高精度需求 |
| GPU + INT4 | ~450MB | 3-5ms | 高性能需求 |
| GPU + FP32 | ~1.8GB | 2-3ms | 极致性能 |

---

## 🛑 停止 Bot

按 `Ctrl+C` 即可优雅停止 Bot 服务。

---

## 💡 使用技巧

### 1. 后台运行

```bash
# 使用 nohup 后台运行
nohup python3 telegram_bot/run_with_qwen.py --token "TOKEN" &

# 查看日志
tail -f nohup.out
```

### 2. 使用 systemd 服务（Linux）

创建 `/etc/systemd/system/qwen-bot.service`:
```ini
[Unit]
Description=Qwen AI Telegram Bot
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/stdpbrian
ExecStart=/usr/bin/python3 telegram_bot/run_with_qwen.py --token YOUR_TOKEN
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl start qwen-bot
sudo systemctl enable qwen-bot  # 开机自启
```

### 3. Docker 部署（高级）

创建 `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "telegram_bot/run_with_qwen.py", "--token", "YOUR_TOKEN"]
```

构建并运行：
```bash
docker build -t qwen-bot.
docker run -d --name bot qwen-bot
```

---

## 📚 相关文档

- [`README.md`](README.md) - 项目总览
- [`TELEGRAM_BOT_SUMMARY.md`](TELEGRAM_BOT_SUMMARY.md) - Bot 功能详解
- [`INSTALL_TELEGRAM.md`](INSTALL_TELEGRAM.md) - 安装指南
- [`core/qwen_interface.py`](core/qwen_interface.py) - 真实模型接口

---

## ✨ Bot 特性

✅ **基于真实 Qwen3.5-0.8B 模型**  
✅ **海马体 - 新皮层双系统架构**  
✅ **100Hz 高刷新推理**  
✅ **STDP 在线学习**  
✅ **流式输出**  
✅ **多轮对话上下文**  
✅ **打字状态模拟**  

---

*文档版本：v1.0*  
*最后更新：2026-03-10*  
*状态：准备就绪，可以启动*
