# Telegram Bot 安装指南

## ⚠️ 重要提示

Telegram Bot 功能需要额外安装 `python-telegram-bot` 依赖包。

---

## 🔧 安装方法

### 方法 1: 使用 pip (推荐)

```bash
cd /Users/hilbert/Desktop/stdpbrian
pip install python-telegram-bot aiohttp
```

### 方法 2: 使用 requirements.txt

```bash
cd /Users/hilbert/Desktop/stdpbrian
pip install -r requirements.txt
```

requirements.txt 已包含:
```
python-telegram-bot>=20.0
aiohttp>=3.8.0
```

### 方法 3: 仅安装 Telegram 相关

如果只想安装 Telegram Bot 相关依赖:

```bash
pip install python-telegram-bot[aiohttp]>=20.0
```

---

## ✅ 验证安装

```bash
python -c "import telegram; print(f'python-telegram-bot version: {telegram.__version__}')"
```

应该输出:
```
python-telegram-bot version: 20.x 或更高
```

---

## 🚀 启动 Bot

安装完成后，运行:

```bash
# 方式 1: 使用 main.py
python main.py --mode telegram

# 方式 2: 使用专用脚本
python telegram_bot/run.py

# 方式 3: 指定 Token
python telegram_bot/run.py --token 7983263905:AAFsMuGRdZzWv7KfUaAkJocu0l7LsHrScuc
```

---

## 🧪 测试

```bash
python telegram_bot/test_bot.py
```

如果看到 "所有测试通过 ✓" 则表示安装成功。

---

## ❓ 常见问题

### Q: 安装失败，提示权限错误

**解决**:
```bash
pip install --user python-telegram-bot aiohttp
```

### Q: 版本冲突

**解决**:
```bash
pip install --upgrade python-telegram-bot
```

### Q: 网络问题无法安装

**解决**: 使用国内镜像
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple python-telegram-bot aiohttp
```

---

## 📦 完整依赖列表

Telegram Bot 需要的完整依赖:

| 包名 | 最低版本 | 用途 |
|------|---------|------|
| python-telegram-bot | 20.0 | Telegram Bot API |
| aiohttp | 3.8.0 | 异步 HTTP 支持 |
| torch | 2.0.0 | PyTorch 框架 |
| transformers | 4.35.0 | 模型加载 |
| numpy | 1.24.0 | 数值计算 |

---

## 🔗 相关链接

- [python-telegram-bot 文档](https://docs.python-telegram-bot.org/)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [@BotFather](https://t.me/BotFather) - 创建和管理 Bot

---

*最后更新：2026-03-09*
