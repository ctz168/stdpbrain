"""
生产级配置文件

将敏感信息（如 Token）和环境配置（如代理）与主代码分离。
"""

# ==================== Telegram Bot 配置 ====================

# 你的 Telegram Bot Token
# 从 @BotFather 获取
TELEGRAM_BOT_TOKEN = "7983263905:AAFsMuGRdZzWv7KfUaAkJocu0l7LsHrScuc"

# 你的网络代理 URL (如果需要)
# 例如: "http://127.0.0.1:7890" 或 "socks5://127.0.0.1:1080"
# 如果你不需要代理，请将其设置为 None
PROXY_URL = None


# ==================== 模型与架构配置 ====================

# 模型路径 (本地)
MODEL_PATH = "./models/Qwen3.5-2B"

# 量化类型 ("INT4", "INT8", "FP16", "FP32", "AUTO")
# 在 macOS/CPU 上，INT4/INT8 会被自动优化或回退
# AUTO: 在 GPU 上使用 INT8，在 CPU 上使用 FP32（避免缓慢的动态量化）
QUANTIZATION = "FP32"

# 设备 ("cuda", "cpu", "mps")
# 留空则自动检测
DEVICE = ""
