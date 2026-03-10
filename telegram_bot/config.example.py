"""
Telegram Bot 配置示例

复制此文件为 config.py 并填入你的配置
"""

# ========== Telegram Bot 配置 ==========

# Bot Token (从 @BotFather 获取)
TELEGRAM_BOT_TOKEN = "7983263905:AAFsMuGRdZzWv7KfUaAkJocu0l7LsHrScuc"

# ========== 流式输出配置 ==========

# 每次输出的 token 数量
STREAM_CHUNK_SIZE = 1

# 输出间隔 (毫秒)
STREAM_DELAY_MS = 50

# 最大生成 token 数
MAX_TOKENS = 200

# ========== 对话配置 ==========

# 最大上下文长度 (轮数)
MAX_CONTEXT_LENGTH = 10

# 会话超时 (秒，超过此时间清除历史)
SESSION_TIMEOUT = 3600

# ========== AI 模型配置 ==========

# 模型路径
MODEL_PATH = "./models/Qwen3.5-0.8B-Base"

# 运行设备 ("cuda" | "cpu")
DEVICE = None  # None 表示自动选择

# 量化类型 ("INT4" | "INT8" | "FP16")
QUANTIZATION = "INT4"

# ========== 日志配置 ==========

# 日志级别 ("DEBUG" | "INFO" | "WARNING" | "ERROR")
LOG_LEVEL = "INFO"

# 日志文件路径
LOG_FILE = "./logs/telegram_bot.log"

# ========== 管理员配置 ==========

# 管理员用户 ID 列表 (可访问/admin 命令)
ADMIN_USER_IDS = []

# 例如：
# ADMIN_USER_IDS = [123456789, 987654321]
