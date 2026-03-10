"""
Telegram Bot 主程序

实现与类人脑AI 模型的交互，支持流式输出
"""

import asyncio
import logging
import time
from typing import Optional, Dict, List
from telegram import Update, Message
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from .stream_handler import StreamHandler, TypingSimulator


# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class BrainAIBot:
    """
    类人脑AI Telegram Bot
    
    功能:
    - 与用户进行自然语言对话
    - 支持流式输出
    - 显示打字状态
    - 多轮对话上下文管理
    """
    
    def __init__(
        self,
        token: str,
        ai_interface=None,
        stream_chunk_size: int = 1,
        stream_delay_ms: int = 50,
        max_context_length: int = 10
    ):
        """
        初始化 Bot
        
        Args:
            token: Telegram Bot Token
            ai_interface: BrainAIInterface 实例
            stream_chunk_size: 流式输出块大小
            stream_delay_ms: 流式延迟 (毫秒)
            max_context_length: 最大上下文长度
        """
        self.token = token
        self.ai = ai_interface
        self.stream_chunk_size = stream_chunk_size
        self.stream_delay_ms = stream_delay_ms
        self.max_context_length = max_context_length
        
        # 用户对话历史
        self.user_history: Dict[int, List[Dict[str, str]]] = {}
        
        # Bot 应用
        self.application: Optional[Application] = None
        
        # 流式处理器
        self.stream_handler: Optional[StreamHandler] = None
        
        # 打字模拟器
        self.typing_simulators: Dict[int, TypingSimulator] = {}
        
        # 初始化流式处理器
        self._init_stream_handler()
    
    def set_ai_interface(self, ai_interface):
        """设置 AI 接口"""
        self.ai = ai_interface
        self._init_stream_handler()
    
    def _init_stream_handler(self):
        """初始化流式处理器"""
        if self.ai:
            self.stream_handler = StreamHandler(
                ai_interface=self.ai,
                chunk_size=self.stream_chunk_size,
                delay_ms=self.stream_delay_ms
            )
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /start 命令"""
        user_id = update.effective_user.id
        
        welcome_message = (
            "🤖 欢迎使用类人脑AI助手！\n\n"
            "✨ 特性:\n"
            "• 基于Qwen3.5-0.8B 模型\n"
            "• 海马体 - 新皮层双系统架构\n"
            "• 100Hz 高刷新推理\n"
            "• STDP 在线学习\n\n"
            "💡 直接发送消息即可与我对话！\n"
            "使用 /help 查看更多帮助"
        )
        
        await update.message.reply_text(welcome_message)
        logger.info(f"User {user_id} started the bot")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /help 命令"""
        help_text = (
            "📚 帮助信息\n\n"
            "可用命令:\n"
            "/start - 重新开始对话\n"
            "/help - 显示帮助信息\n"
            "/clear - 清除对话历史\n"
            "/stats - 查看系统统计\n"
            "/mode - 切换对话模式\n\n"
            "💬 直接发送消息即可与我对话！\n"
            "我会实时显示思考过程 (流式输出)"
        )
        
        await update.message.reply_text(help_text)
        logger.info(f"User {user_id} requested help")
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /clear 命令"""
        user_id = update.effective_user.id
        
        if user_id in self.user_history:
            del self.user_history[user_id]
        
        await update.message.reply_text("✓ 对话历史已清除")
        logger.info(f"Cleared history for user {user_id}")
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /stats 命令"""
        if not self.ai:
            await update.message.reply_text("⚠️ AI 模型未初始化")
            return
        
        try:
            stats = self.ai.get_stats()
            
            stats_text = (
                "📊 系统统计\n\n"
                f"🧠 海马体记忆数：{stats['hippocampus']['num_memories']}\n"
                f"💾 内存使用：{stats['hippocampus']['memory_usage_mb']:.2f}MB\n"
                f"⚡ STDP 周期：{stats['stdp']['cycle_count']}\n"
                f"🔄 推理周期：{stats['refresh_engine']['total_cycles']}\n"
                f"⏱️ 平均延迟：{stats['refresh_engine']['avg_cycle_time_ms']:.2f}ms\n"
                f"🎯 自环周期：{stats['self_loop']['cycle_count']}"
            )
            
            await update.message.reply_text(stats_text)
        except Exception as e:
            await update.message.reply_text(f"❌ 获取统计失败：{e}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理用户消息"""
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        user_message = update.message.text
        
        if not user_message:
            return
        
        logger.info(f"收到用户 {user_id} 消息：{user_message[:50]}...")
        
        # 获取或创建打字模拟器
        if chat_id not in self.typing_simulators:
            self.typing_simulators[chat_id] = TypingSimulator(context.bot, chat_id)
        
        typing = self.typing_simulators[chat_id]
        
        # 开始打字状态
        await typing.start_typing()
        
        try:
            # 获取对话历史并构建提示词 (针对 Base 模型优化)
            system_prompt = "You are a helpful AI assistant. Answer the user accurately and concisely."
            history = self.user_history.get(user_id, [])
            
            # 格式化多轮对话
            formatted_history = ""
            for h in history[-self.max_context_length:]:
                role = "User" if h['role'] == 'user' else "Assistant"
                formatted_history += f"{role}: {h['content']}\n"
            
            full_input = f"{system_prompt}\n\n{formatted_history}User: {user_message}\nAssistant:"
         # 如果没有 AI 接口，返回测试响应
            if not self.ai:
                response = self._get_test_response(user_message)
                await typing.stop_typing()
                await update.message.reply_text(response)
                self._update_history(user_id, user_message, response)
                return
            
            # 使用流式处理器生成响应
            if self.stream_handler is None and self.ai is not None:
                logger.info("StreamHandler 为空，正在重新初始化...")
                self._init_stream_handler()
                
            if self.stream_handler is None:
                logger.error("无法创建 StreamHandler，回退到普通回应")
                response = self.ai.chat(user_message) if self.ai else self._get_test_response(user_message)
                await typing.stop_typing()
                await update.message.reply_text(response)
                self._update_history(user_id, user_message, response)
                return

            await self._handle_stream_generation(
                update=update,
                input_text=full_input,
                typing=typing,
                user_id=user_id,
                user_message=user_message
            )
            
        except Exception as e:
            logger.error(f"处理消息失败：{e}", exc_info=True)
            await typing.stop_typing()
            await update.message.reply_text(f"❌ 处理失败：{str(e)}")
    
    async def _handle_stream_generation(
        self,
        update: Update,
        input_text: str,
        typing: TypingSimulator,
        user_id: int,
        user_message: str
    ):
        """处理流式生成"""
        try:
            # 发送初始消息 (用于后续编辑)
            initial_message = await update.message.reply_text("🤔 思考中...")
            
            full_response = ""
            last_update_time = time.time()
            
            # 流式生成 (带采样参数以防止循环重复)
            stream_gen = self.stream_handler.generate_stream(
                input_text,
                temperature=0.7,
                top_k=50,
                repetition_penalty=1.1
            )
            
            async for chunk in stream_gen:
                full_response += chunk
                
                # 每 500ms 或达到 4000 字符限制时更新一次
                current_time = time.time()
                if current_time - last_update_time > 0.5 or len(full_response) > 4000:
                    last_update_time = current_time
                    
                    # 检查长度限制 (Telegram 限制为 4096)
                    content_to_send = full_response
                    if len(content_to_send) > 4000:
                        content_to_send = content_to_send[:3900] + "\n\n(内容过长，已截断...)"
                    
                    # 编辑消息
                    try:
                        # 尝试使用 Markdown，失败则回退到纯文本
                        await initial_message.edit_text(
                            content_to_send + "▌",
                            parse_mode=None # 暂时关闭 Markdown 以避免格式问题
                        )
                    except Exception as e:
                        logger.warning(f"编辑消息失败：{e}")
                        # 回退：尝试不带光标和 Markdown
                        try:
                            await initial_message.edit_text(content_to_send[:4000])
                        except:
                            pass
                
                # 如果超过限制，主动停止
                if len(full_response) > 4000:
                    logger.warning(f"用户 {user_id} 的响应过长，正在停止生成")
                    break
            
            # 完成
            await typing.stop_typing()
            
            # 最终编辑 (移除光标)
            try:
                final_text = full_response
                if len(final_text) > 4090:
                    final_text = final_text[:4000] + "..."
                await initial_message.edit_text(final_text)
            except Exception as e:
                logger.error(f"最终编辑失败：{e}")
            
            # 更新历史
            self._update_history(user_id, user_message, full_response)
            
            logger.info(f"回复用户 {user_id}: {full_response[:50]}...")
            
        except Exception as e:
            logger.error(f"流式生成失败：{e}", exc_info=True)
            await typing.stop_typing()
            
            # 详细错误日志
            error_msg = str(e)
            if "Message is too long" in error_msg:
                error_msg = "响应内容超过 Telegram 长度限制 (4096 字符)。模型可能陷入了重复生成或输出了过多调试信息。"
                
            await update.message.reply_text(f"❌ 生成失败：{error_msg}")
    
    def _update_history(self, user_id: int, user_message: str, assistant_response: str):
        """更新对话历史"""
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        self.user_history[user_id].append({
            'role': 'user',
            'content': user_message
        })
        self.user_history[user_id].append({
            'role': 'assistant',
            'content': assistant_response
        })
        
        # 保持历史长度
        if len(self.user_history[user_id]) > self.max_context_length * 2:
            self.user_history[user_id] = self.user_history[user_id][-self.max_context_length * 2:]
    
    def _get_test_response(self, message: str) -> str:
        """测试响应 (无 AI 接口时使用)"""
        responses = {
            "你好": "你好！我是类人脑AI 助手。这是一个测试响应。",
            "介绍": "我基于海马体 - 新皮层双系统架构，支持 100Hz 高刷新推理和 STDP 在线学习。",
            "帮助": "直接发送消息即可与我对话！我支持流式输出。"
        }
        
        for key, value in responses.items():
            if key in message:
                return value
        
        return f"收到：{message}。请配置真实的 AI 模型接口以获得更好的响应。"
    
    def run(self):
        """运行 Bot"""
        logger.info("正在启动 Telegram Bot...")
        
        # 创建应用
        self.application = Application.builder().token(self.token).build()
        
        # 添加处理器
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # 启动 Bot
        logger.info("Bot 已启动，正在监听消息...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)
    
    async def start_async(self):
        """异步启动 Bot"""
        logger.info("正在启动 Telegram Bot (异步模式)...")
        
        self.application = Application.builder().token(self.token).build()
        
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        logger.info("Bot 已启动 (异步)")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(allowed_updates=Update.ALL_TYPES)


def create_bot(
    token: str,
    ai_interface=None,
    **kwargs
) -> BrainAIBot:
    """
    快捷创建 Bot 实例
    
    Args:
        token: Telegram Bot Token
        ai_interface: BrainAIInterface 实例
        **kwargs: 其他参数
    
    Returns:
        BrainAIBot 实例
    """
    return BrainAIBot(token=token, ai_interface=ai_interface, **kwargs)
