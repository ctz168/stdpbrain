"""
Telegram Bot 主程序

实现与类人脑AI 模型的交互，支持流式输出
"""

import asyncio
import logging
import time
import random
from typing import Optional, Dict, List
from telegram import Update, Message
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from telegram.request import HTTPXRequest

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
        max_context_length: int = 10,
        proxy_url: Optional[str] = None # 新增代理参数
    ):
        """
        初始化 Bot
        
        Args:
            token: Telegram Bot Token
            ai_interface: BrainAIInterface 实例
            stream_chunk_size: 流式输出块大小
            stream_delay_ms: 流式延迟 (毫秒)
            max_context_length: 最大上下文长度
            proxy_url: HTTP 代理 URL (可选)
        """
        self.token = token
        self.ai = ai_interface
        self.stream_chunk_size = stream_chunk_size
        self.stream_delay_ms = stream_delay_ms
        self.max_context_length = max_context_length
        self.proxy_url = proxy_url
        
        # 用户对话历史
        self.user_history: Dict[int, List[Dict[str, str]]] = {}
        
        # Bot 应用
        self.application: Optional[Application] = None
        
        # 流式处理器
        self.stream_handler: Optional[StreamHandler] = None
        
        # 打字模拟器
        self.typing_simulators: Dict[int, TypingSimulator] = {}
        
        # 记录最近活跃的聊天 ID
        self.last_active_chat_id: Optional[int] = None
        
        # 后台思考任务
        self.thinking_task: Optional[asyncio.Task] = None
        self.is_thinking_enabled = True
        
        # 全局独白缓冲区 (存储最近的内心独白)
        self.monologue_buffer: List[str] = []
        self.max_monologue_buffer = 5
        
        # 初始化流式处理器
        self._init_stream_handler()
    
    async def _background_thinking_loop(self):
        """后台持续自思考流 (内心独白生成 + 主动推送)"""
        logger.info("[Thinking] 后台自思考流已启动")
        
        while self.is_thinking_enabled:
            try:
                if self.ai:
                    # 每 20-40 秒生成一段真实的内心独白
                    delay = random.randint(20, 40)
                    await asyncio.sleep(delay)
                    
                    # 获取最近活跃的用户会话
                    chat_id = self.last_active_chat_id
                    
                    # 如果有活跃会话，则生成并发送独白
                    if chat_id:
                        logger.info(f"[Thinking] 正在生成真实的内心独白...")
                        stats = self.ai.think()
                        monologue = stats.get('monologue', '')
                        
                        if monologue:
                            # 1. 写入缓冲区 (用于被用户输入打断时展示)
                            self.monologue_buffer.append(monologue)
                            if len(self.monologue_buffer) > self.max_monologue_buffer:
                                self.monologue_buffer.pop(0)
                            
                            # 2. 主动推送独白到最近活跃的聊天窗口
                            try:
                                # 以“思考”的格式发送
                                await self.application.bot.send_message(
                                    chat_id=chat_id,
                                    text=f"💭 *[内心独白]*\n_{monologue}_",
                                    parse_mode='Markdown'
                                )
                                logger.info(f"[Monologue Push] Sent to {chat_id}: {monologue[:50]}...")
                            except Exception as e:
                                logger.warning(f"主动推送独白失败: {e}")
                                
                            logger.info(f"[Monologue Internal] {monologue}")
                else:
                    await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"[Thinking] 后台思考异常: {e}")
                await asyncio.sleep(30)

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
        
        # 更新最近活跃的聊天 ID
        self.last_active_chat_id = chat_id
        
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
        """处理流式生成 (包含被用户打断的内心独白链)"""
        try:
            # 1. 构建思维链展示 (包含历史独白和即时打断思考)
            thought_text = "💭 **内心独白流**\n"
            
            # 展示缓冲区中的历史独白
            if self.monologue_buffer:
                for m in self.monologue_buffer:
                    thought_text += f"> ... {m}\n"
            
            # 获取即时的打断思维
            interruption_thought = ""
            if self.ai and hasattr(self.ai, '_generate_thought_process'):
                interruption_thought = self.ai._generate_thought_process(user_message)
                # 过滤掉可能重复的部分，只保留打断逻辑
                formatted_interruption = interruption_thought.replace('\n', '\n> ')
                thought_text += f"> {formatted_interruption}\n"
            
            thought_text += "\n"
            
            # 清空独白缓冲区 (因为已经被用户打断并展示了)
            self.monologue_buffer = []
            
            # 发送初始思考消息
            initial_message = await update.message.reply_text(
                thought_text,
                parse_mode='Markdown'
            )
            
            full_response = ""
            last_update_time = time.time()
            max_chars = 3800 # Telegram 限制为 4096，预留一些空间
            
            # 2. 流式生成
            stream_gen = self.stream_handler.generate_stream(
                input_text,
                temperature=0.8,
                top_k=50,
                repetition_penalty=1.5 # 增加重复惩罚，针对 Base 模型优化
            )
            
            async for chunk in stream_gen:
                full_response += chunk
                
                # 检查长度限制
                if len(full_response) > max_chars:
                    full_response = full_response[:max_chars] + "\n\n⚠️ (内容过长，已自动截断以符合 Telegram 限制)"
                    break
                
                # 定时更新
                current_time = time.time()
                if current_time - last_update_time > 0.8:
                    last_update_time = current_time
                    
                    try:
                        # 合并思考过程和生成内容
                        display_text = f"{thought_text}✨ **回复:**\n{full_response}▌"
                        await initial_message.edit_text(display_text, parse_mode='Markdown')
                    except Exception as e:
                        # 如果 Markdown 报错，回退到纯文本
                        try:
                            await initial_message.edit_text(f"{thought_text}✨ 回复:\n{full_response}▌", parse_mode=None)
                        except:
                            logger.warning(f"流式编辑完全失败: {e}")
            
            # 3. 完成生成
            await typing.stop_typing()
            
            # 最终编辑
            final_display = f"{thought_text}✨ **回复:**\n{full_response}"
            try:
                await initial_message.edit_text(final_display, parse_mode='Markdown')
            except Exception as e:
                try:
                    await initial_message.edit_text(f"{thought_text}✨ 回复:\n{full_response}", parse_mode=None)
                except:
                    logger.error(f"最终编辑完全失败: {e}")
            
            # 4. 更新历史
            self._update_history(user_id, user_message, full_response)
            
            # 注释掉这里的调用，让 AI 接口内部处理 STDP，避免参数缺失错误
            # if self.ai and hasattr(self.ai.stdp_engine, 'step'):
            #     self.ai.stdp_engine.step()
                
            logger.info(f"回复用户 {user_id} 完成 (长度: {len(full_response)})")
            
        except Exception as e:
            logger.error(f"流式生成失败: {e}", exc_info=True)
            await typing.stop_typing()
            await update.message.reply_text(f"❌ 生成失败: {str(e)}")
    
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
        if self.proxy_url:
            request = HTTPXRequest(proxy_url=self.proxy_url)
            self.application = Application.builder().token(self.token).request(request).build()
            logger.info(f"Bot 正在使用代理: {self.proxy_url}")
        else:
            self.application = Application.builder().token(self.token).build()
        
        # 添加处理器
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # 启动后台思考 (需要手动管理循环)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        logger.info("Bot 已启动，正在监听消息...")
        
        # 在 run_polling 之前启动后台任务
        self.is_thinking_enabled = True
        self.application.post_init = self._post_init_hook
        
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)
    
    async def _post_init_hook(self, application: Application):
        """初始化后的钩子，用于启动后台任务"""
        self.thinking_task = asyncio.create_task(self._background_thinking_loop())
        logger.info("后台思考任务已通过 post_init 启动")

    async def start_async(self):
        """异步启动 Bot"""
        logger.info("正在启动 Telegram Bot (异步模式)...")
        
        if self.proxy_url:
            request = HTTPXRequest(proxy_url=self.proxy_url)
            self.application = Application.builder().token(self.token).request(request).build()
            logger.info(f"Bot 正在使用代理: {self.proxy_url}")
        else:
            self.application = Application.builder().token(self.token).build()
        
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        logger.info("Bot 已启动 (异步)")
        await self.application.initialize()
        await self.application.start()
        
        # 启动后台任务
        self.thinking_task = asyncio.create_task(self._background_thinking_loop())
        
        await self.application.updater.start_polling(allowed_updates=Update.ALL_TYPES)


def create_bot(
    token: str,
    ai_interface=None,
    proxy_url: Optional[str] = None, # 新增代理参数
    **kwargs
) -> BrainAIBot:
    """
    快捷创建 Bot 实例
    
    Args:
        token: Telegram Bot Token
        ai_interface: BrainAIInterface 实例
        proxy_url: HTTP 代理 URL (可选)
        **kwargs: 其他参数
    
    Returns:
        BrainAIBot 实例
    """
    return BrainAIBot(token=token, ai_interface=ai_interface, proxy_url=proxy_url, **kwargs)
