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
        proxy_url: Optional[str] = None
    ):
        self.token = token
        self.ai = ai_interface
        self.stream_chunk_size = stream_chunk_size
        self.stream_delay_ms = stream_delay_ms
        self.max_context_length = max_context_length
        self.proxy_url = proxy_url
        
        self.user_history: Dict[int, List[Dict[str, str]]] = {}
        self.application: Optional[Application] = None
        self.stream_handler: Optional[StreamHandler] = None
        self.typing_simulators: Dict[int, TypingSimulator] = {}
        self.last_active_chat_id: Optional[int] = None
        self.thinking_task: Optional[asyncio.Task] = None
        self.is_thinking_enabled = True
        self.monologue_buffer: List[str] = []
        self.max_monologue_buffer = 5
        
        self._init_stream_handler()
    
    async def _background_thinking_loop(self):
        """后台持续自思考流 (内心独白流式生成 + 主动推送)"""
        logger.info("[Thinking] 后台自思考流已启动")
        
        while self.is_thinking_enabled:
            try:
                if self.ai:
                    delay = random.randint(20, 40)
                    await asyncio.sleep(delay)
                    
                    chat_id = self.last_active_chat_id
                    
                    if chat_id:
                        logger.info(f"[Thinking] 正在流式生成内心独白...")
                        
                        # 先发送初始消息
                        try:
                            message = await self.application.bot.send_message(
                                chat_id=chat_id,
                                text="💭 *[内心独白]*\n_思考中..._",
                                parse_mode='Markdown'
                            )
                        except Exception as e:
                            logger.warning(f"发送初始消息失败: {e}")
                            continue
                        
                        # 流式生成独白
                        full_monologue = ""
                        last_update_time = time.time()
                        update_interval = 0.8
                        
                        try:
                            async for chunk in self.ai.generate_monologue_stream(max_tokens=150):
                                full_monologue += chunk
                                
                                current_time = time.time()
                                if current_time - last_update_time > update_interval:
                                    last_update_time = current_time
                                    display_text = full_monologue[:500]
                                    try:
                                        await message.edit_text(
                                            text=f"💭 *[内心独白]*\n_{display_text}▌_",
                                            parse_mode='Markdown'
                                        )
                                    except:
                                        try:
                                            await message.edit_text(
                                                text=f"💭 内心独白:\n{display_text}▌",
                                                parse_mode=None
                                            )
                                        except:
                                            pass
                                            
                        except Exception as e:
                            logger.error(f"流式独白生成失败: {e}")
                            full_monologue = "我正在思考..."
                        
                        # 最终更新
                        if full_monologue:
                            clean_monologue = full_monologue.strip().replace('"', '').replace("'", "")
                            
                            # 写入缓冲区
                            self.monologue_buffer.append(clean_monologue)
                            if len(self.monologue_buffer) > self.max_monologue_buffer:
                                self.monologue_buffer.pop(0)
                            
                            # 最终消息
                            try:
                                await message.edit_text(
                                    text=f"💭 *[内心独白]*\n_{clean_monologue}_",
                                    parse_mode='Markdown'
                                )
                            except:
                                try:
                                    await message.edit_text(
                                        text=f"💭 内心独白:\n{clean_monologue}",
                                        parse_mode=None
                                    )
                                except:
                                    pass
                            
                            logger.info(f"[Monologue Push] Sent to {chat_id}: {clean_monologue[:50]}...")
                else:
                    await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"[Thinking] 后台思考异常: {e}")
                await asyncio.sleep(30)

    def set_ai_interface(self, ai_interface):
        self.ai = ai_interface
        self._init_stream_handler()
    
    def _init_stream_handler(self):
        if self.ai:
            self.stream_handler = StreamHandler(
                ai_interface=self.ai,
                chunk_size=self.stream_chunk_size,
                delay_ms=self.stream_delay_ms
            )
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        help_text = (
            "📚 帮助信息\n\n"
            "可用命令:\n"
            "/start - 重新开始对话\n"
            "/help - 显示帮助信息\n"
            "/clear - 清除对话历史\n"
            "/stats - 查看系统统计\n\n"
            "💬 直接发送消息即可与我对话！\n"
            "我会实时显示思考过程 (流式输出)"
        )
        
        await update.message.reply_text(help_text)
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id in self.user_history:
            del self.user_history[user_id]
        await update.message.reply_text("✓ 对话历史已清除")
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        user_message = update.message.text
        
        self.last_active_chat_id = chat_id
        
        if not user_message:
            return
        
        logger.info(f"收到用户 {user_id} 消息：{user_message[:50]}...")
        
        if chat_id not in self.typing_simulators:
            self.typing_simulators[chat_id] = TypingSimulator(context.bot, chat_id)
        
        typing = self.typing_simulators[chat_id]
        await typing.start_typing()
        
        try:
            system_prompt = "You are a helpful AI assistant. Answer the user accurately and concisely."
            history = self.user_history.get(user_id, [])
            
            formatted_history = ""
            for h in history[-self.max_context_length:]:
                role = "User" if h['role'] == 'user' else "Assistant"
                formatted_history += f"{role}: {h['content']}\n"
            
            full_input = f"{system_prompt}\n\n{formatted_history}User: {user_message}\nAssistant:"
            
            if not self.ai:
                response = self._get_test_response(user_message)
                await typing.stop_typing()
                await update.message.reply_text(response)
                self._update_history(user_id, user_message, response)
                return
            
            if self.stream_handler is None and self.ai is not None:
                self._init_stream_handler()
                
            if self.stream_handler is None:
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
        try:
            thought_text = "💭 **内心独白流**\n"
            
            if self.monologue_buffer:
                for m in self.monologue_buffer:
                    thought_text += f"> ... {m}\n"
            
            interruption_thought = ""
            if self.ai and hasattr(self.ai, '_generate_thought_process'):
                interruption_thought = self.ai._generate_thought_process(user_message)
                formatted_interruption = interruption_thought.replace('\n', '\n> ')
                thought_text += f"> {formatted_interruption}\n"
            
            thought_text += "\n"
            self.monologue_buffer = []
            
            initial_message = await update.message.reply_text(
                thought_text,
                parse_mode='Markdown'
            )
            
            full_response = ""
            last_update_time = time.time()
            max_chars = 3800
            
            stream_gen = self.stream_handler.generate_stream(
                input_text,
                temperature=0.8,
                top_k=50,
                repetition_penalty=1.5
            )
            
            async for chunk in stream_gen:
                full_response += chunk
                
                if len(full_response) > max_chars:
                    full_response = full_response[:max_chars] + "\n\n⚠️ (内容过长，已自动截断)"
                    break
                
                current_time = time.time()
                if current_time - last_update_time > 0.8:
                    last_update_time = current_time
                    
                    try:
                        display_text = f"{thought_text}✨ **回复:**\n{full_response}▌"
                        await initial_message.edit_text(display_text, parse_mode='Markdown')
                    except Exception as e:
                        try:
                            await initial_message.edit_text(f"{thought_text}✨ 回复:\n{full_response}▌", parse_mode=None)
                        except:
                            logger.warning(f"流式编辑失败: {e}")
            
            await typing.stop_typing()
            
            final_display = f"{thought_text}✨ **回复:**\n{full_response}"
            try:
                await initial_message.edit_text(final_display, parse_mode='Markdown')
            except Exception as e:
                try:
                    await initial_message.edit_text(f"{thought_text}✨ 回复:\n{full_response}", parse_mode=None)
                except:
                    logger.error(f"最终编辑失败: {e}")
            
            self._update_history(user_id, user_message, full_response)
            logger.info(f"回复用户 {user_id} 完成 (长度: {len(full_response)})")
            
        except Exception as e:
            logger.error(f"流式生成失败: {e}", exc_info=True)
            await typing.stop_typing()
            await update.message.reply_text(f"❌ 生成失败: {str(e)}")
    
    def _update_history(self, user_id: int, user_message: str, assistant_response: str):
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        self.user_history[user_id].append({'role': 'user', 'content': user_message})
        self.user_history[user_id].append({'role': 'assistant', 'content': assistant_response})
        
        if len(self.user_history[user_id]) > self.max_context_length * 2:
            self.user_history[user_id] = self.user_history[user_id][-self.max_context_length * 2:]
    
    def _get_test_response(self, message: str) -> str:
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
        logger.info("正在启动 Telegram Bot...")
        
        if self.proxy_url:
            request = HTTPXRequest(proxy_url=self.proxy_url)
            self.application = Application.builder().token(self.token).request(request).build()
        else:
            self.application = Application.builder().token(self.token).build()
        
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        self.is_thinking_enabled = True
        self.application.post_init = self._post_init_hook
        
        logger.info("Bot 已启动，正在监听消息...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)
    
    async def _post_init_hook(self, application: Application):
        self.thinking_task = asyncio.create_task(self._background_thinking_loop())
        logger.info("后台思考任务已启动")


def create_bot(
    token: str,
    ai_interface=None,
    proxy_url: Optional[str] = None,
    **kwargs
) -> BrainAIBot:
    return BrainAIBot(token=token, ai_interface=ai_interface, proxy_url=proxy_url, **kwargs)
