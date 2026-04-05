"""
Telegram Bot 主程序

实现与类人脑AI 模型的交互，支持流式输出
"""

import asyncio
import logging
import time
import random
from typing import Optional, Dict, List, Any
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from telegram.request import HTTPXRequest

from .stream_handler import StreamHandler, TypingSimulator

# 导入用户反馈处理器
from core.user_feedback_handler import get_feedback_handler


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
        self.is_user_interacting = False  # 用户正在交互时暂停后台思考
        self.monologue_buffer: List[str] = []
        self.max_monologue_buffer = 5
        
        # 持续潜意识流
        self.pending_user_input: Optional[str] = None  # 待处理的用户输入
        self.subconscious_state: Dict[str, Any] = {}  # 潜意识状态
        self.last_subconscious_update: float = 0  # 上次潜意识更新时间
        
        # 防串线: 按 chat_id 锁定，防止不同聊天交叉
        self._chat_locks: Dict[int, asyncio.Lock] = {}
        # 防 Flood control: 按 chat_id 发送节流
        self._last_edit_times: Dict[int, float] = {}
        self._min_edit_interval: float = 1.5  # Telegram 限制 ~30次/分钟 ≈ 2秒/次
        
        self._init_stream_handler()
        
        # 注册主动消息回调
        if ai_interface and hasattr(ai_interface, 'set_proactive_callback'):
            ai_interface.set_proactive_callback(self._on_proactive_message)
            logger.info("[Bot] 已注册主动消息回调")
    
    async def _background_thinking_loop(self):
        """
        后台持续潜意识流 (永不停止 + 实时决策输出)
        
        核心理念：
        - 潜意识是一个持续运行的流，永不停止
        - 每次生成内容时，实时判断是否输出 (should_speak)
        - 可能思考中不输出，想好了就输出
        - 即使没有用户输入，自己平时在想的时候也可以输出
        """
        logger.info("[Thinking] 后台潜意识流已启动 - 实时决策模式")
        
        while self.is_thinking_enabled:
            try:
                if self.ai:
                    # 检查用户是否正在交互
                    if self.is_user_interacting:
                        logger.debug("[Thinking] 用户正在交互，暂停后台思考")
                        await asyncio.sleep(2)  # 短暂等待
                        continue
                    
                    # 缩短等待时间，让潜意识更频繁地运行
                    delay = random.randint(15, 25)  # 增加到15-25秒，避免过于频繁
                    await asyncio.sleep(delay)
                    
                    # 再次检查用户是否正在交互
                    if self.is_user_interacting:
                        logger.debug("[Thinking] 用户开始交互，取消本轮思考")
                        continue
                    
                    chat_id = self.last_active_chat_id
                    if not chat_id:
                        continue
                    
                    # 检查是否有待处理的用户输入
                    if self.pending_user_input:
                        logger.info(f"[Thinking] 潜意识处理用户输入: {self.pending_user_input[:30]}...")
                        # 潜意识处理用户输入（不发送消息，只更新内部状态）
                        try:
                            self.is_user_interacting = True  # 标记为交互状态
                            async for event in self.ai.chat_stream(self.pending_user_input, []):
                                if event["type"] == "monologue":
                                    logger.debug(f"[潜意识] {event['content'][:50]}...")
                                # 随时检查是否需要停止
                                if not self.is_thinking_enabled:
                                    break
                            self.pending_user_input = None  # 清除待处理输入
                            self.is_user_interacting = False  # 恢复状态
                        except Exception as e:
                            logger.error(f"[Thinking] 潜意识处理输入失败: {e}")
                            self.is_user_interacting = False
                            self.pending_user_input = None
                        continue
                    
                    # ========== 生成自由独白 + 思维修改缓冲区 ==========
                    logger.info(f"[Thinking] 潜意识自由流动...")
                    
                    try:
                        full_monologue = ""
                        draft_buffer = ""  # 草稿缓冲区：可以被思维修改
                        last_update_time = time.time()
                        last_sent_text = ""
                        update_interval = 2.0  # 更新间隔
                        reflect_interval = 35  # 每35个字符反思一次
                        tokens_since_reflect = 0
                        message_sent = False  # 是否已发送消息
                        last_confidence = 0.0  # 上次反思的置信度
                        
                        async for chunk in self.ai.generate_monologue_stream(max_tokens=100):
                            # 检查是否需要停止
                            if not self.is_thinking_enabled or self.is_user_interacting:
                                logger.info("[Thinking] 检测到用户交互或停止信号，中断思考")
                                break
                            
                            full_monologue += chunk
                            draft_buffer += chunk
                            tokens_since_reflect += len(chunk)
                            
                            # ========== 思维反思：修改缓冲区 ==========
                            should_output = False
                            should_revise = False
                            
                            if tokens_since_reflect >= reflect_interval:
                                # 思维反思：审视草稿内容
                                if hasattr(self.ai, 'proactive_generator') and self.ai.proactive_generator is not None:
                                    try:
                                        from core.proactive_intent_generator import ProactiveIntent
                                        context = self.ai._build_proactive_context()
                                        intent, confidence, debug = self.ai.proactive_generator(
                                            self.ai.current_thought_state, context
                                        )
                                        
                                        # 根据置信度判断是否"想清楚"了
                                        if confidence < 0.3:
                                            # 置信度太低，需要修改
                                            should_revise = True
                                            logger.debug(f"[思维修改] 置信度低，重新思考 (conf={confidence:.2f})")
                                        elif intent != ProactiveIntent.SILENCE and confidence > 0.5:
                                            # 置信度高，可以输出
                                            should_output = True
                                            logger.info(f"[想清楚了] 置信度高，输出独白 (conf={confidence:.2f})")
                                        else:
                                            # 中等置信度，继续思考
                                            logger.debug(f"[继续思考] 置信度中等 (conf={confidence:.2f})")
                                        
                                        # 如果需要修改草稿，生成新的思维指导
                                        if should_revise and draft_buffer:
                                            # 基于当前草稿生成改进建议
                                            logger.debug(f"[思维修改] 正在重新思考...")
                                            # 这里可以调用新的思维生成（暂不实现，避免递归）
                                        
                                        last_confidence = confidence
                                        
                                    except Exception as e:
                                        logger.debug(f"思维反思失败: {e}")
                                
                                tokens_since_reflect = 0
                            
                            # 如果决定输出，发送缓冲区内容
                            if should_output and draft_buffer:
                                # 转义 Markdown 特殊字符
                                safe_draft = self._escape_markdown(draft_buffer)
                                
                                # 首次发送消息
                                if not message_sent:
                                    message = await self.application.bot.send_message(
                                        chat_id=chat_id,
                                        text=f"💭 *[潜意识]*\n_{safe_draft}▌_",
                                        parse_mode='Markdown'
                                    )
                                    message_sent = True
                                    last_sent_text = safe_draft
                                else:
                                    # 更新消息
                                    current_time = time.time()
                                    if current_time - last_update_time > update_interval:
                                        safe_full = self._escape_markdown(full_monologue[:400])
                                        new_text = f"💭 *[潜意识]*\n_{safe_full}▌_"
                                        if new_text != last_sent_text:
                                            try:
                                                await self._safe_edit_message(
                                                    message,
                                                    text=new_text,
                                                    parse_mode='Markdown'
                                                )
                                                last_sent_text = new_text
                                                last_update_time = current_time
                                            except Exception:
                                                pass
                                
                                draft_buffer = ""  # 清空缓冲区
                        
                        # 生成结束后，如果有剩余缓冲区内容，强制输出
                        if draft_buffer or full_monologue:
                            clean_monologue = full_monologue.strip()
                            self.monologue_buffer.append(clean_monologue)
                            if len(self.monologue_buffer) > self.max_monologue_buffer:
                                self.monologue_buffer.pop(0)
                            
                            # 转义 Markdown 特殊字符
                            safe_monologue = self._escape_markdown(clean_monologue)
                            final_text = f"💭 *[潜意识]*\n_{safe_monologue}_"
                            
                            if message_sent:
                                # 更新已发送的消息
                                try:
                                    await self._safe_edit_message(
                                        message,
                                        text=final_text,
                                        parse_mode='Markdown'
                                    )
                                except Exception:
                                    pass
                            else:
                                # 如果从未发送过，发送最终结果
                                await self.application.bot.send_message(
                                    chat_id=chat_id,
                                    text=final_text,
                                    parse_mode='Markdown'
                                )
                            
                            logger.info(f"[潜意识] 自由独白: {clean_monologue[:50]}...")
                            
                    except Exception as e:
                        logger.error(f"[Thinking] 潜意识生成失败: {e}")
                        
                else:
                    await asyncio.sleep(10)
                    
            except Exception as e:
                logger.error(f"[Thinking] 后台潜意识异常: {e}")
                await asyncio.sleep(20)

    def set_ai_interface(self, ai_interface):
        self.ai = ai_interface
        self._init_stream_handler()
        
        # 注册主动消息回调
        if hasattr(ai_interface, 'set_proactive_callback'):
            ai_interface.set_proactive_callback(self._on_proactive_message)
            logger.info("[Bot] 已注册主动消息回调")
    
    def _on_proactive_message(self, text: str, is_clarification: bool = False):
        """处理主动消息回调（由 AI 接口异步调用）"""
        import asyncio
        
        async def send_async():
            chat_id = self.last_active_chat_id
            if chat_id and self.application:
                try:
                    prefix = "🤔 *[澄清]*\n" if is_clarification else "💬 *[主动分享]*\n"
                    safe_text = self._escape_markdown(text)
                    await self.application.bot.send_message(
                        chat_id=chat_id,
                        text=f"{prefix}_{safe_text}_",
                        parse_mode='Markdown'
                    )
                    logger.info(f"[Proactive] 主动消息已发送到 {chat_id}")
                except Exception as e:
                    logger.error(f"[Proactive] 发送失败: {e}")
        
        # 获取当前运行的事件循环
        try:
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(send_async(), loop)
        except RuntimeError:
            # 如果没有运行中的循环，尝试创建新的
            try:
                asyncio.run(send_async())
            except Exception as e:
                logger.error(f"[Proactive] 无法发送消息: {e}")
    
    def _init_stream_handler(self):
        if self.ai:
            self.stream_handler = StreamHandler(
                ai_interface=self.ai,
                chunk_size=self.stream_chunk_size,
                delay_ms=self.stream_delay_ms
            )
    
    def _escape_markdown(self, text: str) -> str:
        """
        转义 Telegram Markdown 特殊字符
        
        需要转义的字符：_ * [ ] ( ) ~ ` > # + - = | { } . !
        """
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        
        welcome_message = (
            "🤖 欢迎使用类人脑AI助手！\n\n"
            "[*] 特性:\n"
            "• 基于Qwen3.5-2B 模型\n"
            "• 海马体 - 新皮层双系统架构\n"
            "• 100Hz 高刷新推理\n"
            "• STDP 在线学习\n\n"
            "[*] 直接发送消息即可与我对话！\n"
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
            "/stats - 查看详细系统监控\n"
            "/monitor - 实时状态快照\n"
            "/memory - 查看记忆详情\n\n"
            "📊 监控内容:\n"
            "• 海马体记忆情况\n"
            "• STDP学习状态\n"
            "• 情绪状态\n"
            "• 目标状态\n"
            "• 注意力计算\n"
            "• KV缓存情况\n\n"
            "💬 直接发送消息即可与我对话！"
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
        
        stats = self.ai.get_stats()
        
        # 构建详细的监控消息
        stats_text = "📊 [*系统监控面板*]\n"
        stats_text += "=" * 30 + "\n\n"
        
        # 1. 海马体记忆情况
        hippocampus = stats.get('hippocampus', {})
        stats_text += "🧠 [*海马体记忆*]\n"
        stats_text += f"  记忆数量: {hippocampus.get('num_memories', 0)}\n"
        stats_text += f"  核心记忆: {hippocampus.get('core_memory_count', 0)}\n"
        stats_text += f"  内存使用: {hippocampus.get('memory_usage_mb', 0):.2f}/{hippocampus.get('max_memory_mb', 2.0):.1f}MB\n"
        stats_text += f"  召回次数: {hippocampus.get('recall_count', 0)}\n"
        stats_text += f"  平均激活: {hippocampus.get('avg_activation', 0):.4f}\n"
        stats_text += f"  KV记忆数: {hippocampus.get('kv_memory_count', 0)}\n\n"
        
        # 2. STDP学习情况
        stdp = stats.get('stdp', {})
        stats_text += "🔄 [*STDP学习*]\n"
        stats_text += f"  学习周期: {stdp.get('cycle_count', 0)}\n"
        stats_text += f"  总更新次数: {stdp.get('total_updates', 0)}\n"
        stats_text += f"  LTP增强: {stdp.get('ltp_count', 0)}\n"
        stats_text += f"  LTD抑制: {stdp.get('ltd_count', 0)}\n"
        stats_text += f"  动态权重范数: `{stdp.get('dynamic_weight_norm', 0):.6f}`\n"
        stats_text += f"  最近更新幅度: `{stdp.get('last_update_magnitude', 0):.6f}`\n"
        stats_text += f"  学习率: {stdp.get('learning_rate', 0):.4f}\n\n"
        
        # 3. 情绪状态
        emotion = stats.get('emotion', {})
        if emotion:
            stats_text += "😊 [*情绪状态*]\n"
            stats_text += f"  唤醒度: {emotion.get('arousal', 0.5):.2f} ({emotion.get('energy', 'low')})\n"
            stats_text += f"  效价: {emotion.get('valence', 0.5):.2f} ({emotion.get('state', 'neutral')})\n\n"
        
        # 4. 目标状态
        goal = stats.get('goal', {})
        if goal:
            stats_text += "🎯 [*目标状态*]\n"
            if goal.get('has_goal'):
                stats_text += f"  当前目标: {goal.get('goal_type', 'none')}\n"
                stats_text += f"  描述: {goal.get('goal_description', '')[:30]}...\n"
                stats_text += f"  进度: {goal.get('goal_progress', 0):.1%}\n"
                stats_text += f"  优先级: {goal.get('goal_priority', 0):.2f}\n"
                stats_text += f"  子目标数: {goal.get('sub_goals_count', 0)}\n"
            else:
                stats_text += "  当前无目标\n"
            stats_text += "\n"
        
        # 5. 全局状态
        global_ws = stats.get('global_workspace', {})
        if global_ws:
            stats_text += "🌐 [*全局工作空间*]\n"
            stats_text += f"  活跃状态: {'是' if global_ws.get('is_active') else '否'}\n"
            stats_text += f"  广播次数: {global_ws.get('broadcast_count', 0)}\n"
            stats_text += f"  竞争胜者: {global_ws.get('competition_winner', 'none')}\n\n"
        
        # 6. 注意力计算情况
        attention = stats.get('attention', {})
        if attention:
            stats_text += "👁️ [*注意力机制*]\n"
            stats_text += f"  窗口大小: {attention.get('window_size', 32)}\n"
            stats_text += f"  最大锚点: {attention.get('max_anchors', 5)}\n"
            stats_text += f"  复杂度: {attention.get('attention_complexity', 'O(n×(W+K))')}\n"
            stats_text += f"  KV缓存: {'启用' if attention.get('kv_cache_enabled') else '禁用'}\n\n"
        
        # 7. KV情况
        kv = stats.get('kv', {})
        if kv:
            stats_text += "📦 [*KV缓存*]\n"
            stats_text += f"  活跃KV数: {kv.get('active_kv_count', 0)}\n"
            stats_text += f"  海马体集成: {'启用' if kv.get('kv_enabled') else '禁用'}\n"
            stats_text += f"  滑动窗口: {'启用' if kv.get('sliding_window') else '禁用'}\n"
            stats_text += f"  窗口大小: {kv.get('window_size', 32)}\n\n"
        
        # 8. 最近记忆锚点数值（新增）
        # _last_recalled_memories 在 BrainAIInterface.__init__ 中需要预初始化
        if hasattr(self.ai, '_last_recalled_memories') and self.ai._last_recalled_memories is not None and len(self.ai._last_recalled_memories) > 0:
            stats_text += "⚓ [*最近记忆锚点*]\n"
            for i, mem in enumerate(self.ai._last_recalled_memories[:3], 1):
                semantic = mem.get('semantic_pointer', 'N/A')[:35]
                activation = mem.get('activation_strength', 0)
                stats_text += f"  {i}. {semantic}... ({activation:.4f})\n"
            stats_text += "\n"
        
        # 系统状态
        system = stats.get('system', {})
        stats_text += "⚙️ [*系统状态*]\n"
        stats_text += f"  总周期: {system.get('total_cycles', 0)}\n"
        stats_text += f"  设备: {system.get('device', 'cpu')}\n"
        stats_text += f"  思维状态: {'存在' if system.get('has_thought_state') else '缺失'}\n"
        
        await update.message.reply_text(stats_text, parse_mode='Markdown')
    
    async def monitor_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """实时监控命令 - 显示当前状态快照"""
        if not self.ai:
            await update.message.reply_text("⚠️ AI 模型未初始化")
            return
        
        # 获取当前状态快照
        stats = self.ai.get_stats()
        
        # 构建简洁的实时监控消息
        monitor_text = "🔍 [*实时监控快照*]\n"
        monitor_text += "=" * 30 + "\n\n"
        
        # 当前思维状态
        monologue = stats.get('monologue', {})
        monitor_text += "💭 [*当前思维*]\n"
        monitor_text += f"  状态: {monologue.get('thought_state', 'unknown')}\n"
        monitor_text += f"  情绪: {monologue.get('emotion_state', 'unknown')}\n"
        monitor_text += f"  独白历史: {monologue.get('history_count', 0)}条\n\n"
        
        # 最近记忆（如果有）
        hippocampus = stats.get('hippocampus', {})
        monitor_text += "🧠 [*最近记忆*]\n"
        monitor_text += f"  总数: {hippocampus.get('num_memories', 0)}\n"
        monitor_text += f"  核心记忆: {hippocampus.get('core_memory_count', 0)}\n"
        monitor_text += f"  平均激活: {hippocampus.get('avg_activation', 0):.4f}\n"
        if hippocampus.get('last_recall_time', 0) > 0:
            import time as t
            elapsed = t.time() - hippocampus.get('last_recall_time', 0)
            monitor_text += f"  最近召回: {elapsed:.1f}秒前\n\n"
        
        # 当前学习状态
        stdp = stats.get('stdp', {})
        monitor_text += "📚 [*学习状态*]\n"
        monitor_text += f"  更新次数: {stdp.get('total_updates', 0)}\n"
        monitor_text += f"  权重范数: `{stdp.get('dynamic_weight_norm', 0):.6f}`\n"
        monitor_text += f"  更新幅度: `{stdp.get('last_update_magnitude', 0):.6f}`\n"
        ltp = stdp.get('ltp_count', 0)
        ltd = stdp.get('ltd_count', 0)
        if ltp + ltd > 0:
            ratio = ltp / (ltp + ltd)
            monitor_text += f"  LTP/LTD比: {ratio:.2%}\n\n"
        
        # 当前目标
        goal = stats.get('goal', {})
        if goal and goal.get('has_goal'):
            monitor_text += "🎯 [*当前目标*]\n"
            monitor_text += f"  类型: {goal.get('goal_type', 'none')}\n"
            monitor_text += f"  描述: {goal.get('goal_description', '')[:30]}...\n"
            progress_bar = "█" * int(min(max(goal.get('goal_progress', 0), 0.0), 1.0) * 10)
            progress_bar += "░" * (10 - len(progress_bar))
            monitor_text += f"  进度: [{progress_bar}] {goal.get('goal_progress', 0):.0%}\n"
            monitor_text += f"  优先级: {goal.get('goal_priority', 0):.2f}\n\n"
        
        # 当前情绪
        emotion = stats.get('emotion', {})
        if emotion:
            monitor_text += "😊 [*当前情绪*]\n"
            arousal = emotion.get('arousal', 0.5)
            valence = emotion.get('valence', 0.5)
            arousal = min(max(arousal, 0.0), 1.0)
            valence = min(max(valence, 0.0), 1.0)
            arousal_bar = "█" * int(arousal * 10) + "░" * (10 - int(arousal * 10))
            valence_bar = "█" * int(valence * 10) + "░" * (10 - int(valence * 10))
            monitor_text += f"  唤醒: [{arousal_bar}] {arousal:.2f}\n"
            monitor_text += f"  效价: [{valence_bar}] {valence:.2f}\n\n"
        
        # 最近记忆锚点（新增）
        if hasattr(self.ai, '_last_recalled_memories') and self.ai._last_recalled_memories:
            monitor_text += "⚓ [*记忆锚点*]\n"
            for i, mem in enumerate(self.ai._last_recalled_memories[:2], 1):
                semantic = mem.get('semantic_pointer', 'N/A')[:30]
                activation = mem.get('activation_strength', 0)
                monitor_text += f"  {i}. {semantic}... ({activation:.3f})\n"
            monitor_text += "\n"
        
        # KV状态
        kv = stats.get('kv', {})
        monitor_text += "📦 [*KV状态*]\n"
        monitor_text += f"  活跃: {kv.get('active_kv_count', 0)}\n"
        monitor_text += f"  窗口: {kv.get('window_size', 32)}\n"
        
        # 添加时间戳
        from datetime import datetime
        monitor_text += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        
        await update.message.reply_text(monitor_text, parse_mode='Markdown')
    
    async def memory_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """查看记忆详情命令"""
        if not self.ai:
            await update.message.reply_text("⚠️ AI 模型未初始化")
            return
        
        try:
            # 获取记忆系统详细信息
            # hippocampus 和 ca3_memory 在 BrainAIInterface 和 HippocampusSystem 中已初始化
            ca3 = self.ai.hippocampus.ca3_memory
            memories = list(ca3.memories.values()) if ca3.memories else []
            
            memory_text = "📚 [*记忆系统详情*]\n"
            memory_text += "=" * 30 + "\n\n"
            
            # 统计信息
            memory_text += f"总记忆数: {len(memories)}\n"
            core_memories = [m for m in memories if getattr(m, 'is_core', False)]
            memory_text += f"核心记忆: {len(core_memories)}\n\n"
            
            # 显示最近的5条记忆
            memory_text += "[*最近记忆*]\n"
            recent = sorted(memories, key=lambda m: getattr(m, 'timestamp', 0), reverse=True)[:5]
            for i, mem in enumerate(recent, 1):
                semantic = getattr(mem, 'semantic_pointer', 'N/A')[:50]
                activation = getattr(mem, 'activation_strength', 0)
                is_core = '⭐' if getattr(mem, 'is_core', False) else ''
                memory_text += f"{i}. {is_core}{semantic}...\n"
                memory_text += f"   激活: {activation:.3f}\n"
            
            await update.message.reply_text(memory_text, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ 获取记忆详情失败：{e}")
    
    def _get_chat_lock(self, chat_id: int) -> asyncio.Lock:
        """获取或创建按 chat_id 的锁，防止同聊天内并发串线"""
        if chat_id not in self._chat_locks:
            self._chat_locks[chat_id] = asyncio.Lock()
        return self._chat_locks[chat_id]
    
    def _get_last_edit_time(self, chat_id: int) -> float:
        return self._last_edit_times.get(chat_id, 0.0)
    
    def _set_last_edit_time(self, chat_id: int, t: float):
        self._last_edit_times[chat_id] = t
    
    async def _safe_edit_message(self, message, text: str, parse_mode=None):
        """安全编辑消息，自带 Flood control 退避（按 chat_id 隔离）"""
        chat_id = message.chat_id if hasattr(message, 'chat_id') else 0
        now = time.time()
        elapsed = now - self._get_last_edit_time(chat_id)
        if elapsed < self._min_edit_interval:
            await asyncio.sleep(self._min_edit_interval - elapsed + 0.1)
        try:
            await message.edit_text(text=text, parse_mode=parse_mode)
            self._set_last_edit_time(chat_id, time.time())
            return True
        except Exception as e:
            if "Flood control" in str(e) or "flood" in str(e).lower():
                # 解析 retry_after，默认退避3秒
                retry_after = 3.0
                if hasattr(e, 'parameters') and e.parameters:
                    retry_after = e.parameters.get('retry_after', 3.0)
                logger.warning(f"[Flood Control] 退避 {retry_after:.1f}s")
                await asyncio.sleep(retry_after)
                self._set_last_edit_time(chat_id, time.time())
            else:
                logger.debug(f"[Edit] 非flood错误: {e}")
            return False
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        user_message = update.message.text
        
        # 按 chat_id 加锁，防止同一聊天内消息并发处理导致串线
        lock = self._get_chat_lock(chat_id)
        if lock.locked():
            # 上一条消息还在处理中，跳过（避免串线）
            logger.warning(f"[串线防护] chat_id={chat_id} 上条消息仍在处理，忽略新消息")
            return
        
        async with lock:
            self.last_active_chat_id = chat_id
            
            # 设置用户交互状态（暂停后台思考）
            self.is_user_interacting = True
            try:
                if not user_message:
                    return

                logger.info(f"收到用户 {user_id} 消息：{user_message[:50]}...")
                
                # ========== 检测用户反馈 ==========
                feedback_handler = get_feedback_handler()
                feedback = feedback_handler.detect_feedback(user_message)
                
                if feedback.is_feedback:
                    logger.info(f"[用户反馈] 类型={'正面' if feedback.is_positive else '负面'}, "
                               f"强度={feedback.intensity:.2f}, 关键词={feedback.keywords_matched}")
                    
                    if not feedback.is_positive and feedback.intensity >= 0.7:
                        logger.warning(f"[用户反馈] 检测到强负面反馈，将触发 STDP LTD 学习")
                
                if chat_id not in self.typing_simulators:
                    self.typing_simulators[chat_id] = TypingSimulator(context.bot, chat_id)
                
                typing = self.typing_simulators[chat_id]
                await typing.start_typing()
                
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
                    user_message=user_message,
                    feedback=feedback
                )
                
            except Exception as e:
                logger.error(f"处理消息失败：{e}", exc_info=True)
                try:
                    if 'typing' in locals():
                        await typing.stop_typing()
                    await update.message.reply_text("❌ 处理失败，请稍后重试")
                except Exception:
                    pass
            finally:
                self.is_user_interacting = False

    async def _handle_stream_generation(
        self,
        update: Update,
        input_text: str,
        typing: TypingSimulator,
        user_id: int,
        user_message: str,
        feedback=None  # 新增: 反馈信息
    ):
        try:
            # 获取当前系统状态用于显示
            current_stats = self.ai.get_stats() if self.ai else {}
            emotion = current_stats.get('emotion', {})
            goal = current_stats.get('goal', {})
            stdp = current_stats.get('stdp', {})
            hippocampus = current_stats.get('hippocampus', {})
            
            # ========== 构建详细状态显示 ==========
            status_lines = []
            
            # 1. STDP 学习状态
            if stdp:
                weight_norm = stdp.get('dynamic_weight_norm', 0)
                last_update = stdp.get('last_update_magnitude', 0)
                ltp = stdp.get('ltp_count', 0)
                ltd = stdp.get('ltd_count', 0)
                status_lines.append(f"🔄 STDP权重:{weight_norm:.5f} | 更新:{last_update:.5f} | LTP:{ltp}/LTD:{ltd}")
            
            # 2. 海马体记忆锚点
            if hippocampus:
                num_mem = hippocampus.get('num_memories', 0)
                avg_act = hippocampus.get('avg_activation', 0)
                core_mem = hippocampus.get('core_memory_count', 0)
                status_lines.append(f"🧠 记忆:{num_mem}条 | 激活:{avg_act:.3f} | 核心:{core_mem}")
            
            # 3. 目标状态
            if goal and goal.get('has_goal'):
                goal_type = goal.get('goal_type', '?')
                progress = goal.get('goal_progress', 0)
                priority = goal.get('goal_priority', 0)
                status_lines.append(f"🎯 目标:{goal_type} | 进度:{progress:.0%} | 优先级:{priority:.2f}")
            
            # 4. 情绪状态
            if emotion:
                arousal = emotion.get('arousal', 0.5)
                valence = emotion.get('valence', 0.5)
                status_lines.append(f"😊 情绪 唤醒:{arousal:.2f} 效价:{valence:.2f}")
            
            status_str = "\n".join(status_lines) if status_lines else "初始化中..."
            
            # ========== 召回的记忆详情 ==========
            recalled_memories_str = ""
            if hasattr(self.ai, '_last_recalled_memories') and self.ai._last_recalled_memories:
                recalled_memories_str = "\n📖 *召回记忆:*\n"
                for i, mem in enumerate(self.ai._last_recalled_memories[:3], 1):
                    semantic = mem.get('semantic_pointer', 'N/A')[:40]
                    activation = mem.get('activation_strength', 0)
                    recalled_memories_str += f"  {i}. {semantic}... (激活:{activation:.3f})\n"
            
            # ========== 输入提示词 ==========
            input_preview = user_message[:60] + "..." if len(user_message) > 60 else user_message
            
            # 初始状态消息（大幅增强）
            initial_message = await update.message.reply_text(
                f"💭 *[潜意识消化中...]*\n\n"
                f"📝 *输入:* `{input_preview}`\n\n"
                f"📊 *系统状态:*\n{status_str}\n"
                f"{recalled_memories_str}"
                f"_准备思考..._",
                parse_mode='Markdown'
            )
            
            full_response = ""
            monologue = ""
            safe_monologue = ""
            last_update_time = time.time()
            last_sent_text = ""
            update_interval = 1.2
            
            # 存储生成的记忆（用于最终显示）
            # (reserved for future use)
            
            # 使用新的 chat_stream 接口
            history = self.user_history.get(user_id, [])
            async for event in self.ai.chat_stream(user_message, history):
                if event["type"] == "monologue":
                    monologue = event["content"]
                    
                    # 更新召回记忆信息
                    recalled_mem_str = ""
                    if hasattr(self.ai, '_last_recalled_memories') and self.ai._last_recalled_memories:
                        recalled_mem_str = "\n📖 *召回记忆:*\n"
                        for i, mem in enumerate(self.ai._last_recalled_memories[:3], 1):
                            semantic = mem.get('semantic_pointer', 'N/A')[:40]
                            activation = mem.get('activation_strength', 0)
                            recalled_mem_str += f"  {i}. {semantic}... (激活:{activation:.3f})\n"
                    
                    # 立即显示潜意识
                    # 转义 Markdown 特殊字符
                    safe_monologue = self._escape_markdown(monologue)
                    display_text = (
                        f"💭 *[潜意识]*\n"
                        f"_{safe_monologue}_\n\n"
                        f"📊 *系统状态:*\n{status_str}\n"
                        f"{recalled_mem_str}"
                        f"✨ *[准备回复...]*"
                    )
                    await self._safe_edit_message(initial_message, display_text, parse_mode='Markdown')
                    last_sent_text = display_text
                
                elif event["type"] == "chunk":
                    full_response += event["content"]
                    
                    current_time = time.time()
                    if current_time - last_update_time > update_interval:
                        # 更新召回记忆信息
                        recalled_mem_str = ""
                        if hasattr(self.ai, '_last_recalled_memories') and self.ai._last_recalled_memories is not None and len(self.ai._last_recalled_memories) > 0:
                            recalled_mem_str = "\n📖 *召回记忆:*\n"
                            for i, mem in enumerate(self.ai._last_recalled_memories[:2], 1):
                                semantic = mem.get('semantic_pointer', 'N/A')[:35]
                                recalled_mem_str += f"  {i}. {semantic}...\n"
                        
                        display_text = (
                            f"💭 *[潜意识]*\n"
                            f"_{safe_monologue}_\n\n"
                            f"📊 *系统状态:*\n{status_str}\n"
                            f"{recalled_mem_str}"
                            f"✨ *回复:*\n{self._escape_markdown(full_response)}▌"
                        )
                        if display_text != last_sent_text:
                            ok = await self._safe_edit_message(initial_message, display_text, parse_mode='Markdown')
                            if ok:
                                last_sent_text = display_text
                                last_update_time = current_time
                            else:
                                # Flood control 退避后回退到无 Markdown
                                fallback = f"潜意识:\n{monologue}\n\n回复:\n{full_response}▌"
                                await self._safe_edit_message(initial_message, fallback, parse_mode=None)
                                last_update_time = time.time()
                
                # 处理潜意识刷新事件
                elif event["type"] == "subconscious_refresh":
                    monologue = event["content"]  # 更新当前潜意识内容
                    logger.debug(f"[Bot] 潜意识刷新: {monologue[:30]}...")
            
            await typing.stop_typing()
            
            # ========== 最终显示 - 完整详细信息 ==========
            # 获取最新的状态（包括STDP更新后的）
            final_stats = self.ai.get_stats() if self.ai else {}
            stdp_final = final_stats.get('stdp', {})
            hippocampus_final = final_stats.get('hippocampus', {})
            goal_final = final_stats.get('goal', {})
            
            # 1. STDP 学习详情
            stdp_info = ""
            if stdp_final:
                weight_norm = stdp_final.get('dynamic_weight_norm', 0)
                last_update = stdp_final.get('last_update_magnitude', 0)
                total_updates = stdp_final.get('total_updates', 0)
                ltp = stdp_final.get('ltp_count', 0)
                ltd = stdp_final.get('ltd_count', 0)
                stdp_info = (
                    f"🔄 *STDP学习*\n"
                    f"  权重范数: `{weight_norm:.6f}`\n"
                    f"  更新幅度: `{last_update:.6f}`\n"
                    f"  总更新: {total_updates}次\n"
                    f"  LTP增强: {ltp}次 | LTD抑制: {ltd}次\n"
                )
            
            # 2. 海马体记忆详情
            memory_info = ""
            if hippocampus_final:
                num_mem = hippocampus_final.get('num_memories', 0)
                avg_act = hippocampus_final.get('avg_activation', 0)
                core_mem = hippocampus_final.get('core_memory_count', 0)
                recall_count = hippocampus_final.get('recall_count', 0)
                memory_info = (
                    f"🧠 *海马体记忆*\n"
                    f"  总记忆: {num_mem}条 | 核心记忆: {core_mem}条\n"
                    f"  平均激活: {avg_act:.4f}\n"
                    f"  召回次数: {recall_count}次\n"
                )
            
            # 3. 召回的记忆详情
            recalled_str = ""
            if hasattr(self.ai, '_last_recalled_memories') and self.ai._last_recalled_memories is not None and len(self.ai._last_recalled_memories) > 0:
                recalled_str = "📖 *召回的记忆:*\n"
                for i, mem in enumerate(self.ai._last_recalled_memories[:3], 1):
                    semantic = mem.get('semantic_pointer', 'N/A')[:50]
                    activation = mem.get('activation_strength', 0)
                    recalled_str += f"  {i}. {semantic}\n     激活度: {activation:.4f}\n"
            
            # 4. 记住的记忆（本轮存储的新记忆）
            stored_str = ""
            # hippocampus 和 ca3_memory 都是 BrainAIInterface 的必需组件
            if self.ai.hippocampus is not None and self.ai.hippocampus.ca3_memory is not None:
                # 获取最近存储的记忆
                recent_memories = list(self.ai.hippocampus.ca3_memory.memories.values())
                if recent_memories:
                    # 按时间戳排序，取最新的
                    sorted_memories = sorted(recent_memories, key=lambda m: getattr(m, 'timestamp', 0), reverse=True)[:2]
                    if sorted_memories:
                        stored_str = "💾 *记住的记忆:*\n"
                        for i, mem in enumerate(sorted_memories, 1):
                            semantic = getattr(mem, 'semantic_pointer', 'N/A')[:50]
                            is_core = '⭐核心' if getattr(mem, 'is_core', False) else ''
                            stored_str += f"  {i}. {semantic} {is_core}\n"
            
            # 5. 目标进展详情
            goal_str = ""
            if goal_final and goal_final.get('has_goal'):
                goal_type = goal_final.get('goal_type', '?')
                goal_desc = goal_final.get('goal_description', '')[:40]
                progress = min(max(goal_final.get('goal_progress', 0), 0.0), 1.0)
                priority = goal_final.get('goal_priority', 0)
                progress_bar = "█" * int(progress * 10) + "░" * (10 - int(progress * 10))
                goal_str = (
                    f"🎯 *目标状态*\n"
                    f"  类型: {goal_type}\n"
                    f"  描述: {goal_desc}...\n"
                    f"  进度: [{progress_bar}] {progress:.0%}\n"
                    f"  优先级: {priority:.2f}\n"
                )
            
            # 6. 输入提示词预览
            input_str = f"📝 *输入:* `{user_message[:80]}...`\n" if len(user_message) > 80 else f"📝 *输入:* `{user_message}`\n"
            
            # 构建最终消息
            safe_monologue_final = self._escape_markdown(monologue)
            safe_response = self._escape_markdown(full_response)
            final_display = (
                f"💭 *[潜意识]*\n"
                f"_{safe_monologue_final}_\n\n"
                f"{input_str}\n"
                f"📊 *系统状态详情*\n"
                f"{stdp_info}\n"
                f"{memory_info}\n"
                f"{recalled_str}\n"
                f"{stored_str}\n"
                f"{goal_str}\n"
                f"✨ *回复:*\n{safe_response}"
            )
            
            for attempt in range(3):
                try:
                    await initial_message.edit_text(final_display, parse_mode='Markdown')
                    break
                except Exception:
                    try:
                        # 回退到简化版本
                        simplified = (
                            f"潜意识:\n{monologue}\n\n"
                            f"回复:\n{full_response}\n\n"
                            f"STDP权重: {stdp_final.get('dynamic_weight_norm', 0):.6f}\n"
                            f"记忆: {hippocampus_final.get('num_memories', 0)}条"
                        )
                        await initial_message.edit_text(simplified, parse_mode=None)
                        break
                    except Exception:
                        await asyncio.sleep(1)
            
            self._update_history(user_id, user_message, full_response)
            logger.info(f"回复用户 {user_id} 完成 (长度: {len(full_response)})")
            
            # ========== 新增: 应用用户反馈到 STDP 学习 ==========
            if feedback and feedback.is_feedback and hasattr(self.ai, 'apply_user_feedback'):
                try:
                    _feedback_handler = get_feedback_handler()
                    stdp_reward = _feedback_handler.compute_stdp_reward(feedback)
                    logger.info(f"[STDP学习] 应用用户反馈: reward={stdp_reward:.2f}, "
                               f"feedback={'正面' if feedback.is_positive else '负面'}")
                    
                    # 调用 AI 接口应用反馈
                    self.ai.apply_user_feedback(
                        is_positive=feedback.is_positive,
                        intensity=feedback.intensity,
                        reward=stdp_reward
                    )
                    
                    # 如果是强负面反馈，存储为负样本
                    if _feedback_handler.should_store_as_negative_sample(feedback):
                        logger.warning(f"[STDP学习] 标记上一个回复为负样本")
                        # TODO: 可以在这里存储到负样本数据库
                        
                except Exception as e:
                    logger.error(f"[STDP学习] 应用反馈失败: {e}")
            
            # 重置用户交互状态（允许后台思考继续）
            self.is_user_interacting = False
        
        except Exception as e:
            logger.error(f"流式生成失败: {e}", exc_info=True)
            await typing.stop_typing()
            await update.message.reply_text("❌ 生成失败，请稍后重试")
            self.is_user_interacting = False  # 异常时也重置

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
            request = HTTPXRequest(proxy=self.proxy_url)
            self.application = Application.builder().token(self.token).request(request).post_init(self._post_init_hook).build()
        else:
            self.application = Application.builder().token(self.token).post_init(self._post_init_hook).build()
        
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("monitor", self.monitor_command))
        self.application.add_handler(CommandHandler("memory", self.memory_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        self.is_thinking_enabled = True

        logger.info("Bot 已启动，正在监听消息...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)
    
    async def _post_init_hook(self, application):
        """Post-initialization: start background thinking loop"""
        self.thinking_task = asyncio.create_task(self._background_thinking_loop())
        logger.info("[Bot] 后台思维循环已启动")



