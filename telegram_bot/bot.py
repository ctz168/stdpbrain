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
                        message = await self.application.bot.send_message(
                            chat_id=chat_id,
                            text="💭 *[内心独白]*\n_思考中..._",
                            parse_mode='Markdown'
                        )
                        
                        # 流式生成独白
                        full_monologue = ""
                        last_update_time = time.time()
                        last_sent_text = ""
                        update_interval = 1.5  # 增加间隔以避免 429
                        
                        async for chunk in self.ai.generate_monologue_stream(max_tokens=150):
                            full_monologue += chunk
                            
                            current_time = time.time()
                            if current_time - last_update_time > update_interval:
                                display_text = full_monologue[:500]
                                new_text = f"💭 *[内心独白]*\n_{display_text}▌_"
                                
                                # 只有在内容变化时才更新，避免 400 错误
                                if new_text != last_sent_text:
                                    await message.edit_text(
                                        text=new_text,
                                        parse_mode='Markdown'
                                    )
                                    last_sent_text = new_text
                                    last_update_time = current_time
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
                            final_text = f"💭 *[内心独白]*\n_{clean_monologue}_"
                            if final_text != last_sent_text:
                                await message.edit_text(
                                    text=final_text,
                                    parse_mode='Markdown'
                                )
                            
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
        
        try:
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
            stats_text += f"  平均激活: {hippocampus.get('avg_activation', 0):.3f}\n"
            stats_text += f"  KV记忆数: {hippocampus.get('kv_memory_count', 0)}\n\n"
            
            # 2. STDP学习情况
            stdp = stats.get('stdp', {})
            stats_text += "🔄 [*STDP学习*]\n"
            stats_text += f"  学习周期: {stdp.get('cycle_count', 0)}\n"
            stats_text += f"  总更新次数: {stdp.get('total_updates', 0)}\n"
            stats_text += f"  LTP增强: {stdp.get('ltp_count', 0)}\n"
            stats_text += f"  LTD抑制: {stdp.get('ltd_count', 0)}\n"
            stats_text += f"  动态权重范数: {stdp.get('dynamic_weight_norm', 0):.6f}\n"
            stats_text += f"  最近更新幅度: {stdp.get('last_update_magnitude', 0):.6f}\n"
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
            
            # 系统状态
            system = stats.get('system', {})
            stats_text += "⚙️ [*系统状态*]\n"
            stats_text += f"  总周期: {system.get('total_cycles', 0)}\n"
            stats_text += f"  设备: {system.get('device', 'cpu')}\n"
            stats_text += f"  思维状态: {'存在' if system.get('has_thought_state') else '缺失'}\n"
            
            await update.message.reply_text(stats_text, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ 获取统计失败：{e}")
    
    async def monitor_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """实时监控命令 - 显示当前状态快照"""
        if not self.ai:
            await update.message.reply_text("⚠️ AI 模型未初始化")
            return
        
        try:
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
            if hippocampus.get('last_recall_time', 0) > 0:
                import time as t
                elapsed = t.time() - hippocampus.get('last_recall_time', 0)
                monitor_text += f"  最近召回: {elapsed:.1f}秒前\n\n"
            
            # 当前学习状态
            stdp = stats.get('stdp', {})
            monitor_text += "📚 [*学习状态*]\n"
            monitor_text += f"  更新次数: {stdp.get('total_updates', 0)}\n"
            monitor_text += f"  权重范数: {stdp.get('dynamic_weight_norm', 0):.6f}\n"
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
                progress_bar = "█" * int(goal.get('goal_progress', 0) * 10)
                progress_bar += "░" * (10 - len(progress_bar))
                monitor_text += f"  进度: [{progress_bar}] {goal.get('goal_progress', 0):.0%}\n\n"
            
            # 当前情绪
            emotion = stats.get('emotion', {})
            if emotion:
                monitor_text += "😊 [*当前情绪*]\n"
                arousal = emotion.get('arousal', 0.5)
                valence = emotion.get('valence', 0.5)
                arousal_bar = "█" * int(arousal * 10) + "░" * (10 - int(arousal * 10))
                valence_bar = "█" * int(valence * 10) + "░" * (10 - int(valence * 10))
                monitor_text += f"  唤醒: [{arousal_bar}] {arousal:.2f}\n"
                monitor_text += f"  效价: [{valence_bar}] {valence:.2f}\n\n"
            
            # KV状态
            kv = stats.get('kv', {})
            monitor_text += "📦 [*KV状态*]\n"
            monitor_text += f"  活跃: {kv.get('active_kv_count', 0)}\n"
            monitor_text += f"  窗口: {kv.get('window_size', 32)}\n"
            
            # 添加时间戳
            from datetime import datetime
            monitor_text += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
            
            await update.message.reply_text(monitor_text, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ 获取监控失败：{e}")
    
    async def memory_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """查看记忆详情命令"""
        if not self.ai:
            await update.message.reply_text("⚠️ AI 模型未初始化")
            return
        
        try:
            # 获取记忆系统详细信息
            if hasattr(self.ai, 'hippocampus') and hasattr(self.ai.hippocampus, 'ca3_memory'):
                ca3 = self.ai.hippocampus.ca3_memory
                memories = list(ca3.memories.values()) if hasattr(ca3, 'memories') else []
                
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
            else:
                await update.message.reply_text("❌ 记忆系统未初始化")
        except Exception as e:
            await update.message.reply_text(f"❌ 获取记忆详情失败：{e}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        user_message = update.message.text
        
        self.last_active_chat_id = chat_id
        
        if not user_message:
            return
        
        logger.info(f"收到用户 {user_id} 消息：{user_message[:50]}...")
        
        # ========== 新增: 检测用户反馈 ==========
        feedback_handler = get_feedback_handler()
        feedback = feedback_handler.detect_feedback(user_message)
        
        if feedback.is_feedback:
            logger.info(f"[用户反馈] 类型={'正面' if feedback.is_positive else '负面'}, "
                       f"强度={feedback.intensity:.2f}, 关键词={feedback.keywords_matched}")
            
            # 如果是负面反馈，标记上一个回复需要惩罚
            if not feedback.is_positive and feedback.intensity >= 0.7:
                logger.warning(f"[用户反馈] 检测到强负面反馈，将触发 STDP LTD 学习")
        
        # ========== 新增: 检测用户反馈 ==========
        
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
                user_message=user_message,
                feedback=feedback  # 传递反馈信息
            )
            
        except Exception as e:
            logger.error(f"处理消息失败：{e}", exc_info=True)
            await typing.stop_typing()
            await update.message.reply_text(f"[FAIL] 处理失败：{str(e)}")
    
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
            
            # 构建状态指示器
            status_indicators = []
            if emotion:
                arousal = emotion.get('arousal', 0.5)
                valence = emotion.get('valence', 0.5)
                status_indicators.append(f"情绪:{arousal:.1f}/{valence:.1f}")
            if goal and goal.get('has_goal'):
                status_indicators.append(f"目标:{goal.get('goal_type', '?')}")
            
            status_str = " | ".join(status_indicators) if status_indicators else "初始化"
            
            # 初始状态消息
            initial_message = await update.message.reply_text(
                f"💭 *[潜意识消化中...]*\n"
                f"📊 [{status_str}]\n"
                f"_准备思考..._",
                parse_mode='Markdown'
            )
            
            full_response = ""
            monologue = ""
            last_update_time = time.time()
            last_sent_text = ""
            update_interval = 1.2
            
            # 使用新的 chat_stream 接口
            history = self.user_history.get(user_id, [])
            async for event in self.ai.chat_stream(user_message, history):
                if event["type"] == "monologue":
                    monologue = event["content"]
                    # 立即显示潜意识
                    display_text = (
                        f"💭 *[潜意识]*\n"
                        f"_{monologue}_\n\n"
                        f"📊 [{status_str}]\n"
                        f"✨ *[准备回复...]*"
                    )
                    try:
                        await initial_message.edit_text(display_text, parse_mode='Markdown')
                        last_sent_text = display_text
                    except:
                        pass
                
                elif event["type"] == "chunk":
                    full_response += event["content"]
                    
                    current_time = time.time()
                    if current_time - last_update_time > update_interval:
                        display_text = (
                            f"💭 *[潜意识]*\n"
                            f"_{monologue}_\n\n"
                            f"📊 [{status_str}]\n"
                            f"✨ **回复:**\n{full_response}▌"
                        )
                        if display_text != last_sent_text:
                            try:
                                await initial_message.edit_text(display_text, parse_mode='Markdown')
                                last_sent_text = display_text
                                last_update_time = current_time
                            except Exception as e:
                                if "Flood control exceeded" in str(e):
                                    await asyncio.sleep(2)
                                else:
                                    # 回退到无 Markdown
                                    try:
                                        fallback = f"潜意识:\n{monologue}\n\n回复:\n{full_response}▌"
                                        await initial_message.edit_text(fallback, parse_mode=None)
                                    except:
                                        pass
            
            await typing.stop_typing()
            
            # 最终显示 - 包含学习状态
            stdp = current_stats.get('stdp', {})
            learning_info = f"📚 学习更新: {stdp.get('total_updates', 0)}"
            
            final_display = (
                f"💭 *[潜意识]*\n"
                f"_{monologue}_\n\n"
                f"📊 [{status_str}]\n"
                f"✨ **回复:**\n{full_response}\n\n"
                f"{learning_info}"
            )
            
            for attempt in range(3):
                try:
                    await initial_message.edit_text(final_display, parse_mode='Markdown')
                    break
                except:
                    try:
                        await initial_message.edit_text(
                            f"潜意识:\n{monologue}\n\n回复:\n{full_response}\n\n{learning_info}",
                            parse_mode=None
                        )
                        break
                    except:
                        await asyncio.sleep(1)
            
            self._update_history(user_id, user_message, full_response)
            logger.info(f"回复用户 {user_id} 完成 (长度: {len(full_response)})")
            
            # ========== 新增: 应用用户反馈到 STDP 学习 ==========
            if feedback and feedback.is_feedback and hasattr(self.ai, 'apply_user_feedback'):
                try:
                    stdp_reward = feedback_handler.compute_stdp_reward(feedback)
                    logger.info(f"[STDP学习] 应用用户反馈: reward={stdp_reward:.2f}, "
                               f"feedback={'正面' if feedback.is_positive else '负面'}")
                    
                    # 调用 AI 接口应用反馈
                    self.ai.apply_user_feedback(
                        is_positive=feedback.is_positive,
                        intensity=feedback.intensity,
                        reward=stdp_reward
                    )
                    
                    # 如果是强负面反馈，存储为负样本
                    if feedback_handler.should_store_as_negative_sample(feedback):
                        logger.warning(f"[STDP学习] 标记上一个回复为负样本")
                        # TODO: 可以在这里存储到负样本数据库
                        
                except Exception as e:
                    logger.error(f"[STDP学习] 应用反馈失败: {e}")
            
        except Exception as e:
            logger.error(f"流式生成失败: {e}", exc_info=True)
            await typing.stop_typing()
            await update.message.reply_text(f"[FAIL] 生成失败: {str(e)}")
    
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
            self.application = Application.builder().token(self.token).request(request).build()
        else:
            self.application = Application.builder().token(self.token).build()
        
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("monitor", self.monitor_command))
        self.application.add_handler(CommandHandler("memory", self.memory_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        self.is_thinking_enabled = True
        self.application.post_init = self._post_init_hook
        
        logger.info("Bot 已启动，正在监听消息...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)
    
    async def _post_init_hook(self, application: Application):
        self.thinking_task = asyncio.create_task(self._background_thinking_loop())
        logger.info("后台思考任务已启动")



