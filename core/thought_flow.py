"""
思维流核心引擎 (Thought Flow Engine)

核心特性:
1. 高刷新小数据: 0.8秒刷新，每次2-4 tokens
2. 增量延续: 新思维基于上一思维延续
3. 状态机驱动: 分析→推理→验证→综合
4. 打字机效果: 字符级流式输出
5. 快速响应: 0.3秒内给出填充词反馈
"""

import torch
import random
import time
from typing import Generator, Optional, List, Dict, Any
from collections import deque
from enum import Enum
import threading


class ThoughtState(Enum):
    """思维状态枚举"""
    ANALYZING = "analyzing"      # 分析模式
    REASONING = "reasoning"      # 推理模式
    VERIFYING = "verifying"      # 验证模式
    SYNTHESIZING = "synthesizing" # 综合模式


class ThoughtStateMachine:
    """
    思维状态机
    
    模拟人脑思维状态的自然切换
    """
    
    def __init__(self):
        self.state = ThoughtState.ANALYZING
        self.state_duration = 0  # 当前状态持续周期数
        
        # 状态转换概率矩阵
        self.transitions = {
            ThoughtState.ANALYZING: {
                ThoughtState.REASONING: 0.6,
                ThoughtState.ANALYZING: 0.4
            },
            ThoughtState.REASONING: {
                ThoughtState.VERIFYING: 0.5,
                ThoughtState.REASONING: 0.4,
                ThoughtState.ANALYZING: 0.1
            },
            ThoughtState.VERIFYING: {
                ThoughtState.SYNTHESIZING: 0.6,
                ThoughtState.REASONING: 0.3,
                ThoughtState.VERIFYING: 0.1
            },
            ThoughtState.SYNTHESIZING: {
                ThoughtState.ANALYZING: 0.7,
                ThoughtState.SYNTHESIZING: 0.3
            }
        }
        
        # 状态对应的思维风格前缀
        self.state_prefixes = {
            ThoughtState.ANALYZING: "分析: ",
            ThoughtState.REASONING: "推理: ",
            ThoughtState.VERIFYING: "验证: ",
            ThoughtState.SYNTHESIZING: "综合: "
        }
        
        # 状态对应的思维引导词
        self.state_triggers = {
            ThoughtState.ANALYZING: ["让我分析", "首先看", "分析一下", "梳理"],
            ThoughtState.REASONING: ["因此", "所以", "推导得出", "意味着"],
            ThoughtState.VERIFYING: ["验证", "确认", "检查", "是否成立"],
            ThoughtState.SYNTHESIZING: ["综合来看", "总的来说", "结论是", "归纳"]
        }
    
    def next_state(self) -> ThoughtState:
        """状态转换，返回新状态"""
        # 3-5个周期后考虑切换
        if self.state_duration >= random.randint(3, 5):
            probs = self.transitions[self.state]
            rand = random.random()
            cumulative = 0
            for next_state, prob in probs.items():
                cumulative += prob
                if rand < cumulative:
                    self.state = next_state
                    self.state_duration = 0
                    return self.state
        
        self.state_duration += 1
        return self.state
    
    def get_prefix(self) -> str:
        """获取当前状态的前缀"""
        return self.state_prefixes.get(self.state, "")
    
    def get_trigger(self) -> str:
        """获取当前状态的触发词"""
        triggers = self.state_triggers.get(self.state, ["思考"])
        return random.choice(triggers)


class QuickResponse:
    """
    快速响应机制
    
    模拟人脑的直觉反应，0.3秒内给出简短反馈
    """
    
    def __init__(self):
        # 快速反应填充词库
        self.fillers = {
            "thinking": ["嗯...", "让我想想...", "这个嘛...", "稍等...", "我想想..."],
            "understanding": ["明白了...", "原来如此...", "我理解了...", "好的..."],
            "analyzing": ["分析一下...", "让我看看...", "梳理一下...", "理解中..."],
            "uncertain": ["嗯...不太确定...", "可能...让我想想...", "这个..."],
            "confirming": ["对...", "是的...", "没错...", "确实..."]
        }
        
        # 反应类型检测关键词
        self.trigger_keywords = {
            "understanding": ["是", "对", "好的", "明白", "知道"],
            "analyzing": ["为什么", "怎么", "如何", "分析", "什么"],
            "uncertain": ["可能", "也许", "不确定", "不知道"],
            "confirming": ["真的", "确实", "一定"]
        }
    
    def detect_type(self, text: str) -> str:
        """检测反应类型"""
        text_lower = text.lower()
        for response_type, keywords in self.trigger_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return response_type
        return "thinking"
    
    def get_quick_response(self, user_input: str = "") -> str:
        """获取快速反应填充词"""
        response_type = self.detect_type(user_input)
        fillers = self.fillers.get(response_type, self.fillers["thinking"])
        return random.choice(fillers)


class ThoughtFlowEngine:
    """
    思维流核心引擎
    
    整合三个方案的核心特性:
    - 流式输出 (方案A)
    - 增量思维链 (方案B)
    - 快速响应 (方案C)
    """
    
    def __init__(
        self,
        model_interface,
        refresh_cycle: float = 0.8,
        chunk_size: int = 3,
        context_window: int = 5,
        char_interval_range: tuple = (0.03, 0.1)
    ):
        """
        初始化思维流引擎
        
        Args:
            model_interface: 模型接口
            refresh_cycle: 刷新周期(秒)，默认0.8秒
            chunk_size: 每次生成的token数，默认3
            context_window: 上下文窗口大小，默认5个片段
            char_interval_range: 字符间隔范围(秒)
        """
        self.model = model_interface
        self.refresh_cycle = refresh_cycle
        self.chunk_size = chunk_size
        self.context_window = context_window
        self.char_interval_range = char_interval_range
        
        # 思维流缓冲区
        self.thought_flow: deque = deque(maxlen=context_window)
        
        # 当前思维焦点
        self.current_focus = ""
        
        # 思维状态机
        self.state_machine = ThoughtStateMachine()
        
        # 快速响应器
        self.quick_response = QuickResponse()
        
        # 运行状态
        self.is_running = False
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        # 统计
        self.cycle_count = 0
        self.total_chars_output = 0
    
    def _build_continuation_prompt(self) -> str:
        """
        构建延续性prompt
        
        核心：基于上一思维的结尾延续
        """
        state_prefix = self.state_machine.get_prefix()
        
        if self.thought_flow:
            # 取最近的思维片段构建上下文
            recent = list(self.thought_flow)[-3:]
            context = " → ".join(recent)
            
            # 提取最后一个思维的结尾作为延续点
            last_thought = recent[-1]
            
            # 构建延续性提示
            system_msg = (
                "你是一个持续思考的AI。你的思维是连贯的流。\n"
                "规则：\n"
                "1. 每次只生成2-4个词\n"
                "2. 必须延续上一个思维\n"
                "3. 保持逻辑连贯\n"
                "4. 不要重复，要推进思维"
            )
            
            prompt = f"{system_msg}\n\n思维流: {context}\n继续: "
        else:
            # 初始思维
            trigger = self.state_machine.get_trigger()
            system_msg = (
                "你是一个持续思考的AI。开始一个思维流。\n"
                "规则：每次只生成2-4个词。"
            )
            prompt = f"{system_msg}\n\n开始: {trigger} "
        
        return prompt
    
    def generate_thought_chunk(self) -> Generator[str, None, None]:
        """
        生成一个思维片段
        
        Yields:
            char: 字符级流式输出
        """
        # 构建prompt
        prompt = self._build_continuation_prompt()
        
        try:
            # 优先使用同步流式生成
            if hasattr(self.model, 'generate_stream_sync'):
                for char in self.model.generate_stream_sync(
                    prompt, 
                    max_tokens=self.chunk_size,
                    temperature=0.8
                ):
                    yield char
                    self.total_chars_output += 1
                    # 打字机效果
                    time.sleep(random.uniform(*self.char_interval_range))
            elif hasattr(self.model, 'generate_stream'):
                # 回退到异步流式生成
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def async_gen():
                    async for char in self.model.generate_stream(
                        prompt, 
                        max_tokens=self.chunk_size,
                        temperature=0.8
                    ):
                        yield char
                
                gen = async_gen()
                while True:
                    try:
                        char = loop.run_until_complete(gen.__anext__())
                        yield char
                        self.total_chars_output += 1
                        # 打字机效果
                        time.sleep(random.uniform(*self.char_interval_range))
                    except StopAsyncIteration:
                        break
                
                loop.close()
            else:
                # 最终回退：同步生成 + 模拟流式
                output = self.model.generate(
                    prompt,
                    max_tokens=self.chunk_size,
                    temperature=0.8
                )
                
                text = output.text if hasattr(output, 'text') else str(output)
                
                for char in text:
                    yield char
                    self.total_chars_output += 1
                    time.sleep(random.uniform(*self.char_interval_range))
        
        except Exception as e:
            # 错误时生成默认思维
            default_thoughts = ["...", "继续...", "思考中..."]
            for char in random.choice(default_thoughts):
                yield char
                time.sleep(random.uniform(*self.char_interval_range))
    
    def update_flow(self, chunk: str):
        """更新思维流"""
        if chunk and len(chunk.strip()) > 0:
            self.thought_flow.append(chunk.strip())
            self.current_focus = chunk.strip()[-20:]  # 保存最近焦点
        
        # 状态转换
        self.state_machine.next_state()
        self.cycle_count += 1
    
    def get_quick_response(self, user_input: str = "") -> str:
        """获取快速响应"""
        return self.quick_response.get_quick_response(user_input)
    
    def start_continuous_flow(self, callback=None, duration_minutes: int = 30):
        """
        启动持续思维流
        
        Args:
            callback: 每个思维片段的回调函数
            duration_minutes: 持续时间(分钟)
        """
        self.is_running = True
        self._stop_event.clear()
        self._pause_event.clear()
        
        start_time = time.time()
        duration_seconds = duration_minutes * 60
        
        while self.is_running and not self._stop_event.is_set():
            # 检查暂停
            if self._pause_event.is_set():
                time.sleep(0.1)
                continue
            
            # 检查超时
            if time.time() - start_time > duration_seconds:
                break
            
            # 生成思维片段
            chunk_text = ""
            for char in self.generate_thought_chunk():
                chunk_text += char
                if callback:
                    callback(char)
            
            # 更新流
            self.update_flow(chunk_text)
            
            # 等待下一个周期
            time.sleep(self.refresh_cycle)
    
    def stop(self):
        """停止思维流"""
        self.is_running = False
        self._stop_event.set()
    
    def pause(self):
        """暂停思维流"""
        self._pause_event.set()
    
    def resume(self):
        """恢复思维流"""
        self._pause_event.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'cycle_count': self.cycle_count,
            'total_chars': self.total_chars_output,
            'current_state': self.state_machine.state.value,
            'flow_length': len(self.thought_flow),
            'current_focus': self.current_focus
        }


def create_thought_flow_engine(model_interface, **kwargs) -> ThoughtFlowEngine:
    """创建思维流引擎实例"""
    return ThoughtFlowEngine(model_interface=model_interface, **kwargs)
