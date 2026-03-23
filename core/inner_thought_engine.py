"""
类人脑内心思维独白引擎 (Inner Thought & Monologue Engine)

设计理念:
人的思考和独白是一体的 - 思考时自言自语，独白是思维的外化。
整合主题锚定、联想跳跃、状态切换、流式输出于一体。

核心特性:
1. 统一状态机: 专注→漂移→反思→回归
2. 主题锚定: 思维围绕核心主题展开
3. 联想跳跃: 基于记忆的联想链
4. 流式输出: 打字机效果，可见的思维过程
5. 自闭环优化: 高难度问题自动优化
"""

import torch
import random
import time
import re
import threading
from typing import Generator, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


# ==================== 枚举定义 ====================

class MindState(Enum):
    """思维状态 - 模拟人脑思维的自然流转"""
    FOCUSED = "focused"        # 专注: 深入思考当前主题
    WANDERING = "wandering"    # 漂移: 自由联想，思绪发散
    REFLECTING = "reflecting"  # 反思: 自我审视，元认知
    RESTING = "resting"        # 休息: 后台整理，等待输入


class ThinkingMode(Enum):
    """思维模式 - 影响思考风格"""
    ANALYTICAL = "analytical"      # 分析: 理性分解
    DEDUCTIVE = "deductive"        # 演绎: 逻辑推导
    INDUCTIVE = "inductive"        # 归纳: 总结规律
    CRITICAL = "critical"          # 批判: 质疑审视
    SYNTHESIZING = "synthesizing"  # 综合: 整合信息


# ==================== 数据结构 ====================

@dataclass
class ThoughtTheme:
    """思维主题"""
    content: str                    # 主题内容
    keywords: List[str]             # 关键词
    importance: float = 0.5         # 重要性 (0-1)
    created_time: float = 0.0       # 创建时间
    last_active_time: float = 0.0   # 最后活跃时间
    focus_count: int = 0            # 专注次数
    drift_count: int = 0            # 漂移次数


@dataclass
class ThoughtSegment:
    """思维片段"""
    content: str                    # 内容
    state: MindState                # 思维状态
    mode: ThinkingMode              # 思维模式
    theme: Optional[str] = None     # 所属主题
    timestamp: float = 0.0          # 时间戳
    is_inner_voice: bool = True     # 是否是内心独白


# ==================== 核心引擎 ====================

class InnerThoughtEngine:
    """
    内心思维独白引擎 - 统一的思维系统
    
    整合了:
    - 主题锚定与联想
    - 状态机驱动的思维流转
    - 流式输出
    - 记忆系统连接
    - 自闭环优化触发
    """
    
    def __init__(
        self,
        model_interface,
        hippocampus_system=None,
        self_loop_optimizer=None,
        config=None,
        device: str = "cpu"
    ):
        self.model = model_interface
        self.hippocampus = hippocampus_system
        self.self_loop = self_loop_optimizer
        self.config = config
        self.device = device
        
        # ========== 思维状态 ==========
        self.mind_state = MindState.RESTING
        self.thinking_mode = ThinkingMode.ANALYTICAL
        self.state_duration = 0  # 当前状态持续周期
        
        # ========== 主题系统 ==========
        self.current_theme: Optional[ThoughtTheme] = None
        self.theme_history: deque = deque(maxlen=5)
        
        # ========== 思维流 ==========
        self.thought_flow: deque = deque(maxlen=10)  # 最近的思维片段
        self.current_focus: str = ""  # 当前思维焦点
        self.last_thought: str = ""   # 上一个思维内容
        
        # ========== 联想链 ==========
        self.association_chain: List[str] = []
        self.current_concept: str = ""
        
        # ========== 输出参数 ==========
        self.char_interval = (0.02, 0.06)  # 打字机效果间隔
        
        # ========== 统计 ==========
        self.cycle_count = 0
        self.total_output_chars = 0
        
        # ========== 状态转换矩阵 ==========
        self.state_transitions = {
            MindState.FOCUSED: {
                MindState.WANDERING: 0.20,    # 专注→漂移
                MindState.REFLECTING: 0.15,   # 专注→反思
                MindState.FOCUSED: 0.65       # 保持专注
            },
            MindState.WANDERING: {
                MindState.FOCUSED: 0.35,      # 漂移→回归
                MindState.REFLECTING: 0.20,   # 漂移→反思
                MindState.WANDERING: 0.45     # 继续漂移
            },
            MindState.REFLECTING: {
                MindState.FOCUSED: 0.40,      # 反思→专注
                MindState.RESTING: 0.25,      # 反思→休息
                MindState.REFLECTING: 0.35    # 继续反思
            },
            MindState.RESTING: {
                MindState.FOCUSED: 0.50,      # 休息→专注
                MindState.WANDERING: 0.30,    # 休息→漂移
                MindState.RESTING: 0.20       # 继续休息
            }
        }
        
        # ========== 状态-风格映射 ==========
        self.state_prompts = {
            MindState.FOCUSED: {
                "prefix": "思考中...",
                "triggers": ["深入分析", "仔细想想", "聚焦于", "核心是"]
            },
            MindState.WANDERING: {
                "prefix": "联想到...",
                "triggers": ["说到这个", "顺便想到", "有点像", "让我想起"]
            },
            MindState.REFLECTING: {
                "prefix": "等等...",
                "triggers": ["回顾一下", "这样对吗", "重新审视", "让我确认"]
            },
            MindState.RESTING: {
                "prefix": "...",
                "triggers": ["嗯...", "思考着...", "整理中..."]
            }
        }
        
        self.mode_prompts = {
            ThinkingMode.ANALYTICAL: ["分析", "分解", "梳理", "理解"],
            ThinkingMode.DEDUCTIVE: ["推导", "因此", "所以", "意味着"],
            ThinkingMode.INDUCTIVE: ["归纳", "总结", "规律", "模式"],
            ThinkingMode.CRITICAL: ["质疑", "真的吗", "验证", "确认"],
            ThinkingMode.SYNTHESIZING: ["综合", "整合", "结合", "整体"]
        }
        
        # ========== 初始化 ==========
        self._initialize_mind()
    
    def _initialize_mind(self):
        """初始化思维系统"""
        # 设置初始主题
        initial_themes = [
            "思考当前任务",
            "整理思绪",
            "回顾知识",
            "准备回应"
        ]
        self._set_theme(random.choice(initial_themes))
        
        # 初始状态
        self.mind_state = MindState.RESTING
        self.thinking_mode = ThinkingMode.ANALYTICAL
    
    # ==================== 主题管理 ====================
    
    def _set_theme(self, theme_content: str, importance: float = 0.5):
        """设置新主题"""
        now = time.time()
        keywords = self._extract_keywords(theme_content)
        
        self.current_theme = ThoughtTheme(
            content=theme_content,
            keywords=keywords,
            importance=importance,
            created_time=now,
            last_active_time=now
        )
        self.theme_history.append(self.current_theme)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        if not text or len(text) < 3:
            return []
        
        # 提取2-6个汉字的词组
        pattern = r'[\u4e00-\u9fff]{2,6}'
        matches = re.findall(pattern, text)
        
        # 去重并保留顺序
        seen = set()
        keywords = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                keywords.append(m)
        
        return keywords[:5]  # 最多5个关键词
    
    # ==================== 状态转换 ====================
    
    def _transition_state(self):
        """状态转换"""
        self.state_duration += 1
        
        # 2-4个周期后考虑转换
        if self.state_duration >= random.randint(2, 4):
            probs = self.state_transitions[self.mind_state]
            rand = random.random()
            cumulative = 0
            
            for next_state, prob in probs.items():
                cumulative += prob
                if rand < cumulative:
                    self.mind_state = next_state
                    self.state_duration = 0
                    break
    
    def _select_thinking_mode(self, context: str = ""):
        """根据内容选择思维模式"""
        if not context:
            self.thinking_mode = random.choice(list(ThinkingMode))
            return
        
        # 基于关键词选择模式
        mode_keywords = {
            ThinkingMode.ANALYTICAL: ["分析", "理解", "分解", "梳理"],
            ThinkingMode.DEDUCTIVE: ["推导", "因此", "逻辑", "因果"],
            ThinkingMode.INDUCTIVE: ["总结", "归纳", "规律", "模式"],
            ThinkingMode.CRITICAL: ["质疑", "验证", "确认", "真的"],
            ThinkingMode.SYNTHESIZING: ["综合", "整合", "整体", "结合"]
        }
        
        for mode, keywords in mode_keywords.items():
            if any(kw in context for kw in keywords):
                self.thinking_mode = mode
                return
        
        # 默认随机
        self.thinking_mode = random.choice(list(ThinkingMode))
    
    # ==================== 思维生成 ====================
    
    def generate_inner_thought(
        self,
        external_stimulus: str = "",
        max_tokens: int = 40
    ) -> Generator[str, None, None]:
        """
        生成内心思维独白（流式输出）
        
        Args:
            external_stimulus: 外部刺激（如用户输入）
            max_tokens: 最大token数
        
        Yields:
            char: 字符级流式输出
        """
        self.cycle_count += 1
        
        # 1. 状态转换
        self._transition_state()
        
        # 2. 更新主题
        if external_stimulus:
            self._set_theme(external_stimulus[:50], importance=0.8)
        
        # 3. 构建思维提示
        prompt = self._build_thought_prompt(external_stimulus)
        
        # 4. 获取记忆上下文
        memory_context = self._recall_memory(external_stimulus or self.current_focus)
        
        # 5. 流式生成
        generated_text = ""
        state_prefix = self.state_prompts[self.mind_state]["prefix"]
        
        # 先输出状态前缀
        for char in state_prefix:
            yield char
            generated_text += char
            self.total_output_chars += 1
            time.sleep(random.uniform(*self.char_interval))
        
        # 生成思维内容
        try:
            if hasattr(self.model, 'generate_stream_sync'):
                # 使用流式生成
                full_prompt = f"{memory_context}\n{prompt}" if memory_context else prompt
                for char in self.model.generate_stream_sync(
                    full_prompt, 
                    max_tokens=max_tokens, 
                    temperature=0.8
                ):
                    generated_text += char
                    self.total_output_chars += char
                    yield char
                    time.sleep(random.uniform(*self.char_interval))
            else:
                # 降级：使用预设思维
                thought = self._get_fallback_thought()
                for char in thought:
                    generated_text += char
                    self.total_output_chars += 1
                    yield char
                    time.sleep(random.uniform(*self.char_interval))
        
        except Exception as e:
            # 错误时的优雅降级
            fallback = "思考中..."
            for char in fallback:
                yield char
                generated_text += char
        
        # 6. 记录思维片段
        self._record_thought(generated_text)
        
        # 7. 更新联想链
        self._update_association(generated_text)
    
    def _build_thought_prompt(self, external_stimulus: str = "") -> str:
        """构建思维提示"""
        # 获取当前状态的触发词
        triggers = self.state_prompts[self.mind_state]["triggers"]
        mode_triggers = self.mode_prompts[self.thinking_mode]
        
        trigger = random.choice(triggers)
        mode_trigger = random.choice(mode_triggers)
        
        # 构建提示
        if external_stimulus:
            # 有外部刺激时，围绕刺激展开
            prompt = f"{trigger} {external_stimulus[:30]}... {mode_trigger}"
        elif self.current_theme:
            # 有主题时，围绕主题展开
            prompt = f"{trigger} {self.current_theme.content} {mode_trigger}"
        elif self.last_thought:
            # 延续上一个思维
            prompt = f"延续... {self.last_thought[-20:]} {mode_trigger}"
        else:
            # 初始状态
            prompt = f"{trigger} {mode_trigger}"
        
        return prompt
    
    def _recall_memory(self, query: str) -> str:
        """从海马体召回记忆"""
        if not self.hippocampus or not query:
            return ""
        
        try:
            # 使用模型tokenizer
            if hasattr(self.model, 'tokenizer'):
                input_ids = self.model.tokenizer.encode(query[:30], return_tensors="pt")
                device = getattr(self.model, 'device', 'cpu')
                input_ids = input_ids.to(device)
                
                with torch.no_grad():
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'base_model'):
                        embeddings = self.model.model.base_model.get_input_embeddings()(input_ids)
                        query_features = embeddings.mean(dim=1).squeeze(0)
                    else:
                        return ""
                
                memories = self.hippocampus.recall(query_features, topk=2)
                
                if memories:
                    pointers = [m.get('semantic_pointer', '') for m in memories if m.get('semantic_pointer')]
                    return " | ".join(pointers[:2])
        except Exception:
            pass
        
        return ""
    
    def _get_fallback_thought(self) -> str:
        """获取降级思维"""
        thoughts_by_state = {
            MindState.FOCUSED: [
                "深入思考这个问题...",
                "让我仔细分析一下...",
                "核心要点是什么..."
            ],
            MindState.WANDERING: [
                "这让我联想到...",
                "顺便想到另一个角度...",
                "有点像之前遇到的..."
            ],
            MindState.REFLECTING: [
                "等等，让我确认一下...",
                "这样推理对吗...",
                "重新审视这个观点..."
            ],
            MindState.RESTING: [
                "整理一下思路...",
                "后台处理中...",
                "等待新的输入..."
            ]
        }
        
        thoughts = thoughts_by_state.get(self.mind_state, ["思考中..."])
        return random.choice(thoughts)
    
    def _record_thought(self, content: str):
        """记录思维片段"""
        self.last_thought = content
        
        segment = ThoughtSegment(
            content=content,
            state=self.mind_state,
            mode=self.thinking_mode,
            theme=self.current_theme.content if self.current_theme else None,
            timestamp=time.time()
        )
        self.thought_flow.append(segment)
        
        # 更新主题活跃时间
        if self.current_theme:
            self.current_theme.last_active_time = time.time()
            if self.mind_state == MindState.FOCUSED:
                self.current_theme.focus_count += 1
            elif self.mind_state == MindState.WANDERING:
                self.current_theme.drift_count += 1
    
    def _update_association(self, content: str):
        """更新联想链"""
        keywords = self._extract_keywords(content)
        
        for kw in keywords:
            if kw != self.current_concept:
                self.association_chain.append(kw)
                if len(self.association_chain) > 10:
                    self.association_chain.pop(0)
        
        if keywords:
            self.current_concept = keywords[0]
    
    # ==================== 快速响应 ====================
    
    def get_quick_response(self, user_input: str = "") -> str:
        """获取快速反应填充词"""
        fillers = {
            "thinking": ["嗯...", "让我想想...", "稍等...", "我想想..."],
            "understanding": ["明白了...", "原来如此...", "好的...", "我理解了..."],
            "analyzing": ["分析一下...", "让我看看...", "梳理一下..."],
            "uncertain": ["嗯...不太确定...", "可能..."]
        }
        
        trigger_keywords = {
            "understanding": ["是", "对", "好的", "明白"],
            "analyzing": ["为什么", "怎么", "如何", "什么"],
            "uncertain": ["可能", "也许", "不确定"]
        }
        
        # 检测类型
        response_type = "thinking"
        for rtype, keywords in trigger_keywords.items():
            if any(kw in user_input for kw in keywords):
                response_type = rtype
                break
        
        return random.choice(fillers.get(response_type, fillers["thinking"]))
    
    # ==================== 自闭环优化触发 ====================
    
    def check_self_loop_trigger(self, user_input: str) -> Tuple[bool, str]:
        """
        检查是否应该触发自闭环优化
        
        Returns:
            (should_trigger, mode): 是否触发，以及模式
        """
        if not self.self_loop:
            return False, "self_combine"
        
        mode = self.self_loop.decide_mode(user_input)
        
        # self_game 和 self_eval 模式需要优化
        should_trigger = mode in ["self_game", "self_eval"]
        
        return should_trigger, mode
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "cycle_count": self.cycle_count,
            "total_chars": self.total_output_chars,
            "current_state": self.mind_state.value,
            "current_mode": self.thinking_mode.value,
            "theme": self.current_theme.content if self.current_theme else None,
            "association_chain": self.association_chain[-5:] if self.association_chain else [],
            "thought_segments": len(self.thought_flow)
        }


# ==================== 兼容性别名 ====================

# 兼容旧代码
MonologueEngine = InnerThoughtEngine
ThoughtFlowEngine = InnerThoughtEngine
