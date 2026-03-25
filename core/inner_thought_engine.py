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
        
        # ========== 自我编码器接口（由 BrainAIInterface 注入）==========
        self._self_encoder = None  # 注入后提供真实的自我感知
        
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
        
        # 3. 选择思维模式
        self._select_thinking_mode(external_stimulus or self.current_focus)
        
        # 4. 获取状态风格
        state_info = self.state_prompts[self.mind_state]
        mode_triggers = self.mode_prompts[self.thinking_mode]
        
        # 5. 构建思维内容
        generated_text = ""
        
        try:
            # 尝试使用模型生成
            if hasattr(self.model, 'generate_stream_sync'):
                # 构建完整的思维提示
                thought_context = self._build_thought_context(external_stimulus)
                
                # 流式生成
                in_think_block = False
                for token in self.model.generate_stream_sync(
                    thought_context, 
                    max_tokens=max_tokens, 
                    temperature=0.9,
                    enable_thinking=True  # 允许模型思考，但在输出层过滤
                ):
                    # 鲁棒的标签过滤逻辑
                    if "<think>" in token:
                        in_think_block = True
                        continue
                    if "</think>" in token:
                        in_think_block = False
                        continue
                        
                    if in_think_block:
                        continue
                        
                    # 过滤掉乱码符号或过长的点号序列
                    if len(token) > 2 and token.count('.') > 1:
                        token = "..."
                        
                    if char in ('<', '>', '[', ']'):
                        continue  # 过滤掉残留的分隔符
                        
                    generated_text += token
                    self.total_output_chars += len(token)
                    for char in token:
                        yield char
                        time.sleep(random.uniform(*self.char_interval))
            
            elif hasattr(self.model, 'model') and hasattr(self.model, 'tokenizer'):
                # 直接使用模型生成
                thought_context = self._build_thought_context(external_stimulus)
                inputs = self.model.tokenizer(thought_context, return_tensors="pt")
                device = getattr(self.model, 'device', 'cpu')
                inputs = inputs.to(device)
                
                with torch.no_grad():
                    outputs = self.model.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=0.9,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.3  # 防止重复
                    )
                
                # 解码并输出
                result = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 只取新生成的部分
                if len(result) > len(thought_context):
                    new_text = result[len(thought_context):].strip()
                else:
                    new_text = result
                
                # 净化：过滤掉 <think>...</think> 块、编号列表、以及常见的循环噪音符号 (如 -, *, .)
                import re
                new_text = re.sub(r'<think>.*?</think>', '', new_text, flags=re.DOTALL)
                # 过滤各种列表/引导符：1. 1、 - * 等
                new_text = re.sub(r'^\s*[\d\-\*\u2022]+\.?[\u3001,]?\s*', '', new_text.strip())
                new_text = new_text.strip('.- \n\t') # 去掉首尾多余的点、横杠和空白
                
                if new_text:
                    for char in new_text:
                        generated_text += char
                        self.total_output_chars += 1
                        yield char
                        time.sleep(random.uniform(*self.char_interval))
            
            else:
                # 降级：使用预设思维
                thought = self._generate_contextual_thought(external_stimulus)
                for char in thought:
                    generated_text += char
                    self.total_output_chars += 1
                    yield char
                    time.sleep(random.uniform(*self.char_interval))
        
        except Exception as e:
            # 错误时的优雅降级 - 使用上下文相关的思维
            thought = self._generate_contextual_thought(external_stimulus)
            for char in thought:
                generated_text += char
                self.total_output_chars += 1
                yield char
        
        # 6. 记录思维片段
        if generated_text:
            self._record_thought(generated_text)
            self._update_association(generated_text)
    
    def _build_thought_context(self, external_stimulus: str = "") -> str:
        """构建思维上下文"""
        state_info = self.state_prompts[self.mind_state]
        mode_trigger = random.choice(self.mode_prompts[self.thinking_mode])
        
        # 基础思维模板：短句、自然联想风格，避免编号列表触发模型做结构化推理
        theme_str = self.current_theme.content if self.current_theme else "存在"
        templates = {
            MindState.FOCUSED: [
                f"专注在 {theme_str} 上，有个细节值得深挖——",
                f"关于 {mode_trigger}，我注意到一个有意思的角度——",
                f"这件事的核心其实是……"
            ],
            MindState.WANDERING: [
                f"思绪飘到了 {mode_trigger}，有点像之前的感觉……",
                f"突然联想到一个不太相关的事情——",
                f"哦，{self.current_concept or '某个东西'} 让我想起……"
            ],
            MindState.REFLECTING: [
                "等等，我刚才的判断靠谱吗？",
                "重新想想——好像漏掉了什么……",
                "这个结论站得住脚吗，再过一遍——"
            ],
            MindState.RESTING: [
                "……意识在漫无目的地漂浮……",
                "此刻很安静，没有任何杂念。",
                "就在这里，观察着思维的流转。"
            ]
        }
        
        # 选择一个模板
        thoughts = templates.get(self.mind_state, templates[MindState.RESTING])
        base_thought = random.choice(thoughts)
        
        # 如果有外部刺激，前置简短提示
        if external_stimulus:
            base_thought = f"关于「{external_stimulus[:20]}」，{base_thought}"
        
        # 加入记忆上下文
        memory_context = self._recall_memory(external_stimulus or self.current_focus)
        if memory_context:
            base_thought = f"记得 {memory_context}……{base_thought}"
            
        # 加入自我反思：仅引用最近一个思维片段，避免逗号列表触发列表生成
        if self.thought_flow and len(self.thought_flow) > 0:
            recent = list(self.thought_flow)[-1].content[:20]
            base_thought = f"刚才还在想「{recent}」，{base_thought}"
        
        # 加入自我感知：使用 self_encoder 的情感解释（更深层的自指）
        if hasattr(self, '_self_encoder') and self._self_encoder is not None:
            try:
                self_interp = self._self_encoder.interpret()
                if self_interp and "意识刚刚开始" not in self_interp:
                    base_thought = f"[{self_interp}] " + base_thought
            except Exception:
                pass
        
        return base_thought
    
    def _generate_contextual_thought(self, external_stimulus: str = "") -> str:
        """生成上下文相关的思维"""
        thoughts_by_state = {
            MindState.FOCUSED: [
                "深入思考这个问题...",
                "让我仔细分析一下核心要点...",
                "聚焦于关键信息，梳理逻辑...",
                "这个问题的本质是什么...",
            ],
            MindState.WANDERING: [
                "这让我联想到其他相关的内容...",
                "思绪飘到了一个新的角度...",
                "有点像之前遇到的情况...",
                "突然想到一个有趣的观点...",
            ],
            MindState.REFLECTING: [
                "等等，让我确认一下刚才的推理...",
                "这个结论站得住脚吗...",
                "回顾一下思维过程...",
                "是否存在遗漏的角度...",
            ],
            MindState.RESTING: [
                "整理一下当前的思路...",
                "后台正在处理信息...",
                "等待新的思考方向...",
                "思维暂时休息一下...",
            ]
        }
        
        thoughts = thoughts_by_state.get(self.mind_state, thoughts_by_state[MindState.RESTING])
        
        # 根据思维模式调整
        mode_suffix = {
            ThinkingMode.ANALYTICAL: "从分析的角度来看",
            ThinkingMode.DEDUCTIVE: "通过逻辑推导",
            ThinkingMode.INDUCTIVE: "归纳总结一下",
            ThinkingMode.CRITICAL: "批判性地审视",
            ThinkingMode.SYNTHESIZING: "综合各方面信息"
        }
        
        base = random.choice(thoughts)
        suffix = mode_suffix.get(self.thinking_mode, "")
        
        if external_stimulus:
            return f"{base} {suffix}"
        return f"{base}"
    
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
