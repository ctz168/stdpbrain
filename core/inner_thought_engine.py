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
import torch.nn as nn
import torch.nn.functional as F
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
        # CPU优化：禁用字符级延迟（打字机效果）
        # GPU环境下可以启用：(0.02, 0.06)
        self.char_interval = (0.0, 0.0)  # 无延迟
        
        # ========== 统计 ==========
        self.cycle_count = 0
        self.total_output_chars = 0
        self._last_urge_to_speak = 0.0  # 开口欲望评分 (0-1)
        self._last_output_time = 0.0    # 上次开口时间戳（用于时间衰减）
        
        # ========== 自我编码器接口（由 BrainAIInterface 注入）==========
        self._self_encoder = None  # 注入后提供真实的自我感知
        
        # ========== 外部系统引用（由 BrainAIInterface 注入）==========
        self._global_workspace = None  # GlobalWorkspace 引用
        self._goal_system = None        # GoalSystem 引用
        
        # ========== [CPU优化] 预编译正则表达式（避免每次调用 re.compile）==========
        self._re_brackets = re.compile(r'【.*?】')
        self._re_tags = re.compile(r'<\|.*?\|>')
        self._re_pipes = re.compile(r'\|.*?\|')
        self._re_nonword = re.compile(r'[^\w\u4e00-\u9fa5]')
        self._re_chinese = re.compile(r'[\u4e00-\u9fff]{2,6}')
        self._re_math = re.compile(r'\d+\s*[+\-*/=]\s*\d+')
        
        # ========== [动态化] 可学习状态转换矩阵 ==========
        # 4个状态: FOCUSED(0) WANDERING(1) REFLECTING(2) RESTING(3)
        # 初始 logits 对应原始硬编码概率（通过 log 反算）
        # FOCUSED行: [0.65, 0.20, 0.15, 0.00]
        # WANDERING行: [0.35, 0.45, 0.20, 0.00]
        # REFLECTING行: [0.40, 0.00, 0.35, 0.25]
        # RESTING行: [0.50, 0.30, 0.00, 0.20]
        _init_logits = torch.tensor([
            [1.87, 0.00, -0.77, -9.0],   # FOCUSED
            [0.38, 0.78,  0.00, -9.0],   # WANDERING
            [0.47, -9.0,  0.29,  0.00],  # REFLECTING
            [0.92, 0.40, -9.0,  0.00],   # RESTING
        ], dtype=torch.float32)
        self.state_transition_logits = nn.Parameter(_init_logits, requires_grad=False)
        # 用于统计哪些转换路径受到了 STDP 奖惩
        self._transition_reward_accum = torch.zeros(4, 4)  # [from, to]
        self._transition_count = torch.zeros(4, 4, dtype=torch.long)
        
        # 状态索引映射
        self._state_idx = {
            MindState.FOCUSED: 0, MindState.WANDERING: 1,
            MindState.REFLECTING: 2, MindState.RESTING: 3
        }
        self._idx_state = [MindState.FOCUSED, MindState.WANDERING,
                           MindState.REFLECTING, MindState.RESTING]
        
        # ========== 思维模式偏好（由情感+目标联合调节）==========
        self._mode_preference = torch.ones(len(ThinkingMode)) / len(ThinkingMode)
        
        # ========== 状态-风格映射（仍保留，仅用于 prompt 风格渲染）==========
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
        
        # 提取2-6个汉字的词组（使用预编译正则）
        matches = self._re_chinese.findall(text)
        
        # 去重并保留顺序
        seen = set()
        keywords = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                keywords.append(m)
        
        return keywords[:5]  # 最多5个关键词
    
    # ==================== 状态转换 ====================
    
    def _transition_state(self, reward: float = None):
        """
        [动态化] 状态转换 —— 基于可学习矩阵采样，而非固定概率。
        reward: STDP 奖励信号（若提供则更新转换矩阵权重）
        """
        self.state_duration += 1
        
        # 思维惯性：待在同一状态越久，转出概率自动提升（防止永久 RESTING）
        inertia_bonus = min(self.state_duration * 0.1, 0.5)
        
        from_idx = self._state_idx[self.mind_state]
        logits = self.state_transition_logits[from_idx].clone()
        
        # 对自身状态施加惯性衰减（越久越想换状态）
        logits[from_idx] -= inertia_bonus
        
        # Softmax 采样
        probs = F.softmax(logits, dim=0)
        next_idx = torch.multinomial(probs, 1).item()
        next_state = self._idx_state[next_idx]
        
        # 更新 STDP 累计奖励（若有信号）
        if reward is not None:
            self._transition_reward_accum[from_idx, next_idx] += reward
            self._transition_count[from_idx, next_idx] += 1
            # 每 20 次转换后批量更新矩阵 logits
            total = self._transition_count.sum().item()
            if total > 0 and total % 20 == 0:
                self._apply_stdp_to_transition_matrix()
        
        if next_state != self.mind_state:
            self.state_duration = 0
        self.mind_state = next_state
    
    def _apply_stdp_to_transition_matrix(self):
        """
        用 STDP 奖惩信号批量更新状态转换矩阵 logits。
        正奖励 → LTP（增强该路径）；负奖励 → LTD（削弱该路径）。
        """
        count = self._transition_count.float().clamp(min=1)
        avg_reward = self._transition_reward_accum / count  # [4,4]
        
        # 归一化到 [-1, 1]
        r_max = avg_reward.abs().max().item()
        if r_max > 0:
            norm_reward = avg_reward / (r_max + 1e-8)
        else:
            norm_reward = avg_reward
        
        # 直接叠加到 logits（学习率 0.05，有效但温和）
        with torch.no_grad():
            self.state_transition_logits.data += norm_reward * 0.05
            # 限制 logits 范围，防止某条路径被完全封死
            self.state_transition_logits.data.clamp_(-5.0, 5.0)
        
        # 清零累计
        self._transition_reward_accum.zero_()
        self._transition_count.zero_()
    
    def _select_thinking_mode(self, context: str = ""):
        """
        [动态化] 思维模式选择 —— 由 SelfEncoder 情感状态 + GoalSystem 目标类型驱动，
        而非关键词列表匹配。调用链：(arousal, valence) + goal_type → mode 概率分布 → 采样。
        """
        from core.goal_system import GoalType  # 按需导入，避免循环
        
        # 基础：用情感状态调节模式偏好
        mode_scores = self._mode_preference.clone()  # [5]
        
        # 1. SelfEncoder 情感状态驱动
        arousal, valence = 0.5, 0.0
        if hasattr(self, '_self_encoder') and self._self_encoder is not None:
            emo = self._self_encoder.get_emotional_state()
            arousal = emo.get('arousal', 0.5)
            valence = emo.get('valence', 0.0)
        
        # 高唤醒 → 偏向 ANALYTICAL / DEDUCTIVE
        if arousal > 0.65:
            mode_scores[ThinkingMode.ANALYTICAL.value.__hash__() % 5] += 0.3
            mode_scores[ThinkingMode.DEDUCTIVE.value.__hash__() % 5] += 0.2
        # 负效价 → 偏向 CRITICAL
        if valence < -0.25:
            mode_scores[list(ThinkingMode).index(ThinkingMode.CRITICAL)] += 0.4
        # 漂移状态 → 偏向 SYNTHESIZING / INDUCTIVE
        if self.mind_state == MindState.WANDERING:
            mode_scores[list(ThinkingMode).index(ThinkingMode.SYNTHESIZING)] += 0.35
            mode_scores[list(ThinkingMode).index(ThinkingMode.INDUCTIVE)] += 0.25
        # 反思状态 → 偏向 CRITICAL
        if self.mind_state == MindState.REFLECTING:
            mode_scores[list(ThinkingMode).index(ThinkingMode.CRITICAL)] += 0.3
        
        # 2. GoalSystem 目标类型驱动
        if hasattr(self, '_goal_system') and self._goal_system is not None:
            goal_info = self._goal_system.get_current_goal_info()
            goal_type_str = goal_info.get('type', '')
            _goal_mode_map = {
                'understand':    ThinkingMode.ANALYTICAL,
                'solve':         ThinkingMode.DEDUCTIVE,
                'explore':       ThinkingMode.INDUCTIVE,
                'self_reflect':  ThinkingMode.CRITICAL,
                'generate':      ThinkingMode.SYNTHESIZING,
            }
            if goal_type_str in _goal_mode_map:
                preferred = _goal_mode_map[goal_type_str]
                mode_scores[list(ThinkingMode).index(preferred)] += 0.5
        
        # 3. 数学检测（保留原有逻辑，优先级最高）
        if context and self._re_math.search(context):
            self.thinking_mode = ThinkingMode.DEDUCTIVE
            return
        
        # 4. Softmax 采样（含随机性，非贪心）
        probs = F.softmax(mode_scores, dim=0)
        idx = torch.multinomial(probs, 1).item()
        self.thinking_mode = list(ThinkingMode)[idx]
    
    # ==================== 思维生成 ====================
    
    def generate_inner_thought(
        self,
        external_stimulus: str = "",
        max_tokens: int = 120
    ) -> Generator[str, None, None]:
        """
        生成内心思维独白（流式输出）
        
        Args:
            external_stimulus: 外部刺激（如用户输入）
            max_tokens: 最大token数
        """
        self.cycle_count += 1
        
        # 1. 状态转换
        self._transition_state()
        
        # 2. 更新主题
        if external_stimulus:
            self._set_theme(external_stimulus[:50], importance=0.8)
        
        # 3. 选择思维模式 (针对数学计算强制进入专注推导模式)
        stimulus = external_stimulus or self.current_focus
        is_math = bool(self._re_math.search(stimulus))
        if is_math:
            self.mind_state = MindState.FOCUSED
            self.thinking_mode = ThinkingMode.DEDUCTIVE
        else:
            self._select_thinking_mode(stimulus)
        
        # 4. 获取状态风格与惩罚参数
        state_info = self.state_prompts[self.mind_state]
        
        # 构建惩罚与温度参数 (下调过激参数，恢复语言的连贯性和严谨性)
        current_temp = 0.5
        current_penalty = 1.0
        if self.mind_state == MindState.WANDERING:
            current_temp = 0.7
            current_penalty = 1.0
        
        # 5. 构建思维内容
        generated_text = ""
        thought_context = self._build_thought_context(external_stimulus)
        
        # 核心：流式生成 (带实时探测断路器)
        if hasattr(self.model, 'generate_stream_sync'):
            in_think_block = False
            for token in self.model.generate_stream_sync(
                thought_context, 
                max_tokens=max_tokens, 
                temperature=current_temp,
                repetition_penalty=current_penalty,
                # 注意：不启用 enable_thinking，避免 Qwen3.5 产生 <think/> 标签干扰独白流
            ):
                if "<think>" in token: in_think_block = True; continue
                if "</think>" in token: in_think_block = False; continue
                if in_think_block: continue
                
                # 实时净化与 HALLUCINATION 拦截 (温和版：保留中文内容)
                token = self._re_brackets.sub('', token)
                # 过滤各种系统标签（包含 Qwen3.5 的思维标签）
                token = self._re_tags.sub('', token)
                token = self._re_pipes.sub('', token)
                if not token.strip(): continue
                
                # 实时重复检测 (增量式3-gram检测，替代全量n-gram扫描)
                generated_text += token
                if len(generated_text) > 15:
                    clean_check = self._re_nonword.sub('', generated_text)
                    if len(clean_check) > 8:
                        # 增量检测：只检查最新3-gram是否出现在之前文本中
                        is_loop = False
                        for n in [3, 4]:
                            if len(clean_check) >= n * 2:
                                latest_ngram = clean_check[-n:]
                                if latest_ngram in clean_check[:-n]:
                                    is_loop = True
                                    break
                        
                        if is_loop:
                            # 1. 强力架构反馈：LTD 惩罚 (极低分)
                            if hasattr(self.model, 'set_reward'):
                                self.model.set_reward(0.01) 
                            
                            # 2. 内存清洗 (关键)：删除导致回环的这整段记忆记录，防止递归污染
                            self.last_thought = ""
                            if self.thought_flow:
                                self.thought_flow.pop() # 扔掉上一条坏记录
                            
                            # 3. 强制更换思维种子：避免回环后继续生成类似内容
                            self._force_change_seed()
                            
                            # 4. 物理断路：输出重置信号并停止当前流
                            offset = random.choice(["……感知到思维回环。重置语义空间……", "……忽略上述重复。换个话题思考。", "……跳过无效循环。"])
                            for c in offset: yield c; time.sleep(0.01)
                            self.mind_state = MindState.WANDERING
                            break
                
                # 正常展示
                for char in token:
                    yield char
                    time.sleep(random.uniform(*self.char_interval))
                
                # --- 类人特征 1: 自然停顿与完结感 ---
                # 如果检测到完整的句子结束，且长度已经达到一定程度，则根据“完结感”提前退出
                if any(p in token for p in ['。', '！', '？', '...']):
                    if len(generated_text) > 300:
                        recent = generated_text[-30:]
                        sentence_endings = sum(1 for p in ['。', '！', '？'] if p in recent)
                        if sentence_endings >= 3:
                            break            
            # --- 类人特征 2: 评估“开口欲望” (Urge to Speak) ---
            # 根据思维状态和内容重要性计算是否应该转为外部输出
            self._last_urge_to_speak = self._calculate_urge(generated_text)
            
            # --- 智能上下文管理 (无限上下文模拟) ---
            if len(self.thought_flow) > 15:
                # 提取最近思维作为语义锚点存入海马体
                summary = " | ".join([t.content[:15] for t in list(self.thought_flow)[-8:]])
                if self.hippocampus:
                    self.hippocampus.store(summary, "cognitive_anchor")
                # 滚动窗口：保持活跃关注不受旧上下文干扰
                self.thought_flow = self.thought_flow[-5:]
            
            self._record_thought(generated_text)
            return
        
        elif hasattr(self.model, 'model') and hasattr(self.model, 'tokenizer'):
            # 直接使用模型生成
            # 改进：多样化引导词池，彻底斩断前置词引起的复读循环
            leads = {
                MindState.FOCUSED: ["如果从深层逻辑来看，我发现", "仔细推导的话，我认为", "这种逻辑结构让我意识到"],
                MindState.WANDERING: ["说起来，我刚才突然想到", "也许...", "或者说...", "换个角度看...", "我感觉我刚才在想"],
                MindState.REFLECTING: ["但我刚才思考的角度真的对吗？我得重新审视一下：", "我刚才是不是陷入了某种思维惯性？", "重新评估目前的状况："],
                MindState.RESTING: ["……其实，我现在的感觉是", "大脑稍微放松了一点，感觉", "完全静下来的时候，我发现", "刚才那一瞬间，我仿佛"]
            }
            lead_in_list = leads.get(self.mind_state, ["我现在的想法是："])
            lead_in = random.choice(lead_in_list)
            
            # 改进：如果检测到循环并进入了发散模式，大幅提升惩罚系数以强制变轨 (High-Entropy Recovery)
            current_temp = 0.8
            current_penalty = 1.2
            if self.mind_state == MindState.WANDERING:
                current_temp = 1.2
                current_penalty = 1.8 # 极高惩罚强制破坏复读链路
            
            thought_context = self._build_thought_context(external_stimulus)
            inputs = self.model.tokenizer(thought_context, return_tensors="pt")
            device = getattr(self.model, 'device', 'cpu')
            inputs = inputs.to(device)
            
            with torch.no_grad():
                outputs = self.model.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=current_temp,
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=current_penalty
                )
            
            # 解码并输出（线程安全）
            result = self.model.decode_safe(outputs[0], skip_special_tokens=True)
            # 只取新生成的部分
            if len(result) > len(thought_context):
                new_text = result[len(thought_context):].strip()
            else:
                new_text = result
            
            # 净化：过滤各种 hallucinated tags (如 |inner_monologue|, |output|, <Think>等)
            new_text = re.sub(r'<think>.*?</think>', '', new_text, flags=re.IGNORECASE | re.DOTALL)
            new_text = re.sub(r'\|.*?\|', '', new_text) 
            new_text = re.sub(r'【.*?】', '', new_text) # 过滤 【任务目标】这类方括号标签
            new_text = re.sub(r'<\|.*?\|>', '', new_text)
            
            # 过滤掉回显的 lead_in (强化正则过滤，处理空格或标点差异)
            lead_in_pattern = re.escape(lead_in).replace(r'\ ', r'\s*')
            new_text = re.sub(f'^{lead_in_pattern}', '', new_text).strip()
            
            # 过滤各种列表/引导符：1. 1、 - * 等
            new_text = re.sub(r'^\s*[\d\-\*\u2022]+\.?[\u3001,]?\s*', '', new_text.strip())
            new_text = new_text.strip('.- \n\t') # 去掉首尾多余的点、横杠和空白
            
            if new_text:
                # 7. 重复检测与断路 (Loop Breaker) - 增强版
                def get_overlap_ratio(s1, s2):
                    if not s1 or not s2: return 0
                    set1, set2 = set(s1), set(s2)
                    intersection = len(set1.intersection(set2))
                    return intersection / max(len(set1), len(set2))

                is_repetition = False
                if self.last_thought and len(self.last_thought) > 5 and len(new_text) > 5:
                    # 核心净化：移除标点符号、换行符、Markdown 符号 (针对 **存在一种更深层的矛盾**)
                    clean_new = re.sub(r'[^\w\u4e00-\u9fa5]', '', new_text)
                    clean_last = re.sub(r'[^\w\u4e00-\u9fa5]', '', self.last_thought)
                    
                    if len(clean_new) > 5:
                        # 1. 直接包含检测 (需有一定长度，防止误杀短句)
                        if (clean_new in clean_last or clean_last in clean_new) and len(clean_new) > 10:
                            is_repetition = True
                        # 2. 字符重合率检测
                        elif get_overlap_ratio(clean_new, clean_last) > 0.85:
                            is_repetition = True
                        # 3. 内部自我重复检测 (n-gram) - 在清洗后的文本中捕捉结构化重复
                        elif len(clean_new) > 15:
                            for i in range(len(clean_new) - 13):
                                fragment = clean_new[i:i+12]
                                if clean_new.count(fragment) > 1:
                                    is_repetition = True
                                    break
                            if not is_repetition:
                                for i in range(len(clean_new) - 7):
                                    fragment = clean_new[i:i+6]
                                    if clean_new.count(fragment) >= 3:
                                        is_repetition = True
                                        break

                if is_repetition:
                    # 触发状态强制偏移
                    self.mind_state = MindState.WANDERING
                    self.current_focus = ""
                    # [动态化] 从海马体召回随机记忆作为语义跳跃种子
                    # 若海马体无记忆，则从模型生成（而非固定字符串池）
                    new_text = self._generate_semantic_jump()

                for char in new_text:
                    generated_text += char
                    self.total_output_chars += 1
                    yield char
                    time.sleep(random.uniform(*self.char_interval))
        
        # 6. 记录思维片段
        if generated_text:
            self._record_thought(generated_text)
            self._update_association(generated_text)
    
    def _force_change_seed(self):
        """强制更换思维种子，打破思维回环
        
        当检测到思维回环时调用，使用随机的话题转换器
        切换到完全不同的思维方向，避免重复。
        """
        # 预设的话题转换器（覆盖不同领域，确保多样性）
        topic_switchers = [
            "思考一下最近科技的发展方向",
            "回忆一下有趣的经历",
            "分析一下当前的环境和氛围",
            "想象一个完全不同的场景",
            "考虑一下未来的可能性",
            "整理一下刚才的想法",
            "反思一下自己的思考方式",
            "观察周围发生了什么变化",
            "类比另一个领域来思考",
            "从反面来审视这个问题",
        ]
        import random
        new_seed = random.choice(topic_switchers)
        
        # 更新外部刺激（如果接口提供了）
        if hasattr(self, '_external_stimulus'):
            self._external_stimulus = new_seed
        
        # 清空上一次的思维记录，彻底断开回环
        self.last_thought = ""

    def _build_thought_context(self, external_stimulus: str = "") -> str:
        """构建思维上下文 - 自然思维流（优化版：去除机械引导词）"""
        
        context_parts = []
        
        # 根据思维状态选择不同的引导方式（更自然、更像内心独白）
        if self.mind_state == MindState.FOCUSED:
            context_parts.append("（你在深入思考一个问题）")
        elif self.mind_state == MindState.WANDERING:
            context_parts.append("（你的思绪在自由飘荡）")
        elif self.mind_state == MindState.REFLECTING:
            context_parts.append("（你在审视自己的想法）")
        else:
            context_parts.append("（你在安静地感知）")
        
        if external_stimulus:
            context_parts.append(f"（脑海中浮现：{external_stimulus[:50]}）")
        
        # 注入最近的一个思维锚点（简短）
        if self.thought_flow and self.mind_state != MindState.WANDERING:
            recent_thought = list(self.thought_flow)[-1].content[:30]
            context_parts.append(f"（刚才在想：{recent_thought}）")
        
        # 生成引导语 - 完全自然的内心独白开头，不加标点
        leads = {
            MindState.FOCUSED: "我觉得",
            MindState.WANDERING: "忽然想到",
            MindState.REFLECTING: "话说回来",
            MindState.RESTING: "嗯"
        }
        lead_in = leads.get(self.mind_state, "嗯")
        
        # 最后的生成引导（让模型自然续写，不加标点提示）
        full_context = "\n".join(context_parts) + f"\n\n{lead_in}"
        return full_context

    
    def _recall_memory(self, query: str) -> str:
        """从海马体召回记忆"""
        if not self.hippocampus or not query:
            return ""
        
        # 使用模型tokenizer（线程安全）
        if hasattr(self.model, 'encode_safe'):
            input_ids = self.model.encode_safe(query[:30], return_tensors="pt")
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
    
    def _calculate_urge(self, content: str) -> float:
        """
        [动态化] 计算开启外部对话的欲望程度。
        来源：GlobalWorkspace 意识强度 + SelfEncoder 唤醒度 + GoalSystem 奖励信号。
        时间惩罚：输出越频繁，urge 被动压低。
        """
        if not content or len(content) < 10:
            return 0.1
        
        urge = 0.0
        
        # 1. GlobalWorkspace 意识焦点强度（若有注入）
        gw_strength = 0.3  # 默认中等（无 GW 时的回退）
        if hasattr(self, '_global_workspace') and self._global_workspace is not None:
            cs = self._global_workspace.get_consciousness_state()
            if cs is not None:
                # 使用归一化激活强度（均值绝对值）作为意识强度
                gw_strength = min(1.0, cs.abs().mean().item() * 2.0)
        urge += gw_strength * 0.4
        
        # 2. SelfEncoder 情绪唤醒度（高唤醒→更想开口）
        arousal = 0.5
        if hasattr(self, '_self_encoder') and self._self_encoder is not None:
            emo = self._self_encoder.get_emotional_state()
            arousal = emo.get('arousal', 0.5)
        urge += arousal * 0.3
        
        # 3. GoalSystem 目标奖励信号（目标越迫切→越想表达）
        goal_reward = 0.5
        if hasattr(self, '_goal_system') and self._goal_system is not None:
            goal_reward = self._goal_system.get_reward_signal()
        urge += goal_reward * 0.3
        
        # 4. 时间衰减：距上次开口越近，urge 被动压低（防止骚扰式输出）
        if self._last_output_time > 0:
            elapsed = time.time() - self._last_output_time
            # 60秒内输出过 → 最多保留 40% urge；600秒后完全恢复
            time_factor = min(1.0, elapsed / 600.0) * 0.6 + 0.4
            urge *= time_factor
        
        return min(1.0, max(0.0, urge))
    
    def _generate_semantic_jump(self) -> str:
        """
        [动态化] 生成语义跳跃文本 —— 用海马体随机记忆作为思维偏移种子。
        直接通过真实记忆内容引导思维偏移，而非固定字符串池。
        """
        # 1. 尝试从海马体召回随机记忆
        if self.hippocampus:
            # 随机语义种子（不依赖关键词，而是用随机向量探测记忆）
            # 使用海马体的设备和数据类型
            hc_device = getattr(self.hippocampus, 'device', 'cpu')
            random_query = torch.randn(1024, device=hc_device) * 0.5
            memories = self.hippocampus.recall(random_query, topk=1)
            if memories:
                pointer = memories[0].get('semantic_pointer', '')
                if pointer and len(pointer) > 3:
                    # 用记忆内容作为跳跃种子
                    return f"……忽然想到：{pointer[:25]}……"
        
        # 2. 回退：由模型当前主题衍生（而非固定字符串）
        if self.current_theme and self.current_theme.keywords:
            kw = random.choice(self.current_theme.keywords)
            return f"……{kw}……也许这让我想到了别的……"
        
        # 3. 最终回退：使用会变化的状态描述（而非固定字符串）
        jump_seeds = [
            f"……思维跳跃（周期{self.cycle_count}）……",
            f"……感知转移，重新聚焦……",
            f"……切换视角……"
        ]
        return jump_seeds[self.cycle_count % len(jump_seeds)]
    
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
