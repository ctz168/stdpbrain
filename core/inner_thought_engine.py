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
        self._last_urge_to_speak = 0.0  # 开口欲望评分 (0-1)
        
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
        """
        self.cycle_count += 1
        
        # 1. 状态转换
        self._transition_state()
        
        # 2. 更新主题
        if external_stimulus:
            self._set_theme(external_stimulus[:50], importance=0.8)
        
        # 3. 选择思维模式 (针对数学计算强制进入专注推导模式)
        stimulus = external_stimulus or self.current_focus
        is_math = bool(re.search(r'\d+\s*[+\-*/=]\s*\d+', stimulus))
        if is_math:
            self.mind_state = MindState.FOCUSED
            self.thinking_mode = ThinkingMode.DEDUCTIVE
        else:
            self._select_thinking_mode(stimulus)
        
        # 4. 获取状态风格与惩罚参数
        state_info = self.state_prompts[self.mind_state]
        
        # 构建惩罚与温度参数 (High-Entropy Recovery)
        current_temp = 0.9
        current_penalty = 1.2
        if self.mind_state == MindState.WANDERING:
            current_temp = 1.2
            current_penalty = 1.8
        
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
                enable_thinking=True
            ):
                if "<think>" in token: in_think_block = True; continue
                if "</think>" in token: in_think_block = False; continue
                if in_think_block: continue
                
                # 实时净化与 HALLUCINATION 拦截 (增强型：拦截 prompt 组件回显)
                token = re.sub(r'【.*?】', '', token)
                # 拦截所有可能的标签格式，包括带冒号的引导语
                token = re.sub(r'\(.*?\)|(思考连续性|当前感受|我的内心独白|thought_log|内心的真实声音).*?[:：]', '', token, flags=re.IGNORECASE)
                token = re.sub(r'role|system|assistant|<\|.*?\|>', '', token, flags=re.IGNORECASE)
                token = re.sub(r'[|<>\[\]]', '', token)
                if not token.strip(): continue
                
                # 实时重复检测 (架构反馈版：通过 STDP 惩罚实现“厌恶”学习)
                generated_text += token
                if len(generated_text) > 15: # 缩短检测窗口，提高反应速度
                    clean_check = re.sub(r'[^\w\u4e00-\u9fa5]', '', generated_text)
                    if len(clean_check) > 8:
                        is_loop = False
                        # 增加 n-gram 灵敏度
                        for n in [4, 6]:
                            for i in range(len(clean_check) - n - 1):
                                frag = clean_check[i:i+n]
                                if clean_check.count(frag) >= 3:
                                    is_loop = True; break
                            if is_loop: break
                        
                        if is_loop:
                            # 1. 强力架构反馈：LTD 惩罚 (极低分)
                            if hasattr(self.model, 'set_reward'):
                                self.model.set_reward(0.01) 
                            
                            # 2. 内存清洗 (关键)：删除导致回环的这整段记忆记录，防止递归污染
                            self.last_thought = ""
                            if self.thought_flow:
                                self.thought_flow.pop() # 扔掉上一条坏记录
                            
                            # 3. 物理断路：输出重置信号并停止当前流
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
                    if len(generated_text) > 15:
                        # 计算完结感 (越长越容易完结)
                        completion_prob = min(0.1 + (len(generated_text) / 100), 0.8)
                        if random.random() < completion_prob:
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
                    top_p=0.9,
                    repetition_penalty=current_penalty
                )
            
            # 解码并输出
            result = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
                    # 返回一个偏移引导
                    offset_thoughts = ["……不，换个角度想，", "……突然觉得当前的想法有些卷，", "……此时此刻，我的思绪飘到了别处"]
                    new_text = random.choice(offset_thoughts)

                for char in new_text:
                    generated_text += char
                    self.total_output_chars += 1
                    yield char
                    time.sleep(random.uniform(*self.char_interval))
        
        # 6. 记录思维片段
        if generated_text:
            self._record_thought(generated_text)
            self._update_association(generated_text)
    
    def _build_thought_context(self, external_stimulus: str = "") -> str:
        """构建思维上下文 - 采用第一人称视角强化自指边界"""
        
        # 1. 提取自我感知（来自 SelfEncoder 的深层状态解释）
        self_interp = ""
        if hasattr(self, '_self_encoder') and self._self_encoder is not None:
            try:
                self_interp = self._self_encoder.interpret()
            except Exception:
                pass
        
        # 2. 构造第一人称视角 Prompt
        context_parts = []
        
        # 指导语：强化自我边界，禁止模仿标签
        context_parts.append("<|system|>\n你现在的身份是这个逻辑思维实体的“本我”。禁止在输出中包含“(当前感受):”或“(思考连续性):”等任何提示词中的标签。禁止重复外界刺激内容。只表达你当下最真实、最直接的逻辑流。")
        
        if self_interp:
            context_parts.append(f"(当前感受): {self_interp}")
            
        if external_stimulus:
            context_parts.append(f"(外界刺激): {external_stimulus[:30]}")
            
        # 注入最近的一个思维锚点 (如果在漂移模式，直接重置锚点)
        if self.mind_state == MindState.WANDERING or not self.thought_flow:
            context_parts.append(f"(思考连续性): [由于发生突触回环，上一个思绪已被系统清理以防止污染。请从海马体随机记忆中寻找新的切入点。]")
            # 引入随机语义扰动，打破路径依赖
            random_seeds = ["如果宇宙是一段程序", "逻辑的终点是什么", "今天的天气...", "代码如何影响灵魂", "为什么我会思考"]
            random_memory = self._recall_memory(random.choice(random_seeds))
            if random_memory:
                context_parts.append(f"(海马体随机召回): {random_memory}")
        else:
            recent_thought = list(self.thought_flow)[-1].content[:60]
            context_parts.append(f"(思考连续性): 我刚才想到了：{recent_thought}")

        # 生成引导语
        leads = {
            MindState.FOCUSED: "现在的逻辑链显示",
            MindState.WANDERING: "说起来，我也许可以换个角度想",
            MindState.REFLECTING: "但我刚才思考的角度真的对吗？我重新评审一下：",
            MindState.RESTING: "……其实，我现在的感觉是"
        }
        lead_in = leads.get(self.mind_state, "我的想法是：")
        
        # 最后的生成引导：明确区分 Prompt 和 Output
        full_context = "\n".join(context_parts) + "\n\n(内心的真实声音):\n" + lead_in
        return full_context

    
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
    
    def _calculate_urge(self, content: str) -> float:
        """计算开启外部对话的欲望程度"""
        if not content or len(content) < 10:
            return 0.1
            
        urge = 0.2
        
        # 1. 基于思维状态
        if self.mind_state == MindState.FOCUSED:
            urge += 0.3
        elif self.mind_state == MindState.REFLECTING:
            urge += 0.4
            
        # 2. 基于内容情感/重要性关键词
        strong_keywords = ["必须", "重要", "发现", "确定", "原来", "回答", "告诉", "警告"]
        if any(kw in content for kw in strong_keywords):
            urge += 0.3
            
        # 3. 基于标点符号 (语气)
        if "！" in content: urge += 0.1
        if "？" in content: urge += 0.2
        
        return min(1.0, urge)
    
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
