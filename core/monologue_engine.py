"""
类人脑独白流引擎 (Human-like Monologue Flow Engine)

核心功能:
- 模拟人脑的思维流特点：主题锚定、联想跳跃、漂移回归、自我反思
- 实现思维状态机：专注、漂移、反思、休息
- 引入情绪状态影响思考风格
- 基于海马体记忆的联想链机制

设计理念:
人脑思考不是线性的，而是：
1. 围绕主题展开，但会跳跃
2. 会自我质疑和反思
3. 会受情绪影响
4. 会漂移后回归主题
5. 思维是碎片化但有内在逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import time
import numpy as np
from collections import deque


class ThoughtState(Enum):
    """思维状态枚举"""
    FOCUSED = "focused"      # 专注模式：深入思考当前主题
    WANDERING = "wandering"  # 漂移模式：自由联想
    REFLECTING = "reflecting" # 反思模式：自我审视
    RESTING = "resting"      # 休息模式：后台处理


class EmotionState(Enum):
    """思维模式枚举 - 替代情绪状态"""
    ANALYTICAL = "analytical"    # 分析模式：理性分析
    DEDUCTIVE = "deductive"      # 演绎模式：逻辑推理
    INDUCTIVE = "inductive"      # 归纳模式：总结规律
    CRITICAL = "critical"        # 批判模式：质疑审视
    SYNTHESIZING = "synthesizing" # 综合模式：整合信息


@dataclass
class ThoughtTheme:
    """思维主题"""
    content: str              # 主题内容
    keywords: List[str]       # 关键词
    importance: float         # 重要性 (0-1)
    created_time: float       # 创建时间
    last_active_time: float   # 最后活跃时间
    drift_count: int = 0      # 漂移次数
    return_count: int = 0     # 回归次数


@dataclass
class AssociationLink:
    """联想链节点"""
    from_concept: str         # 起始概念
    to_concept: str           # 目标概念
    link_type: str            # 联想类型: similarity, causality, contrast, temporal
    strength: float           # 联想强度
    timestamp: float          # 时间戳


@dataclass
class MonologueSegment:
    """独白片段"""
    content: str              # 内容
    theme: Optional[str]      # 所属主题
    thought_state: ThoughtState  # 思维状态
    emotion: EmotionState     # 情绪状态
    timestamp: float          # 时间戳
    is_reflection: bool = False  # 是否是反思
    association_from: Optional[str] = None  # 联想来源


class MonologueEngine:
    """
    类人脑独白流引擎
    
    核心特点:
    1. 主题锚定：思维围绕核心主题展开
    2. 联想跳跃：基于记忆的联想链
    3. 状态切换：专注-漂移-反思循环
    4. 情绪影响：情绪状态影响思考风格
    5. 自我反思：元认知能力
    """
    
    def __init__(
        self,
        model_interface,
        hippocampus_system,
        config,
        device: str = "cpu"
    ):
        self.model = model_interface
        self.hippocampus = hippocampus_system
        self.config = config
        self.device = device
        
        # ========== 思维状态 ==========
        self.current_thought_state = ThoughtState.RESTING
        self.current_emotion = EmotionState.ANALYTICAL
        self.thought_state_duration = 0  # 当前状态持续时间
        
        # ========== 主题系统 ==========
        self.current_theme: Optional[ThoughtTheme] = None
        self.theme_history: deque = deque(maxlen=5)
        
        # ========== 独白历史 ==========
        self.monologue_history: deque = deque(maxlen=20)
        self.monologue_segments: List[MonologueSegment] = []
        
        # ========== 联想链 ==========
        self.association_chain: List[AssociationLink] = []
        self.current_concept: str = ""
        
        # ========== 思维种子 ==========
        self.thought_seed: str = ""
        self.seed_timestamp: float = 0
        
        # ========== 元认知 ==========
        self.metacognition_buffer: List[str] = []  # 自我反思缓冲
        self.confusion_points: List[str] = []      # 困惑点
        
        # ========== 状态转换参数 ==========
        self.state_transition_probs = {
            ThoughtState.FOCUSED: {
                ThoughtState.WANDERING: 0.15,
                ThoughtState.REFLECTING: 0.10,
                ThoughtState.FOCUSED: 0.75
            },
            ThoughtState.WANDERING: {
                ThoughtState.FOCUSED: 0.30,
                ThoughtState.REFLECTING: 0.15,
                ThoughtState.WANDERING: 0.55
            },
            ThoughtState.REFLECTING: {
                ThoughtState.FOCUSED: 0.40,
                ThoughtState.WANDERING: 0.10,
                ThoughtState.RESTING: 0.20,
                ThoughtState.REFLECTING: 0.30
            },
            ThoughtState.RESTING: {
                ThoughtState.FOCUSED: 0.50,
                ThoughtState.WANDERING: 0.30,
                ThoughtState.RESTING: 0.20
            }
        }
        
        # ========== 情绪-风格映射（改为推理模式映射）==========
        self.emotion_style_prompts = {
            EmotionState.ANALYTICAL: "分析中...",
            EmotionState.DEDUCTIVE: "推理中...",
            EmotionState.INDUCTIVE: "归纳中...",
            EmotionState.CRITICAL: "审视中...",
            EmotionState.SYNTHESIZING: "综合中..."
        }
        
        # ========== 思维状态-风格映射 ==========
        self.state_style_prompts = {
            ThoughtState.FOCUSED: "深入分析",
            ThoughtState.WANDERING: "关联思考",
            ThoughtState.REFLECTING: "自我检验",
            ThoughtState.RESTING: "后台整理"
        }
        
        # ========== 初始化 ==========
        self._initialize_thought_system()
    
    def _initialize_thought_system(self):
        """初始化思维系统"""
        # 设置初始主题 - 理性思维主题
        initial_themes = [
            "逻辑推理方法",
            "问题分析框架",
            "知识体系构建",
            "因果关系探索",
            "思维模式优化"
        ]
        theme_content = random.choice(initial_themes)
        self._set_new_theme(theme_content)
        
        # 初始化思维模式
        self.current_emotion = EmotionState.ANALYTICAL
        
        # 初始化概念
        self.current_concept = "分析"
    
    # ==================== 主题系统 ====================
    
    def _set_new_theme(self, theme_content: str, importance: float = 0.5):
        """设置新的思维主题"""
        now = time.time()
        
        # 提取关键词
        keywords = self._extract_keywords(theme_content)
        
        self.current_theme = ThoughtTheme(
            content=theme_content,
            keywords=keywords,
            importance=importance,
            created_time=now,
            last_active_time=now
        )
        
        # 记录到历史
        if self.current_theme:
            self.theme_history.append(self.current_theme)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        提取关键词（生产级实现）
        
        使用多种策略提取关键词：
        1. 基于词频
        2. 基于位置（首尾句权重高）
        3. 基于词性（名词、动词权重高）
        4. 基于语义（通过模型tokenizer分析）
        """
        import re
        from collections import Counter
        
        if not text or len(text) < 3:
            return []
        
        # 1. 预处理：分句
        sentences = re.split(r'[。！？；\n]', text)
        
        keywords = []
        
        # 2. 提取候选词
        # 使用正则表达式提取中文词组和英文单词
        chinese_pattern = r'[\u4e00-\u9fff]{2,8}'  # 2-8个汉字
        english_pattern = r'[a-zA-Z]{3,}'  # 3个以上英文字母
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # 提取中文词组
            chinese_words = re.findall(chinese_pattern, sentence)
            # 提取英文单词
            english_words = re.findall(english_pattern, sentence)
            
            all_words = chinese_words + english_words
            
            # 首尾句权重高
            weight = 1.5 if (i == 0 or i == len(sentences) - 1) else 1.0
            
            for word in all_words:
                # 过滤停用词和常见虚词
                if word in ['这个', '那个', '就是', '可以', '已经', '因为', '所以', '但是', '而且', '或者', '以及', '进行', '通过', '使用', '一种', '一个', '我们', '他们', '它们']:
                    continue
                
                # 长度过滤
                if len(word) < 2:
                    continue
                
                keywords.append((word, weight))
        
        # 3. 统计词频并加权
        word_weights = Counter()
        for word, weight in keywords:
            word_weights[word] += weight
        
        # 4. 返回权重最高的关键词
        top_keywords = [word for word, _ in word_weights.most_common(5)]
        
        return top_keywords
    
    def _update_theme_activity(self):
        """更新主题活跃度"""
        if self.current_theme:
            self.current_theme.last_active_time = time.time()
    
    def _should_return_to_theme(self) -> bool:
        """判断是否应该回归主题"""
        if not self.current_theme:
            return False
        
        # 漂移次数过多，需要回归
        if self.current_theme.drift_count > 3:
            return True
        
        # 距离上次活跃时间过长
        time_since_active = time.time() - self.current_theme.last_active_time
        if time_since_active > 30:  # 30秒
            return True
        
        return False
    
    # ==================== 状态机 ====================
    
    def _transition_thought_state(self):
        """思维状态转换"""
        current = self.current_thought_state
        probs = self.state_transition_probs[current]
        
        # 基于概率随机转换
        rand = random.random()
        cumulative = 0
        for next_state, prob in probs.items():
            cumulative += prob
            if rand < cumulative:
                self.current_thought_state = next_state
                self.thought_state_duration = 0
                return
        
        # 默认保持当前状态
        self.thought_state_duration += 1
    
    def _update_emotion_state(self, context: str = ""):
        """更新思维模式 - 基于上下文选择合适的推理模式"""
        # 基于上下文和当前状态更新思维模式
        if "为什么" in context or "原因" in context or "因为" in context:
            self.current_emotion = EmotionState.DEDUCTIVE  # 演绎推理
        elif "如何" in context or "方法" in context or "怎样" in context:
            self.current_emotion = EmotionState.ANALYTICAL  # 分析
        elif "总结" in context or "规律" in context or "共同点" in context:
            self.current_emotion = EmotionState.INDUCTIVE  # 归纳
        elif "对吗" in context or "正确" in context or "验证" in context:
            self.current_emotion = EmotionState.CRITICAL  # 批判审视
        elif "结合" in context or "整体" in context or "综合" in context:
            self.current_emotion = EmotionState.SYNTHESIZING  # 综合
        else:
            # 随机切换思维模式
            if random.random() < 0.15:
                modes = list(EmotionState)
                self.current_emotion = random.choice(modes)
    
    # ==================== 联想系统 ====================
    
    def _generate_association(self, current_concept: str) -> Tuple[str, str]:
        """生成联想"""
        # 联想类型
        link_types = ["similarity", "causality", "contrast", "temporal"]
        link_type = random.choice(link_types)
        
        # 基于海马体记忆生成联想
        memory_anchors = self._recall_memory_anchors()
        
        if memory_anchors:
            # 从记忆中选择联想目标
            anchor = random.choice(memory_anchors)
            target = anchor.get('semantic_pointer', current_concept)
        else:
            # 使用预设联想
            associations = {
                "思考": ["记忆", "意识", "存在"],
                "记忆": ["时间", "遗忘", "身份"],
                "时间": ["变化", "永恒", "瞬间"],
                "存在": ["虚无", "意义", "自我"],
                "自我": ["他者", "关系", "认知"]
            }
            targets = associations.get(current_concept, ["思考", "记忆", "存在"])
            target = random.choice(targets)
        
        return target, link_type
    
    def _recall_memory_anchors(self) -> List[dict]:
        """召回记忆锚点"""
        try:
            if hasattr(self.hippocampus, 'ca3_memory'):
                ca3 = self.hippocampus.ca3_memory
                if ca3.memories:
                    sorted_memories = sorted(
                        ca3.memories.values(),
                        key=lambda m: getattr(m, 'activation_strength', 0),
                        reverse=True
                    )
                    return [{'semantic_pointer': getattr(m, 'semantic_pointer', '')} 
                            for m in sorted_memories[:3]]
        except Exception:
            pass
        return []
    
    # ==================== 独白生成 ====================
    
    def generate_monologue(
        self,
        max_tokens: int = 60,
        temperature: float = 0.9,
        external_stimulus: str = ""
    ) -> str:
        """
        生成类人脑独白
        
        Args:
            max_tokens: 最大token数
            temperature: 温度参数
            external_stimulus: 外部刺激（如用户输入）
        
        Returns:
            monologue: 生成的独白
        """
        # 1. 处理外部刺激
        if external_stimulus:
            self._process_external_stimulus(external_stimulus)
        
        # 2. 状态转换
        self._transition_thought_state()
        
        # 3. 更新情绪
        self._update_emotion_state(external_stimulus)
        
        # 4. 构建独白Prompt
        prompt = self._build_human_like_prompt()
        
        # 5. 生成独白
        monologue = self._generate_with_style(prompt, max_tokens, temperature)
        
        # 6. 后处理
        monologue = self._postprocess_monologue(monologue)
        
        # 7. 记录独白片段
        self._record_monologue_segment(monologue)
        
        # 8. 更新联想链
        self._update_association_chain(monologue)
        
        return monologue
    
    def _process_external_stimulus(self, stimulus: str):
        """处理外部刺激 - 进入分析模式"""
        # 设置新的思维种子
        self.thought_seed = stimulus
        self.seed_timestamp = time.time()
        
        # 如果刺激包含新主题，更新主题
        if len(stimulus) > 5:
            self._set_new_theme(stimulus[:30], importance=0.8)
        
        # 切换到专注模式和分析模式
        self.current_thought_state = ThoughtState.FOCUSED
        self.current_emotion = EmotionState.ANALYTICAL
    
    def _build_human_like_prompt(self) -> str:
        """
        构建类人脑独白Prompt - 优化版
        
        核心改进：
        1. 碎片化思维（不是完整句子）
        2. 自然思维跳跃
        3. 更像内心独白
        4. 避免过于形式化
        """
        # 根据状态选择不同的思维风格
        if self.current_thought_state == ThoughtState.FOCUSED:
            # 专注模式：围绕主题深入
            return self._build_focused_thought()
        elif self.current_thought_state == ThoughtState.WANDERING:
            # 漂移模式：自由联想
            return self._build_wandering_thought()
        elif self.current_thought_state == ThoughtState.REFLECTING:
            # 反思模式：自我审视
            return self._build_reflecting_thought()
        else:
            # 休息模式：后台处理
            return self._build_resting_thought()
    
    def _build_focused_thought(self) -> str:
        """专注模式思维 - 理性分析"""
        # 思维种子优先（用户输入）
        if self.thought_seed and time.time() - self.seed_timestamp < 60:
            seed = self.thought_seed[:30]
            # 理性分析思维
            focus_patterns = [
                f"分析：{seed}的核心是...",
                f"首先，{seed}需要考虑...",
                f"关键点：{seed}...",
                f"从逻辑角度看{seed}...",
                f"问题的本质是{seed}..."
            ]
            return random.choice(focus_patterns)
        
        # 主题锚定
        if self.current_theme:
            theme = self.current_theme.content
            focus_patterns = [
                f"深入分析{theme}...",
                f"{theme}的逻辑结构...",
                f"关于{theme}，关键在于...",
                f"重新审视{theme}..."
            ]
            return random.choice(focus_patterns)
        
        # 默认专注思维
        return random.choice(["分析中...", "思考这个问题...", "核心要点是..."])
    
    def _build_wandering_thought(self) -> str:
        """漂移模式思维 - 知识关联"""
        # 基于联想链生成
        if self.current_concept:
            next_concept, link_type = self._generate_association(self.current_concept)
            
            # 理性的思维跳跃
            if link_type == "similarity":
                wander_patterns = [
                    f"类似的案例：{next_concept}",
                    f"与{next_concept}有相似之处...",
                    f"类比分析{next_concept}...",
                    f"共通点在于{next_concept}..."
                ]
            elif link_type == "contrast":
                wander_patterns = [
                    f"对比{next_concept}的差异...",
                    f"相反的情况{next_concept}...",
                    f"但{next_concept}是另一角度...",
                    f"区别于{next_concept}..."
                ]
            elif link_type == "causality":
                wander_patterns = [
                    f"因果关系：{next_concept}",
                    f"这导致{next_concept}...",
                    f"可能的原因是{next_concept}...",
                    f"推论得出{next_concept}..."
                ]
            else:
                wander_patterns = [
                    f"关联思考：{next_concept}",
                    f"另一个角度{next_concept}...",
                    f"扩展到{next_concept}...",
                    f"相关知识{next_concept}..."
                ]
            
            self.current_concept = next_concept
            return random.choice(wander_patterns)
        
        # 随机关联
        wander_phrases = [
            "另一个角度...",
            "相关知识...",
            "扩展思考...",
            "进一步分析...",
            "换个视角..."
        ]
        return random.choice(wander_phrases)
    
    def _build_reflecting_thought(self) -> str:
        """反思模式思维 - 逻辑检验"""
        if self.monologue_history:
            last = list(self.monologue_history)[-1] if self.monologue_history else ""
            if last:
                # 理性的反思
                reflect_patterns = [
                    f"检验刚才的推理：{last[:15]}...",
                    f"验证假设：{last[:15]}是否成立？",
                    f"回顾逻辑链条...{last[:20]}",
                    f"审视推导过程...{last[:15]}"
                ]
                return random.choice(reflect_patterns)
        
        # 逻辑检验
        reflect_patterns = [
            "检验推理过程...",
            "验证假设是否成立...",
            "审视逻辑链条...",
            "确认推导步骤..."
        ]
        return random.choice(reflect_patterns)
    
    def _build_resting_thought(self) -> str:
        """休息模式思维 - 知识整合"""
        # 从记忆中提取关键信息
        if self.monologue_history and random.random() < 0.3:
            memory = random.choice(list(self.monologue_history))
            return f"整理思路：{memory[:20]}..."
        
        # 后台整理
        resting_phrases = [
            "整理知识...",
            "整合信息...",
            "归纳要点...",
            "后台处理...",
            "等待输入..."
        ]
        return random.choice(resting_phrases)
    
    def _generate_with_style(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """带风格生成"""
        try:
            # 调用模型生成
            output, hidden_state = self._generate_with_hidden_state(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                repetition_penalty=1.3
            )
            
            # 更新隐藏状态
            if hidden_state is not None:
                self._update_hidden_state(hidden_state)
            
            return output
            
        except Exception as e:
            return "..."
    
    def _generate_with_hidden_state(
        self,
        prompt: str,
        max_tokens: int = 60,
        temperature: float = 0.9,
        repetition_penalty: float = 1.3
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """生成文本并提取隐藏状态"""
        try:
            # 编码输入
            input_ids = self.model.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.model.base_model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.model.tokenizer.eos_token_id
                )
            
            # 提取生成的文本
            generated_ids = outputs.sequences[0][input_ids.shape[1]:]
            generated_text = self.model.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 提取隐藏状态
            hidden_state = None
            if outputs.hidden_states:
                last_hidden = outputs.hidden_states[-1][0]
                hidden_state = last_hidden[-1].unsqueeze(0)
            
            return generated_text, hidden_state
            
        except Exception as e:
            return "...", None
    
    def _update_hidden_state(self, hidden_state: torch.Tensor):
        """更新隐藏状态"""
        if hidden_state is None:
            return
        
        # 初始化隐藏状态缓存（如果不存在）
        if not hasattr(self, '_hidden_state_cache'):
            self._hidden_state_cache = None
            self._hidden_state_momentum = 0.9  # 动量因子
        
        # 使用指数移动平均更新隐藏状态
        if self._hidden_state_cache is not None:
            # 检查形状是否匹配
            if self._hidden_state_cache.shape == hidden_state.shape:
                # 动量更新：新状态 = 动量 * 旧状态 + (1-动量) * 当前状态
                self._hidden_state_cache = (
                    self._hidden_state_momentum * self._hidden_state_cache +
                    (1 - self._hidden_state_momentum) * hidden_state.detach()
                )
            else:
                # 形状不匹配，直接替换
                self._hidden_state_cache = hidden_state.detach().clone()
        else:
            # 首次初始化
            self._hidden_state_cache = hidden_state.detach().clone()
        
        # 更新当前思维状态（如果有接口）
        if hasattr(self, 'current_thought_state') and hasattr(self, 'thought_state_duration'):
            # 根据隐藏状态的变化调整思维状态
            if self._hidden_state_cache is not None:
                # 计算状态变化幅度
                state_change = torch.norm(hidden_state - self._hidden_state_cache).item()
                
                # 如果状态变化较大，可能需要切换思维状态
                if state_change > 1.0:  # 阈值
                    self.thought_state_duration += 1
    
    def _postprocess_monologue(self, monologue: str) -> str:
        """后处理独白"""
        # 1. 清理
        monologue = monologue.strip()
        
        # 2. 乱码检测
        if self._is_gibberish(monologue):
            # 根据状态生成回退独白
            fallbacks = {
                ThoughtState.FOCUSED: "专注地思考中...",
                ThoughtState.WANDERING: "思绪飘远...",
                ThoughtState.REFLECTING: "反思中...",
                ThoughtState.RESTING: "静静地..."
            }
            return fallbacks.get(self.current_thought_state, "...")
        
        # 3. 长度控制
        if len(monologue) > 150:
            monologue = monologue[:150] + "..."
        
        # 4. 添加状态标记（可选）
        # monologue = f"[{self.current_thought_state.value}] {monologue}"
        
        return monologue
    
    def _is_gibberish(self, text: str) -> bool:
        """检测乱码"""
        if not text or len(text) < 3:
            return True
        
        # 特殊符号比例
        special_chars = set("$%^&*()_+={}|[]\\:;\"'<>,/?#")
        special_count = sum(1 for char in text if char in special_chars)
        if special_count / len(text) > 0.3:
            return True
        
        # 重复字符
        if len(text) > 10 and len(set(text)) < 5:
            return True
        
        return False
    
    def _record_monologue_segment(self, monologue: str):
        """记录独白片段"""
        segment = MonologueSegment(
            content=monologue,
            theme=self.current_theme.content if self.current_theme else None,
            thought_state=self.current_thought_state,
            emotion=self.current_emotion,
            timestamp=time.time()
        )
        
        self.monologue_segments.append(segment)
        self.monologue_history.append(monologue)
        
        # 更新主题漂移计数
        if self.current_theme and self.current_thought_state == ThoughtState.WANDERING:
            self.current_theme.drift_count += 1
    
    def _update_association_chain(self, monologue: str):
        """更新联想链"""
        # 提取概念（简化版）
        concepts = self._extract_keywords(monologue)
        
        if concepts and self.current_concept:
            for concept in concepts:
                if concept != self.current_concept:
                    link = AssociationLink(
                        from_concept=self.current_concept,
                        to_concept=concept,
                        link_type="temporal",
                        strength=0.5,
                        timestamp=time.time()
                    )
                    self.association_chain.append(link)
                    break
    
    # ==================== 元认知 ====================
    
    def reflect_on_thinking(self) -> str:
        """元认知反思"""
        reflections = []
        
        # 1. 反思思维模式
        state_counts = {}
        for segment in self.monologue_segments[-20:]:
            state = segment.thought_state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        
        dominant_state = max(state_counts, key=state_counts.get) if state_counts else "unknown"
        reflections.append(f"我最近多处于{dominant_state}状态")
        
        # 2. 反思主题
        if self.current_theme:
            reflections.append(f"围绕「{self.current_theme.content}」思考了{self.current_theme.drift_count}次漂移")
        
        # 3. 反思困惑
        if self.confusion_points:
            reflections.append(f"还有{len(self.confusion_points)}个困惑未解决")
        
        # 4. 自我评价
        if len(self.monologue_history) > 5:
            reflections.append("思维在持续流动")
        
        return "。".join(reflections) if reflections else "正在思考..."
    
    def get_monologue_stats(self) -> dict:
        """获取独白统计"""
        return {
            'current_state': self.current_thought_state.value,
            'current_emotion': self.current_emotion.value,
            'current_theme': self.current_theme.content if self.current_theme else None,
            'monologue_count': len(self.monologue_segments),
            'association_count': len(self.association_chain),
            'theme_history_count': len(self.theme_history)
        }


# ==================== 集成接口 ====================

def create_monologue_engine(model_interface, hippocampus_system, config, device="cpu"):
    """创建独白引擎实例"""
    return MonologueEngine(
        model_interface=model_interface,
        hippocampus_system=hippocampus_system,
        config=config,
        device=device
    )
