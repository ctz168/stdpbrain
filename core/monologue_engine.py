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
    """情绪状态枚举"""
    CURIOUS = "curious"      # 好奇
    CONFUSED = "confused"    # 困惑
    EXCITED = "excited"      # 兴奋
    CALM = "calm"           # 平静
    THOUGHTFUL = "thoughtful" # 沉思


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
        self.current_emotion = EmotionState.CALM
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
        
        # ========== 情绪-风格映射 ==========
        self.emotion_style_prompts = {
            EmotionState.CURIOUS: "好奇地探索...",
            EmotionState.CONFUSED: "困惑地思考...",
            EmotionState.EXCITED: "兴奋地联想...",
            EmotionState.CALM: "平静地沉思...",
            EmotionState.THOUGHTFUL: "深沉地反思..."
        }
        
        # ========== 思维状态-风格映射 ==========
        self.state_style_prompts = {
            ThoughtState.FOCUSED: "专注于",
            ThoughtState.WANDERING: "思绪飘向",
            ThoughtState.REFLECTING: "反思刚才的想法",
            ThoughtState.RESTING: "静静地"
        }
        
        # ========== 初始化 ==========
        self._initialize_thought_system()
    
    def _initialize_thought_system(self):
        """初始化思维系统"""
        # 设置初始主题
        initial_themes = [
            "存在的意义",
            "思维的奥秘",
            "记忆与时间",
            "自我与他者",
            "知识的边界"
        ]
        theme_content = random.choice(initial_themes)
        self._set_new_theme(theme_content)
        
        # 设置初始情绪
        self.current_emotion = EmotionState.CALM
        
        # 初始化概念
        self.current_concept = "思考"
    
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
        """提取关键词（简化版）"""
        # 简单分词，实际应用中可以使用更复杂的NLP
        words = text.replace("的", " ").replace("与", " ").split()
        return [w for w in words if len(w) > 1][:5]
    
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
        """更新情绪状态"""
        # 基于上下文和当前状态更新情绪
        if "？" in context or "困惑" in context or "不明白" in context:
            self.current_emotion = EmotionState.CONFUSED
        elif "！" in context or "发现" in context or "原来" in context:
            self.current_emotion = EmotionState.EXCITED
        elif "思考" in context or "反思" in context:
            self.current_emotion = EmotionState.THOUGHTFUL
        elif "为什么" in context or "如何" in context:
            self.current_emotion = EmotionState.CURIOUS
        else:
            # 随机小幅度变化
            if random.random() < 0.2:
                emotions = list(EmotionState)
                self.current_emotion = random.choice(emotions)
    
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
        """处理外部刺激"""
        # 设置新的思维种子
        self.thought_seed = stimulus
        self.seed_timestamp = time.time()
        
        # 如果刺激包含新主题，更新主题
        if len(stimulus) > 5:
            self._set_new_theme(stimulus[:30], importance=0.8)
        
        # 切换到专注模式
        self.current_thought_state = ThoughtState.FOCUSED
        self.current_emotion = EmotionState.CURIOUS
    
    def _build_human_like_prompt(self) -> str:
        """构建类人脑独白Prompt"""
        parts = []
        
        # 1. 状态和情绪前缀
        state_prefix = self.state_style_prompts[self.current_thought_state]
        emotion_prefix = self.emotion_style_prompts[self.current_emotion]
        
        # 2. 主题锚定
        if self.current_theme and self.current_thought_state == ThoughtState.FOCUSED:
            parts.append(f"关于「{self.current_theme.content}」")
        
        # 3. 联想链
        if self.current_concept and self.current_thought_state == ThoughtState.WANDERING:
            next_concept, link_type = self._generate_association(self.current_concept)
            if link_type == "similarity":
                parts.append(f"从{self.current_concept}联想到相似的{next_concept}")
            elif link_type == "contrast":
                parts.append(f"想到{self.current_concept}，却又想到相反的{next_concept}")
            elif link_type == "causality":
                parts.append(f"{self.current_concept}让我想到{next_concept}")
            else:
                parts.append(f"从{self.current_concept}到{next_concept}")
            self.current_concept = next_concept
        
        # 4. 反思模式
        if self.current_thought_state == ThoughtState.REFLECTING:
            if self.monologue_history:
                last_thought = list(self.monologue_history)[-1] if self.monologue_history else ""
                parts.append(f"反思刚才的想法：「{last_thought[:30]}...」")
            parts.append("我在想...")
        
        # 5. 回归主题
        if self._should_return_to_theme() and self.current_theme:
            parts.append(f"回到「{self.current_theme.content}」")
            self.current_theme.return_count += 1
            self.current_theme.drift_count = 0
        
        # 6. 思维种子
        if self.thought_seed and time.time() - self.seed_timestamp < 60:
            parts.append(f"用户说：{self.thought_seed[:20]}")
        
        # 7. 历史上下文
        if self.monologue_history and self.current_thought_state != ThoughtState.REFLECTING:
            recent = list(self.monologue_history)[-1] if self.monologue_history else ""
            if recent and len(recent) > 10:
                parts.append(f"刚才想到{recent[-30:]}")
        
        # 8. 构建最终Prompt
        if parts:
            prompt = "，".join(parts) + "..."
        else:
            # 默认Prompt
            default_prompts = [
                "此刻我在想...",
                "忽然想到...",
                "思维在流动...",
                "静静地思考...",
                "回忆起..."
            ]
            prompt = random.choice(default_prompts)
        
        return prompt
    
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
        # 这里可以添加更复杂的状态更新逻辑
        pass
    
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
