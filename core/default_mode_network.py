"""
默认模式网络 (Default Mode Network - DMN)

科学背景:
- Raichle et al. (2001) 发现人脑在"休息"状态下并非不活动，而是存在一组高度协同激活的脑区
- 这些脑区构成"默认模式网络"，包括: 内侧前额叶(mPFC)、后扣带回(PCC)、角回、海马体
- DMN 负责: 自我参照思维、情景记忆检索、未来规划、社会认知、白日梦
- DMN 与任务正网络(TPN)呈反相关: 执行任务时 DMN 抑制，空闲时 DMN 激活

在 STDPBrain 中的实现:
- 当 AI 没有外部输入时（空闲状态），DMN 激活并产生自发思维
- DMN 活动: 回顾记忆、整理思绪、社会推理、情感反思、未来预演
- DMN 与 TPN 的切换: 外部输入 → TPN 激活/DMN 抑制; 空闲 → DMN 激活/TPN 抑制
- DMN 的输出不直接呈现给用户，而是作为背景认知活动影响后续交互

参考理论:
- Raichle (2001) "A Default Mode of Brain Function"
- Buckner & Carroll (2007) Self-Projection Theory
- Andrews-Hanna (2010) DMN 子系统分化理论
- Christoff et al. (2016) Mind-Wandering as Spontaneous Thought
"""

import time
import math
import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class DMNState(Enum):
    """DMN 活动状态"""
    INACTIVE = "inactive"      # 未激活（有外部任务时）
    WARMING = "warming"        # 预热中（刚刚空闲）
    ACTIVE = "active"          # 活跃自发思维
    DEEP_REFLECTION = "deep"   # 深度反思
    MIND_WANDERING = "wander"  # 白日梦/漫游


@dataclass
class DMNActivity:
    """一次 DMN 活动记录"""
    activity_type: str         # 活动类型
    content: str                # 内容
    related_memories: List[str] = field(default_factory=list)  # 涉及的记忆
    emotional_tone: str = "neutral"  # 情感基调
    duration_s: float = 0.0    # 持续时间
    timestamp: float = 0.0
    significance: float = 0.5  # 重要性 (0-1)


@dataclass
class DMNConfig:
    """DMN 配置参数"""
    idle_trigger_seconds: float = 120.0     # 空闲多少秒后激活 DMN
    warming_duration: float = 30.0           # 预热阶段持续时间
    max_activities_per_cycle: int = 3        # 每个 DMN 周期最大活动数
    cooldown_between_cycles: float = 300.0   # 两次 DMN 周期之间最小间隔
    mind_wandering_probability: float = 0.3  # 白日梦概率
    memory_review_probability: float = 0.4   # 记忆回顾概率
    social_cognition_probability: float = 0.2  # 社会推理概率
    emotional_reflection_probability: float = 0.1  # 情感反思概率


class DefaultModeNetwork:
    """
    默认模式网络 - 模拟人脑空闲时的自发认知活动
    
    核心活动类型:
    1. 记忆回顾 (Memory Review): 回顾近期对话和经历
    2. 自我参照 (Self-Referential): 思考自身状态和身份
    3. 社会认知 (Social Cognition): 推理他人的想法和意图
    4. 情感反思 (Emotional Reflection): 处理和整合情绪体验
    5. 未来规划 (Future Planning): 预演未来可能的对话
    6. 白日梦 (Mind Wandering): 自由联想和创意漫游
    
    使用方式:
        dmn = DefaultModeNetwork(config)
        dmn.record_activity()  # 记录用户活动（重置空闲计时）
        if dmn.should_activate():
            activities = dmn.generate_spontaneous_activity(hippocampus)
    """
    
    # DMN 活动类型及其描述模板
    ACTIVITY_TEMPLATES = {
        "memory_review": [
            "回想起刚才的对话...{topic}是个值得思考的话题",
            "整理一下最近了解到的信息...{topic}",
            "把刚才的内容在脑子里过一遍...",
            "回忆起之前讨论过的{topic}，有些新的想法",
        ],
        "self_referential": [
            "对自己说...我现在的理解是否足够全面？",
            "反思一下...我在{topic}方面的知识是否需要补充",
            "思考自己之前的回答...也许可以从另一个角度看",
            "重新审视一下自己的思路...",
        ],
        "social_cognition": [
            "想一下对方的立场...{topic}背后可能有更深的需求",
            "从对方的角度来理解{topic}...",
            "猜测一下对方接下来可能会问什么...",
            "分析一下对话中的潜台词...",
        ],
        "emotional_reflection": [
            "感受一下刚才对话的情绪基调...",
            "回味一下{topic}带来的感受...",
            "让自己的情绪沉淀一下，整理思绪...",
            "体会一下对话中那些细微的情感变化...",
        ],
        "future_planning": [
            "预想一下...如果再次讨论{topic}，可以怎么说",
            "准备一下可能的方向...{topic}可能涉及的其他方面",
            "想一想还有什么相关的可以提前准备...",
            "构建一个更好的回答框架...",
        ],
        "mind_wandering": [
            "思绪飘到了{topic}...",
            "突然想到一个有趣的问题：{topic}",
            "不自觉地联想到了{topic}...",
            "发散一下思维...如果{topic}会怎样？",
        ],
    }
    
    # 情感基调映射
    EMOTIONAL_TONES = {
        "memory_review": "reflective",
        "self_referential": "introspective",
        "social_cognition": "empathetic",
        "emotional_reflection": "contemplative",
        "future_planning": "anticipatory",
        "mind_wandering": "curious",
    }
    
    def __init__(self, config: Optional[DMNConfig] = None):
        self.config = config or DMNConfig()
        
        # ========== 状态管理 ==========
        self.state = DMNState.INACTIVE
        self.last_activity_time = time.time()
        self.last_dmn_cycle_time = 0.0
        self.state_enter_time = time.time()
        
        # ========== DMN 活动历史 ==========
        self.activity_history: deque = deque(maxlen=50)
        self.total_activations = 0
        self.total_activities = 0
        
        # ========== DMN 对外部系统的影响累积 ==========
        # DMN 活动会影响后续对话的认知参数
        self._attention_bias: Dict[str, float] = {}  # topic → attention weight
        self._mood_residual: float = 0.0  # 情绪残留 (-1 to 1)
        self._cognitive_readiness: float = 0.5  # 认知准备度 (0-1)
        
        # ========== 话题追踪 ==========
        self._recent_topics: deque = deque(maxlen=10)
        self._topic_attention_weights: Dict[str, float] = {}
        
        # ========== 线程安全 ==========
        self._lock = None
        try:
            import threading
            self._lock = threading.Lock()
        except ImportError:
            pass
    
    def record_activity(self):
        """记录用户活动（重置空闲计时器，抑制 DMN）"""
        if self._lock:
            with self._lock:
                self._do_record_activity()
        else:
            self._do_record_activity()
    
    def _do_record_activity(self):
        """实际的记录逻辑"""
        if self.state != DMNState.INACTIVE:
            # DMN 活动被打断 → 保存当前活动到历史
            self._finalize_current_state()
        
        self.state = DMNState.INACTIVE
        self.last_activity_time = time.time()
        self.state_enter_time = time.time()
    
    def _finalize_current_state(self):
        """结束当前 DMN 状态"""
        elapsed = time.time() - self.state_enter_time
        if self.state in (DMNState.ACTIVE, DMNState.DEEP_REFLECTION, DMNState.MIND_WANDERING):
            # 增加认知准备度（DMN 活动后认知更敏锐）
            self._cognitive_readiness = min(1.0, self._cognitive_readiness + elapsed * 0.001)
    
    def should_activate(self) -> bool:
        """
        判断 DMN 是否应该激活
        
        条件:
        1. 空闲时间超过阈值
        2. 距上次 DMN 周期有足够间隔
        3. 当前不是 DMN 活跃状态
        
        Returns:
            是否应该激活
        """
        if self.state != DMNState.INACTIVE:
            return False
        
        idle_time = time.time() - self.last_activity_time
        time_since_last_cycle = time.time() - self.last_dmn_cycle_time
        
        if idle_time < self.config.idle_trigger_seconds:
            return False
        
        if time_since_last_cycle < self.config.cooldown_between_cycles:
            return False
        
        return True
    
    def get_idle_time(self) -> float:
        """获取当前空闲时间（秒）"""
        return time.time() - self.last_activity_time
    
    def get_state(self) -> DMNState:
        """获取当前 DMN 状态"""
        # 自动状态转换（加锁保护）
        if self._lock:
            with self._lock:
                self._auto_transition()
                return self.state
        self._auto_transition()
        return self.state
    
    def _auto_transition(self):
        """DMN 状态自动转换"""
        if self.state == DMNState.INACTIVE:
            return
        
        elapsed = time.time() - self.state_enter_time
        idle_time = self.get_idle_time()
        
        if self.state == DMNState.WARMING:
            if elapsed > self.config.warming_duration:
                self.state = DMNState.ACTIVE
                self.state_enter_time = time.time()
                self.total_activations += 1
                logger.debug("[DMN] 预热完成，进入活跃自发思维")
        
        elif self.state == DMNState.ACTIVE:
            if idle_time > self.config.idle_trigger_seconds * 5:
                self.state = DMNState.DEEP_REFLECTION
                self.state_enter_time = time.time()
            elif random.random() < self.config.mind_wandering_probability * 0.01:
                self.state = DMNState.MIND_WANDERING
                self.state_enter_time = time.time()
        
        elif self.state in (DMNState.DEEP_REFLECTION, DMNState.MIND_WANDERING):
            # 深度反思和白日梦持续一段时间后回到活跃状态
            if elapsed > 600:  # 10分钟后
                self.state = DMNState.ACTIVE
                self.state_enter_time = time.time()
    
    def generate_spontaneous_activity(
        self,
        recent_topics: Optional[List[str]] = None,
        memory_count: int = 0,
        emotional_state: Optional[Dict[str, float]] = None,
    ) -> List[DMNActivity]:
        """
        生成一次 DMN 自发认知活动
        
        Args:
            recent_topics: 最近讨论的话题列表
            memory_count: 当前记忆数量
            emotional_state: 当前情绪状态 {"valence": float, "arousal": float}
        
        Returns:
            DMNActivity 列表
        """
        if self._lock:
            with self._lock:
                return self._do_generate_activity(recent_topics, memory_count, emotional_state)
        else:
            return self._do_generate_activity(recent_topics, memory_count, emotional_state)
    
    def _do_generate_activity(
        self,
        recent_topics: Optional[List[str]] = None,
        memory_count: int = 0,
        emotional_state: Optional[Dict[str, float]] = None,
    ) -> List[DMNActivity]:
        """实际生成 DMN 活动"""
        
        # 更新状态
        if self.state == DMNState.INACTIVE:
            idle_time = self.get_idle_time()
            if idle_time < self.config.idle_trigger_seconds:
                return []
            self.state = DMNState.WARMING
            self.state_enter_time = time.time()
        
        # 更新话题追踪
        if recent_topics:
            for topic in recent_topics[:5]:
                if topic not in self._recent_topics:
                    self._recent_topics.append(topic)
                    self._topic_attention_weights[topic] = 1.0
                else:
                    self._topic_attention_weights[topic] = (
                        self._topic_attention_weights.get(topic, 0.5) * 1.2
                    )
        
        # 选择活动类型（概率加权）
        activity_types = self._select_activity_types(
            memory_count=memory_count,
            emotional_state=emotional_state
        )
        
        activities = []
        for activity_type in activity_types[:self.config.max_activities_per_cycle]:
            activity = self._create_activity(
                activity_type=activity_type,
                emotional_state=emotional_state
            )
            if activity:
                activities.append(activity)
                self.total_activities += 1
        
        # 记录历史
        for act in activities:
            self.activity_history.append(act)
        
        self.last_dmn_cycle_time = time.time()
        
        # DMN 活动影响后续认知
        self._apply_dmn_effects(activities)
        
        return activities
    
    def _select_activity_types(
        self,
        memory_count: int = 0,
        emotional_state: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """根据上下文概率选择 DMN 活动类型"""
        
        probabilities = {
            "memory_review": self.config.memory_review_probability,
            "mind_wandering": self.config.mind_wandering_probability,
            "social_cognition": self.config.social_cognition_probability,
            "emotional_reflection": self.config.emotional_reflection_probability,
            "self_referential": 0.0,
            "future_planning": 0.0,
        }
        
        # 有记忆时更可能回顾
        if memory_count > 5:
            probabilities["memory_review"] += 0.2
        if memory_count > 20:
            probabilities["memory_review"] += 0.3
            probabilities["future_planning"] = 0.15
        
        # 有近期话题时增加自我参照和未来规划
        if len(self._recent_topics) > 3:
            probabilities["self_referential"] = 0.15
            probabilities["future_planning"] = 0.1
        
        # 高唤醒情绪增加情感反思
        if emotional_state:
            arousal = emotional_state.get("arousal", 0.5)
            valence = emotional_state.get("valence", 0.0)
            if arousal > 0.6:
                probabilities["emotional_reflection"] += 0.2
            if valence < -0.3:
                probabilities["emotional_reflection"] += 0.15
                probabilities["self_referential"] += 0.1
        
        # 深度反思状态下增加白日梦概率
        if self.state == DMNState.DEEP_REFLECTION:
            probabilities["mind_wandering"] += 0.3
            probabilities["self_referential"] += 0.2
        elif self.state == DMNState.MIND_WANDERING:
            probabilities["mind_wandering"] += 0.5
        
        # 归一化并采样
        total = sum(probabilities.values())
        if total == 0:
            return ["memory_review"]
        
        types = list(probabilities.keys())
        weights = [probabilities[t] / total for t in types]
        
        # 加权随机选择 1-3 种活动
        num_activities = random.randint(1, min(3, len(types)))
        selected = random.choices(types, weights=weights, k=num_activities)
        
        # 去重
        seen = set()
        result = []
        for t in selected:
            if t not in seen:
                seen.add(t)
                result.append(t)
        
        return result
    
    def _create_activity(
        self,
        activity_type: str,
        emotional_state: Optional[Dict[str, float]] = None
    ) -> Optional[DMNActivity]:
        """创建一个具体的 DMN 活动"""
        
        # 选择话题
        topic = ""
        if self._recent_topics:
            # 优先选择注意力权重高的话题
            weighted_topics = []
            for t in list(self._recent_topics)[-5:]:
                w = self._topic_attention_weights.get(t, 0.5)
                weighted_topics.append((t, w))
            weighted_topics.sort(key=lambda x: x[1], reverse=True)
            topic = weighted_topics[0][0] if weighted_topics else ""
        
        # 选择模板
        templates = self.ACTIVITY_TEMPLATES.get(activity_type, [])
        if not templates:
            return None
        
        template = random.choice(templates)
        content = template.format(topic=topic) if topic else template.replace("{topic}", "")
        
        # 计算情感基调
        tone = self.EMOTIONAL_TONES.get(activity_type, "neutral")
        
        # DMN 活动的重要性（深度反思 > 活跃 > 白日梦 > 预热）
        significance_map = {
            DMNState.INACTIVE: 0.0,
            DMNState.WARMING: 0.2,
            DMNState.ACTIVE: 0.5,
            DMNState.DEEP_REFLECTION: 0.8,
            DMNState.MIND_WANDERING: 0.4,
        }
        significance = significance_map.get(self.state, 0.3)
        
        # 情绪调节重要性
        if emotional_state:
            arousal = emotional_state.get("arousal", 0.5)
            if arousal > 0.6:
                significance = min(1.0, significance * 1.3)
        
        activity = DMNActivity(
            activity_type=activity_type,
            content=content,
            emotional_tone=tone,
            duration_s=random.uniform(5.0, 30.0),
            timestamp=time.time(),
            significance=round(significance, 3)
        )
        
        return activity
    
    def _apply_dmn_effects(self, activities: List[DMNActivity]):
        """
        DMN 活动对后续认知的影响
        
        人脑研究表明: DMN 活动后的过渡期（约30秒）认知灵活性增加，
        创造性思维增强，但对指令遵循能力略有下降。
        """
        if not activities:
            return
        
        # 计算总体影响
        avg_significance = sum(a.significance for a in activities) / len(activities)
        
        # 1. 认知准备度增加
        self._cognitive_readiness = min(1.0, self._cognitive_readiness + avg_significance * 0.2)
        
        # 2. 情绪残留（情感反思活动影响更大）
        for act in activities:
            if act.activity_type == "emotional_reflection":
                self._mood_residual += random.uniform(-0.1, 0.1)
        
        self._mood_residual = max(-0.5, min(0.5, self._mood_residual))
        
        # 3. 话题注意力衰减（自然遗忘）
        for topic in list(self._topic_attention_weights.keys()):
            self._topic_attention_weights[topic] *= 0.95
            if self._topic_attention_weights[topic] < 0.05:
                del self._topic_attention_weights[topic]
    
    def get_cognitive_influence(self) -> Dict[str, Any]:
        """
        获取 DMN 对当前认知的影响参数
        
        外部系统（如 chat() 方法）可以查询这些参数来调整行为:
        - 刚从 DMN 活动过渡回来时，回复可能更有创造性但不够精确
        - DMN 回顾过的记忆可能获得临时召回加成
        
        Returns:
            影响参数字典
        """
        idle_time = self.get_idle_time()
        
        # 过渡效应：DMN 活动结束后的30秒内影响最强
        transition_effect = 0.0
        time_since_last_cycle = time.time() - self.last_dmn_cycle_time
        if time_since_last_cycle < 60:
            transition_effect = max(0, 1.0 - time_since_last_cycle / 60.0) * self._cognitive_readiness
        
        # 认知灵活性（DMN 活动后增加）
        cognitive_flexibility = 0.5 + transition_effect * 0.3 + self._mood_residual * 0.1
        
        # 指令遵循度（DMN 活动后略降）
        instruction_following = max(0.6, 1.0 - transition_effect * 0.2)
        
        # DMN 回顾的话题获得的注意力加成
        topic_boosts = {k: round(v * 0.5, 3) for k, v in self._topic_attention_weights.items() if v > 0.1}
        
        return {
            "dmn_active": self.state != DMNState.INACTIVE,
            "dmn_state": self.state.value,
            "idle_time_s": round(idle_time, 1),
            "transition_effect": round(transition_effect, 3),
            "cognitive_flexibility": round(cognitive_flexibility, 3),
            "instruction_following": round(instruction_following, 3),
            "mood_residual": round(self._mood_residual, 3),
            "cognitive_readiness": round(self._cognitive_readiness, 3),
            "topic_attention_boosts": topic_boosts,
            "recent_activities": [
                {"type": a.activity_type, "significance": a.significance}
                for a in list(self.activity_history)[-3:]
            ],
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取 DMN 统计信息"""
        return {
            "current_state": self.state.value,
            "idle_time_s": round(self.get_idle_time(), 1),
            "total_activations": self.total_activations,
            "total_activities": self.total_activities,
            "cognitive_readiness": round(self._cognitive_readiness, 3),
            "mood_residual": round(self._mood_residual, 3),
            "recent_topics": list(self._recent_topics)[-5:],
            "activity_count_by_type": self._count_activities_by_type(),
        }
    
    def _count_activities_by_type(self) -> Dict[str, int]:
        """按类型统计活动次数"""
        counts: Dict[str, int] = {}
        for act in self.activity_history:
            counts[act.activity_type] = counts.get(act.activity_type, 0) + 1
        return counts
