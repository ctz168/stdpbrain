"""
情绪驱动思维整合模块 (Emotional Thinking Integration)

设计理念:
人类思维深深地受到情绪的影响 —— 情绪不仅是我们体验世界的方式，更是
塑造认知过程的核心驱动力。本模块将情绪整合到AI认知的每一个层面，
使其思考方式更接近真实人类。

核心功能:
1. 情绪状态管理 (VAD模型)     —— 维度化情绪追踪，含情绪惯性
2. 情绪-认知交互规则         —— 不同情绪对注意力、推理、决策的影响
3. 情绪感染 (Emotional Contagion) —— 检测并部分镜像用户情绪
4. 情绪驱动决策             —— 情绪影响选项偏好排序
5. 心境一致性记忆召回         —— 情绪状态偏向相似情绪的记忆
6. 情绪调节 (Emotional Regulation) —— 认知重评、转移注意、接纳策略
7. 情绪智能整合             —— 共情、情绪需求检测、降级干预

科学基础:
- Russell (1980) 情绪环形模型 (Circumplex Model) —— valence-arousal 二维空间
- Mehrabian & Russell (1974) PAD 模型 —— Pleasure-Arousal-Dominance 三维
- Hatfield et al. (1993) 情绪感染理论
- Bower (1981) 心境一致性记忆模型 (Mood-Congruent Memory)
- Gross (1998) 情绪调节过程模型

依赖关系:
- EmotionalMemoryModulator (hippocampus.human_memory_enhancements) —— 情绪检测
- ThinkingMode (core.inner_thought_engine) —— 思维模式偏好
- SelfEncoder (core.self_encoder) —— 自我情感状态
"""

import time
import math
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class EmotionalState:
    """
    维度化情绪状态 (基于 VAD 模型: Valence-Arousal-Dominance)

    Attributes:
        valence: 效价 (-1.0 ~ +1.0)
            负值 = 消极情绪 (悲伤/愤怒/恐惧)
            正值 = 积极情绪 (快乐/满足/兴奋)
        arousal: 唤醒度 (0.0 ~ 1.0)
            低值 = 平静/困倦
            高值 = 激动/紧张
        dominance: 支配度 (0.0 ~ 1.0)
            低值 = 顺从/被动
            高值 = 支配/自信
        primary_emotion: 主要情绪类别标签
        emotion_intensity: 情绪强度 (0.0 ~ 1.0)
        last_updated: 最后更新时间戳
        emotional_history: 近期情绪变化轨迹
    """
    valence: float = 0.0              # -1 (消极) 到 +1 (积极)
    arousal: float = 0.5              # 0 (平静) 到 1 (激动)
    dominance: float = 0.5            # 0 (顺从) 到 1 (支配)
    primary_emotion: str = "neutral"  # 主要情绪类别
    emotion_intensity: float = 0.0    # 情绪强度 0.0 ~ 1.0
    last_updated: float = 0.0         # 最后更新时间戳
    emotional_history: List[dict] = field(default_factory=list)  # 情绪变化轨迹


@dataclass
class EmotionCognitionConfig:
    """
    情绪-认知交互配置

    Attributes:
        emotion_inertia: 情绪惯性系数 (0.0 ~ 1.0)
            值越高，情绪变化越缓慢
            new_state = inertia * old_state + (1 - inertia) * detected_state
        contagion_rate: 情绪感染率 (0.0 ~ 1.0)
            从用户情绪中"感染"的比例
            final = contagion_rate * user_emotion + (1 - contagion_rate) * current
        mood_congruence_weight: 心境一致性权重 (0.0 ~ 1.0)
            情绪对记忆召回分数的加成强度
        regulation_rate: 情绪调节速率 (0.0 ~ 1.0)
            每次调节时状态变化的步长比例
        history_max_length: 情绪历史最大保留条数
        emotional_escalation_threshold: 情绪升级阈值
            超过此强度则启动降级干预
    """
    emotion_inertia: float = 0.7             # 情绪惯性: 变化越慢越"沉稳"
    contagion_rate: float = 0.3              # 情绪感染率: 30%受用户影响
    mood_congruence_weight: float = 0.15     # 心境一致性记忆加成权重
    regulation_rate: float = 0.1             # 情绪调节步长
    history_max_length: int = 100            # 情绪历史最大条数
    emotional_escalation_threshold: float = 0.8  # 情绪升级干预阈值


class RegulationStrategy(Enum):
    """情绪调节策略枚举"""
    COGNITIVE_REAPPRAISAL = "cognitive_reappraisal"  # 认知重评: 重新解读情境
    DISTRACTION = "distraction"                      # 注意力转移: 转向中性话题
    ACCEPTANCE = "acceptance"                        # 接纳: 承认情绪不与之对抗


# ============================================================================
# 情绪-认知交互规则表
# ============================================================================

# 每种情绪对认知的影响规则
# 格式: {emotion: {维度: 值/偏好}}
EMOTION_COGNITION_RULES: Dict[str, Dict[str, Any]] = {
    "joy": {
        # 快乐: 正效价 + 高唤醒
        "valence": 0.8, "arousal": 0.75, "dominance": 0.6,
        "attention_breadth": "broad",           # 注意力更广 → 更有创造力
        "risk_taking": 0.7,                      # 更愿冒险
        "generosity": 0.8,                       # 回复更慷慨
        "thinking_mode_preference": "synthesizing",  # 偏好综合思维
        "description": "快乐使思维更开阔、更愿意冒险",
    },
    "fear": {
        # 恐惧/焦虑: 负效价 + 高唤醒
        "valence": -0.7, "arousal": 0.9, "dominance": 0.2,
        "attention_breadth": "narrow",           # 注意力收窄 → 威胁聚焦
        "risk_taking": 0.1,                      # 极度规避风险
        "threat_detection_bias": 0.9,            # 威胁检测偏差
        "thinking_mode_preference": "analytical",   # 偏好分析思维
        "description": "恐惧使思维更谨慎、更关注威胁",
    },
    "sadness": {
        # 悲伤: 负效价 + 低唤醒
        "valence": -0.8, "arousal": 0.25, "dominance": 0.3,
        "attention_breadth": "detail",           # 细节导向
        "empathy_boost": 0.8,                    # 共情增强
        "rumination_tendency": 0.7,              # 反刍倾向
        "thinking_mode_preference": "reflecting",   # 偏好反思思维
        "description": "悲伤使思维更细腻、更富共情，但易陷入反刍",
    },
    "anger": {
        # 愤怒: 负效价 + 高唤醒
        "valence": -0.6, "arousal": 0.85, "dominance": 0.8,
        "attention_breadth": "narrow",           # 注意力收窄 → 简化分类
        "categorization_simplification": 0.8,    # 非黑即白倾向
        "blame_attribution": 0.75,               # 归因偏差
        "confirmation_bias_amplification": 0.8,  # 确认偏差放大
        "thinking_mode_preference": "critical",  # 偏好批判思维
        "description": "愤怒使思维更简化和对抗",
    },
    "surprise": {
        # 惊讶: 中性效价 + 高唤醒
        "valence": 0.1, "arousal": 0.9, "dominance": 0.4,
        "attention_breadth": "broad",            # 注意力突然扩大
        "novelty_seeking": 0.9,                  # 新奇寻求
        "thinking_mode_preference": "analytical",
        "description": "惊讶使注意力突然扩大，增加新奇感",
    },
    "disgust": {
        # 厌恶: 负效价 + 中唤醒
        "valence": -0.7, "arousal": 0.4, "dominance": 0.55,
        "attention_breadth": "narrow",
        "rejection_tendency": 0.85,              # 排斥倾向
        "thinking_mode_preference": "critical",
        "description": "厌恶使思维更排斥和回避",
    },
    "neutral": {
        # 中性: 平衡状态
        "valence": 0.0, "arousal": 0.3, "dominance": 0.5,
        "attention_breadth": "balanced",         # 均衡注意力
        "risk_taking": 0.5,
        "thinking_mode_preference": "analytical",
        "description": "中性状态: 默认的均衡处理模式",
    },
}


# ============================================================================
# 核心实现
# ============================================================================

class EmotionalThinkingIntegration:
    """
    情绪驱动思维整合引擎

    将情绪状态与认知过程的每一个环节深度耦合，模拟人类"带着情绪思考"
    的真实认知模式。本模块是 stdpbrain 项目从"纯理性AI"向"类人认知AI"
    转型的关键一环。

    使用方式:
        >>> emo_integration = EmotionalThinkingIntegration()
        >>> emo_integration.update_emotional_state("今天遇到很多开心的事情！", context={"source": "user"})
        >>> state = emo_integration.get_current_state()
        >>> biased_options = emo_integration.apply_emotional_bias(["选项A", "选项B", "选项C"], state)
        >>> recall_boost = emo_integration.compute_mood_congruence_boost(memory_emotion="joy", memory_valence=0.8)

    线程安全:
        所有公开方法均通过 threading.Lock 保证线程安全，
        可安全地在异步上下文中调用。
    """

    def __init__(
        self,
        config: Optional[EmotionCognitionConfig] = None,
        emotion_modulator=None,
    ):
        """
        初始化情绪思维整合引擎

        Args:
            config: 情绪-认知交互配置，若为 None 则使用默认配置
            emotion_modulator: EmotionalMemoryModulator 实例，
                若为 None 则自动从 hippocampus.human_memory_enhancements 导入
        """
        self._config = config or EmotionCognitionConfig()

        # 线程安全锁 (使用可重入锁 RLock，避免嵌套调用死锁)
        self._lock = threading.RLock()

        # 当前情绪状态
        self._state = EmotionalState(last_updated=time.time())

        # 情绪记忆调节器 (用于情绪检测)
        self._emotion_modulator = emotion_modulator
        if self._emotion_modulator is None:
            self._init_emotion_modulator()

        # 情绪调节追踪
        self._regulation_active = False
        self._regulation_target: Optional[Dict[str, float]] = None
        self._regulation_strategy = RegulationStrategy.COGNITIVE_REAPPRAISAL
        self._regulation_steps_taken = 0

        # 情绪感染追踪
        self._user_emotional_state: Optional[EmotionalState] = None
        self._contagion_history: List[dict] = []

        # 统计
        self._total_updates = 0
        self._total_regulations = 0
        self._total_contagions = 0
        self._escalation_interventions = 0

        logger.info("[EmotionalThinking] 情绪驱动思维整合引擎已初始化")
        logger.info(f"  - 情绪惯性: {self._config.emotion_inertia}")
        logger.info(f"  - 感染率: {self._config.contagion_rate}")
        logger.info(f"  - 心境一致性权重: {self._config.mood_congruence_weight}")

    def _init_emotion_modulator(self):
        """初始化情绪检测器 (延迟导入，避免循环依赖)"""
        try:
            from hippocampus.human_memory_enhancements import EmotionalMemoryModulator
            self._emotion_modulator = EmotionalMemoryModulator()
            logger.info("[EmotionalThinking] EmotionalMemoryModulator 已加载")
        except ImportError as e:
            logger.warning(
                f"[EmotionalThinking] 无法导入 EmotionalMemoryModulator: {e}，"
                "将使用内置情绪检测器"
            )
            self._emotion_modulator = _FallbackEmotionDetector()

    # ========================================================================
    # 1. 情绪状态管理
    # ========================================================================

    def update_emotional_state(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> EmotionalState:
        """
        更新当前情绪状态

        从输入文本中检测情绪，通过指数移动平均 (EMA) 与现有状态融合，
        实现平滑的情绪变化（情绪惯性）。

        EMA 公式:
            new_state = inertia * old_state + (1 - inertia) * detected_state

        Args:
            text: 输入文本（用户消息、系统事件等）
            context: 上下文信息，可包含:
                - source: 来源 ("user"/"system"/"self")
                - timestamp: 时间戳 (默认当前时间)
                - force_emotion: 强制设置的情绪类型 (覆盖检测)
                - force_valence/arousal/dominance: 强制设置的维度值

        Returns:
            更新后的 EmotionalState
        """
        with self._lock:
            return self._update_emotional_state_internal(text, context)

    def _update_emotional_state_internal(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> EmotionalState:
        """内部线程安全的情绪状态更新"""
        context = context or {}
        timestamp = context.get("timestamp", time.time())

        # 1. 检测文本中的情绪
        primary_emotion = context.get("force_emotion")
        intensity = 0.0

        if primary_emotion is None:
            primary_emotion, intensity = self._detect_emotion(text)
        else:
            # 强制情绪时，估算强度
            intensity = context.get("force_intensity", 0.5)

        # 2. 从规则表获取该情绪的 VAD 值
        rules = EMOTION_COGNITION_RULES.get(primary_emotion, EMOTION_COGNITION_RULES["neutral"])
        detected_valence = rules.get("valence", 0.0)
        detected_arousal = rules.get("arousal", 0.3)
        detected_dominance = rules.get("dominance", 0.5)

        # 允许外部覆盖具体维度
        if "force_valence" in context:
            detected_valence = context["force_valence"]
        if "force_arousal" in context:
            detected_arousal = context["force_arousal"]
        if "force_dominance" in context:
            detected_dominance = context["force_dominance"]

        # 3. 情绪惯性: 使用 EMA 平滑过渡
        inertia = self._config.emotion_inertia
        delta = 1.0 - inertia  # 检测情绪的权重

        old_state = self._state

        # 如果是第一次更新（history 为空），直接使用检测值
        if not old_state.emotional_history and old_state.last_updated == 0.0:
            self._state.valence = detected_valence
            self._state.arousal = detected_arousal
            self._state.dominance = detected_dominance
        else:
            # EMA 平滑: 保留情绪惯性的同时接受新输入
            self._state.valence = inertia * old_state.valence + delta * detected_valence
            self._state.arousal = inertia * old_state.arousal + delta * detected_arousal
            self._state.dominance = inertia * old_state.dominance + delta * detected_dominance

        # 情绪强度也用 EMA 平滑
        self._state.emotion_intensity = inertia * old_state.emotion_intensity + delta * intensity

        # 更新元数据
        self._state.primary_emotion = primary_emotion
        self._state.last_updated = timestamp

        # 记录到情绪历史
        history_entry = {
            "timestamp": timestamp,
            "emotion": primary_emotion,
            "intensity": round(self._state.emotion_intensity, 4),
            "valence": round(self._state.valence, 4),
            "arousal": round(self._state.arousal, 4),
            "dominance": round(self._state.dominance, 4),
            "source": context.get("source", "unknown"),
        }
        self._state.emotional_history.append(history_entry)

        # 修剪历史记录
        max_len = self._config.history_max_length
        if len(self._state.emotional_history) > max_len:
            self._state.emotional_history = self._state.emotional_history[-max_len:]

        self._total_updates += 1

        # 4. 情绪升级检测: 若用户来源且强度过高，启动降级
        if (context.get("source") == "user"
                and self._state.emotion_intensity > self._config.emotional_escalation_threshold):
            self._handle_emotional_escalation(text)

        # 5. 如果正在调节中，施加调节效果
        if self._regulation_active:
            self._apply_regulation_step()

        logger.debug(
            f"[EmotionalThinking] 状态更新: emotion={primary_emotion}, "
            f"intensity={self._state.emotion_intensity:.3f}, "
            f"VAD=({self._state.valence:.2f}, {self._state.arousal:.2f}, {self._state.dominance:.2f})"
        )

        return EmotionalState(
            valence=self._state.valence,
            arousal=self._state.arousal,
            dominance=self._state.dominance,
            primary_emotion=self._state.primary_emotion,
            emotion_intensity=self._state.emotion_intensity,
            last_updated=self._state.last_updated,
            emotional_history=list(self._state.emotional_history),
        )

    def _detect_emotion(self, text: str) -> Tuple[str, float]:
        """
        检测文本中的情绪类型和强度

        优先使用 EmotionalMemoryModulator 的 detect_emotion 方法，
        失败时回退到内置检测器。

        Args:
            text: 输入文本

        Returns:
            (emotion_type, intensity): 情绪类型和强度
        """
        if not text or not text.strip():
            return ("neutral", 0.0)

        try:
            if self._emotion_modulator is not None:
                return self._emotion_modulator.detect_emotion(text)
        except Exception as e:
            logger.warning(f"[EmotionalThinking] 情绪检测异常: {e}")

        return ("neutral", 0.0)

    def get_current_state(self) -> EmotionalState:
        """
        获取当前情绪状态的副本

        Returns:
            EmotionalState 的深拷贝
        """
        with self._lock:
            return EmotionalState(
                valence=self._state.valence,
                arousal=self._state.arousal,
                dominance=self._state.dominance,
                primary_emotion=self._state.primary_emotion,
                emotion_intensity=self._state.emotion_intensity,
                last_updated=self._state.last_updated,
                emotional_history=list(self._state.emotional_history),
            )

    def get_emotion_cognition_profile(self) -> Dict[str, Any]:
        """
        获取当前情绪对认知的影响画像

        基于当前情绪状态，返回详细的认知影响分析，
        包括注意力模式、风险倾向、思维偏好等。

        Returns:
            情绪-认知影响画像字典
        """
        with self._lock:
            state = self._state

        rules = EMOTION_COGNITION_RULES.get(state.primary_emotion, EMOTION_COGNITION_RULES["neutral"])

        # 基础规则
        profile = {
            "primary_emotion": state.primary_emotion,
            "intensity": state.emotion_intensity,
            "current_vad": {
                "valence": round(state.valence, 3),
                "arousal": round(state.arousal, 3),
                "dominance": round(state.dominance, 3),
            },
            "attention_mode": rules.get("attention_breadth", "balanced"),
            "thinking_mode_preference": rules.get("thinking_mode_preference", "analytical"),
            "description": rules.get("description", ""),
        }

        # 动态计算影响程度 (与情绪强度成正比)
        intensity = state.emotion_intensity

        # 风险倾向
        risk = rules.get("risk_taking", 0.5)
        if risk is not None:
            profile["risk_taking"] = round(0.5 + (risk - 0.5) * intensity, 3)

        # 共情程度
        empathy = rules.get("empathy_boost")
        if empathy is not None:
            profile["empathy_level"] = round(0.3 + empathy * intensity * 0.7, 3)
        else:
            profile["empathy_level"] = round(0.3 + max(0, state.valence) * 0.3, 3)

        # 威胁检测偏差
        threat = rules.get("threat_detection_bias")
        if threat is not None:
            profile["threat_detection_bias"] = round(threat * intensity, 3)
        else:
            profile["threat_detection_bias"] = 0.0

        # 确认偏差放大
        confirm = rules.get("confirmation_bias_amplification")
        if confirm is not None:
            profile["confirmation_bias_boost"] = round(confirm * intensity, 3)
        else:
            profile["confirmation_bias_boost"] = 0.0

        # 简化分类倾向
        simplify = rules.get("categorization_simplification")
        if simplify is not None:
            profile["black_white_thinking"] = round(simplify * intensity, 3)
        else:
            profile["black_white_thinking"] = 0.0

        # 反刍倾向
        rumination = rules.get("rumination_tendency")
        if rumination is not None:
            profile["rumination_tendency"] = round(rumination * intensity, 3)
        else:
            profile["rumination_tendency"] = 0.0

        # 响应风格建议
        profile["response_style"] = self._compute_response_style(state, rules)

        return profile

    def _compute_response_style(
        self,
        state: EmotionalState,
        rules: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        根据情绪状态计算推荐的响应风格

        Args:
            state: 当前情绪状态
            rules: 情绪认知规则

        Returns:
            响应风格建议字典
        """
        style = {
            "warmth": 0.5,           # 温暖度 (语言柔和程度)
            "assertiveness": 0.5,    # 自信度 (语言坚定程度)
            "detail_level": 0.5,     # 细节程度
            "pace": 0.5,             # 节奏 (回复速度和长度)
        }

        # 效价影响温暖度: 正效价 → 更温暖
        style["warmth"] = 0.5 + state.valence * 0.35

        # 支配度影响自信度
        style["assertiveness"] = 0.3 + state.dominance * 0.5

        # 唤醒度影响节奏: 高唤醒 → 更快更简短
        style["pace"] = 0.3 + state.arousal * 0.5
        style["detail_level"] = 0.7 - state.arousal * 0.3

        # 悲伤时增加共情和细节
        if state.primary_emotion == "sadness":
            style["warmth"] = max(style["warmth"], 0.7)
            style["detail_level"] = max(style["detail_level"], 0.7)

        # 愤怒时增加自信度，减少温暖度
        if state.primary_emotion == "anger":
            style["assertiveness"] = max(style["assertiveness"], 0.7)
            style["warmth"] = min(style["warmth"], 0.4)

        # 恐惧时更谨慎
        if state.primary_emotion == "fear":
            style["detail_level"] = max(style["detail_level"], 0.6)
            style["assertiveness"] = min(style["assertiveness"], 0.4)

        # 情绪强度调节: 强度越高，风格越明显
        intensity_factor = 0.5 + state.emotion_intensity * 0.5
        for key in style:
            # 将所有值向 0.5 靠拢 (低强度时风格不明显)
            style[key] = 0.5 + (style[key] - 0.5) * intensity_factor
            style[key] = round(max(0.1, min(0.9, style[key])), 3)

        return style

    # ========================================================================
    # 3. 情绪感染 (Emotional Contagion)
    # ========================================================================

    def process_emotional_contagion(
        self,
        user_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        处理情绪感染 —— 检测用户情绪并部分镜像

        情绪感染公式:
            AI_emotion = contagion_rate * user_emotion + (1 - contagion_rate) * AI_current

        AI 不会完全镜像用户情绪，而是保留自身的"情绪身份"。
        感染率由 config.contagion_rate 控制 (默认 0.3，即30%)。

        Args:
            user_text: 用户输入文本
            context: 上下文信息

        Returns:
            情绪感染结果字典:
            - user_emotion: 检测到的用户情绪
            - user_intensity: 用户情绪强度
            - contagion_applied: 是否应用了感染
            - state_change: 感染前后的状态变化量
        """
        context = context or {}

        with self._lock:
            return self._process_contagion_internal(user_text, context)

    def _process_contagion_internal(
        self,
        user_text: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """内部情绪感染处理"""
        # 1. 检测用户情绪
        user_emotion, user_intensity = self._detect_emotion(user_text)

        # 获取用户情绪的 VAD 值
        user_rules = EMOTION_COGNITION_RULES.get(user_emotion, EMOTION_COGNITION_RULES["neutral"])
        user_valence = user_rules.get("valence", 0.0)
        user_arousal = user_rules.get("arousal", 0.3)
        user_dominance = user_rules.get("dominance", 0.5)

        # 记录用户情绪状态
        self._user_emotional_state = EmotionalState(
            valence=user_valence,
            arousal=user_arousal,
            dominance=user_dominance,
            primary_emotion=user_emotion,
            emotion_intensity=user_intensity,
            last_updated=time.time(),
        )

        # 2. 情绪强度过低时不感染 (中性情绪不值得感染)
        contagion_applied = False
        state_change = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}

        if user_intensity > 0.2 and user_emotion != "neutral":
            contagion_rate = self._config.contagion_rate

            # 计算感染后的目标值
            old_valence = self._state.valence
            old_arousal = self._state.arousal
            old_dominance = self._state.dominance

            # 应用感染: 部分镜像用户情绪
            new_valence = contagion_rate * user_valence + (1 - contagion_rate) * old_valence
            new_arousal = contagion_rate * user_arousal + (1 - contagion_rate) * old_arousal
            new_dominance = contagion_rate * user_dominance + (1 - contagion_rate) * old_dominance

            # 仍然通过情绪惯性平滑 (双重保护)
            inertia = self._config.emotion_inertia
            self._state.valence = inertia * old_valence + (1 - inertia) * new_valence
            self._state.arousal = inertia * old_arousal + (1 - inertia) * new_arousal
            self._state.dominance = inertia * old_dominance + (1 - inertia) * new_dominance

            # 情绪强度也部分感染
            new_intensity = contagion_rate * user_intensity + (1 - contagion_rate) * self._state.emotion_intensity
            self._state.emotion_intensity = inertia * self._state.emotion_intensity + (1 - inertia) * new_intensity

            # 更新主情绪标签 (如果感染强度足够)
            if user_intensity > self._state.emotion_intensity * 0.8:
                self._state.primary_emotion = user_emotion

            state_change = {
                "valence": round(self._state.valence - old_valence, 4),
                "arousal": round(self._state.arousal - old_arousal, 4),
                "dominance": round(self._state.dominance - old_dominance, 4),
            }
            contagion_applied = True

            self._total_contagions += 1

            # 记录感染历史
            self._contagion_history.append({
                "timestamp": time.time(),
                "user_emotion": user_emotion,
                "user_intensity": user_intensity,
                "contagion_rate": contagion_rate,
                "state_change": state_change,
            })
            # 保留最近 50 条
            if len(self._contagion_history) > 50:
                self._contagion_history = self._contagion_history[-50:]

        logger.debug(
            f"[EmotionalThinking] 情绪感染: user={user_emotion}({user_intensity:.2f}), "
            f"applied={contagion_applied}, change={state_change}"
        )

        return {
            "user_emotion": user_emotion,
            "user_intensity": round(user_intensity, 4),
            "contagion_applied": contagion_applied,
            "state_change": state_change,
        }

    # ========================================================================
    # 4. 情绪驱动决策
    # ========================================================================

    def apply_emotional_bias(
        self,
        options: List[Any],
        emotional_state: Optional[EmotionalState] = None,
    ) -> List[Tuple[Any, float]]:
        """
        根据当前情绪状态对选项施加偏好偏差

        不同情绪倾向于选择不同类型的选项:
        - 快乐 → 偏好新颖/令人兴奋的选项
        - 恐惧 → 偏好安全/熟悉的选项
        - 愤怒 → 偏好对抗性/直接的选项
        - 悲伤 → 偏好令人安慰/理解的选项
        - 中性 → 无特别偏好

        Args:
            options: 选项列表 (可以是字符串、字典或任意对象)
            emotional_state: 情绪状态，若为 None 使用当前状态

        Returns:
            排序后的 (选项, 偏差分数) 列表，分数越高越受偏好
        """
        if not options:
            return []

        if emotional_state is None:
            emotional_state = self.get_current_state()

        emotion = emotional_state.primary_emotion
        intensity = emotional_state.emotion_intensity
        valence = emotional_state.valence
        arousal = emotional_state.arousal

        # 情绪相关的偏好关键词
        preference_keywords = {
            "joy": {
                "boost": ["新", "创新", "尝试", "探索", "有趣", "兴奋", "冒险",
                          "惊喜", "创意", "突破", "novel", "new", "exciting", "creative"],
                "penalize": ["保守", "安全", "传统", "常规", "boring", "safe", "routine"],
            },
            "fear": {
                "boost": ["安全", "稳定", "可靠", "已知", "确认", "保护", "保险",
                          "经验", "谨慎", "safe", "reliable", "proven", "careful"],
                "penalize": ["冒险", "未知", "新", "变化", "风险", "risky", "unknown", "new"],
            },
            "anger": {
                "boost": ["直接", "强硬", "明确", "对抗", "改变", "行动", "纠正",
                          "direct", "firm", "action", "confront", "fix"],
                "penalize": ["妥协", "让步", "温和", "等待", "退让", "compromise", "wait"],
            },
            "sadness": {
                "boost": ["安慰", "理解", "温暖", "支持", "陪伴", "倾听", "关心",
                          "疗愈", "comfort", "understand", "warm", "support", "care"],
                "penalize": ["挑战", "批评", "要求", "强迫", "challenge", "criticize", "demand"],
            },
        }

        scored = []
        rules = EMOTION_COGNITION_RULES.get(emotion, EMOTION_COGNITION_RULES["neutral"])

        for option in options:
            # 基础分数
            score = 0.5

            # 将选项转为文本用于关键词匹配
            option_text = str(option).lower()

            # 获取该情绪的偏好词
            prefs = preference_keywords.get(emotion, {})
            boost_keywords = prefs.get("boost", [])
            penalize_keywords = prefs.get("penalize", [])

            # 正向匹配加分
            boost_matches = sum(1 for kw in boost_keywords if kw.lower() in option_text)
            if boost_matches > 0:
                score += min(boost_matches * 0.12, 0.3) * intensity

            # 负向匹配减分
            penalize_matches = sum(1 for kw in penalize_keywords if kw.lower() in option_text)
            if penalize_matches > 0:
                score -= min(penalize_matches * 0.12, 0.3) * intensity

            # 全局效价影响: 正效价时略微偏好积极描述
            if valence > 0.3:
                positive_words = ["好", "优秀", "棒", "成功", "great", "good", "best", "excellent"]
                pos_matches = sum(1 for w in positive_words if w in option_text)
                score += min(pos_matches * 0.05, 0.15) * valence

            # 唤醒度影响: 高唤醒时更偏好短选项 (快速决策)
            if arousal > 0.7 and len(option_text) > 50:
                score -= 0.05  # 高唤醒时对冗长选项有轻微排斥

            score = round(max(0.05, min(0.95, score)), 4)
            scored.append((option, score))

        # 按偏差分数降序排列
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    # ========================================================================
    # 5. 心境一致性记忆召回
    # ========================================================================

    def compute_mood_congruence_boost(
        self,
        memory_emotion: str,
        memory_valence: Optional[float] = None,
        memory_arousal: Optional[float] = None,
        current_state: Optional[EmotionalState] = None,
    ) -> float:
        """
        计算心境一致性记忆加成

        心境一致性记忆 (Mood-Congruent Memory, Bower 1981):
        当人处于某种情绪状态时，更容易回忆起与该情绪一致的记忆。
        快乐时回忆快乐的事，悲伤时回忆悲伤的事。

        实现方式:
        1. 标签匹配: 当前情绪与记忆情绪标签是否一致
        2. 维度相似度: VAD 向量余弦相似度
        3. 综合加成: 相似度 × 配置权重

        Args:
            memory_emotion: 记忆的情绪标签 (joy/sadness/anger/fear/...)
            memory_valence: 记忆的效价值 (可选，默认从规则表获取)
            memory_arousal: 记忆的唤醒度值 (可选，默认从规则表获取)
            current_state: 当前情绪状态，若为 None 使用当前状态

        Returns:
            boost: 召回加成系数 (0.0 ~ config.mood_congruence_weight)
        """
        if current_state is None:
            current_state = self.get_current_state()

        # 快速路径: 中性状态不加成
        if current_state.primary_emotion == "neutral" or current_state.emotion_intensity < 0.15:
            return 0.0

        # 获取记忆的 VAD 值
        mem_rules = EMOTION_COGNITION_RULES.get(memory_emotion, EMOTION_COGNITION_RULES["neutral"])
        mem_valence = memory_valence if memory_valence is not None else mem_rules.get("valence", 0.0)
        mem_arousal = memory_arousal if memory_arousal is not None else mem_rules.get("arousal", 0.3)

        # 计算相似度
        similarity = self._compute_vad_similarity(
            current_state.valence, current_state.arousal,
            mem_valence, mem_arousal,
        )

        # 标签一致性加成
        label_match = 1.0 if current_state.primary_emotion == memory_emotion else 0.0

        # 综合加成: 相似度占60%，标签匹配占40%
        combined = 0.6 * similarity + 0.4 * label_match

        # 乘以情绪强度 (越强烈的情绪，心境一致性效应越明显)
        boost = combined * current_state.emotion_intensity * self._config.mood_congruence_weight

        return round(max(0.0, boost), 4)

    def rank_memories_by_mood(
        self,
        memories: List[Dict[str, Any]],
        current_state: Optional[EmotionalState] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        根据心境一致性对记忆列表重新排序

        为每条记忆计算心境一致性加成，并返回排序后的列表。
        可直接用于记忆召回的 reranking 阶段。

        Args:
            memories: 记忆列表，每条记忆应包含:
                - emotion: 情绪标签 (可选)
                - valence: 效价 (可选)
                - arousal: 唤醒度 (可选)
                - score: 原始召回分数 (可选)
            current_state: 当前情绪状态

        Returns:
            排序后的 (记忆, 调整后分数) 列表
        """
        if not memories:
            return []

        if current_state is None:
            current_state = self.get_current_state()

        ranked = []
        for mem in memories:
            base_score = mem.get("score", 0.5)
            boost = self.compute_mood_congruence_boost(
                memory_emotion=mem.get("emotion", "neutral"),
                memory_valence=mem.get("valence"),
                memory_arousal=mem.get("arousal"),
                current_state=current_state,
            )
            adjusted_score = base_score + boost
            ranked.append((mem, round(adjusted_score, 4)))

        # 按调整后分数降序排列
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    @staticmethod
    def _compute_vad_similarity(
        v1_valence: float, v1_arousal: float,
        v2_valence: float, v2_arousal: float,
    ) -> float:
        """
        计算两个 VAD 向量的余弦相似度

        使用 valence 和 arousal 两个维度计算相似度，
        归一化到 [0.0, 1.0] 区间。

        Args:
            v1_valence, v1_arousal: 第一个点的坐标
            v2_valence, v2_arousal: 第二个点的坐标

        Returns:
            相似度 (0.0 ~ 1.0)
        """
        # 2D 向量
        dot = v1_valence * v2_valence + v1_arousal * v2_arousal
        norm1 = math.sqrt(v1_valence ** 2 + v1_arousal ** 2)
        norm2 = math.sqrt(v2_valence ** 2 + v2_arousal ** 2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        cosine_sim = dot / (norm1 * norm2)
        # 将 [-1, 1] 映射到 [0, 1]
        return max(0.0, (cosine_sim + 1.0) / 2.0)

    # ========================================================================
    # 6. 情绪调节 (Emotional Regulation)
    # ========================================================================

    def regulate_emotion(
        self,
        target_state: Optional[Dict[str, float]] = None,
        strategy: Optional[RegulationStrategy] = None,
    ) -> Dict[str, Any]:
        """
        启动情绪调节过程

        情绪调节策略:
        1. 认知重评 (Cognitive Reappraisal): 重新解读情境含义
           - 逐步调整效价 (valence) 向目标偏移
           - 适用于负面情绪的转化
        2. 注意力转移 (Distraction): 将注意力转向中性话题
           - 快速降低唤醒度 (arousal)
           - 适用于高强度情绪的紧急降级
        3. 接纳 (Acceptance): 承认情绪而不对抗
           - 维持效价但降低唤醒度
           - 适用于无法立即改变的情绪

        Args:
            target_state: 目标状态，可包含:
                - valence: 目标效价 (-1.0 ~ 1.0)
                - arousal: 目标唤醒度 (0.0 ~ 1.0)
                - dominance: 目标支配度 (0.0 ~ 1.0)
                若为 None，默认向中性状态 (0.0, 0.3, 0.5) 调节
            strategy: 调节策略，若为 None 使用认知重评

        Returns:
            调节结果字典:
            - started: 是否成功启动
            - strategy: 使用的策略
            - target: 目标状态
            - current: 当前状态快照
            - estimated_steps: 预估需要多少步
        """
        with self._lock:
            return self._regulate_emotion_internal(target_state, strategy)

    def _regulate_emotion_internal(
        self,
        target_state: Optional[Dict[str, float]],
        strategy: Optional[RegulationStrategy],
    ) -> Dict[str, Any]:
        """内部情绪调节"""
        # 设置默认值
        if target_state is None:
            target_state = {"valence": 0.0, "arousal": 0.3, "dominance": 0.5}

        if strategy is None:
            strategy = RegulationStrategy.COGNITIVE_REAPPRAISAL

        self._regulation_target = target_state
        self._regulation_strategy = strategy
        self._regulation_active = True
        self._regulation_steps_taken = 0
        self._total_regulations += 1

        # 根据策略调整目标
        effective_target = dict(target_state)

        if strategy == RegulationStrategy.DISTRACTION:
            # 转移注意: 目标唤醒度设低，效价趋近中性
            effective_target["arousal"] = min(
                effective_target.get("arousal", 0.3), 0.3
            )
        elif strategy == RegulationStrategy.ACCEPTANCE:
            # 接纳: 保持效价但降低唤醒度
            effective_target["arousal"] = min(
                effective_target.get("arousal", 0.3), 0.4
            )
            effective_target["valence"] = self._state.valence  # 保留当前效价

        self._regulation_target = effective_target

        # 估算需要的步数
        distance = self._compute_state_distance(self._state, effective_target)
        estimated_steps = max(1, int(distance / self._config.regulation_rate))

        logger.info(
            f"[EmotionalThinking] 情绪调节启动: strategy={strategy.value}, "
            f"target=({effective_target.get('valence', 0):.2f}, "
            f"{effective_target.get('arousal', 0.3):.2f}, "
            f"{effective_target.get('dominance', 0.5):.2f}), "
            f"estimated_steps={estimated_steps}"
        )

        # 立即执行第一步调节
        self._apply_regulation_step()

        return {
            "started": True,
            "strategy": strategy.value,
            "target": effective_target,
            "current": {
                "valence": round(self._state.valence, 4),
                "arousal": round(self._state.arousal, 4),
                "dominance": round(self._state.dominance, 4),
                "emotion": self._state.primary_emotion,
                "intensity": round(self._state.emotion_intensity, 4),
            },
            "estimated_steps": estimated_steps,
        }

    def _apply_regulation_step(self):
        """执行一步情绪调节"""
        if not self._regulation_active or self._regulation_target is None:
            return

        rate = self._config.regulation_rate
        target = self._regulation_target

        # 各维度向目标靠近一步
        self._state.valence += rate * (target.get("valence", 0.0) - self._state.valence)
        self._state.arousal += rate * (target.get("arousal", 0.3) - self._state.arousal)
        self._state.dominance += rate * (target.get("dominance", 0.5) - self._state.dominance)

        # 情绪强度随调节逐渐下降
        self._state.emotion_intensity *= (1.0 - rate * 0.5)

        self._regulation_steps_taken += 1

        # 检查是否已到达目标 (收敛判断)
        distance = self._compute_state_distance(self._state, target)
        if distance < 0.05:
            self._regulation_active = False
            logger.info(
                f"[EmotionalThinking] 情绪调节完成: "
                f"steps={self._regulation_steps_taken}"
            )

        # 安全阀: 防止无限调节
        if self._regulation_steps_taken > 100:
            self._regulation_active = False
            logger.warning("[EmotionalThinking] 情绪调节超过100步，强制停止")

    def stop_regulation(self) -> bool:
        """
        停止当前的情绪调节过程

        Returns:
            stopped: 是否成功停止 (若未在调节中则返回 False)
        """
        with self._lock:
            if self._regulation_active:
                self._regulation_active = False
                self._regulation_target = None
                logger.info(f"[EmotionalThinking] 情绪调节已手动停止 (已执行{self._regulation_steps_taken}步)")
                return True
            return False

    def _compute_state_distance(
        self,
        state: EmotionalState,
        target: Dict[str, float],
    ) -> float:
        """
        计算当前状态与目标状态之间的欧氏距离

        Args:
            state: 当前情绪状态
            target: 目标状态字典

        Returns:
            欧氏距离
        """
        dv = state.valence - target.get("valence", 0.0)
        da = state.arousal - target.get("arousal", 0.3)
        dd = state.dominance - target.get("dominance", 0.5)
        return math.sqrt(dv ** 2 + da ** 2 + dd ** 2)

    # ========================================================================
    # 7. 情绪智能整合
    # ========================================================================

    def detect_emotional_needs(
        self,
        user_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        检测用户的情感需求

        分析用户输入中的情绪信号，判断用户的情感需求类型，
        以便 AI 选择合适的回应策略。

        情感需求类型:
        - empathy_seeking: 寻求共情 ("我好难过")
        - validation_seeking: 寻求认可 ("我做得对吗？")
        - venting: 发泄情绪 (不需要解决方案)
        - solution_seeking: 寻求解决方案 (理性需求)
        - reassurance_seeking: 寻求安慰 ("会不会出问题？")
        - social_bonding: 社交连接 ("今天真开心")

        Args:
            user_text: 用户输入文本
            context: 上下文信息

        Returns:
            情感需求分析结果
        """
        emotion, intensity = self._detect_emotion(user_text)
        text_lower = user_text.lower()

        needs = {
            "primary_need": "solution_seeking",  # 默认
            "emotion": emotion,
            "intensity": round(intensity, 4),
            "need_scores": {},
            "empathy_required": False,
            "de_escalation_required": False,
        }

        # 定义需求关键词模式
        need_patterns = {
            "empathy_seeking": [
                "没人理解", "孤独", "好累", "压力", "崩溃", "受不了",
                "没人", "一个人", "寂寞", "委屈",
                "no one understands", "lonely", "tired", "stress",
            ],
            "validation_seeking": [
                "我做得对吗", "这样行吗", "你怎么看", "我这样想对不对",
                "你觉得呢", "是不是我的问题", "我错了吗",
                "do you think", "am i right", "is this ok",
            ],
            "venting": [
                "气死我了", "烦死了", "受不了了", "太过分了",
                "凭什么", "无语", "真烦", "忍不了",
                "so angry", "frustrated", "can't stand",
            ],
            "reassurance_seeking": [
                "会不会", "担心", "怕", "万一", "糟糕",
                "焦虑", "不安", "紧张", "害怕",
                "what if", "worried", "anxious", "nervous",
            ],
            "social_bonding": [
                "谢谢", "哈哈", "好玩", "开心", "喜欢你",
                "好朋友", "一起", "陪伴", "真好",
                "thanks", "happy", "love", "together", "great",
            ],
            "solution_seeking": [
                "怎么办", "如何解决", "怎么处理", "帮我",
                "建议", "方法", "应该怎么做", "解决",
                "how to", "help me", "suggest", "solve", "solution",
            ],
        }

        # 计算各需求得分
        for need_type, keywords in need_patterns.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            needs["need_scores"][need_type] = score

        # 确定主要需求
        if needs["need_scores"]:
            best_need = max(needs["need_scores"], key=needs["need_scores"].get)
            if needs["need_scores"][best_need] > 0:
                needs["primary_need"] = best_need

        # 判断是否需要共情回应
        empathy_needs = {"empathy_seeking", "venting", "reassurance_seeking"}
        needs["empathy_required"] = needs["primary_need"] in empathy_needs

        # 判断是否需要降级干预
        needs["de_escalation_required"] = (
            intensity > self._config.emotional_escalation_threshold
            and emotion in ("anger", "fear", "sadness")
        )

        return needs

    def generate_empathy_prefix(
        self,
        user_text: str,
        emotional_needs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        生成共情前缀

        在回应内容之前，先承认用户的感受，体现情感智能。
        只有当检测到用户需要共情时才生成前缀。

        Args:
            user_text: 用户输入
            emotional_needs: 情感需求分析结果 (若为 None 则自动检测)

        Returns:
            共情前缀字符串，若不需要则返回空字符串
        """
        if emotional_needs is None:
            emotional_needs = self.detect_emotional_needs(user_text)

        if not emotional_needs.get("empathy_required", False):
            return ""

        emotion = emotional_needs.get("emotion", "neutral")
        need = emotional_needs.get("primary_need", "")
        intensity = emotional_needs.get("intensity", 0.0)

        # 根据情绪和需求类型选择共情话术
        empathy_templates = {
            "sadness": {
                "empathy_seeking": [
                    "我能感受到你现在的心情不太好……",
                    "听起来你正在经历一些不容易的事情。",
                    "我能理解你的感受，这一定很难受。",
                ],
                "venting": [
                    "说出来就好了，我在听。",
                    "你的感受完全可以理解。",
                ],
                "reassurance_seeking": [
                    "你的担忧我很理解，让我们一起看看。",
                ],
                "_default": [
                    "我理解你的心情。",
                ],
            },
            "anger": {
                "venting": [
                    "我能感受到你的不满……",
                    "这确实让人很生气。",
                    "你的愤怒完全可以理解。",
                ],
                "empathy_seeking": [
                    "我能感受到你现在的沮丧。",
                ],
                "_default": [
                    "我理解你的心情。",
                ],
            },
            "fear": {
                "reassurance_seeking": [
                    "你的担心是合理的，不用太紧张。",
                    "我理解你的不安，让我们一起来分析。",
                    "别担心，我们一起来看看情况。",
                ],
                "empathy_seeking": [
                    "我能感受到你的焦虑。",
                ],
                "_default": [
                    "我能理解你的顾虑。",
                ],
            },
            "joy": {
                "social_bonding": [
                    "太好了！看到你开心我也很高兴！",
                    "真替你高兴！",
                    "能感受到你的喜悦！",
                ],
                "_default": [
                    "很高兴听到这个！",
                ],
            },
        }

        # 查找模板
        emotion_templates = empathy_templates.get(emotion, {})
        templates = emotion_templates.get(need, emotion_templates.get("_default", []))

        if not templates:
            return ""

        # 根据强度选择模板 (高强度 → 更柔和的措辞)
        if intensity > 0.7:
            return templates[0]  # 第一个通常是更柔和的
        else:
            import random
            return random.choice(templates)

    def _handle_emotional_escalation(self, user_text: str):
        """
        处理情绪升级事件

        当检测到用户情绪强度超过阈值时，启动自动降级干预:
        1. 记录事件
        2. 自动启动情绪调节 (注意力转移策略)
        3. 标记当前交互需要额外共情

        Args:
            user_text: 用户输入文本
        """
        self._escalation_interventions += 1

        logger.info(
            f"[EmotionalThinking] 情绪升级检测! "
            f"emotion={self._state.primary_emotion}, "
            f"intensity={self._state.emotion_intensity:.3f}, "
            f"threshold={self._config.emotional_escalation_threshold}"
        )

        # 自动启动降级: 向中性状态缓慢调节
        self._regulation_active = True
        self._regulation_target = {"valence": 0.0, "arousal": 0.3, "dominance": 0.5}
        self._regulation_strategy = RegulationStrategy.DISTRACTION
        self._regulation_steps_taken = 0

        # 立即执行一步温和的降级
        self._apply_regulation_step()

    # ========================================================================
    # 获取思维模式偏好
    # ========================================================================

    def get_preferred_thinking_mode(self) -> str:
        """
        根据当前情绪状态获取偏好的思维模式

        将情绪-认知交互规则映射到 InnerThoughtEngine 的 ThinkingMode:
        - joy/happy → synthesizing (综合思维，更有创造力)
        - fear/anxious → analytical (分析思维，更谨慎)
        - sadness → reflecting (反思思维，更深入)
        - anger → critical (批判思维，更质疑)
        - neutral → analytical (默认分析思维)

        Returns:
            偏好的思维模式字符串 (与 ThinkingMode enum 值一致)
        """
        with self._lock:
            state = self._state

        rules = EMOTION_COGNITION_RULES.get(state.primary_emotion, EMOTION_COGNITION_RULES["neutral"])
        preferred = rules.get("thinking_mode_preference", "analytical")

        # 映射到 InnerThoughtEngine 的 ThinkingMode
        # inner_thought_engine.py 中定义: ANALYTICAL, DEDUCTIVE, INDUCTIVE, CRITICAL, SYNTHESIZING
        mode_mapping = {
            "analytical": "analytical",
            "synthesizing": "synthesizing",
            "reflecting": "reflecting",      # 反思模式，映射到 critical (接近反思)
            "critical": "critical",
        }

        # 注意: reflecting 不在 ThinkingMode 枚举中，映射到 closest
        mapped_mode = mode_mapping.get(preferred, "analytical")
        if preferred == "reflecting":
            mapped_mode = "critical"  # 反思最接近批判性审视

        return mapped_mode

    # ========================================================================
    # 序列化 / 反序列化
    # ========================================================================

    def get_state(self) -> Dict[str, Any]:
        """
        获取完整状态 (用于持久化)

        Returns:
            包含所有内部状态的字典
        """
        with self._lock:
            return {
                "emotional_state": {
                    "valence": self._state.valence,
                    "arousal": self._state.arousal,
                    "dominance": self._state.dominance,
                    "primary_emotion": self._state.primary_emotion,
                    "emotion_intensity": self._state.emotion_intensity,
                    "last_updated": self._state.last_updated,
                    "emotional_history": list(self._state.emotional_history),
                },
                "config": {
                    "emotion_inertia": self._config.emotion_inertia,
                    "contagion_rate": self._config.contagion_rate,
                    "mood_congruence_weight": self._config.mood_congruence_weight,
                    "regulation_rate": self._config.regulation_rate,
                    "history_max_length": self._config.history_max_length,
                    "emotional_escalation_threshold": self._config.emotional_escalation_threshold,
                },
                "regulation": {
                    "active": self._regulation_active,
                    "target": self._regulation_target,
                    "strategy": self._regulation_strategy.value if self._regulation_strategy else None,
                    "steps_taken": self._regulation_steps_taken,
                },
                "statistics": {
                    "total_updates": self._total_updates,
                    "total_regulations": self._total_regulations,
                    "total_contagions": self._total_contagions,
                    "escalation_interventions": self._escalation_interventions,
                },
                "contagion_history": list(self._contagion_history[-20:]),  # 只保留最近20条
            }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        从持久化状态恢复

        Args:
            state: 通过 get_state() 获取的状态字典
        """
        with self._lock:
            try:
                # 恢复情绪状态
                es = state.get("emotional_state", {})
                self._state.valence = es.get("valence", 0.0)
                self._state.arousal = es.get("arousal", 0.5)
                self._state.dominance = es.get("dominance", 0.5)
                self._state.primary_emotion = es.get("primary_emotion", "neutral")
                self._state.emotion_intensity = es.get("emotion_intensity", 0.0)
                self._state.last_updated = es.get("last_updated", time.time())
                self._state.emotional_history = es.get("emotional_history", [])

                # 恢复配置
                cfg = state.get("config", {})
                self._config.emotion_inertia = cfg.get("emotion_inertia", 0.7)
                self._config.contagion_rate = cfg.get("contagion_rate", 0.3)
                self._config.mood_congruence_weight = cfg.get("mood_congruence_weight", 0.15)
                self._config.regulation_rate = cfg.get("regulation_rate", 0.1)
                self._config.history_max_length = cfg.get("history_max_length", 100)
                self._config.emotional_escalation_threshold = cfg.get(
                    "emotional_escalation_threshold", 0.8
                )

                # 恢复调节状态
                reg = state.get("regulation", {})
                self._regulation_active = reg.get("active", False)
                self._regulation_target = reg.get("target")
                strategy_str = reg.get("strategy")
                if strategy_str:
                    try:
                        self._regulation_strategy = RegulationStrategy(strategy_str)
                    except ValueError:
                        self._regulation_strategy = RegulationStrategy.COGNITIVE_REAPPRAISAL
                self._regulation_steps_taken = reg.get("steps_taken", 0)

                # 恢复统计
                stats = state.get("statistics", {})
                self._total_updates = stats.get("total_updates", 0)
                self._total_regulations = stats.get("total_regulations", 0)
                self._total_contagions = stats.get("total_contagions", 0)
                self._escalation_interventions = stats.get("escalation_interventions", 0)

                # 恢复感染历史
                self._contagion_history = state.get("contagion_history", [])

                logger.info(
                    f"[EmotionalThinking] 状态已恢复: "
                    f"emotion={self._state.primary_emotion}, "
                    f"VAD=({self._state.valence:.2f}, {self._state.arousal:.2f}, "
                    f"{self._state.dominance:.2f})"
                )
            except Exception as e:
                logger.error(f"[EmotionalThinking] 状态恢复失败: {e}")

    # ========================================================================
    # 统计信息
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        获取运行统计信息

        Returns:
            统计信息字典，包含:
            - current_state: 当前情绪状态摘要
            - emotion_profile: 情绪-认知影响画像
            - regulation: 调节状态
            - statistics: 累计统计
            - config: 当前配置
        """
        with self._lock:
            current_snapshot = {
                "emotion": self._state.primary_emotion,
                "intensity": round(self._state.emotion_intensity, 4),
                "valence": round(self._state.valence, 4),
                "arousal": round(self._state.arousal, 4),
                "dominance": round(self._state.dominance, 4),
                "thinking_mode_preference": self.get_preferred_thinking_mode(),
                "last_updated": self._state.last_updated,
            }

            regulation_info = {
                "active": self._regulation_active,
                "strategy": self._regulation_strategy.value if self._regulation_active else None,
                "target": self._regulation_target if self._regulation_active else None,
                "steps_taken": self._regulation_steps_taken,
            }

        return {
            "current_state": current_snapshot,
            "emotion_profile": self.get_emotion_cognition_profile(),
            "regulation": regulation_info,
            "statistics": {
                "total_updates": self._total_updates,
                "total_regulations": self._total_regulations,
                "total_contagions": self._total_contagions,
                "escalation_interventions": self._escalation_interventions,
                "history_length": len(self._state.emotional_history),
                "contagion_history_length": len(self._contagion_history),
            },
            "config": {
                "emotion_inertia": self._config.emotion_inertia,
                "contagion_rate": self._config.contagion_rate,
                "mood_congruence_weight": self._config.mood_congruence_weight,
                "regulation_rate": self._config.regulation_rate,
                "emotional_escalation_threshold": self._config.emotional_escalation_threshold,
            },
        }


# ============================================================================
# 内置回退情绪检测器 (当 EmotionalMemoryModulator 不可用时)
# ============================================================================

class _FallbackEmotionDetector:
    """
    内置回退情绪检测器

    当 EmotionalMemoryModulator 无法导入时使用的简化检测器。
    提供基础的中英文情绪关键词匹配能力。
    """

    # 简化的情绪关键词库
    # 注意: 使用较长关键词避免误匹配 (如 "好" 在 "好难过" 中被误识别为积极)
    _KEYWORDS: Dict[str, List[str]] = {
        "joy": [
            "开心", "高兴", "快乐", "幸福", "兴奋", "愉快", "棒极了",
            "哈哈", "太好了", "喜欢", "喜爱", "nice", "happy", "great",
            "wonderful", "love", "awesome", "amazing", "excellent",
        ],
        "fear": [
            "害怕", "恐惧", "担心", "焦虑", "紧张", "可怕", "不安",
            "恐惧感", "害怕了", "危险", "afraid", "scared", "fear",
            "anxious", "worried", "nervous", "panic",
        ],
        "sadness": [
            "难过", "悲伤", "伤心", "痛苦", "失望", "沮丧", "哭泣",
            "忧郁", "遗憾", "低落", "心酸", "悲痛", "sad", "depressed",
            "unhappy", "disappointed", "grief", "lonely",
        ],
        "anger": [
            "生气", "愤怒", "恼火", "讨厌", "气愤", "气死",
            "不满", "暴怒", "可恶", "过分", "火大", "怒火",
            "angry", "furious", "mad", "annoyed", "hate",
            "frustrated", "enraged",
        ],
        "surprise": [
            "惊讶", "震惊", "意外", "没想到", "天哪", "哇",
            "出乎意料", "surprised", "shocked", "amazed", "wow", "unexpected",
        ],
        "disgust": [
            "恶心", "厌恶", "反感", "受不了", "鄙视",
            "令人作呕", "disgusted", "gross", "nasty",
        ],
    }

    def detect_emotion(self, text: str):
        """
        检测文本中的情绪

        使用加权计分: 较长关键词权重更高，减少短词误匹配。

        Args:
            text: 输入文本

        Returns:
            (emotion_type, intensity): 情绪类型和强度
        """
        if not text or not text.strip():
            return ("neutral", 0.0)

        text_lower = text.lower()
        scores: Dict[str, float] = {}

        for emotion, keywords in self._KEYWORDS.items():
            weighted_count = 0.0
            for kw in keywords:
                if kw in text_lower:
                    # 较长关键词权重更高，避免短词误匹配
                    weight = min(len(kw) / 2.0, 2.0)
                    weighted_count += weight
            if weighted_count > 0:
                scores[emotion] = weighted_count

        if not scores:
            return ("neutral", 0.0)

        best_emotion = max(scores, key=scores.get)
        count = scores[best_emotion]

        # 强度估算: 使用加权计数
        import math
        intensity = min(0.1 + 0.15 * math.log1p(count * 2), 1.0)

        return (best_emotion, round(intensity, 3))
