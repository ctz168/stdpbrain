"""
记忆重构引擎 - Memory Reconstruction Engine

基于认知神经科学中的「建构性记忆」理论（Constructive Memory Theory），
模拟人类记忆的碎片化存储与重构式回忆过程。

科学背景:
    - 人类记忆不是像录像带一样精确回放，而是从碎片中动态重构
    - Bartlett (1932): 记忆是"重建过去"而非"再现过去"
    - Schacter (1999): 记忆的七宗罪 — 包括失忆、歪曲、暗示敏感性等
    - Loftus (1975): 虚假记忆范式 — 外部暗示可以修改重构过程
    - 每次回忆都是一次"重新编码"，记忆会随时间被微调

核心机制:
    1. 碎片化提取: 将完整记忆拆解为 Who/What/Where/When/Why/How/Emotion 碎片
    2. 模板化重构: 使用中文叙事模板将碎片组装为连贯记忆叙述
    3. 置信度追踪: 区分直接回忆 / 碎片拼合 / 联想推理 / 缺口填补
    4. 时间排序: 从多个记忆碎片中重建事件序列
    5. 记忆扭曲: 模拟人类记忆随时间的自然偏差
    6. 智能缓存: 避免重复重构，提升响应效率

依赖:
    - ca3_memory.EpisodicMemory: 情景记忆数据结构
    - human_memory_enhancements: 情绪检测与语境分析
"""

import time
import re
import math
import hashlib
import logging
import uuid
from enum import Enum
from typing import (
    Dict, List, Optional, Tuple, Any, Set, Union,
    NamedTuple, Sequence
)
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ============================================================================
# 枚举与基础数据结构
# ============================================================================

class FragmentType(Enum):
    """
    记忆碎片类型枚举

    对应人类记忆编码中的核心维度:
    - WHO: 人物/实体（谁参与了这件事）
    - WHAT: 事件/动作（发生了什么）
    - WHERE: 地点/场景（在哪里发生的）
    - WHEN: 时间（何时发生的）
    - WHY: 原因/动机（为什么会发生）
    - HOW: 方式/过程（是如何发生的）
    - EMOTION: 情绪（当时的感受）
    """
    WHO = "who"              # 谁（人物/实体）
    WHAT = "what"            # 什么（事件/动作）
    WHERE = "where"          # 哪里（地点/场景）
    WHEN = "when"            # 何时（时间参考）
    WHY = "why"              # 为什么（原因/动机）
    HOW = "how"              # 怎么（方式/过程）
    EMOTION = "emotion"      # 情绪（当时的感受）


class ConfidenceLevel(Enum):
    """
    重构置信度等级

    模拟人类对不同记忆来源的"确定感":
    - DIRECT: 直接回忆 — 记忆清晰完整，直接提取 (0.8-1.0)
    - FRAGMENT: 碎片拼合 — 从多个碎片中拼合，有少量推断 (0.5-0.8)
    - ASSOCIATIVE: 联想推理 — 基于关联信息推断，不是直接记忆 (0.3-0.5)
    - GAP_FILL: 缺口填补 — 无法回忆，用合理推测补充 (0.1-0.3)
    """
    DIRECT = "direct"
    FRAGMENT = "fragment"
    ASSOCIATIVE = "associative"
    GAP_FILL = "gap_fill"

    @property
    def score_range(self) -> Tuple[float, float]:
        """该等级对应的置信度分数范围"""
        ranges = {
            ConfidenceLevel.DIRECT: (0.8, 1.0),
            ConfidenceLevel.FRAGMENT: (0.5, 0.8),
            ConfidenceLevel.ASSOCIATIVE: (0.3, 0.5),
            ConfidenceLevel.GAP_FILL: (0.1, 0.3),
        }
        return ranges[self]

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """根据置信度分数判断等级"""
        if score >= 0.8:
            return cls.DIRECT
        elif score >= 0.5:
            return cls.FRAGMENT
        elif score >= 0.3:
            return cls.ASSOCIATIVE
        else:
            return cls.GAP_FILL


class ReconstructionTemplate(Enum):
    """
    记忆重构模板类型

    支持多种中文叙事模板，模拟人类不同的回忆方式:
    """
    EPISODIC = "episodic"            # 情景回忆：时间线式叙述
    FACTUAL = "factual"              # 事实陈述：知识点式回忆
    EMOTIONAL = "emotional"          # 情感回忆：以情绪为中心
    CAUSAL = "causal"                # 因果回忆：强调原因与结果
    SEQUENCE = "sequence"            # 序列回忆：多个事件的时序排列


# ============================================================================
# 碎片数据结构
# ============================================================================

@dataclass
class MemoryFragment:
    """
    记忆碎片 — 记忆的最小存储与检索单元

    每个碎片代表记忆的一个维度信息（如"谁"、"什么"、"在哪里"）。
    碎片独立存储，通过 source_memory_id 关联到原始记忆。

    Attributes:
        fragment_id: 碎片唯一标识
        fragment_type: 碎片类型（Who/What/Where/...）
        content: 碎片内容文本
        source_memory_id: 来源记忆 ID（可多个，竖线分隔）
        confidence: 该碎片的置信度 (0.0-1.0)
        confidence_level: 置信度等级
        created_time: 碎片创建时间戳（秒）
        access_count: 被访问的次数
        last_access_time: 最后访问时间
        metadata: 扩展元数据（如情绪强度、位置信息等）
    """
    fragment_id: str
    fragment_type: FragmentType
    content: str
    source_memory_id: str
    confidence: float = 1.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.DIRECT
    created_time: float = 0.0
    access_count: int = 0
    last_access_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        if self.created_time <= 0:
            self.created_time = time.time()
        self.confidence_level = ConfidenceLevel.from_score(self.confidence)

    def touch(self) -> None:
        """记录一次访问"""
        self.access_count += 1
        self.last_access_time = time.time()

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'fragment_id': self.fragment_id,
            'fragment_type': self.fragment_type.value,
            'content': self.content,
            'source_memory_id': self.source_memory_id,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value,
            'created_time': self.created_time,
            'access_count': self.access_count,
            'last_access_time': self.last_access_time,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryFragment":
        """从字典反序列化"""
        return cls(
            fragment_id=data.get('fragment_id', str(uuid.uuid4())),
            fragment_type=FragmentType(data.get('fragment_type', 'what')),
            content=data.get('content', ''),
            source_memory_id=data.get('source_memory_id', ''),
            confidence=data.get('confidence', 1.0),
            confidence_level=ConfidenceLevel(data.get('confidence_level', 'direct')),
            created_time=data.get('created_time', time.time()),
            access_count=data.get('access_count', 0),
            last_access_time=data.get('last_access_time', 0.0),
            metadata=data.get('metadata', {}),
        )


@dataclass
class ConfidenceBreakdown:
    """
    重构置信度细目 — 追踪每个维度的置信度

    用于向调用方暴露哪些部分是确定的，哪些是推断的。
    """
    who_confidence: float = 0.0
    what_confidence: float = 0.0
    where_confidence: float = 0.0
    when_confidence: float = 0.0
    why_confidence: float = 0.0
    how_confidence: float = 0.0
    emotion_confidence: float = 0.0

    @property
    def overall_confidence(self) -> float:
        """计算总体置信度 — 取所有维度的加权平均"""
        weights = {
            'who': 0.15, 'what': 0.25, 'where': 0.10,
            'when': 0.10, 'why': 0.15, 'how': 0.10, 'emotion': 0.15,
        }
        total = 0.0
        weight_sum = 0.0
        for attr, weight in weights.items():
            val = getattr(self, f"{attr}_confidence", 0.0)
            total += val * weight
            weight_sum += weight
        return round(total / weight_sum, 4) if weight_sum > 0 else 0.0

    @property
    def gap_count(self) -> int:
        """统计低置信度维度数量（置信度 < 0.3 视为缺口）"""
        attrs = ['who', 'what', 'where', 'when', 'why', 'how', 'emotion']
        return sum(1 for a in attrs if getattr(self, f"{a}_confidence", 0.0) < 0.3)

    def to_dict(self) -> dict:
        return {
            'who_confidence': self.who_confidence,
            'what_confidence': self.what_confidence,
            'where_confidence': self.where_confidence,
            'when_confidence': self.when_confidence,
            'why_confidence': self.why_confidence,
            'how_confidence': self.how_confidence,
            'emotion_confidence': self.emotion_confidence,
            'overall_confidence': self.overall_confidence,
            'gap_count': self.gap_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConfidenceBreakdown":
        return cls(
            who_confidence=data.get('who_confidence', 0.0),
            what_confidence=data.get('what_confidence', 0.0),
            where_confidence=data.get('where_confidence', 0.0),
            when_confidence=data.get('when_confidence', 0.0),
            why_confidence=data.get('why_confidence', 0.0),
            how_confidence=data.get('how_confidence', 0.0),
            emotion_confidence=data.get('emotion_confidence', 0.0),
        )


@dataclass
class ReconstructedMemory:
    """
    重构后的记忆 — 引擎的输出产物

    与原始 EpisodicMemory 不同，重构记忆是从碎片中组装的，
    并带有完整的置信度标注和元数据。

    Attributes:
        reconstruction_id: 重构结果唯一 ID
        narrative: 重构后的叙事文本（中文自然语言）
        fragments_used: 使用的碎片列表
        source_memory_ids: 来源记忆 ID 集合
        overall_confidence: 总体置信度
        confidence_breakdown: 各维度置信度细目
        template_used: 使用的重构模板
        is_reconstructed: 是否经过重构（vs 直接回放）
        temporal_order: 时间排序后的碎片序列
        inferred_elements: 推断/填补的元素列表
        distortion_applied: 是否应用了记忆扭曲
        distortion_metadata: 扭曲相关元数据
        created_time: 重构时间
        query: 触发重构的原始查询
        metadata: 扩展元数据
    """
    reconstruction_id: str = ""
    narrative: str = ""
    fragments_used: List[MemoryFragment] = field(default_factory=list)
    source_memory_ids: List[str] = field(default_factory=list)
    overall_confidence: float = 0.0
    confidence_breakdown: Optional[ConfidenceBreakdown] = None
    template_used: ReconstructionTemplate = ReconstructionTemplate.EPISODIC
    is_reconstructed: bool = True
    temporal_order: List[str] = field(default_factory=list)
    inferred_elements: List[Dict[str, Any]] = field(default_factory=list)
    distortion_applied: bool = False
    distortion_metadata: Dict[str, Any] = field(default_factory=dict)
    created_time: float = 0.0
    query: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.reconstruction_id:
            self.reconstruction_id = f"recon_{uuid.uuid4().hex[:12]}"
        if self.created_time <= 0:
            self.created_time = time.time()
        if self.confidence_breakdown is None:
            self.confidence_breakdown = ConfidenceBreakdown()

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'reconstruction_id': self.reconstruction_id,
            'narrative': self.narrative,
            'fragments_used': [f.to_dict() for f in self.fragments_used],
            'source_memory_ids': self.source_memory_ids,
            'overall_confidence': self.overall_confidence,
            'confidence_breakdown': self.confidence_breakdown.to_dict() if self.confidence_breakdown else None,
            'template_used': self.template_used.value,
            'is_reconstructed': self.is_reconstructed,
            'temporal_order': self.temporal_order,
            'inferred_elements': self.inferred_elements,
            'distortion_applied': self.distortion_applied,
            'distortion_metadata': self.distortion_metadata,
            'created_time': self.created_time,
            'query': self.query,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReconstructedMemory":
        """从字典反序列化"""
        fragments = [
            MemoryFragment.from_dict(f) for f in data.get('fragments_used', [])
        ]
        breakdown = None
        if data.get('confidence_breakdown'):
            breakdown = ConfidenceBreakdown.from_dict(data['confidence_breakdown'])

        return cls(
            reconstruction_id=data.get('reconstruction_id', ''),
            narrative=data.get('narrative', ''),
            fragments_used=fragments,
            source_memory_ids=data.get('source_memory_ids', []),
            overall_confidence=data.get('overall_confidence', 0.0),
            confidence_breakdown=breakdown,
            template_used=ReconstructionTemplate(data.get('template_used', 'episodic')),
            is_reconstructed=data.get('is_reconstructed', True),
            temporal_order=data.get('temporal_order', []),
            inferred_elements=data.get('inferred_elements', []),
            distortion_applied=data.get('distortion_applied', False),
            distortion_metadata=data.get('distortion_metadata', {}),
            created_time=data.get('created_time', time.time()),
            query=data.get('query', ''),
            metadata=data.get('metadata', {}),
        )


# ============================================================================
# 重构模板定义
# ============================================================================

class ReconstructionTemplates:
    """
    中文记忆重构模板集

    人类回忆不是像搜索结果一样返回原始数据，
    而是用自然的语言模式重新叙述。这些模板模拟了常见的中文回忆句式。

    每个模板包含:
    - template: 格式字符串，使用 {placeholder} 占位符
    - required_slots: 必须填写的槽位列表
    - optional_slots: 可选槽位（缺失时自动跳过对应子句）
    - fallback: 碎片不足时的简化模板
    """

    TEMPLATES = {
        ReconstructionTemplate.EPISODIC: {
            "name": "情景回忆模板",
            "template": "我记得{time_ref}，{person}在{location}{action_desc}，当时{emotion_desc}。",
            "slots": {
                "time_ref": FragmentType.WHEN,
                "person": FragmentType.WHO,
                "location": FragmentType.WHERE,
                "action_desc": FragmentType.WHAT,
                "emotion_desc": FragmentType.EMOTION,
            },
            "optional_slots": ["time_ref", "location", "emotion_desc"],
            "fallback": "我记得{person}{action_desc}。",
        },
        ReconstructionTemplate.FACTUAL: {
            "name": "事实陈述模板",
            "template": "关于{topic}，我知道{fact}。",
            "slots": {
                "topic": FragmentType.WHAT,
                "fact": FragmentType.WHAT,
            },
            "optional_slots": [],
            "fallback": "我记得{fact}。",
        },
        ReconstructionTemplate.EMOTIONAL: {
            "name": "情感回忆模板",
            "template": "那次经历让我感到{emotion}，因为{reason}。{detail}",
            "slots": {
                "emotion": FragmentType.EMOTION,
                "reason": FragmentType.WHY,
                "detail": FragmentType.WHAT,
            },
            "optional_slots": ["reason", "detail"],
            "fallback": "我记得那种{emotion}的感觉。",
        },
        ReconstructionTemplate.CAUSAL: {
            "name": "因果回忆模板",
            "template": "因为{cause}，所以{result}，{how_desc}。",
            "slots": {
                "cause": FragmentType.WHY,
                "result": FragmentType.WHAT,
                "how_desc": FragmentType.HOW,
            },
            "optional_slots": ["how_desc"],
            "fallback": "因为{cause}，所以{result}。",
        },
        ReconstructionTemplate.SEQUENCE: {
            "name": "序列回忆模板",
            "template": "{event_list}",
            "slots": {
                "event_list": FragmentType.WHAT,
            },
            "optional_slots": [],
            "fallback": "{event_list}",
        },
    }

    @classmethod
    def get_template(cls, template_type: ReconstructionTemplate) -> dict:
        """获取指定模板的配置"""
        return cls.TEMPLATES.get(template_type, cls.TEMPLATES[ReconstructionTemplate.EPISODIC])

    @classmethod
    def render(
        cls,
        template_type: ReconstructionTemplate,
        slot_values: Dict[str, str],
    ) -> str:
        """
        使用槽位值渲染模板

        Args:
            template_type: 模板类型
            slot_values: 槽位名 → 值 的映射

        Returns:
            渲染后的中文叙述文本
        """
        config = cls.get_template(template_type)
        template_str = config["template"]
        optional_slots = config.get("optional_slots", [])
        fallback_str = config.get("fallback", template_str)

        # 检查必填槽位
        required_slots = [s for s in config["slots"] if s not in optional_slots]
        missing_required = [s for s in required_slots if not slot_values.get(s)]

        if missing_required:
            # 必填槽位缺失 → 使用简化 fallback 模板
            try:
                result = fallback_str.format(**slot_values)
                return result
            except KeyError:
                # fallback 也失败 → 拼接可用碎片
                parts = [v for v in slot_values.values() if v]
                return "、".join(parts) + "。" if parts else ""

        # 移除可选槽位中的空值
        render_values = {}
        for slot_name, value in slot_values.items():
            if slot_name in optional_slots and not value:
                # 可选槽位为空 → 替换为空字符串
                render_values[slot_name] = ""
            else:
                render_values[slot_name] = value

        try:
            result = template_str.format(**render_values)
        except KeyError:
            result = fallback_str.format(**render_values)

        # 清理多余标点（连续句号、逗号等）
        result = re.sub(r'，\s*，', '，', result)
        result = re.sub(r'，\s*。', '。', result)
        result = re.sub(r'。+', '。', result)
        result = re.sub(r'\s+', ' ', result).strip()

        return result


# ============================================================================
# 记忆扭曲引擎
# ============================================================================

class MemoryDistortionEngine:
    """
    记忆扭曲引擎 — 模拟人类记忆的自然偏差

    科学依据:
    - Loftus & Palmer (1974): 误导信息效应 — 外部信息可修改记忆
    - Roediger & McDermott (1995): DRM范式 — 联想可以创造虚假记忆
    - Bartlett (1932): 记忆随时间被"合理化"（趋向文化/个人图式）
    - Anderson (2013): 记忆的"修正"机制 — 每次回忆都微调记忆内容

    实现策略:
    - 时间衰减扭曲: 记忆越旧，扭曲越大（模拟遗忘导致的不确定性）
    - 情绪染色: 当前情绪状态会影响记忆内容的情感色调
    - 语义平滑: 零碎记忆会被自动"圆滑化"，补全缺失的过渡
    - 强度衰减: 细节的锐利度随时间降低，模糊化

    设计原则:
    - 扭曲是温和的，不会改变记忆的核心事实
    - 扭曲程度与记忆年龄正相关
    - 扭曲可以被禁用（通过参数控制）
    """

    # 扭曲配置
    DISTORTION_CAP = 0.3           # 最大扭曲程度（30%，保持核心事实不变）
    TIME_DECAY_FACTOR = 0.000001   # 时间衰减因子（每秒）
    EMOTION_DISTORTION_FACTOR = 0.15  # 情绪染色强度

    def __init__(
        self,
        enabled: bool = True,
        max_distortion: float = DISTORTION_CAP,
        time_decay_factor: float = TIME_DECAY_FACTOR,
        emotion_distortion_factor: float = EMOTION_DISTORTION_FACTOR,
    ):
        """
        初始化记忆扭曲引擎

        Args:
            enabled: 是否启用扭曲模拟
            max_distortion: 最大扭曲程度 (0.0-1.0)
            time_decay_factor: 时间衰减因子
            emotion_distortion_factor: 情绪染色因子
        """
        self.enabled = enabled
        self.max_distortion = max_distortion
        self.time_decay_factor = time_decay_factor
        self.emotion_distortion_factor = emotion_distortion_factor

    def compute_distortion_amount(
        self,
        memory_age_seconds: float,
        current_emotion: str = "neutral",
        emotion_intensity: float = 0.0,
    ) -> float:
        """
        计算给定记忆应施加的扭曲量

        扭曲量 = 时间衰减贡献 + 情绪染色贡献（有上限）

        Args:
            memory_age_seconds: 记忆年龄（秒）
            current_emotion: 当前情绪状态
            emotion_intensity: 当前情绪强度 (0.0-1.0)

        Returns:
            distortion_amount: 扭曲量 (0.0 - max_distortion)
        """
        if not self.enabled:
            return 0.0

        # 1. 时间衰减贡献 — 对数增长，模拟"老记忆更模糊"
        age_days = memory_age_seconds / 86400.0
        time_distortion = self.time_decay_factor * memory_age_seconds
        # 使用对数限制增长速率
        time_distortion = min(time_distortion, 0.2 * (1 - math.exp(-age_days / 30.0)))

        # 2. 情绪染色贡献
        emotion_distortion = 0.0
        if current_emotion != "neutral" and emotion_intensity > 0:
            # 高唤醒情绪产生更多染色
            high_arousal = {"joy", "fear", "anger", "surprise"}
            arousal_bonus = 1.5 if current_emotion in high_arousal else 1.0
            emotion_distortion = (
                self.emotion_distortion_factor
                * emotion_intensity
                * arousal_bonus
            )

        # 3. 合并并限制
        total = time_distortion + emotion_distortion
        return min(total, self.max_distortion)

    def apply_distortion(
        self,
        content: str,
        distortion_amount: float,
        current_emotion: str = "neutral",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        对记忆内容施加温和扭曲

        当前实现的扭曲策略:
        - 情绪词汇微调: 在描述中融入当前情绪的近义词
        - 细节模糊化: 长文本截取核心部分，模拟细节遗忘
        - 语序微调: 在不改变语义的前提下微调表达

        注意: 扭曲是概率性的，不是每次都会发生

        Args:
            content: 原始记忆内容
            distortion_amount: 扭曲量 (0.0-1.0)
            current_emotion: 当前情绪状态

        Returns:
            (distorted_content, metadata): 扭曲后的内容和元数据
        """
        if not self.enabled or distortion_amount < 0.01 or not content:
            return content, {"distorted": False, "amount": 0.0}

        metadata = {
            "distorted": False,
            "amount": distortion_amount,
            "strategies_applied": [],
        }

        result = content

        # 策略1: 细节模糊化 — 对长文本进行温和截取
        if len(result) > 50 and distortion_amount > 0.05:
            # 保留核心部分，移除过于具体的细节
            sentences = re.split(r'[，。；！？]', result)
            if len(sentences) > 3:
                # 保留首句和末句（人类通常记得开头和结尾）
                keep_count = max(2, int(len(sentences) * (1 - distortion_amount * 0.5)))
                keep_indices = sorted(set([0, -1] + list(range(1, keep_count - 1))))
                kept = [sentences[i] for i in keep_indices if 0 <= i < len(sentences) and sentences[i]]
                result = "，".join(kept)
                metadata["strategies_applied"].append("detail_blur")

        # 策略2: 语气润色 — 基于当前情绪添加轻微语气词
        if current_emotion != "neutral" and distortion_amount > 0.08:
            emotion_filler = {
                "joy": "好像", "fear": "感觉", "sadness": "印象中",
                "anger": "记得", "surprise": "好像", "disgust": "记得",
            }
            filler = emotion_filler.get(current_emotion, "")
            if filler and not result.startswith(filler):
                # 概率性地在句首添加语气修饰
                import random
                if random.random() < distortion_amount * 2:
                    result = f"{filler}，{result}"
                    metadata["strategies_applied"].append("emotion_filler")

        # 策略3: 添加不确定性标记 — 模拟记忆的不确定感
        if distortion_amount > 0.15:
            import random
            uncertain_markers = ["大概", "可能", "似乎", "应该"]
            marker = uncertain_markers[0]  # 用确定性最高的
            if random.random() < distortion_amount:
                # 在适当位置插入不确定性标记
                if "，" in result:
                    parts = result.split("，", 1)
                    result = f"{parts[0]}，{marker}{parts[1]}"
                else:
                    result = f"{marker}{result}"
                metadata["strategies_applied"].append("uncertainty_marker")

        metadata["distorted"] = bool(metadata["strategies_applied"])
        return result, metadata


# ============================================================================
# 核心引擎
# ============================================================================

class MemoryReconstructionEngine:
    """
    记忆重构引擎 — 核心类

    负责将存储的 EpisodicMemory 转化为类似人类的重构式回忆。
    不直接操作 EpisodicMemory 对象，而是接收它们作为输入，
    输出 ReconstructedMemory 对象。

    工作流程:
        1. extract_fragments() — 将记忆拆解为碎片
        2. reconstruct_memory() — 从碎片中重构连贯叙述
        3. 可选: apply_distortion() — 模拟记忆扭曲

    设计原则:
        - 纯函数式: 不修改输入记忆，所有结果都是新创建的
        - 幂等性: 相同输入产生相同输出（通过缓存保证）
        - 延迟计算: 只在需要时才执行重构
        - 可序列化: 所有状态都可以 get_state/set_state

    用法示例:
        >>> engine = MemoryReconstructionEngine()
        >>> fragments = engine.extract_fragments(episodic_memory)
        >>> result = engine.reconstruct_memory("你还记得我叫什么", [episodic_memory])
        >>> print(result.narrative)
        >>> print(f"置信度: {result.overall_confidence:.2f}")
    """

    # 碎片提取模式 — 用于从文本中识别不同维度的信息
    FRAGMENT_PATTERNS = {
        FragmentType.WHO: {
            "name": "人物/实体",
            "patterns": [
                r"我叫([\u4e00-\u9fa5a-zA-Z]{2,4})",
                r"我是([\u4e00-\u9fa5a-zA-Z]{2,8}?)(?=[，。,.\s]|$)",
                r"我的名字(?:是|叫)([\u4e00-\u9fa5a-zA-Z]{2,4})",
                r"([\u4e00-\u9fa5]{2,3})(?:告诉|跟|给)(?:我|你)",
                r"(?:和|跟)([\u4e00-\u9fa5]{2,4})(?:一起|一起|去)",
                r"([\u4e00-\u9fa5]{2,4})(?:说|说|觉得|认为)",
            ],
        },
        FragmentType.WHAT: {
            "name": "事件/动作",
            "patterns": [
                r"(?:去|到|在)([\u4e00-\u9fa5a-zA-Z]{2,15}?)(?:的|了|过|玩|工作|上学)",
                r"(?:喜欢|爱|讨厌|爱好)([\u4e00-\u9fa5a-zA-Z]{2,15})",
                r"(?:做|干|从事)([\u4e00-\u9fa5a-zA-Z]{2,15}?)(?:工作|职业|行业)",
                r"(?:买了|买了|换了|用了)([\u4e00-\u9fa5a-zA-Z]{2,15})",
            ],
        },
        FragmentType.WHERE: {
            "name": "地点/场景",
            "patterns": [
                r"来自([\u4e00-\u9fa5a-zA-Z]{2,10})",
                r"住在([\u4e00-\u9fa5a-zA-Z]{2,10})",
                r"在([\u4e00-\u9fa5a-zA-Z]{2,10}?)(?:工作|生活|上学|上班|玩)",
                r"(?:去|到|从)([\u4e00-\u9fa5a-zA-Z]{2,10}?)(?:旅行|旅游|出差|玩|回来)",
                r"([\u4e00-\u9fa5]{2,6}?)(?:市|省|区|县|镇|村|路|街|号)",
            ],
        },
        FragmentType.WHEN: {
            "name": "时间参考",
            "patterns": [
                r"(\d{4}年\d{1,2}月\d{1,2}[日号]?)",
                r"(?:去年|今年|明年|前年|后年)([\u4e00-\u9fa5]{0,5})",
                r"(?:昨天|今天|明天|前天|后天)([\u4e00-\u9fa5]{0,5})",
                r"(?:上|下|这)(?:周|月|次|个)(?:[\u4e00-\u9fa5]{0,5})",
                r"(?:大前天|大后天|刚才|刚刚|最近|之前|以前|之前|上次|之前)",
                r"(\d{1,2})月(\d{1,2})[日号]?",
            ],
        },
        FragmentType.WHY: {
            "name": "原因/动机",
            "patterns": [
                r"因为([\u4e00-\u9fa5a-zA-Z，。]{2,30}?)(?:所以|因此|导致)",
                r"由于([\u4e00-\u9fa5a-zA-Z，。]{2,30})",
                r"(?:为了|是为了)([\u4e00-\u9fa5a-zA-Z，。]{2,30})",
                r"(?:所以|因此|导致|因为(?:此)?)?([\u4e00-\u9fa5a-zA-Z]{2,20})(?:的原因|的原因)",
            ],
        },
        FragmentType.HOW: {
            "name": "方式/过程",
            "patterns": [
                r"(?:通过|用|使用)([\u4e00-\u9fa5a-zA-Z]{2,20}?)(?:来|去|完成|实现|做)",
                r"(?:一步一步|慢慢|逐渐|终于|最终)([\u4e00-\u9fa5a-zA-Z]{2,20})",
                r"(?:方法|方式|途径)(?:是|为)([\u4e00-\u9fa5a-zA-Z]{2,20})",
            ],
        },
        FragmentType.EMOTION: {
            "name": "情绪",
            "patterns": [
                r"(?:感觉|觉得|感到|认为)([\u4e00-\u9fa5a-zA-Z]{2,10})",
                r"(?:很|非常|特别|挺|蛮)([\u4e00-\u9fa5]{2,6})",
                r"(?:开心|高兴|快乐|幸福|悲伤|难过|焦虑|担心|害怕|紧张|激动|兴奋|失望|沮丧|满足|欣慰|期待|后悔|遗憾)",
            ],
        },
    }

    def __init__(
        self,
        enable_distortion: bool = True,
        max_distortion: float = 0.3,
        cache_enabled: bool = True,
        cache_max_size: int = 200,
        cache_ttl_seconds: float = 300.0,
    ):
        """
        初始化记忆重构引擎

        Args:
            enable_distortion: 是否启用记忆扭曲模拟
            max_distortion: 最大扭曲程度 (0.0-1.0)
            cache_enabled: 是否启用重构结果缓存
            cache_max_size: 缓存最大条目数
            cache_ttl_seconds: 缓存过期时间（秒）
        """
        # 碎片存储: memory_id → List[MemoryFragment]
        self._fragment_store: Dict[str, List[MemoryFragment]] = {}

        # 扭曲引擎
        self._distortion_engine = MemoryDistortionEngine(
            enabled=enable_distortion,
            max_distortion=max_distortion,
        )

        # 重构结果缓存: cache_key → (ReconstructedMemory, timestamp)
        self._cache: Dict[str, Tuple[ReconstructedMemory, float]] = {}
        self._cache_enabled = cache_enabled
        self._cache_max_size = cache_max_size
        self._cache_ttl = cache_ttl_seconds

        # 统计信息
        self._stats = {
            "total_extractions": 0,
            "total_reconstructions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_confidence": 0.0,
            "distortion_count": 0,
        }

        logger.info(
            f"[记忆重构引擎] 初始化完成 | "
            f"扭曲={'启用' if enable_distortion else '禁用'} | "
            f"缓存={'启用' if cache_enabled else '禁用'}"
        )

    # ====================================================================
    # 1. 碎片提取
    # ====================================================================

    def extract_fragments(
        self,
        memory: Any,
    ) -> List[MemoryFragment]:
        """
        将单条记忆拆解为关键碎片

        从 EpisodicMemory 对象中提取七维碎片:
        Who（人物）、What（事件）、Where（地点）、When（时间）、
        Why（原因）、How（方式）、Emotion（情绪）

        碎片提取策略:
        1. 优先从结构化字段提取（key_entities, semantic_summary, emotion_tag）
        2. 其次用正则模式从原始文本中提取
        3. 最后对提取结果去重、评级

        Args:
            memory: EpisodicMemory 对象（或具有类似属性的字典）

        Returns:
            fragments: 提取到的碎片列表
        """
        self._stats["total_extractions"] += 1

        # 兼容 dict 和 EpisodicMemory 对象
        if isinstance(memory, dict):
            mem_id = memory.get("memory_id", "unknown")
            content = memory.get("content", "") or memory.get("user_input", "")
            summary = memory.get("semantic_summary", "")
            entities = memory.get("key_entities", "")
            emotion_tag = memory.get("emotion_tag", "中性")
            emotion_type = memory.get("emotion_type", "neutral")
            emotion_intensity = float(memory.get("emotion_intensity", 0.0))
            timestamp = memory.get("timestamp", 0)
            is_core = memory.get("is_core", False)
            user_input = memory.get("user_input", "")
        else:
            mem_id = getattr(memory, "memory_id", "unknown")
            content = getattr(memory, "content", "") or getattr(memory, "user_input", "")
            summary = getattr(memory, "semantic_summary", "")
            entities = getattr(memory, "key_entities", "")
            emotion_tag = getattr(memory, "emotion_tag", "中性")
            emotion_type = getattr(memory, "emotion_type", "neutral")
            emotion_intensity = float(getattr(memory, "emotion_intensity", 0.0))
            timestamp = getattr(memory, "timestamp", 0)
            is_core = getattr(memory, "is_core", False)
            user_input = getattr(memory, "user_input", "")

        # 拼接所有可用的文本源
        full_text = " ".join(filter(None, [content, summary, entities, user_input]))

        if not full_text.strip():
            logger.debug(f"[碎片提取] 记忆 {mem_id} 无可用文本，跳过")
            return []

        fragments: List[MemoryFragment] = []
        seen_contents: Dict[FragmentType, Set[str]] = {ft: set() for ft in FragmentType}

        def _add_fragment(
            frag_type: FragmentType,
            frag_content: str,
            confidence: float = 0.8,
            source: str = "pattern",
        ) -> None:
            """内部辅助方法：去重添加碎片"""
            if not frag_content or not frag_content.strip():
                return
            frag_content = frag_content.strip()
            if frag_content in seen_contents[frag_type]:
                return
            seen_contents[frag_type].add(frag_content)

            fid = f"frag_{mem_id}_{frag_type.value}_{hashlib.md5(frag_content.encode()).hexdigest()[:8]}"
            frag = MemoryFragment(
                fragment_id=fid,
                fragment_type=frag_type,
                content=frag_content,
                source_memory_id=mem_id,
                confidence=min(confidence, 1.0),
                metadata={"extraction_source": source, "is_core": is_core},
            )
            fragments.append(frag)

        # ---- 1. 从结构化字段提取 ----

        # WHO: 从 key_entities 中提取人名相关实体
        if entities:
            entity_parts = [e.strip() for e in entities.split("|") if e.strip()]
            for entity in entity_parts:
                if entity.startswith("name:"):
                    name_val = entity.replace("name:", "").strip()
                    _add_fragment(FragmentType.WHO, name_val, confidence=0.95, source="entity_field")

        # EMOTION: 直接从 emotion_tag / emotion_type 获取
        if emotion_tag and emotion_tag != "中性":
            emotion_desc = self._emotion_to_chinese(emotion_type, emotion_tag, emotion_intensity)
            _add_fragment(FragmentType.EMOTION, emotion_desc, confidence=0.9, source="emotion_field")

        # ---- 2. 从文本中用正则模式提取 ----

        for frag_type, config in self.FRAGMENT_PATTERNS.items():
            for pattern in config["patterns"]:
                try:
                    matches = re.finditer(pattern, full_text)
                    for match in matches:
                        # 提取匹配内容（合并所有捕获组）
                        groups = match.groups()
                        if groups:
                            matched_text = "".join(g for g in groups if g is not None).strip()
                        else:
                            matched_text = match.group(0).strip()

                        if matched_text:
                            # 根据碎片类型设定基础置信度
                            base_confidence = {
                                FragmentType.WHO: 0.75,
                                FragmentType.WHAT: 0.65,
                                FragmentType.WHERE: 0.70,
                                FragmentType.WHEN: 0.80,
                                FragmentType.WHY: 0.55,
                                FragmentType.HOW: 0.50,
                                FragmentType.EMOTION: 0.70,
                            }
                            conf = base_confidence.get(frag_type, 0.6)
                            # 核心记忆置信度加成
                            if is_core:
                                conf = min(conf + 0.15, 1.0)
                            _add_fragment(frag_type, matched_text, confidence=conf, source="regex")
                except re.error:
                    continue

        # ---- 3. 从语义摘要中提取补充信息 ----

        if summary and summary != full_text:
            for frag_type, config in self.FRAGMENT_PATTERNS.items():
                for pattern in config["patterns"]:
                    try:
                        match = re.search(pattern, summary)
                        if match:
                            groups = match.groups()
                            matched_text = "".join(g for g in groups if g is not None).strip()
                            if matched_text:
                                _add_fragment(frag_type, matched_text, confidence=0.6, source="summary")
                    except re.error:
                        continue

        # ---- 4. 智能推断: 当碎片过少时，从整体内容中推断 ----

        type_counts = {ft: 0 for ft in FragmentType}
        for f in fragments:
            type_counts[f.fragment_type] += 1

        # 如果核心维度（WHO, WHAT）缺失，从内容中提取关键片段
        if type_counts[FragmentType.WHAT] == 0 and content:
            # 从原始内容中截取核心事件描述
            core_content = self._extract_core_event(content)
            if core_content:
                _add_fragment(
                    FragmentType.WHAT, core_content,
                    confidence=0.5, source="fallback_extraction"
                )

        # 缓存碎片到碎片存储
        if fragments:
            self._fragment_store[mem_id] = fragments

        logger.debug(
            f"[碎片提取] 记忆 {mem_id} → 提取 {len(fragments)} 个碎片 | "
            f"分布: {dict((ft.value, cnt) for ft, cnt in type_counts.items() if cnt > 0)}"
        )

        return fragments

    def _extract_core_event(self, text: str) -> str:
        """
        从文本中提取核心事件描述

        策略: 取第一个完整句子（去除客套语后）
        """
        if not text:
            return ""

        # 分句
        sentences = re.split(r'[。！？\n]', text)
        # 过滤太短或纯客套的句子
        skip_prefixes = {"你好", "请问", "谢谢", "嗯", "好的", "是的", "不是"}
        for sent in sentences:
            sent = sent.strip()
            if len(sent) >= 4 and not any(sent.startswith(p) for p in skip_prefixes):
                return sent[:60]

        # 如果没有合适的句子，返回截取的前面部分
        return text[:50].strip() if text else ""

    def _emotion_to_chinese(
        self,
        emotion_type: str,
        emotion_tag: str,
        intensity: float,
    ) -> str:
        """
        将情绪类型和标签转换为中文自然语言描述

        Args:
            emotion_type: 英文情绪类型 (joy/fear/sadness/...)
            emotion_tag: 中文情绪标签
            intensity: 情绪强度 (0.0-1.0)

        Returns:
            中文情绪描述
        """
        if not emotion_tag or emotion_tag == "中性":
            return ""

        # 强度修饰词
        if intensity >= 0.7:
            modifier = "非常"
        elif intensity >= 0.4:
            modifier = "有些"
        else:
            modifier = "略微"

        # 情绪类型到自然描述的映射
        emotion_desc_map = {
            "joy": "开心",
            "fear": "害怕",
            "sadness": "难过",
            "anger": "生气",
            "surprise": "惊讶",
            "disgust": "不舒服",
            "neutral": "",
        }

        base_emotion = emotion_desc_map.get(emotion_type, emotion_tag)

        if not base_emotion:
            return emotion_tag

        return f"{modifier}{base_emotion}"

    # ====================================================================
    # 2. 记忆重构
    # ====================================================================

    def reconstruct_memory(
        self,
        query: str,
        relevant_memories: Sequence[Any],
        current_emotion: str = "neutral",
        emotion_intensity: float = 0.0,
        template_hint: Optional[ReconstructionTemplate] = None,
        enable_distortion: Optional[bool] = None,
    ) -> ReconstructedMemory:
        """
        从相关记忆碎片中重构连贯的记忆叙述

        这是引擎的核心方法。它:
        1. 对每条记忆提取碎片
        2. 按维度聚合碎片
        3. 选择最佳碎片
        4. 使用中文模板生成叙述
        5. 计算置信度
        6. 可选地应用记忆扭曲

        Args:
            query: 触发重构的查询文本
            relevant_memories: 相关记忆列表（EpisodicMemory 或 dict）
            current_emotion: 当前情绪状态
            emotion_intensity: 当前情绪强度
            template_hint: 建议使用的模板（None=自动选择）
            enable_distortion: 是否应用记忆扭曲（None=使用引擎默认配置）

        Returns:
            ReconstructedMemory: 重构后的记忆对象
        """
        self._stats["total_reconstructions"] += 1

        # 检查缓存
        cache_key = self._compute_cache_key(query, relevant_memories, current_emotion)
        if self._cache_enabled and cache_key in self._cache:
            cached_result, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                self._stats["cache_hits"] += 1
                logger.debug(f"[记忆重构] 缓存命中: {cache_key}")
                return cached_result
            else:
                # 缓存过期，移除
                del self._cache[cache_key]

        self._stats["cache_misses"] += 1

        # Step 1: 提取所有碎片
        all_fragments: List[MemoryFragment] = []
        source_ids: Set[str] = set()

        for memory in relevant_memories:
            mem_id = (
                memory.memory_id if hasattr(memory, "memory_id")
                else memory.get("memory_id", "unknown")
            )
            source_ids.add(mem_id)

            # 尝试从缓存中获取碎片
            if mem_id in self._fragment_store:
                fragments = self._fragment_store[mem_id]
            else:
                fragments = self.extract_fragments(memory)

            all_fragments.extend(fragments)

        if not all_fragments:
            # 无碎片可重构 → 返回空结果
            result = ReconstructedMemory(
                query=query,
                narrative="",
                source_memory_ids=list(source_ids),
                overall_confidence=0.0,
                is_reconstructed=False,
            )
            return result

        # Step 2: 按碎片类型聚合，选择最佳碎片
        best_fragments = self._select_best_fragments(all_fragments, query)

        # Step 3: 选择重构模板
        template = template_hint or self._auto_select_template(best_fragments, query)

        # Step 4: 生成叙述
        narrative = self._generate_narrative(best_fragments, template)

        # Step 5: 计算置信度
        breakdown = self._compute_confidence_breakdown(best_fragments)
        overall_conf = breakdown.overall_confidence

        # Step 6: 标记推断元素
        inferred_elements = self._identify_inferred_elements(best_fragments)

        # Step 7: 时间排序
        temporal_order = self._order_fragments_temporally(best_fragments)

        # Step 8: 应用记忆扭曲（可选）
        should_distort = (
            enable_distortion if enable_distortion is not None
            else self._distortion_engine.enabled
        )
        distortion_applied = False
        distortion_metadata = {}

        if should_distort and narrative:
            # 计算记忆年龄（使用最古老记忆的时间戳）
            oldest_age = self._get_oldest_memory_age(relevant_memories)
            distortion_amount = self._distortion_engine.compute_distortion_amount(
                memory_age_seconds=oldest_age,
                current_emotion=current_emotion,
                emotion_intensity=emotion_intensity,
            )
            if distortion_amount > 0.01:
                narrative, distortion_metadata = self._distortion_engine.apply_distortion(
                    content=narrative,
                    distortion_amount=distortion_amount,
                    current_emotion=current_emotion,
                )
                distortion_applied = distortion_metadata.get("distorted", False)
                if distortion_applied:
                    self._stats["distortion_count"] += 1

        # 构建结果
        result = ReconstructedMemory(
            narrative=narrative,
            fragments_used=best_fragments,
            source_memory_ids=list(source_ids),
            overall_confidence=overall_conf,
            confidence_breakdown=breakdown,
            template_used=template,
            is_reconstructed=True,
            temporal_order=temporal_order,
            inferred_elements=inferred_elements,
            distortion_applied=distortion_applied,
            distortion_metadata=distortion_metadata,
            query=query,
        )

        # 更新缓存
        if self._cache_enabled:
            self._cache_result(cache_key, result)

        # 更新统计
        self._update_avg_confidence(overall_conf)

        logger.debug(
            f"[记忆重构] 完成 | 模板={template.value} | "
            f"置信度={overall_conf:.2f} | 碎片数={len(best_fragments)} | "
            f"推断={len(inferred_elements)} | 扭曲={distortion_applied}"
        )

        return result

    def reconstruct_sequence(
        self,
        memories: Sequence[Any],
        query: str = "",
    ) -> List[ReconstructedMemory]:
        """
        从多条记忆中重构事件序列

        按时间排序对多条记忆分别重构，并返回有序列表。
        适用于"你还记得我们上次聊了什么"这类需要按时间回忆的场景。

        Args:
            memories: 记忆列表（需可按时间排序）
            query: 查询文本

        Returns:
            按时间排序的重构记忆列表
        """
        if not memories:
            return []

        # 按时间戳排序
        def _get_timestamp(m):
            if hasattr(m, "timestamp"):
                return m.timestamp
            return m.get("timestamp", 0)

        sorted_memories = sorted(memories, key=_get_timestamp)

        results = []
        for i, memory in enumerate(sorted_memories):
            # 为序列中的每条记忆使用序列模板
            result = self.reconstruct_memory(
                query=query or f"回忆事件序列 #{i+1}",
                relevant_memories=[memory],
                template_hint=ReconstructionTemplate.SEQUENCE,
                enable_distortion=False,  # 序列重构不应用扭曲
            )
            results.append(result)

        return results

    # ====================================================================
    # 内部方法 — 碎片选择与聚合
    # ====================================================================

    def _select_best_fragments(
        self,
        fragments: List[MemoryFragment],
        query: str = "",
    ) -> List[MemoryFragment]:
        """
        从所有碎片中为每个维度选择最佳碎片

        选择策略:
        1. 按碎片类型分组
        2. 每组内按置信度降序排序
        3. 每组取前 N 个碎片（N=1-2，取决于碎片丰富度）
        4. 可选: 根据查询关键词对碎片进行相关性加权

        Args:
            fragments: 所有候选碎片
            query: 查询文本（用于相关性加权）

        Returns:
            best_fragments: 精选碎片列表
        """
        # 按类型分组
        grouped: Dict[FragmentType, List[MemoryFragment]] = {}
        for frag in fragments:
            grouped.setdefault(frag.fragment_type, []).append(frag)

        best = []
        for frag_type, type_fragments in grouped.items():
            # 按置信度降序排序
            type_fragments.sort(key=lambda f: f.confidence, reverse=True)

            # 如果有查询文本，用关键词匹配做二次排序
            if query:
                type_fragments = self._rank_by_query_relevance(type_fragments, query)

            # 每组取 top 1-2
            take_count = min(len(type_fragments), 2)
            best.extend(type_fragments[:take_count])

            # 标记碎片被使用
            for f in type_fragments[:take_count]:
                f.touch()

        return best

    def _rank_by_query_relevance(
        self,
        fragments: List[MemoryFragment],
        query: str,
    ) -> List[MemoryFragment]:
        """
        根据查询关键词对碎片进行相关性加权排序

        如果碎片内容中包含查询中的关键词，给予额外加分。
        """
        # 提取查询中的关键词（简单中文分词）
        query_chars = set(query)
        relevant_chars = set("的了吗吧呢啊是在有不也这那")  # 过滤停用词
        query_keywords = query_chars - relevant_chars

        if not query_keywords:
            return fragments

        scored = []
        for frag in fragments:
            frag_chars = set(frag.content)
            overlap = len(frag_chars & query_keywords)
            relevance_bonus = overlap * 0.05
            scored.append((frag, frag.confidence + relevance_bonus))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored]

    # ====================================================================
    # 内部方法 — 模板选择与叙述生成
    # ====================================================================

    def _auto_select_template(
        self,
        fragments: List[MemoryFragment],
        query: str = "",
    ) -> ReconstructionTemplate:
        """
        自动选择最合适的重构模板

        选择策略:
        - 如果有 WHEN + WHO + WHAT → 情景回忆模板（最常见）
        - 如果有 EMOTION + WHY → 情感回忆模板
        - 如果有 WHY + HOW → 因果回忆模板
        - 如果查询包含"怎么/如何" → 因果模板
        - 如果查询包含"感觉/心情" → 情感模板
        - 如果碎片仅覆盖少数维度 → 事实陈述模板
        - 默认 → 情景回忆模板

        Args:
            fragments: 可用碎片列表
            query: 查询文本

        Returns:
            template: 选中的模板类型
        """
        type_set = {f.fragment_type for f in fragments}

        # 基于查询意图的模板选择
        if query:
            emotion_keywords = {"感觉", "心情", "情绪", "难过", "开心", "害怕", "生气"}
            how_keywords = {"怎么", "如何", "为什么", "原因", "导致"}
            sequence_keywords = {"上次", "之前", "之前", "第一次", "最后", "过程", "步骤"}

            if any(kw in query for kw in emotion_keywords):
                return ReconstructionTemplate.EMOTIONAL
            if any(kw in query for kw in how_keywords):
                return ReconstructionTemplate.CAUSAL
            if any(kw in query for kw in sequence_keywords):
                return ReconstructionTemplate.SEQUENCE

        # 基于碎片覆盖度的选择
        has_time = FragmentType.WHEN in type_set
        has_who = FragmentType.WHO in type_set
        has_what = FragmentType.WHAT in type_set
        has_emotion = FragmentType.EMOTION in type_set
        has_why = FragmentType.WHY in type_set
        has_how = FragmentType.HOW in type_set

        if has_emotion and has_why:
            return ReconstructionTemplate.EMOTIONAL
        if has_why and has_how:
            return ReconstructionTemplate.CAUSAL
        if has_time and has_who and has_what:
            return ReconstructionTemplate.EPISODIC

        # 碎片维度不足 → 使用事实模板
        if len(type_set) <= 2:
            return ReconstructionTemplate.FACTUAL

        return ReconstructionTemplate.EPISODIC

    def _generate_narrative(
        self,
        fragments: List[MemoryFragment],
        template: ReconstructionTemplate,
    ) -> str:
        """
        使用选定模板和碎片生成中文叙述

        将碎片内容映射到模板槽位中。每个碎片类型对应特定的槽位。

        Args:
            fragments: 精选碎片列表
            template: 重构模板

        Returns:
            narrative: 生成的中文叙述
        """
        # 按类型组织碎片内容
        type_contents: Dict[FragmentType, str] = {}
        for frag in fragments:
            if frag.fragment_type not in type_contents:
                type_contents[frag.fragment_type] = frag.content
            # 第一个（最高置信度）碎片优先

        # 准备槽位值
        slot_values: Dict[str, str] = {}

        # 情景回忆模板的槽位映射
        if template == ReconstructionTemplate.EPISODIC:
            slot_values["time_ref"] = type_contents.get(FragmentType.WHEN, "")
            slot_values["person"] = type_contents.get(FragmentType.WHO, "")
            slot_values["location"] = type_contents.get(FragmentType.WHERE, "")
            slot_values["action_desc"] = self._format_action(
                type_contents.get(FragmentType.WHAT, ""),
                type_contents.get(FragmentType.HOW, ""),
            )
            slot_values["emotion_desc"] = self._format_emotion_clause(
                type_contents.get(FragmentType.EMOTION, ""),
            )

        elif template == ReconstructionTemplate.FACTUAL:
            topic = type_contents.get(FragmentType.WHAT, "") or type_contents.get(FragmentType.WHO, "")
            fact = type_contents.get(FragmentType.WHAT, "")
            slot_values["topic"] = topic
            slot_values["fact"] = fact

        elif template == ReconstructionTemplate.EMOTIONAL:
            slot_values["emotion"] = type_contents.get(FragmentType.EMOTION, "")
            slot_values["reason"] = type_contents.get(FragmentType.WHY, "")
            slot_values["detail"] = type_contents.get(FragmentType.WHAT, "")

        elif template == ReconstructionTemplate.CAUSAL:
            slot_values["cause"] = type_contents.get(FragmentType.WHY, "")
            slot_values["result"] = type_contents.get(FragmentType.WHAT, "")
            slot_values["how_desc"] = type_contents.get(FragmentType.HOW, "")

        elif template == ReconstructionTemplate.SEQUENCE:
            # 序列模板: 将所有 WHAT 碎片拼接
            what_fragments = [f for f in fragments if f.fragment_type == FragmentType.WHAT]
            event_list = "，然后".join(f.content for f in what_fragments) if what_fragments else ""
            slot_values["event_list"] = event_list

        # 渲染模板
        narrative = ReconstructionTemplates.render(template, slot_values)

        # 如果模板渲染失败或为空，尝试简单拼接
        if not narrative.strip():
            parts = []
            for frag in fragments:
                if frag.content:
                    parts.append(frag.content)
            narrative = "、".join(parts) + "。" if parts else ""

        return narrative

    def _format_action(self, what: str, how: str = "") -> str:
        """
        格式化动作描述

        将 WHAT 和 HOW 碎片合并为自然的动作描述。
        """
        if what and how:
            return f"{how}{what}"
        return what or ""

    def _format_emotion_clause(self, emotion: str) -> str:
        """
        格式化情绪子句

        将情绪碎片转换为自然的情绪描述子句。
        """
        if not emotion:
            return ""
        # 确保以"的"结尾，便于模板拼接
        if not emotion.endswith("的"):
            return f"感觉{emotion}的"
        return f"感觉{emotion}"

    # ====================================================================
    # 内部方法 — 置信度计算
    # ====================================================================

    def _compute_confidence_breakdown(
        self,
        fragments: List[MemoryFragment],
    ) -> ConfidenceBreakdown:
        """
        计算各维度的置信度细目

        对于每个碎片维度:
        - 取该维度所有碎片的最高置信度
        - 如果该维度无碎片，置信度为 0.0

        Args:
            fragments: 使用的碎片列表

        Returns:
            breakdown: 置信度细目
        """
        breakdown = ConfidenceBreakdown()

        # 维度映射
        type_to_attr = {
            FragmentType.WHO: "who_confidence",
            FragmentType.WHAT: "what_confidence",
            FragmentType.WHERE: "where_confidence",
            FragmentType.WHEN: "when_confidence",
            FragmentType.WHY: "why_confidence",
            FragmentType.HOW: "how_confidence",
            FragmentType.EMOTION: "emotion_confidence",
        }

        # 计算每个维度的最大置信度
        for frag in fragments:
            attr = type_to_attr.get(frag.fragment_type)
            if attr:
                current = getattr(breakdown, attr)
                if frag.confidence > current:
                    setattr(breakdown, attr, frag.confidence)

        return breakdown

    def _identify_inferred_elements(
        self,
        fragments: List[MemoryFragment],
    ) -> List[Dict[str, Any]]:
        """
        识别并标记推断/填补的元素

        置信度低于 0.5 的元素视为推断而非直接回忆。

        Args:
            fragments: 使用的碎片列表

        Returns:
            inferred: 推断元素列表
        """
        inferred = []
        for frag in fragments:
            if frag.confidence < 0.5:
                inferred.append({
                    "element_type": frag.fragment_type.value,
                    "content": frag.content,
                    "confidence": frag.confidence,
                    "confidence_level": frag.confidence_level.value,
                    "source": frag.metadata.get("extraction_source", "unknown"),
                })
        return inferred

    # ====================================================================
    # 内部方法 — 时间排序
    # ====================================================================

    def _order_fragments_temporally(
        self,
        fragments: List[MemoryFragment],
    ) -> List[str]:
        """
        对碎片按时间顺序排列

        排序规则:
        1. 有明确 WHEN 碎片的排在前面（提供时间锚点）
        2. WHO 碎片次之（确定主体）
        3. WHAT 碎片居中（核心事件）
        4. EMOTION 碎片靠后（通常附属于事件）
        5. WHY/HOW 碎片最后（解释性信息）

        Args:
            fragments: 碎片列表

        Returns:
            ordered_ids: 按时间顺序排列的碎片 ID 列表
        """
        # 维度优先级（模拟人类回忆的典型顺序）
        priority = {
            FragmentType.WHEN: 1,
            FragmentType.WHO: 2,
            FragmentType.WHERE: 3,
            FragmentType.WHAT: 4,
            FragmentType.HOW: 5,
            FragmentType.WHY: 6,
            FragmentType.EMOTION: 7,
        }

        sorted_fragments = sorted(
            fragments,
            key=lambda f: (
                priority.get(f.fragment_type, 99),
                -f.confidence,  # 同类型内高置信度优先
            ),
        )

        return [f.fragment_id for f in sorted_fragments]

    # ====================================================================
    # 内部方法 — 冲突处理
    # ====================================================================

    def resolve_conflicts(
        self,
        fragments: List[MemoryFragment],
    ) -> List[MemoryFragment]:
        """
        处理来自不同记忆的碎片冲突

        当同一维度存在多个碎片且内容不一致时:
        1. 优先选择置信度最高的
        2. 核心记忆碎片优先
        3. 最近访问的碎片优先
        4. 如果差异过大，保留多个版本并标记为"可能"

        Args:
            fragments: 候选碎片列表

        Returns:
            resolved: 解决冲突后的碎片列表
        """
        # 按类型分组
        grouped: Dict[FragmentType, List[MemoryFragment]] = {}
        for frag in fragments:
            grouped.setdefault(frag.fragment_type, []).append(frag)

        resolved = []
        for frag_type, type_frags in grouped.items():
            if len(type_frags) == 1:
                resolved.append(type_frags[0])
                continue

            # 多个碎片 → 需要冲突处理
            # 按优先级排序: 置信度 > is_core > 最近访问
            sorted_frags = sorted(
                type_frags,
                key=lambda f: (
                    f.confidence,
                    1.0 if f.metadata.get("is_core", False) else 0.0,
                    f.last_access_time,
                ),
                reverse=True,
            )

            # 检查内容相似度
            best = sorted_frags[0]
            resolved.append(best)

            # 如果第二好的碎片内容差异很大，添加为"可能"版本
            if len(sorted_frags) > 1:
                second = sorted_frags[1]
                similarity = self._compute_text_similarity(best.content, second.content)
                if similarity < 0.3:  # 内容差异大
                    # 将第二碎片标记为低置信度的备选
                    alt_frag = MemoryFragment(
                        fragment_id=f"{second.fragment_id}_alt",
                        fragment_type=second.fragment_type,
                        content=f"（可能是: {second.content}）",
                        source_memory_id=second.source_memory_id,
                        confidence=second.confidence * 0.5,
                        metadata={"is_alternative": True, "primary_fragment_id": best.fragment_id},
                    )
                    resolved.append(alt_frag)

        return resolved

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两段文本的简单相似度（字符级 Jaccard）

        不依赖外部 embedding，使用轻量级的字符集合相似度。
        """
        if not text1 or not text2:
            return 0.0

        # 中文字符级 bigram
        def _bigrams(text):
            return set(text[i:i+2] for i in range(len(text) - 1))

        bigrams1 = _bigrams(text1)
        bigrams2 = _bigrams(text2)

        if not bigrams1 or not bigrams2:
            return 0.0

        intersection = bigrams1 & bigrams2
        union = bigrams1 | bigrams2

        return len(intersection) / len(union) if union else 0.0

    # ====================================================================
    # 内部方法 — 缓存与辅助
    # ====================================================================

    def _compute_cache_key(
        self,
        query: str,
        memories: Sequence[Any],
        emotion: str,
    ) -> str:
        """
        计算缓存键

        基于查询文本 + 记忆 ID 列表 + 情绪状态的确定性哈希
        """
        memory_ids = []
        for m in memories:
            mid = m.memory_id if hasattr(m, "memory_id") else m.get("memory_id", "?")
            memory_ids.append(str(mid))

        raw = f"{query}|{'|'.join(memory_ids)}|{emotion}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    def _cache_result(self, cache_key: str, result: ReconstructedMemory) -> None:
        """缓存重构结果"""
        if not self._cache_enabled:
            return

        # LRU 淘汰
        if len(self._cache) >= self._cache_max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        self._cache[cache_key] = (result, time.time())

    def _get_oldest_memory_age(self, memories: Sequence[Any]) -> float:
        """获取最古老记忆的年龄（秒）"""
        if not memories:
            return 0.0

        now_ms = time.time() * 1000
        min_timestamp = float("inf")

        for m in memories:
            ts = (
                m.timestamp if hasattr(m, "timestamp")
                else m.get("timestamp", now_ms)
            )
            if ts < min_timestamp:
                min_timestamp = ts

        if min_timestamp == float("inf"):
            return 0.0

        return (now_ms - min_timestamp) / 1000.0

    def _update_avg_confidence(self, new_confidence: float) -> None:
        """更新平均置信度统计（在线平均）"""
        total = self._stats["total_reconstructions"]
        if total <= 0:
            return
        old_avg = self._stats["avg_confidence"]
        self._stats["avg_confidence"] = old_avg + (new_confidence - old_avg) / total

    # ====================================================================
    # 批量重构
    # ====================================================================

    def batch_reconstruct(
        self,
        queries_and_memories: List[Tuple[str, Sequence[Any]]],
        current_emotion: str = "neutral",
        emotion_intensity: float = 0.0,
    ) -> List[ReconstructedMemory]:
        """
        批量重构记忆

        对多个查询-记忆对进行批量重构，共享碎片缓存。

        Args:
            queries_and_memories: [(query, memories), ...] 列表
            current_emotion: 当前情绪状态
            emotion_intensity: 情绪强度

        Returns:
            results: 重构结果列表
        """
        results = []
        for query, memories in queries_and_memories:
            result = self.reconstruct_memory(
                query=query,
                relevant_memories=memories,
                current_emotion=current_emotion,
                emotion_intensity=emotion_intensity,
            )
            results.append(result)
        return results

    # ====================================================================
    # 序列化 / 状态管理
    # ====================================================================

    def get_state(self) -> dict:
        """
        获取引擎完整状态（用于持久化）

        Returns:
            state: 包含碎片存储、缓存、统计等所有内部状态
        """
        fragment_store_dict = {}
        for mem_id, frag_list in self._fragment_store.items():
            fragment_store_dict[mem_id] = [f.to_dict() for f in frag_list]

        cache_dict = {}
        for key, (result, ts) in self._cache.items():
            cache_dict[key] = {
                "result": result.to_dict(),
                "timestamp": ts,
            }

        return {
            "fragment_store": fragment_store_dict,
            "cache": cache_dict,
            "stats": dict(self._stats),
            "distortion_config": {
                "enabled": self._distortion_engine.enabled,
                "max_distortion": self._distortion_engine.max_distortion,
            },
            "cache_config": {
                "enabled": self._cache_enabled,
                "max_size": self._cache_max_size,
                "ttl_seconds": self._cache_ttl,
            },
        }

    def set_state(self, state: dict) -> None:
        """
        从状态字典恢复引擎

        Args:
            state: 由 get_state() 返回的状态字典
        """
        # 恢复碎片存储
        self._fragment_store.clear()
        for mem_id, frag_dicts in state.get("fragment_store", {}).items():
            self._fragment_store[mem_id] = [
                MemoryFragment.from_dict(fd) for fd in frag_dicts
            ]

        # 恢复缓存
        self._cache.clear()
        for key, cache_entry in state.get("cache", {}).items():
            result = ReconstructedMemory.from_dict(cache_entry["result"])
            ts = cache_entry.get("timestamp", time.time())
            self._cache[key] = (result, ts)

        # 恢复统计
        self._stats.update(state.get("stats", {}))

        # 恢复扭曲配置
        dist_config = state.get("distortion_config", {})
        self._distortion_engine.enabled = dist_config.get("enabled", True)
        self._distortion_engine.max_distortion = dist_config.get("max_distortion", 0.3)

        # 恢复缓存配置
        cache_config = state.get("cache_config", {})
        self._cache_enabled = cache_config.get("enabled", True)
        self._cache_max_size = cache_config.get("max_size", 200)
        self._cache_ttl = cache_config.get("ttl_seconds", 300.0)

        logger.info(
            f"[记忆重构引擎] 状态恢复完成 | "
            f"碎片存储: {len(self._fragment_store)} 条记忆 | "
            f"缓存: {len(self._cache)} 条"
        )

    def clear_cache(self) -> int:
        """
        清空重构结果缓存

        Returns:
            cleared_count: 清除的缓存条目数
        """
        count = len(self._cache)
        self._cache.clear()
        logger.debug(f"[记忆重构引擎] 缓存已清空，共 {count} 条")
        return count

    def clear_fragment_store(self) -> int:
        """
        清空碎片存储

        Returns:
            cleared_count: 清除的碎片存储条目数
        """
        count = len(self._fragment_store)
        self._fragment_store.clear()
        logger.debug(f"[记忆重构引擎] 碎片存储已清空，共 {count} 条记忆")
        return count

    def reset(self) -> None:
        """完全重置引擎到初始状态"""
        self._fragment_store.clear()
        self._cache.clear()
        self._stats = {
            "total_extractions": 0,
            "total_reconstructions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_confidence": 0.0,
            "distortion_count": 0,
        }
        logger.info("[记忆重构引擎] 已完全重置")

    # ====================================================================
    # 统计与调试
    # ====================================================================

    def get_stats(self) -> dict:
        """
        获取引擎运行统计

        Returns:
            stats: 统计信息字典
        """
        return {
            "total_extractions": self._stats["total_extractions"],
            "total_reconstructions": self._stats["total_reconstructions"],
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "cache_hit_rate": (
                self._stats["cache_hits"]
                / max(self._stats["cache_hits"] + self._stats["cache_misses"], 1)
            ),
            "avg_confidence": round(self._stats["avg_confidence"], 4),
            "distortion_count": self._stats["distortion_count"],
            "fragment_store_size": len(self._fragment_store),
            "cache_size": len(self._cache),
            "total_fragments": sum(len(f) for f in self._fragment_store.values()),
            "distortion_enabled": self._distortion_engine.enabled,
            "cache_enabled": self._cache_enabled,
        }

    def get_fragment_summary(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定记忆的碎片摘要

        Args:
            memory_id: 记忆 ID

        Returns:
            summary: 碎片摘要字典，记忆不存在返回 None
        """
        fragments = self._fragment_store.get(memory_id)
        if not fragments:
            return None

        type_counts: Dict[str, int] = {}
        avg_confidence: Dict[str, float] = {}
        for frag in fragments:
            ft = frag.fragment_type.value
            type_counts[ft] = type_counts.get(ft, 0) + 1
            if ft not in avg_confidence:
                avg_confidence[ft] = 0.0
            avg_confidence[ft] += frag.confidence

        for ft in avg_confidence:
            count = type_counts[ft]
            avg_confidence[ft] = round(avg_confidence[ft] / count, 3) if count > 0 else 0.0

        return {
            "memory_id": memory_id,
            "total_fragments": len(fragments),
            "type_distribution": type_counts,
            "avg_confidence_by_type": avg_confidence,
            "fragments": [f.to_dict() for f in fragments],
        }

    def diagnose_reconstruction(
        self,
        query: str,
        relevant_memories: Sequence[Any],
    ) -> Dict[str, Any]:
        """
        诊断模式 — 返回重构过程的详细中间结果

        用于调试和理解重构引擎的决策过程。

        Args:
            query: 查询文本
            relevant_memories: 相关记忆列表

        Returns:
            diagnosis: 包含各步骤详细信息的诊断报告
        """
        # Step 1: 碎片提取
        all_fragments = []
        for memory in relevant_memories:
            frags = self.extract_fragments(memory)
            all_fragments.extend(frags)

        # Step 2: 碎片选择
        best_fragments = self._select_best_fragments(all_fragments, query)

        # Step 3: 冲突检测
        type_counts: Dict[str, int] = {}
        for f in all_fragments:
            ft = f.fragment_type.value
            type_counts[ft] = type_counts.get(ft, 0) + 1

        conflicts = []
        for ft, count in type_counts.items():
            if count > 2:
                conflicts.append({
                    "fragment_type": ft,
                    "count": count,
                    "contents": [
                        f.content for f in all_fragments
                        if f.fragment_type.value == ft
                    ],
                })

        # Step 4: 模板选择
        selected_template = self._auto_select_template(best_fragments, query)

        # Step 5: 置信度分析
        breakdown = self._compute_confidence_breakdown(best_fragments)

        return {
            "query": query,
            "input_memories_count": len(relevant_memories),
            "total_fragments_extracted": len(all_fragments),
            "fragments_by_type": type_counts,
            "selected_fragments": [f.to_dict() for f in best_fragments],
            "conflicts_detected": conflicts,
            "selected_template": selected_template.value,
            "confidence_breakdown": breakdown.to_dict(),
            "gap_analysis": {
                "missing_dimensions": [
                    ft for ft, cnt in type_counts.items()
                    if cnt == 0
                ],
                "weak_dimensions": [
                    ft for ft, cnt in type_counts.items()
                    if cnt > 0 and cnt <= 1
                ],
            },
            "cache_status": {
                "enabled": self._cache_enabled,
                "size": len(self._cache),
                "max_size": self._cache_max_size,
            },
        }
