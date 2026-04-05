"""
联想记忆网络 - Associative Memory Network

模拟人脑联想记忆机制的核心组件。
人类记忆并非孤立存储，而是通过丰富的关联网络形成"联想"能力：
- 听到"巴黎"会联想到"埃菲尔铁塔"、"浪漫"、"法国"
- 看到"下雨"会联想到"上次淋雨"、"带伞"

本模块实现：
1. 自动关联检测（语义/情绪/时间/实体/因果）
2. 关联图存储与查询（邻接表 + BFS 遍历）
3. 联想扩散激活（Spreading Activation，模拟人脑联想思维）
4. 赫布学习法则（"共同激活的神经元会连接更紧密"）
5. 记忆干扰检测（新记忆覆盖旧记忆的预警）
6. LRU 风格的弱关联修剪（防止内存爆炸）

神经科学基础：
- CA3 区域是海马体的联想网络核心，支持模式补全
- 赫布定律（Hebbian Learning）：同时激活的突触会被强化
- 扩散激活理论（Spreading Activation）：激活会在关联网络中扩散衰减
- 记忆干扰（Interference）：相似记忆之间的竞争和覆盖
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import enum
import math
import heapq
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import deque, OrderedDict
import logging

# 尝试从同包导入已有模块（运行时导入，避免循环依赖）
try:
    from .memory_layers import MemoryTier
except ImportError:
    MemoryTier = None

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举 & 数据结构
# =============================================================================

class AssociationType(enum.Enum):
    """关联类型枚举 - 定义记忆之间关联的不同维度"""
    SEMANTIC = "semantic"       # 语义关联：基于语义 embedding 余弦相似度
    EMOTIONAL = "emotional"     # 情绪关联：相同或相似的情绪类型 + 高情绪强度
    TEMPORAL = "temporal"       # 时间关联：在短时间内发生的记忆
    ENTITY = "entity"           # 实体关联：共享关键实体（人名、地名等）
    CAUSAL = "causal"           # 因果关联：一个记忆的内容导致/解释另一个
    CONTRAST = "contrast"       # 对比关联：内容相反或矛盾


@dataclass
class Association:
    """
    关联边数据结构 - 表示两条记忆之间的关联关系

    Attributes:
        target_memory_id: 目标记忆 ID（关联指向的记忆）
        association_type: 关联类型（语义/情绪/时间/实体/因果/对比）
        strength: 关联强度 [0.0, 1.0]，越强表示关联越紧密
        created_time: 关联创建时间（Unix 时间戳，秒）
        last_accessed_time: 最后一次被访问/使用的时间
        co_activation_count: 共激活计数（两条记忆同时被召回的次数）
        metadata: 附加元数据（如具体的相似度分数、共享实体列表等）
    """
    target_memory_id: str
    association_type: AssociationType
    strength: float = 0.5
    created_time: float = 0.0
    last_accessed_time: float = 0.0
    co_activation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为字典（避免 enum 和复杂类型）"""
        return {
            'target_memory_id': self.target_memory_id,
            'association_type': self.association_type.value,
            'strength': round(self.strength, 6),
            'created_time': self.created_time,
            'last_accessed_time': self.last_accessed_time,
            'co_activation_count': self.co_activation_count,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Association':
        """从字典反序列化"""
        return cls(
            target_memory_id=data['target_memory_id'],
            association_type=AssociationType(data['association_type']),
            strength=data.get('strength', 0.5),
            created_time=data.get('created_time', 0.0),
            last_accessed_time=data.get('last_accessed_time', 0.0),
            co_activation_count=data.get('co_activation_count', 0),
            metadata=data.get('metadata', {}),
        )


@dataclass
class InterferenceWarning:
    """
    记忆干扰警告 - 当新记忆可能覆盖/混淆旧记忆时产生

    Attributes:
        new_memory_id: 新记忆 ID
        existing_memory_id: 被干扰的旧记忆 ID
        interference_score: 干扰分数 [0.0, 1.0]，越高表示干扰越严重
        interference_types: 导致干扰的关联类型列表
        suggestion: 建议的处理措施
    """
    new_memory_id: str
    existing_memory_id: str
    interference_score: float
    interference_types: List[str] = field(default_factory=list)
    suggestion: str = ""

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'new_memory_id': self.new_memory_id,
            'existing_memory_id': self.existing_memory_id,
            'interference_score': round(self.interference_score, 6),
            'interference_types': self.interference_types,
            'suggestion': self.suggestion,
        }


# =============================================================================
# 联想记忆网络核心类
# =============================================================================

class AssociativeMemoryNetwork:
    """
    联想记忆网络（Associative Memory Network）

    核心功能：
    - 当新记忆存储时，自动检测并创建与已有记忆的关联
    - 支持基于关联图的联想召回（BFS 遍历、扩散激活）
    - 赫布学习：共同召回的记忆之间的关联会自动增强
    - 时间衰减：长期不使用的关联会逐渐弱化
    - 干扰检测：警告新记忆可能覆盖相似旧记忆

    内存管理：
    - 每条记忆最多保持 MAX_ASSOCIATIONS_PER_MEMORY 条关联（默认500）
    - 弱关联会被 LRU 策略修剪
    - 关联图使用邻接表存储（Dict[str, List[Association]]）

    使用示例：
        >>> network = AssociativeMemoryNetwork()
        >>> # 检测并创建关联
        >>> new_associations = network.detect_and_create_associations(new_memory, all_memories)
        >>> # 联想召回
        >>> related = network.get_associated_memories("mem_001", max_depth=2)
        >>> # 扩散激活（模拟人类自由联想）
        >>> activated = network.spread_activation(["mem_001", "mem_005"], iterations=3)
        >>> # 记录共同召回（赫布学习）
        >>> network.record_co_recall(["mem_001", "mem_005", "mem_012"])
    """

    # ======================== 配置常量 ========================
    MAX_ASSOCIATIONS_PER_MEMORY: int = 500    # 每条记忆的最大关联数
    DEFAULT_SEMANTIC_THRESHOLD: float = 0.6   # 语义关联的余弦相似度阈值
    DEFAULT_EMOTION_INTENSITY_THRESHOLD: float = 0.5  # 情绪关联的最小情绪强度
    DEFAULT_TEMPORAL_WINDOW_S: float = 300.0  # 时间关联的时间窗口（5分钟 = 300秒）
    DEFAULT_LEARNING_RATE: float = 0.1        # 赫布学习速率
    DEFAULT_DECAY_RATE: float = 0.995         # 关联强度衰减率（每次调用 decay 时）
    MIN_ASSOCIATION_STRENGTH: float = 0.01    # 关联强度的最小值（低于此值将被修剪）
    DEFAULT_SPREAD_DECAY: float = 0.5         # 扩散激活的衰减系数

    def __init__(
        self,
        max_associations: int = 500,
        semantic_threshold: float = 0.6,
        emotion_intensity_threshold: float = 0.5,
        temporal_window_s: float = 300.0,
        learning_rate: float = 0.1,
        decay_rate: float = 0.995,
        min_strength: float = 0.01,
        spread_decay: float = 0.5,
    ):
        """
        初始化联想记忆网络

        Args:
            max_associations: 每条记忆的最大关联数量（LRU 修剪阈值）
            semantic_threshold: 语义关联的余弦相似度阈值 [0, 1]
            emotion_intensity_threshold: 情绪关联的最小情绪强度 [0, 1]
            temporal_window_s: 时间关联的时间窗口（秒）
            learning_rate: 赫布学习速率 (0, 1]
            decay_rate: 关联强度衰减率 (0, 1]
            min_strength: 关联强度低于此值将被自动修剪
            spread_decay: 扩散激活的衰减系数 (0, 1]
        """
        # ======================== 配置 ========================
        self.max_associations = max_associations
        self.semantic_threshold = semantic_threshold
        self.emotion_intensity_threshold = emotion_intensity_threshold
        self.temporal_window_s = temporal_window_s
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.min_strength = min_strength
        self.spread_decay = spread_decay

        # ======================== 关联图（邻接表）========================
        # adjacency_graph[source_memory_id] = List[Association]
        self.adjacency_graph: Dict[str, List[Association]] = {}

        # ======================== 反向索引（加速查询）========================
        # 用于快速查找哪些记忆关联到某个目标记忆
        self._reverse_index: Dict[str, Set[str]] = {}

        # ======================== 统计信息 ========================
        self._total_associations_created: int = 0
        self._total_hebbian_updates: int = 0
        self._total_pruned: int = 0
        self._last_decay_time: float = 0.0

        logger.info(
            f"[联想记忆网络] 初始化完成: "
            f"最大关联数={max_associations}, "
            f"语义阈值={semantic_threshold}, "
            f"时间窗口={temporal_window_s}s, "
            f"学习率={learning_rate}"
        )

    # =========================================================================
    # 1. 自动关联检测
    # =========================================================================

    def detect_and_create_associations(
        self,
        new_memory: Any,
        all_memories: Dict[str, Any],
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Association]:
        """
        自动检测并创建新记忆与现有记忆之间的关联

        对新记忆依次执行以下检测：
        1. 语义关联：基于 semantic_embedding 的余弦相似度
        2. 情绪关联：相同/相似情绪类型 + 高强度
        3. 时间关联：时间戳在时间窗口内
        4. 实体关联：共享关键实体
        5. 因果关联：内容存在因果关系
        6. 对比关联：内容相反

        Args:
            new_memory: 新记忆对象（EpisodicMemory 或类似结构）
            all_memories: 所有现有记忆字典 {memory_id: memory_object}
            exclude_ids: 需要排除的记忆 ID 集合（如自身）

        Returns:
            新创建的关联列表
        """
        if exclude_ids is None:
            exclude_ids = set()

        # 排除自身
        new_id = getattr(new_memory, 'memory_id', '')
        if new_id:
            exclude_ids.add(new_id)

        new_associations: List[Association] = []
        now = time.time()

        # 新记忆的基础信息（缓存提取，避免重复 getattr）
        new_embedding = getattr(new_memory, 'semantic_embedding', None)
        new_emotion_type = getattr(new_memory, 'emotion_type', 'neutral')
        new_emotion_intensity = getattr(new_memory, 'emotion_intensity', 0.0)
        new_timestamp_ms = getattr(new_memory, 'timestamp', 0)
        new_key_entities = getattr(new_memory, 'key_entities', '')
        new_emotion_tag = getattr(new_memory, 'emotion_tag', '中性')
        new_content = getattr(new_memory, 'content', '') or getattr(new_memory, 'semantic_summary', '')
        new_user_input = getattr(new_memory, 'user_input', '')

        for mem_id, existing_mem in all_memories.items():
            if mem_id in exclude_ids:
                continue

            # 提取现有记忆的信息
            existing_embedding = getattr(existing_mem, 'semantic_embedding', None)
            existing_emotion_type = getattr(existing_mem, 'emotion_type', 'neutral')
            existing_emotion_intensity = getattr(existing_mem, 'emotion_intensity', 0.0)
            existing_timestamp_ms = getattr(existing_mem, 'timestamp', 0)
            existing_key_entities = getattr(existing_mem, 'key_entities', '')
            existing_emotion_tag = getattr(existing_mem, 'emotion_tag', '中性')
            existing_content = getattr(existing_mem, 'content', '') or getattr(existing_mem, 'semantic_summary', '')

            # 累计关联分数和类型
            detected_types: List[Tuple[AssociationType, float, Dict[str, Any]]] = []

            # ---- 1. 语义关联检测 ----
            semantic_score = self._detect_semantic_association(
                new_embedding, existing_embedding
            )
            if semantic_score > self.semantic_threshold:
                detected_types.append((
                    AssociationType.SEMANTIC,
                    semantic_score,
                    {'cosine_similarity': round(semantic_score, 4)},
                ))

            # ---- 2. 情绪关联检测 ----
            emotion_score = self._detect_emotional_association(
                new_emotion_type, new_emotion_intensity,
                new_emotion_tag,
                existing_emotion_type, existing_emotion_intensity,
                existing_emotion_tag,
            )
            if emotion_score > 0:
                detected_types.append((
                    AssociationType.EMOTIONAL,
                    emotion_score,
                    {
                        'new_emotion': new_emotion_type,
                        'existing_emotion': existing_emotion_type,
                        'new_tag': new_emotion_tag,
                        'existing_tag': existing_emotion_tag,
                    },
                ))

            # ---- 3. 时间关联检测 ----
            temporal_score = self._detect_temporal_association(
                new_timestamp_ms, existing_timestamp_ms
            )
            if temporal_score > 0:
                detected_types.append((
                    AssociationType.TEMPORAL,
                    temporal_score,
                    {'time_diff_s': round(abs(new_timestamp_ms - existing_timestamp_ms) / 1000.0, 2)},
                ))

            # ---- 4. 实体关联检测 ----
            entity_score, shared_entities = self._detect_entity_association(
                new_key_entities, existing_key_entities
            )
            if entity_score > 0:
                detected_types.append((
                    AssociationType.ENTITY,
                    entity_score,
                    {'shared_entities': shared_entities},
                ))

            # ---- 5. 因果关联检测 ----
            causal_score = self._detect_causal_association(
                new_content, new_user_input,
                existing_content,
                getattr(existing_mem, 'user_input', ''),
            )
            if causal_score > 0:
                detected_types.append((
                    AssociationType.CAUSAL,
                    causal_score,
                    {},
                ))

            # ---- 6. 对比关联检测 ----
            contrast_score = self._detect_contrast_association(
                new_emotion_tag, new_content,
                existing_emotion_tag, existing_content,
            )
            if contrast_score > 0:
                detected_types.append((
                    AssociationType.CONTRAST,
                    contrast_score,
                    {},
                ))

            # ---- 为每种检测到的关联类型创建关联边 ----
            for assoc_type, score, metadata in detected_types:
                # 综合强度 = 检测得分（已归一化到 0-1）
                strength = min(max(score, 0.05), 1.0)

                association = Association(
                    target_memory_id=mem_id,
                    association_type=assoc_type,
                    strength=strength,
                    created_time=now,
                    last_accessed_time=now,
                    co_activation_count=0,
                    metadata=metadata,
                )

                # 双向创建关联（A→B 和 B→A）
                self._add_association(new_id, association)
                # 反向关联（弱化一点，避免过度双向放大）
                reverse_strength = strength * 0.9
                reverse_assoc = Association(
                    target_memory_id=new_id,
                    association_type=assoc_type,
                    strength=reverse_strength,
                    created_time=now,
                    last_accessed_time=now,
                    co_activation_count=0,
                    metadata=metadata,
                )
                self._add_association(mem_id, reverse_assoc)

                new_associations.append(association)
                self._total_associations_created += 2  # 双向计数

        # 对新记忆的关联列表执行 LRU 修剪
        self._prune_weak_associations(new_id)

        logger.debug(
            f"[联想记忆网络] 为记忆 {new_id} 创建了 "
            f"{len(new_associations)} 种关联（共 {len(new_associations) * 2} 条边）"
        )

        return new_associations

    def _detect_semantic_association(
        self,
        emb_a: Optional[torch.Tensor],
        emb_b: Optional[torch.Tensor],
    ) -> float:
        """
        语义关联检测 - 基于 embedding 余弦相似度

        当两个记忆的语义 embedding 余弦相似度超过阈值时，
        认为它们在语义上相关。

        Args:
            emb_a: 记忆 A 的语义 embedding
            emb_b: 记忆 B 的语义 embedding

        Returns:
            余弦相似度 [0.0, 1.0]，如果无法计算则返回 0.0
        """
        if emb_a is None or emb_b is None:
            return 0.0

        try:
            # 确保是 1D 向量
            a = emb_a.flatten().float()
            b = emb_b.flatten().float()

            # 维度不一致时截断到较小维度
            if a.shape[0] != b.shape[0]:
                min_dim = min(a.shape[0], b.shape[0])
                a = a[:min_dim]
                b = b[:min_dim]

            # L2 归一化
            a_norm = F.normalize(a.unsqueeze(0), p=2, dim=-1)
            b_norm = F.normalize(b.unsqueeze(0), p=2, dim=-1)

            sim = F.cosine_similarity(a_norm, b_norm).item()
            return max(0.0, min(1.0, sim))
        except Exception as e:
            logger.debug(f"[联想记忆网络] 语义关联计算失败: {e}")
            return 0.0

    def _detect_emotional_association(
        self,
        emotion_type_a: str,
        emotion_intensity_a: float,
        emotion_tag_a: str,
        emotion_type_b: str,
        emotion_intensity_b: float,
        emotion_tag_b: str,
    ) -> float:
        """
        情绪关联检测 - 基于情绪类型相似性 + 情绪强度

        规则：
        - 两个记忆都满足最小情绪强度阈值
        - 情绪类型相同或属于同一极性（正面/负面/中性）

        Args:
            emotion_type_a/b: 情绪类型字符串（如 "joy", "sad", "neutral"）
            emotion_intensity_a/b: 情绪强度 [0.0, 1.0]
            emotion_tag_a/b: 情绪标签（如 "正面", "负面", "中性"）

        Returns:
            情绪关联分数 [0.0, 1.0]
        """
        # 两个记忆都需满足最小情绪强度
        if (emotion_intensity_a < self.emotion_intensity_threshold or
                emotion_intensity_b < self.emotion_intensity_threshold):
            return 0.0

        # 情绪极性映射
        POSITIVE_EMOTIONS = {
            'joy', 'happy', 'excited', 'love', 'gratitude',
            'pride', 'hope', 'surprise_positive', '正面',
            '开心', '高兴', '喜欢', '爱', '幸福', '满足',
        }
        NEGATIVE_EMOTIONS = {
            'sad', 'angry', 'fear', 'disgust', 'shame', 'guilt',
            'anxiety', 'jealousy', 'surprise_negative', '负面',
            '难过', '悲伤', '焦虑', '压力', '恐惧', '愤怒',
        }
        NEUTRAL_EMOTIONS = {
            'neutral', 'calm', 'bored', 'indifferent', '中性',
        }

        def get_polarity(etype: str) -> str:
            """获取情绪极性"""
            etype_lower = etype.lower().strip()
            if etype_lower in POSITIVE_EMOTIONS:
                return 'positive'
            elif etype_lower in NEGATIVE_EMOTIONS:
                return 'negative'
            return 'neutral'

        # 同时检查 emotion_type 和 emotion_tag 的极性
        polarity_a = get_polarity(emotion_type_a)
        if polarity_a == 'neutral' and emotion_tag_a:
            polarity_a = get_polarity(emotion_tag_a)

        polarity_b = get_polarity(emotion_type_b)
        if polarity_b == 'neutral' and emotion_tag_b:
            polarity_b = get_polarity(emotion_tag_b)

        # 情绪极性相同 → 关联
        if polarity_a == polarity_b and polarity_a != 'neutral':
            # 关联强度 = 两个情绪强度的几何平均
            score = math.sqrt(emotion_intensity_a * emotion_intensity_b)
            # 如果类型完全相同，额外加分
            if emotion_type_a.lower() == emotion_type_b.lower():
                score = min(score * 1.2, 1.0)
            return min(max(score, 0.0), 1.0)

        # 情绪极性不同（如正面 vs 负面）→ 不算情绪关联（由对比关联处理）
        return 0.0

    def _detect_temporal_association(
        self,
        timestamp_a_ms: int,
        timestamp_b_ms: int,
    ) -> float:
        """
        时间关联检测 - 基于时间戳接近程度

        在时间窗口内的记忆被认为在时间上相关（通常是一段连续对话的多个记忆）。
        距离越近，关联越强。

        Args:
            timestamp_a_ms: 记忆 A 的时间戳（毫秒）
            timestamp_b_ms: 记忆 B 的时间戳（毫秒）

        Returns:
            时间关联分数 [0.0, 1.0]
        """
        if timestamp_a_ms == 0 or timestamp_b_ms == 0:
            return 0.0

        time_diff_s = abs(timestamp_a_ms - timestamp_b_ms) / 1000.0

        if time_diff_s > self.temporal_window_s:
            return 0.0

        # 线性衰减：越近越强
        score = 1.0 - (time_diff_s / self.temporal_window_s)
        # 加上非线性衰减让非常接近的记忆关联更强
        score = score ** 0.7
        return min(max(score, 0.0), 1.0)

    def _detect_entity_association(
        self,
        key_entities_a: str,
        key_entities_b: str,
    ) -> Tuple[float, List[str]]:
        """
        实体关联检测 - 基于共享关键实体

        提取两条记忆的关键实体字符串，计算共享实体数量。
        实体通过管道符 "|" 或逗号分隔。

        Args:
            key_entities_a: 记忆 A 的关键实体字符串（如 "name:小明 | city:北京"）
            key_entities_b: 记忆 B 的关键实体字符串

        Returns:
            (关联分数, 共享实体列表)
        """
        if not key_entities_a or not key_entities_b:
            return 0.0, []

        # 分割实体（支持管道符、逗号、中文逗号）
        import re
        entities_a = set()
        entities_b = set()

        for segment in re.split(r'[|,，]', key_entities_a):
            segment = segment.strip()
            if segment:
                entities_a.add(segment.lower())

        for segment in re.split(r'[|,，]', key_entities_b):
            segment = segment.strip()
            if segment:
                entities_b.add(segment.lower())

        # 找共享实体
        shared = entities_a & entities_b
        if not shared:
            return 0.0, []

        # 分数 = 共享实体数 / min(A总数, B总数)，避免被大量实体的记忆稀释
        min_count = min(len(entities_a), len(entities_b))
        if min_count == 0:
            return 0.0, []

        score = len(shared) / min_count
        return min(score, 1.0), list(shared)

    def _detect_causal_association(
        self,
        content_a: str,
        user_input_a: str,
        content_b: str,
        user_input_b: str,
    ) -> float:
        """
        因果关联检测 - 基于内容因果关系

        通过关键词模式检测记忆之间是否存在因果关系：
        - A 的内容导致了 B（A 中有"因为"、"所以"、"导致"等词）
        - B 的内容是对 A 的原因（B 中提到 A 的问题的解决方案）

        这是一个轻量级的基于规则的因果检测，不依赖外部NLP模型。

        Args:
            content_a: 记忆 A 的内容/语义摘要
            user_input_a: 记忆 A 的用户输入
            content_b: 记忆 B 的内容/语义摘要
            user_input_b: 记忆 B 的用户输入

        Returns:
            因果关联分数 [0.0, 1.0]
        """
        import re

        # 合并内容和用户输入以增加检测范围
        text_a = (content_a + " " + user_input_a).strip()
        text_b = (content_b + " " + user_input_b).strip()

        if not text_a or not text_b:
            return 0.0

        # 因果关系关键词模式
        CAUSE_PATTERNS = [
            r'因为.*?(?:所以|导致|使得|造成)',
            r'由于.*?(?:所以|因此|导致|使得)',
            r'原因[是为].*?结果',
            r'(?:所以|因此|导致|使得).*?因为',
            r'(?:之所以|因为).*?(?:是因为|由于)',
        ]

        # 问题-解决方案模式
        SOLUTION_PATTERNS = [
            r'(?:解决|处理|办法|方法|如何).*?(?:问题|困难)',
            r'(?:问题|困难|烦恼).*?(?:解决|处理|克服)',
        ]

        score = 0.0

        # 检测 A→B 因果关系
        for pattern in CAUSE_PATTERNS:
            if re.search(pattern, text_a):
                score += 0.3
                break

        # 检测 B→A 因果关系
        for pattern in CAUSE_PATTERNS:
            if re.search(pattern, text_b):
                score += 0.3
                break

        # 检测问题-解决方案关系
        for pattern in SOLUTION_PATTERNS:
            if re.search(pattern, text_a) or re.search(pattern, text_b):
                score += 0.2
                break

        return min(score, 1.0)

    def _detect_contrast_association(
        self,
        emotion_tag_a: str,
        content_a: str,
        emotion_tag_b: str,
        content_b: str,
    ) -> float:
        """
        对比关联检测 - 基于情绪标签或内容的对立关系

        当两条记忆的情绪极性相反（正面 vs 负面），
        或者内容中包含明确的对比表达时，建立对比关联。

        Args:
            emotion_tag_a/b: 情绪标签（"正面"/"负面"/"中性"）
            content_a/b: 记忆内容

        Returns:
            对比关联分数 [0.0, 1.0]
        """
        score = 0.0

        # 情绪极性对比
        POSITIVE_TAGS = {'正面', '开心', '高兴', '喜欢', '美好', '幸福'}
        NEGATIVE_TAGS = {'负面', '难过', '悲伤', '焦虑', '压力', '痛苦', '烦恼'}

        is_a_positive = emotion_tag_a in POSITIVE_TAGS
        is_a_negative = emotion_tag_a in NEGATIVE_TAGS
        is_b_positive = emotion_tag_b in POSITIVE_TAGS
        is_b_negative = emotion_tag_b in NEGATIVE_TAGS

        if (is_a_positive and is_b_negative) or (is_a_negative and is_b_positive):
            score += 0.5

        # 内容中的对比关键词
        import re
        contrast_patterns = [
            r'(?:但是|可是|然而|不过|相反|反之|却|而)',
            r'(?:对比|比较|不同|差异|区别)',
            r'(?:之前|以前|原来).*?(?:现在|如今|后来)',
            r'(?:改变|转变|变成|变成)',
        ]

        combined_text = content_a + " " + content_b
        for pattern in contrast_patterns:
            if re.search(pattern, combined_text):
                score += 0.3
                break

        return min(score, 1.0)

    # =========================================================================
    # 2. 关联图操作（内部方法）
    # =========================================================================

    def _add_association(self, source_id: str, association: Association) -> bool:
        """
        向关联图中添加一条关联边

        如果已存在相同类型和目标的关联，则更新其强度（取较大值）。
        否则创建新关联。

        Args:
            source_id: 源记忆 ID
            association: 关联对象

        Returns:
            是否成功添加
        """
        if not source_id or not association.target_memory_id:
            return False

        # 初始化邻接表
        if source_id not in self.adjacency_graph:
            self.adjacency_graph[source_id] = []

        # 检查是否已存在相同类型的目标关联
        existing_list = self.adjacency_graph[source_id]
        for existing in existing_list:
            if (existing.target_memory_id == association.target_memory_id and
                    existing.association_type == association.association_type):
                # 更新：取较大的强度值
                existing.strength = max(existing.strength, association.strength)
                existing.last_accessed_time = max(
                    existing.last_accessed_time, association.created_time
                )
                # 合并元数据
                existing.metadata.update(association.metadata)
                # 更新反向索引
                self._reverse_index.setdefault(
                    association.target_memory_id, set()
                ).add(source_id)
                return True

        # 新建关联
        existing_list.append(association)

        # 更新反向索引
        self._reverse_index.setdefault(
            association.target_memory_id, set()
        ).add(source_id)

        return True

    def _prune_weak_associations(self, memory_id: str) -> int:
        """
        LRU 风格修剪：移除指定记忆的最弱关联

        修剪策略：
        1. 先移除强度低于 min_strength 的关联
        2. 如果仍超过 max_associations，按 (strength * recency) 排序后移除最弱的

        Args:
            memory_id: 记忆 ID

        Returns:
            被修剪的关联数量
        """
        if memory_id not in self.adjacency_graph:
            return 0

        associations = self.adjacency_graph[memory_id]
        pruned = 0
        now = time.time()

        # 第一步：移除低于最小强度的关联
        to_remove = []
        for i, assoc in enumerate(associations):
            if assoc.strength < self.min_strength:
                to_remove.append(i)

        # 逆序删除以保持索引有效
        for i in reversed(to_remove):
            removed = associations.pop(i)
            # 更新反向索引
            if removed.target_memory_id in self._reverse_index:
                self._reverse_index[removed.target_memory_id].discard(memory_id)
            pruned += 1

        # 第二步：如果仍然超过上限，按综合评分修剪
        if len(associations) > self.max_associations:
            # 综合评分 = 强度 * 时间衰减因子
            def score_assoc(a: Association) -> float:
                age = now - a.last_accessed_time
                recency = 1.0 / (1.0 + age / 86400.0)  # 每天衰减一半
                return a.strength * recency

            associations.sort(key=score_assoc, reverse=True)
            excess = len(associations) - self.max_associations
            removed_list = associations[self.max_associations:]
            self.adjacency_graph[memory_id] = associations[:self.max_associations]

            for removed in removed_list:
                if removed.target_memory_id in self._reverse_index:
                    self._reverse_index[removed.target_memory_id].discard(memory_id)
                pruned += 1

        self._total_pruned += pruned
        return pruned

    def remove_memory_associations(self, memory_id: str) -> int:
        """
        完全移除一条记忆的所有关联（通常在记忆被删除时调用）

        同时清理关联图和反向索引。

        Args:
            memory_id: 要移除的记忆 ID

        Returns:
            被移除的关联总数
        """
        removed_count = 0

        # 1. 移除该记忆作为源的所有关联
        if memory_id in self.adjacency_graph:
            removed_count += len(self.adjacency_graph[memory_id])
            del self.adjacency_graph[memory_id]

        # 2. 移除其他记忆指向该记忆的关联
        if memory_id in self._reverse_index:
            source_ids = self._reverse_index[memory_id].copy()
            for source_id in source_ids:
                if source_id in self.adjacency_graph:
                    original_len = len(self.adjacency_graph[source_id])
                    self.adjacency_graph[source_id] = [
                        a for a in self.adjacency_graph[source_id]
                        if a.target_memory_id != memory_id
                    ]
                    removed_count += original_len - len(self.adjacency_graph[source_id])
            del self._reverse_index[memory_id]

        if removed_count > 0:
            logger.debug(f"[联想记忆网络] 移除记忆 {memory_id} 的 {removed_count} 条关联")

        return removed_count

    # =========================================================================
    # 3. 关联查询（联想召回增强）
    # =========================================================================

    def get_associated_memories(
        self,
        memory_id: str,
        max_depth: int = 2,
        max_results: int = 50,
        min_strength: float = 0.0,
        type_filter: Optional[Set[AssociationType]] = None,
    ) -> List[Tuple[str, int, float]]:
        """
        获取与指定记忆关联的所有记忆（BFS 遍历关联图）

        广度优先搜索关联图，返回所有可达的记忆。
        结果按 (memory_id, depth, cumulative_strength) 排序。

        Args:
            memory_id: 起始记忆 ID
            max_depth: BFS 最大深度（1=直接关联，2=二阶关联）
            max_results: 最大返回数量
            min_strength: 最小关联强度过滤
            type_filter: 只返回指定类型的关联（None=全部）

        Returns:
            [(memory_id, depth, cumulative_strength), ...] 按 depth 和 strength 排序
        """
        if memory_id not in self.adjacency_graph:
            return []

        visited: Set[str] = {memory_id}
        results: List[Tuple[str, int, float]] = []
        queue: deque = deque()

        # 初始化队列：(memory_id, depth, cumulative_strength)
        for assoc in self.adjacency_graph.get(memory_id, []):
            if (assoc.strength >= min_strength and
                    (type_filter is None or assoc.association_type in type_filter)):
                queue.append((assoc.target_memory_id, 1, assoc.strength))
                assoc.last_accessed_time = time.time()  # 更新访问时间

        while queue and len(results) < max_results:
            current_id, depth, cum_strength = queue.popleft()

            if current_id in visited:
                continue
            visited.add(current_id)

            results.append((current_id, depth, round(cum_strength, 6)))

            # 继续搜索下一层
            if depth < max_depth and current_id in self.adjacency_graph:
                for assoc in self.adjacency_graph[current_id]:
                    if (assoc.target_memory_id not in visited and
                            assoc.strength >= min_strength and
                            (type_filter is None or assoc.association_type in type_filter)):
                        # 累积强度 = 父强度 × 当前边强度（衰减传播）
                        new_cum_strength = cum_strength * assoc.strength
                        if new_cum_strength >= min_strength:
                            queue.append((
                                assoc.target_memory_id,
                                depth + 1,
                                new_cum_strength,
                            ))
                            assoc.last_accessed_time = time.time()

        # 排序：先按深度升序，再按累积强度降序
        results.sort(key=lambda x: (x[1], -x[2]))
        return results[:max_results]

    def spread_activation(
        self,
        seed_memory_ids: List[str],
        iterations: int = 3,
        activation_decay: float = 0.5,
        min_activation: float = 0.01,
        max_results: int = 30,
    ) -> Dict[str, float]:
        """
        扩散激活算法（Spreading Activation）

        模拟人脑的联想思维过程：
        从种子记忆开始，激活值沿关联边扩散传播，每经过一层衰减一次。
        多条路径到达同一记忆时，激活值累加（而非覆盖）。

        这类似于人类"自由联想"的思维模式：
        "下雨" → "伞" → "上次淋雨" → "感冒" → "药" ...

        Args:
            seed_memory_ids: 种子记忆 ID 列表（初始激活的记忆）
            iterations: 扩散迭代次数（即 BFS 的最大层数）
            activation_decay: 每层扩散的衰减系数 [0, 1]
                - 0.5 表示每层衰减一半（适中）
                - 0.7 表示扩散较远（弱关联也有影响）
                - 0.3 表示扩散很近（只关注强关联）
            min_activation: 最终激活值低于此阈值的记忆被过滤
            max_results: 最大返回数量

        Returns:
            {memory_id: activation_value, ...} 按激活值降序排列
        """
        if not seed_memory_ids:
            return {}

        now = time.time()
        # 激活值字典
        activations: Dict[str, float] = {}
        # 已访问集合（每轮迭代重置，因为允许同一条记忆被多条路径激活）
        globally_visited: Set[str] = set(seed_memory_ids)

        # 初始化种子记忆的激活值
        for mem_id in seed_memory_ids:
            activations[mem_id] = 1.0

        # 迭代扩散
        for iteration in range(iterations):
            new_activations: Dict[str, float] = {}

            # 收集当前轮所有有激活值的记忆
            current_active = {
                mid: act for mid, act in activations.items() if act > min_activation
            }

            if not current_active:
                break

            # 对每个有激活值的记忆，向其邻居扩散
            for source_id, source_activation in current_active.items():
                neighbors = self.adjacency_graph.get(source_id, [])

                for assoc in neighbors:
                    target_id = assoc.target_memory_id
                    if target_id in globally_visited:
                        # 已被激活过的记忆，只更新激活值（不再次扩散）
                        continue

                    # 扩散激活值 = 源激活 × 关联强度 × 衰减系数
                    spread_value = source_activation * assoc.strength * activation_decay

                    if spread_value >= min_activation:
                        # 多路径激活值累加（关键：允许不同路径对同一记忆的激活叠加）
                        if target_id in new_activations:
                            new_activations[target_id] += spread_value
                        else:
                            new_activations[target_id] = spread_value

                        # 标记为已扩散到（不再作为新一轮的源）
                        globally_visited.add(target_id)

                        # 更新关联的访问时间
                        assoc.last_accessed_time = now

            # 合并新的激活值到总激活字典
            for mid, act in new_activations.items():
                activations[mid] = activations.get(mid, 0.0) + act

        # 过滤低激活值，排除种子记忆本身
        results = {
            mid: round(act, 6)
            for mid, act in activations.items()
            if act >= min_activation and mid not in set(seed_memory_ids)
        }

        # 按激活值降序排序
        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1], reverse=True)[:max_results]
        )

        if sorted_results:
            logger.debug(
                f"[联想记忆网络] 扩散激活: 种子={seed_memory_ids[:3]}..., "
                f"迭代={iterations}, 激活记忆数={len(sorted_results)}"
            )

        return sorted_results

    def find_bridge_memories(
        self,
        mem_id_a: str,
        mem_id_b: str,
        max_depth: int = 3,
        min_path_strength: float = 0.1,
    ) -> List[List[str]]:
        """
        查找桥接记忆 - 找到连接两个无直接关联的记忆的中间记忆

        类似社交网络中的"共同好友"或知识图谱中的"最短路径"。
        这对于理解两个看似无关的记忆之间的关系非常有用。

        例如：
        - 记忆A："我在北京工作"
        - 记忆B："我爱吃烤鸭"
        - 桥接记忆C："北京烤鸭很有名" → A→C→B

        使用 BFS 在关联图中搜索最短路径。

        Args:
            mem_id_a: 起始记忆 ID
            mem_id_b: 目标记忆 ID
            max_depth: 最大搜索深度
            min_path_strength: 路径最小强度阈值

        Returns:
            [[path_memory_ids], ...] 按路径强度排序的路径列表
        """
        if mem_id_a == mem_id_b:
            return [[mem_id_a]]

        # 首先检查是否有直接关联
        for assoc in self.adjacency_graph.get(mem_id_a, []):
            if assoc.target_memory_id == mem_id_b and assoc.strength >= min_path_strength:
                return [[mem_id_a, mem_id_b]]

        # BFS 搜索最短路径
        visited: Set[str] = {mem_id_a}
        queue: deque = deque()
        # 队列元素: (current_node, path, cumulative_strength)

        for assoc in self.adjacency_graph.get(mem_id_a, []):
            if assoc.strength >= min_path_strength and assoc.target_memory_id not in visited:
                new_path = [mem_id_a, assoc.target_memory_id]
                queue.append((assoc.target_memory_id, new_path, assoc.strength))
                visited.add(assoc.target_memory_id)

        found_paths: List[Tuple[List[str], float]] = []

        while queue:
            current_id, path, cum_strength = queue.popleft()

            # 检查是否到达目标
            if current_id == mem_id_b:
                found_paths.append((path, cum_strength))
                continue  # 继续搜索其他路径

            # 超过最大深度则跳过
            if len(path) >= max_depth:
                continue

            # 扩展邻居
            for assoc in self.adjacency_graph.get(current_id, []):
                next_id = assoc.target_memory_id
                if next_id not in visited and assoc.strength >= min_path_strength:
                    new_strength = cum_strength * assoc.strength
                    if new_strength >= min_path_strength:
                        new_path = path + [next_id]
                        queue.append((next_id, new_path, new_strength))
                        visited.add(next_id)

        # 按路径强度降序排序
        found_paths.sort(key=lambda x: x[1], reverse=True)

        # 只返回路径（不含强度分数）
        result = [p[0] for p in found_paths]

        if result:
            logger.debug(
                f"[联想记忆网络] 找到 {len(result)} 条桥接路径: "
                f"{mem_id_a} → ... → {mem_id_b}"
            )

        return result

    # =========================================================================
    # 4. 赫布学习（Hebbian Learning）
    # =========================================================================

    def record_co_recall(self, recalled_ids: List[str]) -> Dict[str, int]:
        """
        记录共同召回事件（赫布学习）

        当多条记忆在同一时刻被共同召回时，
        它们之间的关联会被加强 —— 这就是赫布定律的核心：
        "一起激活的神经元会连接得更紧密"。

        对于 recalled_ids 中的每对记忆 (A, B)：
        1. 如果 A→B 关联存在，增加其 co_activation_count 并加强 strength
        2. 如果不存在关联，可以选择创建新的 SEMANTIC 类型关联

        Args:
            recalled_ids: 被共同召回的记忆 ID 列表

        Returns:
            {association_key: 更新次数} 更新统计
        """
        if len(recalled_ids) < 2:
            return {}

        now = time.time()
        update_count: Dict[str, int] = {}

        # 对每对记忆执行赫布更新
        for i in range(len(recalled_ids)):
            for j in range(i + 1, len(recalled_ids)):
                id_a = recalled_ids[i]
                id_b = recalled_ids[j]

                if id_a == id_b:
                    continue

                # 双向更新
                for source_id, target_id in [(id_a, id_b), (id_b, id_a)]:
                    key = f"{source_id}→{target_id}"
                    updated = self._hebbian_strengthen(source_id, target_id, now)
                    if updated:
                        update_count[key] = update_count.get(key, 0) + 1
                        self._total_hebbian_updates += 1

        if update_count:
            logger.debug(
                f"[联想记忆网络] 赫布学习: {len(recalled_ids)} 条记忆共同召回, "
                f"更新 {len(update_count)} 条关联"
            )

        return update_count

    def _hebbian_strengthen(
        self,
        source_id: str,
        target_id: str,
        current_time: float,
    ) -> bool:
        """
        赫布法则强化 - 加强指定记忆对之间的关联

        公式: new_strength = old_strength + learning_rate * (1 - old_strength)

        这个公式确保：
        - 弱关联增长更快（有更多提升空间）
        - 强关联增长更慢（逐渐饱和到 1.0）
        - 强度永远不会超过 1.0

        Args:
            source_id: 源记忆 ID
            target_id: 目标记忆 ID
            current_time: 当前时间戳

        Returns:
            是否成功更新
        """
        if source_id not in self.adjacency_graph:
            return False

        for assoc in self.adjacency_graph[source_id]:
            if assoc.target_memory_id == target_id:
                # 赫布法则：new = old + rate * (1 - old)
                delta = self.learning_rate * (1.0 - assoc.strength)
                assoc.strength = min(assoc.strength + delta, 1.0)
                assoc.co_activation_count += 1
                assoc.last_accessed_time = current_time
                return True

        return False

    def decay_associations(self) -> Dict[str, Any]:
        """
        对所有关联应用时间衰减

        未被频繁使用的关联会逐渐弱化，模拟人脑中不常用的连接逐渐消失。
        衰减公式: strength *= decay_rate
        衰减后低于 min_strength 的关联将被标记为待修剪。

        建议在每次记忆固化周期（如 SWR 巩固时）调用此方法。

        Returns:
            {'decayed': int, 'to_prune': int, 'stats': {...}}
        """
        now = time.time()
        decayed_count = 0
        to_prune_count = 0
        total_before = 0
        total_after = 0

        for source_id, associations in self.adjacency_graph.items():
            for assoc in associations:
                total_before += assoc.strength

                # 时间衰减
                assoc.strength *= self.decay_rate

                # 额外衰减：长时间未被访问的关联衰减更快
                time_since_access = now - assoc.last_accessed_time
                if time_since_access > 86400.0:  # 超过1天未被使用
                    extra_decay = 1.0 - 0.001 * min(time_since_access / 86400.0, 0.5)
                    assoc.strength *= extra_decay

                total_after += assoc.strength
                decayed_count += 1

                # 标记低于阈值的关联
                if assoc.strength < self.min_strength:
                    to_prune_count += 1

        # 执行修剪
        for source_id in list(self.adjacency_graph.keys()):
            self._prune_weak_associations(source_id)

        self._last_decay_time = now

        result = {
            'decayed': decayed_count,
            'to_prune': to_prune_count,
            'total_strength_before': round(total_before, 2),
            'total_strength_after': round(total_after, 2),
            'timestamp': now,
        }

        logger.debug(
            f"[联想记忆网络] 关联衰减: {decayed_count} 条衰减, "
            f"{to_prune_count} 条待修剪"
        )

        return result

    # =========================================================================
    # 5. 记忆干扰检测
    # =========================================================================

    def detect_interference(
        self,
        new_memory: Any,
        all_memories: Dict[str, Any],
        interference_threshold: float = 0.7,
    ) -> List[InterferenceWarning]:
        """
        记忆干扰检测 - 检测新记忆是否可能干扰（覆盖/混淆）已有记忆

        在认知心理学中，记忆干扰（Proactive/Retroactive Interference）
        是指相似的新旧记忆之间会相互干扰：
        - 顺向干扰：旧记忆干扰新记忆的学习
        - 逆向干扰：新记忆干扰旧记忆的回忆

        当新记忆与旧记忆过于相似时，可能导致：
        1. 旧记忆被新记忆覆盖
        2. 两段记忆在召回时混淆
        3. 信息准确性下降

        Args:
            new_memory: 新记忆对象
            all_memories: 所有现有记忆字典
            interference_threshold: 干扰分数阈值

        Returns:
            干扰警告列表
        """
        warnings: List[InterferenceWarning] = []
        new_id = getattr(new_memory, 'memory_id', '')

        new_embedding = getattr(new_memory, 'semantic_embedding', None)
        new_key_entities = getattr(new_memory, 'key_entities', '')
        new_emotion_type = getattr(new_memory, 'emotion_type', 'neutral')

        for mem_id, existing_mem in all_memories.items():
            if mem_id == new_id:
                continue

            interference_score = 0.0
            interference_types: List[str] = []

            # 1. 语义相似度（最大的干扰来源）
            existing_embedding = getattr(existing_mem, 'semantic_embedding', None)
            semantic_sim = self._detect_semantic_association(
                new_embedding, existing_embedding
            )
            if semantic_sim > 0.8:  # 非常高的语义相似度
                interference_score += semantic_sim * 0.4
                interference_types.append('semantic')

            # 2. 实体重叠（会导致实体信息混淆）
            entity_score, shared_entities = self._detect_entity_association(
                new_key_entities,
                getattr(existing_mem, 'key_entities', ''),
            )
            if entity_score > 0.5:
                interference_score += entity_score * 0.3
                interference_types.append('entity')

            # 3. 情绪相同（同情绪不同内容的记忆容易混淆）
            existing_emotion_type = getattr(existing_mem, 'emotion_type', 'neutral')
            if (new_emotion_type == existing_emotion_type and
                    new_emotion_type != 'neutral'):
                interference_score += 0.2
                interference_types.append('emotional')

            # 4. 检查已有旧记忆是否与新记忆的已有强关联重叠
            # （如果 A-B 强关联，新记忆 C 也与 A 强关联，C 可能会"窃取"A-B 关联）
            existing_associations = self.adjacency_graph.get(mem_id, [])
            for assoc in existing_associations:
                if assoc.strength > 0.7:
                    interference_score += 0.1
                    interference_types.append('association_overlap')
                    break

            # 生成干扰警告
            if interference_score >= interference_threshold:
                # 生成建议
                suggestion = self._generate_consolidation_suggestion(
                    new_memory, existing_mem, interference_types, semantic_sim
                )

                warning = InterferenceWarning(
                    new_memory_id=new_id,
                    existing_memory_id=mem_id,
                    interference_score=round(interference_score, 4),
                    interference_types=interference_types,
                    suggestion=suggestion,
                )
                warnings.append(warning)

        # 按干扰分数排序
        warnings.sort(key=lambda w: w.interference_score, reverse=True)

        if warnings:
            logger.warning(
                f"[联想记忆网络] 检测到 {len(warnings)} 个记忆干扰警告 "
                f"（最高干扰分数: {warnings[0].interference_score:.3f}）"
            )

        return warnings

    def _generate_consolidation_suggestion(
        self,
        new_memory: Any,
        existing_memory: Any,
        interference_types: List[str],
        semantic_similarity: float,
    ) -> str:
        """
        生成记忆巩固建议

        根据干扰类型和严重程度，给出具体的处理建议。

        Args:
            new_memory: 新记忆
            existing_memory: 旧记忆
            interference_types: 干扰类型列表
            semantic_similarity: 语义相似度

        Returns:
            建议字符串
        """
        suggestions = []

        if semantic_similarity > 0.9:
            suggestions.append("新旧记忆语义高度重叠，建议合并为一条记忆（保留最新版本的关键信息）")

        if 'entity' in interference_types:
            suggestions.append("关键实体冲突，建议核实新旧记忆中的实体信息是否一致")

        if 'emotional' in interference_types:
            suggestions.append("同情绪不同内容，建议在记忆中添加时间戳或上下文区分")

        if 'association_overlap' in interference_types:
            suggestions.append("可能窃取已有记忆的关联，建议显式创建新旧记忆之间的关联以保留上下文")

        if semantic_similarity > 0.8 and semantic_similarity <= 0.9:
            suggestions.append("语义相似但不完全相同，建议保留两条记忆但建立对比关联")

        if not suggestions:
            return "建议持续监控，暂无紧急处理需求"

        return "；".join(suggestions)

    # =========================================================================
    # 6. 序列化（get_state / set_state）
    # =========================================================================

    def get_state(self) -> dict:
        """
        获取联想记忆网络的完整状态（用于持久化）

        将关联图序列化为可 JSON 序列化的字典格式。

        Returns:
            {
                'adjacency_graph': {source_id: [association_dict, ...], ...},
                'config': {...},
                'stats': {...},
            }
        """
        # 序列化关联图
        graph_dict = {}
        for source_id, associations in self.adjacency_graph.items():
            graph_dict[source_id] = [a.to_dict() for a in associations]

        state = {
            'adjacency_graph': graph_dict,
            'config': {
                'max_associations': self.max_associations,
                'semantic_threshold': self.semantic_threshold,
                'emotion_intensity_threshold': self.emotion_intensity_threshold,
                'temporal_window_s': self.temporal_window_s,
                'learning_rate': self.learning_rate,
                'decay_rate': self.decay_rate,
                'min_strength': self.min_strength,
                'spread_decay': self.spread_decay,
            },
            'stats': {
                'total_associations_created': self._total_associations_created,
                'total_hebbian_updates': self._total_hebbian_updates,
                'total_pruned': self._total_pruned,
                'last_decay_time': self._last_decay_time,
                'total_memories_with_associations': len(self.adjacency_graph),
                'total_association_edges': sum(
                    len(assocs) for assocs in self.adjacency_graph.values()
                ),
            },
        }

        return state

    def set_state(self, state: dict):
        """
        从状态字典恢复联想记忆网络（用于加载持久化数据）

        Args:
            state: get_state() 返回的状态字典
        """
        # 恢复配置
        config = state.get('config', {})
        self.max_associations = config.get('max_associations', self.max_associations)
        self.semantic_threshold = config.get('semantic_threshold', self.semantic_threshold)
        self.emotion_intensity_threshold = config.get(
            'emotion_intensity_threshold', self.emotion_intensity_threshold
        )
        self.temporal_window_s = config.get('temporal_window_s', self.temporal_window_s)
        self.learning_rate = config.get('learning_rate', self.learning_rate)
        self.decay_rate = config.get('decay_rate', self.decay_rate)
        self.min_strength = config.get('min_strength', self.min_strength)
        self.spread_decay = config.get('spread_decay', self.spread_decay)

        # 恢复关联图
        self.adjacency_graph = {}
        self._reverse_index = {}

        graph_dict = state.get('adjacency_graph', {})
        for source_id, assoc_dicts in graph_dict.items():
            associations = []
            for assoc_dict in assoc_dicts:
                assoc = Association.from_dict(assoc_dict)
                associations.append(assoc)

                # 重建反向索引
                self._reverse_index.setdefault(
                    assoc.target_memory_id, set()
                ).add(source_id)

            self.adjacency_graph[source_id] = associations

        # 恢复统计信息
        stats = state.get('stats', {})
        self._total_associations_created = stats.get(
            'total_associations_created', 0
        )
        self._total_hebbian_updates = stats.get('total_hebbian_updates', 0)
        self._total_pruned = stats.get('total_pruned', 0)
        self._last_decay_time = stats.get('last_decay_time', 0.0)

        logger.info(
            f"[联想记忆网络] 状态恢复完成: "
            f"{len(self.adjacency_graph)} 条记忆有关联, "
            f"共 {sum(len(a) for a in self.adjacency_graph.values())} 条关联边"
        )

    # =========================================================================
    # 7. 统计与诊断
    # =========================================================================

    def get_stats(self) -> dict:
        """
        获取联想记忆网络的统计信息

        Returns:
            {
                'total_memories_with_associations': int,  # 有关联的记忆总数
                'total_association_edges': int,            # 关联边总数
                'avg_associations_per_memory': float,      # 平均每条记忆的关联数
                'max_associations_for_single': int,        # 单条记忆的最大关联数
                'total_associations_created': int,         # 累计创建的关联数
                'total_hebbian_updates': int,              # 累计赫布更新次数
                'total_pruned': int,                       # 累计修剪的关联数
                'association_type_distribution': dict,     # 各类型关联数量分布
                'avg_strength': float,                     # 平均关联强度
                'strong_associations_count': int,          # 强关联数量（>0.7）
            }
        """
        total_edges = sum(len(a) for a in self.adjacency_graph.values())
        total_memories = len(self.adjacency_graph)

        if total_memories == 0:
            return {
                'total_memories_with_associations': 0,
                'total_association_edges': 0,
                'avg_associations_per_memory': 0.0,
                'max_associations_for_single': 0,
                'total_associations_created': self._total_associations_created,
                'total_hebbian_updates': self._total_hebbian_updates,
                'total_pruned': self._total_pruned,
                'association_type_distribution': {},
                'avg_strength': 0.0,
                'strong_associations_count': 0,
            }

        # 关联类型分布
        type_dist: Dict[str, int] = {}
        total_strength = 0.0
        strong_count = 0
        max_assoc_count = 0

        for associations in self.adjacency_graph.values():
            max_assoc_count = max(max_assoc_count, len(associations))
            for assoc in associations:
                type_name = assoc.association_type.value
                type_dist[type_name] = type_dist.get(type_name, 0) + 1
                total_strength += assoc.strength
                if assoc.strength > 0.7:
                    strong_count += 1

        return {
            'total_memories_with_associations': total_memories,
            'total_association_edges': total_edges,
            'avg_associations_per_memory': round(total_edges / total_memories, 2),
            'max_associations_for_single': max_assoc_count,
            'total_associations_created': self._total_associations_created,
            'total_hebbian_updates': self._total_hebbian_updates,
            'total_pruned': self._total_pruned,
            'association_type_distribution': type_dist,
            'avg_strength': round(total_strength / max(total_edges, 1), 4),
            'strong_associations_count': strong_count,
        }

    def get_association_types_for_memory(
        self, memory_id: str
    ) -> Dict[str, List[str]]:
        """
        获取指定记忆的所有关联类型及其对应的目标记忆

        Args:
            memory_id: 记忆 ID

        Returns:
            {association_type: [target_memory_id, ...], ...}
        """
        result: Dict[str, List[str]] = {}
        for assoc in self.adjacency_graph.get(memory_id, []):
            type_name = assoc.association_type.value
            if type_name not in result:
                result[type_name] = []
            result[type_name].append(assoc.target_memory_id)
        return result

    def get_association_strength(
        self, source_id: str, target_id: str
    ) -> float:
        """
        获取两条记忆之间的最大关联强度

        Args:
            source_id: 源记忆 ID
            target_id: 目标记忆 ID

        Returns:
            最大关联强度（0.0 如果无关联）
        """
        for assoc in self.adjacency_graph.get(source_id, []):
            if assoc.target_memory_id == target_id:
                return assoc.strength
        return 0.0

    def reset(self):
        """重置联想记忆网络到初始状态"""
        self.adjacency_graph.clear()
        self._reverse_index.clear()
        self._total_associations_created = 0
        self._total_hebbian_updates = 0
        self._total_pruned = 0
        self._last_decay_time = 0.0
        logger.info("[联想记忆网络] 已重置")

    def get_memory_association_summary(
        self, memory_id: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取指定记忆的关联摘要（用于调试/展示）

        Args:
            memory_id: 记忆 ID
            top_k: 返回前 K 条最强关联

        Returns:
            [{'target_id': str, 'type': str, 'strength': float, ...}, ...]
        """
        associations = self.adjacency_graph.get(memory_id, [])
        # 按强度降序排序
        sorted_assocs = sorted(associations, key=lambda a: a.strength, reverse=True)

        return [
            {
                'target_id': a.target_memory_id,
                'type': a.association_type.value,
                'strength': round(a.strength, 4),
                'co_activation_count': a.co_activation_count,
                'created_time': a.created_time,
                'metadata': a.metadata,
            }
            for a in sorted_assocs[:top_k]
        ]
