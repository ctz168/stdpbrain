"""
创造性洞察引擎 (Creative Insight Engine)

设计理念:
模拟人类的创造性思维过程 —— 从发散思维到"灵光一现"的洞察时刻。
人类最独特的认知能力之一就是将看似无关的概念建立联系，
产生新颖且有价值的想法（即 "Aha Moment"）。

核心模块:
1. DivergentThinking        - 发散思维: 从单一问题生成多种解决方案
2. RemoteAssociationTest    - 远程联想测试: Mednick's RAT，寻找三词之间的隐含关联
3. InsightDetection         - 洞察检测: 识别"灵光一现"时刻，跟踪洞察历史
4. CreativeCombination      - 创意组合: 将两个概念融合为新颖组合
5. IncubationSystem         - 孵化系统: "睡一觉再说"，延迟创造性问题解决
6. MetaphorGeneration       - 隐喻生成: 跨领域类比，"X像Y因为..."
7. InnerThoughtIntegration  - 思维引擎集成: 在漫游/反思状态注入创造性火花

参考理论:
- Mednick (1962) Remote Associates Test (远程联想测试)
- Guilford (1967) Divergent Thinking (发散思维)
- Wallas (1926) Creative Process: 准备→孵化→启发→验证
- Koestler (1964) Bisociation Theory (双联想理论)
- Finke et al. (1992) Creative Cognition Approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import math
import time
import random
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


logger = logging.getLogger(__name__)


# ==================== 枚举与常量 ====================

class ThinkingStrategy(Enum):
    """发散思维策略"""
    RANDOM_ASSOCIATION = "random_association"    # 随机联想: 将主题与随机概念连接
    ANALOGY_MAPPING = "analogy_mapping"          # 类比映射: "X像Y因为..."
    REVERSAL = "reversal"                        # 逆向思考: "如果反过来呢？"
    COMBINATION = "combination"                  # 组合创新: "把A和B结合起来呢？"
    SCALING = "scaling"                          # 尺度变化: "放大/缩小10倍会怎样？"
    PERSPECTIVE_SHIFT = "perspective_shift"      # 视角转换: "孩子/专家/外星人怎么看？"


class CombinationTemplate(Enum):
    """创意组合模板"""
    APPLICATION = "application"          # 应用型: "A使用B"
    ANALOGY = "analogy"                  # 类比型: "A像B"
    FUSION = "fusion"                    # 融合型: "A遇上B"
    IMPROVEMENT = "improvement"          # 改良型: "A但带有B"
    PERSPECTIVE = "perspective"          # 透视型: "通过B的视角看A"


# ==================== 数据结构 ====================

@dataclass
class CreativeIdea:
    """
    创意数据结构 - 存储一个发散思维产生的想法

    Attributes:
        content: 想法内容
        strategy: 生成该想法的思维策略
        novelty_score: 新颖度 (0-1)，越高表示越独特
        coherence_score: 连贯性 (0-1)，越高表示越合理
        utility_score: 实用度 (0-1)，越高表示越有用
        source_concepts: 输入概念列表
        timestamp: 创建时间戳
    """
    content: str
    strategy: str = ""
    novelty_score: float = 0.5
    coherence_score: float = 0.5
    utility_score: float = 0.5
    source_concepts: List[str] = field(default_factory=list)
    timestamp: float = 0.0

    def overall_score(self) -> float:
        """综合评分 = 新颖度 × 0.4 + 连贯性 × 0.3 + 实用度 × 0.3"""
        return self.novelty_score * 0.4 + self.coherence_score * 0.3 + self.utility_score * 0.3

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "content": self.content,
            "strategy": self.strategy,
            "novelty_score": round(self.novelty_score, 4),
            "coherence_score": round(self.coherence_score, 4),
            "utility_score": round(self.utility_score, 4),
            "overall_score": round(self.overall_score(), 4),
            "source_concepts": self.source_concepts,
            "timestamp": self.timestamp,
        }


@dataclass
class InsightRecord:
    """
    洞察记录 - 一次"灵光一现"的完整记录

    Attributes:
        insight: 洞察内容
        connecting_concepts: 被连接的无关概念
        novelty: 新颖度
        utility: 实用度
        incubation_time: 孵化时间（秒），该问题被搁置了多久
        timestamp: 洞察发生时间戳
    """
    insight: str
    connecting_concepts: List[str] = field(default_factory=list)
    novelty: float = 0.0
    utility: float = 0.0
    incubation_time: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "insight": self.insight,
            "connecting_concepts": self.connecting_concepts,
            "novelty": round(self.novelty, 4),
            "utility": round(self.utility, 4),
            "incubation_time": round(self.incubation_time, 2),
            "timestamp": self.timestamp,
        }


@dataclass
class IncubationProblem:
    """
    孵化中的问题 - 等待"灵光一现"的未解决问题

    Attributes:
        problem: 问题描述
        context: 上下文信息
        start_time: 开始孵化时间戳
        attempts: 已尝试解决次数
        solution: 解决方案（若已找到）
        resolved: 是否已解决
    """
    problem: str
    context: str = ""
    start_time: float = 0.0
    attempts: int = 0
    solution: Optional[str] = None
    resolved: bool = False

    def incubation_elapsed(self) -> float:
        """已孵化时长（秒）"""
        if self.resolved:
            return 0.0
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "problem": self.problem,
            "context": self.context,
            "start_time": self.start_time,
            "attempts": self.attempts,
            "solution": self.solution,
            "resolved": self.resolved,
            "incubation_elapsed": round(self.incubation_elapsed(), 2),
        }


@dataclass
class RemoteAssociationResult:
    """远程联想结果"""
    associate_word: str
    connections: Dict[str, float] = field(default_factory=dict)  # word → similarity
    minimax_score: float = 0.0


# ==================== 核心引擎 ====================

class CreativeInsightEngine(nn.Module):
    """
    创造性洞察引擎 - 让 AI 拥有人类般的创造性思维

    核心能力:
    1. 发散思维: 从一个主题出发，用多种策略生成多样想法
    2. 远程联想: 寻找三个看似无关词语之间的隐含联系
    3. 洞察检测: 识别并记录"灵光一现"时刻
    4. 创意组合: 用多种模板融合两个不同概念
    5. 孵化系统: "睡一觉再说"——延迟创造性问题解决
    6. 隐喻生成: 创建跨领域的隐喻类比

    设计原则:
    - 全部本地计算，不依赖外部 API
    - 使用 torch 张量做 embedding 距离运算
    - 轻量级：不阻塞主思维循环
    - 与 InnerThoughtEngine 的漫游/反思状态集成
    """

    # 常量：概念空间维度（用于 hash-based embedding）
    EMBEDDING_DIM = 128

    # 孵化相关常量
    MAX_INCUBATION_TIME = 86400.0    # 最大孵化时间 24 小时（秒）
    OPTIMAL_INCUBATION_TIME = 1800.0  # 最佳孵化时间 30 分钟（秒）
    INCUBATION_DECAY_START = 7200.0   # 孵化效果开始衰减 2 小时

    # 洞察阈值
    INSIGHT_NOVELTY_THRESHOLD = 0.6   # 新颖度阈值
    INSIGHT_UTILITY_THRESHOLD = 0.4   # 实用度阈值
    INSIGHT_DOMAIN_DISTANCE = 0.7     # 被连接领域间的最小距离

    def __init__(
        self,
        embedding_dim: int = 128,
        max_incubation_problems: int = 20,
        insight_history_size: int = 50,
        device: str = "cpu",
    ):
        """
        初始化创造性洞察引擎

        Args:
            embedding_dim: 概念 embedding 维度
            max_incubation_problems: 最大同时孵化的未解决问题数
            insight_history_size: 洞察历史记录最大条数
            device: 计算设备
        """
        super().__init__()

        self.device = device
        self.embedding_dim = embedding_dim

        # ========== 概念 embedding 空间 ==========
        # 使用可训练的 embedding 层作为概念空间
        # 初始时为空，随着使用逐步填充
        self._concept_embeddings: Dict[str, torch.Tensor] = {}
        self._concept_domains: Dict[str, str] = {}  # concept → domain

        # ========== 洞察记录 ==========
        self._insight_history: deque = deque(maxlen=insight_history_size)
        self._total_insights = 0

        # ========== 孵化系统 ==========
        self._incubation_problems: List[IncubationProblem] = []
        self.max_incubation_problems = max_incubation_problems

        # ========== 领域/概念知识库（用于联想和组合）==========
        self._domain_concepts: Dict[str, List[str]] = {
            "自然": ["山", "海", "风", "雨", "雪", "日", "月", "星", "云", "河",
                     "花", "树", "鸟", "鱼", "石", "火", "水", "土", "光", "影"],
            "技术": ["算法", "数据", "网络", "代码", "芯片", "信号", "存储",
                     "计算", "通信", "接口", "协议", "架构", "优化", "编译", "加密"],
            "情感": ["爱", "恨", "喜", "怒", "哀", "乐", "恐惧", "希望",
                     "孤独", "温暖", "怀念", "期待", "勇气", "温柔", "感动"],
            "社会": ["家庭", "友谊", "教育", "文化", "经济", "政治", "历史",
                     "法律", "艺术", "音乐", "文学", "哲学", "宗教", "传统", "变革"],
            "科学": ["原子", "分子", "细胞", "基因", "进化", "引力", "能量",
                     "波动", "量子", "相对论", "生态", "天文", "化学", "物理", "生物"],
            "生活": ["饮食", "运动", "旅行", "睡眠", "阅读", "写作", "绘画",
                     "舞蹈", "游戏", "手工", "烹饪", "园艺", "冥想", "社交", "冒险"],
            "抽象": ["时间", "空间", "因果", "自由", "公平", "真理", "美",
                     "和谐", "秩序", "混沌", "变化", "永恒", "无限", "虚无", "意义"],
            "商业": ["市场", "品牌", "创新", "策略", "效率", "价值", "增长",
                     "竞争", "合作", "信任", "领导力", "决策", "执行", "转型", "生态"],
        }

        # ========== 概念联想映射（共现/语义关联）==========
        self._association_links: Dict[str, List[Tuple[str, float]]] = {}

        # ========== 隐喻模板库 ==========
        self._metaphor_templates = {
            "structural": [
                "{source}就像{target}——{explanation}",
                "{source}是{target}的一种{explanation}",
                "如果把{source}比作{target}，那么{explanation}",
            ],
            "functional": [
                "{source}的作用就像{target}一样，{explanation}",
                "{source}和{target}异曲同工：{explanation}",
                "{source}之于{domain_a}，如同{target}之于{domain_b}，{explanation}",
            ],
            "emotional": [
                "{source}给我{target}般的感受，{explanation}",
                "{source}的灵魂中藏着{target}的影子，{explanation}",
                "理解{source}的关键，在于感受{target}的{explanation}",
            ],
        }

        # ========== 发散思维的领域词池 ==========
        self._random_concept_pool: List[str] = []
        for concepts in self._domain_concepts.values():
            self._random_concept_pool.extend(concepts)

        # ========== 统计计数器 ==========
        self._brainstorm_count = 0
        self._combination_count = 0
        self._metaphor_count = 0
        self._incubation_resolution_count = 0

        # ========== 预编译正则 ==========
        self._re_chinese_chars = re.compile(r'[\u4e00-\u9fff]+')
        self._re_punctuation = re.compile(
            r'[，。！？、；：\u201c\u201d\u2018\u2019（）【】\s,.\-!?;:()\[\]{}<>]'
        )

        # ========== MindState 引用（延迟注入）==========
        self._mind_state = None  # 由外部注入

        # ========== 初始化基础概念 embedding ==========
        self._initialize_base_embeddings()

    def _initialize_base_embeddings(self):
        """初始化基础概念 embedding（基于 hash 的确定性向量）"""
        for domain, concepts in self._domain_concepts.items():
            for concept in concepts:
                self._get_or_create_embedding(concept)
                self._concept_domains[concept] = domain

        # 建立领域内联想链接（同领域概念之间有基础关联）
        for domain, concepts in self._domain_concepts.items():
            for i, c1 in enumerate(concepts):
                for j, c2 in enumerate(concepts):
                    if i != j:
                        # 同领域内概念有中等关联强度
                        if c1 not in self._association_links:
                            self._association_links[c1] = []
                        self._association_links[c1].append((c2, 0.5))

    # ==================== 1. 概念 Embedding ====================

    def _get_or_create_embedding(self, concept: str) -> torch.Tensor:
        """
        获取或创建概念的 embedding 向量

        使用基于 hash 的确定性方法生成 embedding：
        - 相同概念总是产生相同向量
        - 不同概念产生不同向量
        - 维度固定为 EMBEDDING_DIM

        Args:
            concept: 概念文本

        Returns:
            归一化后的 embedding 张量 [embedding_dim]
        """
        if concept in self._concept_embeddings:
            return self._concept_embeddings[concept]

        # 基于 hash 生成确定性伪随机向量
        concept_key = concept.strip().lower()
        hash_bytes = hashlib.sha256(concept_key.encode('utf-8')).digest()

        # 用 hash 字节生成浮点向量
        vec = []
        for i in range(self.embedding_dim):
            byte_idx = i % len(hash_bytes)
            # 组合多个字节产生更均匀的分布
            val = (hash_bytes[byte_idx] * (i + 1) +
                   hash_bytes[(byte_idx + 1) % len(hash_bytes)] +
                   hash_bytes[(byte_idx + 3) % len(hash_bytes)]) / 765.0
            vec.append(val * 2.0 - 1.0)  # 映射到 [-1, 1]

        embedding = torch.tensor(vec, dtype=torch.float32, device=self.device)

        # L2 归一化
        norm = torch.norm(embedding, p=2)
        if norm > 0:
            embedding = embedding / norm

        self._concept_embeddings[concept] = embedding
        return embedding

    def _register_concept_domain(self, concept: str, domain: str):
        """注册概念所属领域"""
        self._concept_domains[concept.strip()] = domain.strip()

    def _compute_similarity(self, concept_a: str, concept_b: str) -> float:
        """
        计算两个概念之间的余弦相似度

        Args:
            concept_a: 概念A
            concept_b: 概念B

        Returns:
            余弦相似度 [-1, 1]
        """
        emb_a = self._get_or_create_embedding(concept_a)
        emb_b = self._get_or_create_embedding(concept_b)

        # 两者都已归一化，直接点积即余弦相似度
        sim = torch.dot(emb_a, emb_b).item()
        return max(-1.0, min(1.0, sim))

    def _compute_concept_distance(self, concept_a: str, concept_b: str) -> float:
        """
        计算两个概念之间的距离（1 - 余弦相似度）

        Args:
            concept_a: 概念A
            concept_b: 概念B

        Returns:
            距离 [0, 2]，0 表示完全相同
        """
        return 1.0 - self._compute_similarity(concept_a, concept_b)

    def _find_associated_concepts(self, concept: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        查找与给定概念最相关的其他概念

        Args:
            concept: 查询概念
            top_k: 返回前 k 个最相关概念

        Returns:
            [(概念, 相似度)] 列表，按相似度降序
        """
        query_emb = self._get_or_create_embedding(concept)
        results = []

        for other_concept, other_emb in self._concept_embeddings.items():
            if other_concept == concept:
                continue
            sim = torch.dot(query_emb, other_emb).item()
            results.append((other_concept, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _are_different_domains(self, concepts: List[str]) -> bool:
        """
        检查一组概念是否来自至少两个不同领域

        Args:
            concepts: 概念列表

        Returns:
            如果概念跨越至少2个不同领域则返回 True
        """
        domains = set()
        for c in concepts:
            c_stripped = c.strip()
            if c_stripped in self._concept_domains:
                domains.add(self._concept_domains[c_stripped])

        return len(domains) >= 2

    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词（中文2-4字词组）"""
        matches = self._re_chinese_chars.findall(text)
        # 保留2-4字词组
        keywords = []
        seen = set()
        for m in matches:
            if len(m) < 2 or len(m) > 4:
                continue
            if m not in seen:
                seen.add(m)
                keywords.append(m)
        return keywords[:8]

    # ==================== 2. 发散思维模块 ====================

    def brainstorm(
        self,
        topic: str,
        num_ideas: int = 5,
        strategies: Optional[List[ThinkingStrategy]] = None,
    ) -> List[CreativeIdea]:
        """
        发散思维 - 从一个主题生成多种不同角度的想法

        人类面对一个问题时，能够从多个角度思考：
        - 随机联想：将主题与随机概念连接
        - 类比映射："X像Y因为..."
        - 逆向思考："如果反过来呢？"
        - 组合创新："把A和B结合起来呢？"
        - 尺度变化："放大/缩小10倍会怎样？"
        - 视角转换："孩子/专家/外星人怎么看？"

        Args:
            topic: 主题/问题
            num_ideas: 目标生成想法数量
            strategies: 指定使用的思维策略列表，None 则随机选择

        Returns:
            CreativeIdea 列表，按综合评分降序排列
        """
        self._brainstorm_count += 1
        now = time.time()

        if strategies is None:
            # 从所有策略中随机选择（确保多样性）
            available = list(ThinkingStrategy)
            num_strategies = min(num_ideas, len(available))
            strategies = random.sample(available, num_strategies)
            # 如果需要更多想法，允许重复策略
            while len(strategies) < num_ideas:
                strategies.append(random.choice(available))

        ideas: List[CreativeIdea] = []

        for i in range(num_ideas):
            strategy = strategies[i] if i < len(strategies) else random.choice(strategies)
            idea = self._apply_strategy(topic, strategy)
            idea.timestamp = now
            idea.source_concepts = [topic]
            ideas.append(idea)

        # 按综合评分降序排列
        ideas.sort(key=lambda x: x.overall_score(), reverse=True)
        return ideas[:num_ideas]

    def _apply_strategy(self, topic: str, strategy: ThinkingStrategy) -> CreativeIdea:
        """
        应用单个思维策略生成想法

        Args:
            topic: 主题
            strategy: 思维策略

        Returns:
            CreativeIdea
        """
        if strategy == ThinkingStrategy.RANDOM_ASSOCIATION:
            return self._strategy_random_association(topic)
        elif strategy == ThinkingStrategy.ANALOGY_MAPPING:
            return self._strategy_analogy_mapping(topic)
        elif strategy == ThinkingStrategy.REVERSAL:
            return self._strategy_reversal(topic)
        elif strategy == ThinkingStrategy.COMBINATION:
            return self._strategy_combination(topic)
        elif strategy == ThinkingStrategy.SCALING:
            return self._strategy_scaling(topic)
        elif strategy == ThinkingStrategy.PERSPECTIVE_SHIFT:
            return self._strategy_perspective_shift(topic)
        else:
            return self._strategy_random_association(topic)

    def _strategy_random_association(self, topic: str) -> CreativeIdea:
        """
        随机联想策略：将主题与随机概念连接

        模拟人类思维中"说X想到Y"的自由联想过程。
        通过概念空间中的距离找到有一定距离但不太远的关联。

        Args:
            topic: 主题

        Returns:
            CreativeIdea
        """
        # 从随机概念池中选取一个与主题有一定距离的概念
        candidates = []
        for concept in self._random_concept_pool:
            dist = self._compute_concept_distance(topic, concept)
            # 选择中等距离的概念（太近=太普通，太远=无关联）
            if 0.5 < dist < 1.2:
                candidates.append((concept, dist))

        if not candidates:
            # 回退：使用任意概念
            random_concept = random.choice(self._random_concept_pool)
        else:
            # 从候选中随机选取
            random_concept, _ = random.choice(candidates)

        # 找到关联概念之间的语义桥梁
        bridge = self._find_bridge_concept(topic, random_concept)

        if bridge:
            content = f"{topic}让我联想到{random_concept}——它们之间通过「{bridge}」产生联系，也许可以从中获得启发"
            novelty = 0.6 + random.uniform(0, 0.2)
            coherence = 0.5 + random.uniform(0, 0.2)
        else:
            content = f"{topic}让我意外地想到{random_concept}——这两个看似无关的事物，或许存在某种隐藏的联系"
            novelty = 0.7 + random.uniform(0, 0.2)
            coherence = 0.3 + random.uniform(0, 0.2)

        return CreativeIdea(
            content=content,
            strategy="random_association",
            novelty_score=round(min(1.0, novelty), 4),
            coherence_score=round(min(1.0, coherence), 4),
            utility_score=round(0.3 + random.uniform(0, 0.3), 4),
        )

    def _strategy_analogy_mapping(self, topic: str) -> CreativeIdea:
        """
        类比映射策略："X像Y因为..."

        寻找与主题在结构上相似的领域，建立类比关系。
        这是人类创造性思维中最常见的方式之一。

        Args:
            topic: 主题

        Returns:
            CreativeIdea
        """
        # 找到一个与主题结构相似但领域不同的概念
        topic_emb = self._get_or_create_embedding(topic)
        topic_domain = self._concept_domains.get(topic, "")

        best_analogy = None
        best_similarity = -1.0

        for concept, emb in self._concept_embeddings.items():
            if concept == topic:
                continue
            concept_domain = self._concept_domains.get(concept, "")
            # 优先选择不同领域的概念（跨域类比更有创造性）
            if concept_domain != topic_domain and topic_domain:
                sim = torch.dot(topic_emb, emb).item()
                if sim > best_similarity and sim < 0.85:  # 不能太相似
                    best_similarity = sim
                    best_analogy = concept

        if best_analogy is None:
            # 回退：从同领域找
            for concept, emb in self._concept_embeddings.items():
                if concept == topic:
                    continue
                sim = torch.dot(topic_emb, emb).item()
                if sim > best_similarity and 0.3 < sim < 0.85:
                    best_similarity = sim
                    best_analogy = concept

        if best_analogy:
            # 找到共同的结构特征
            shared_features = self._find_shared_features(topic, best_analogy)
            feature_str = "、".join(shared_features) if shared_features else "某种结构上的相似性"
            content = f"「{topic}」和「{best_analogy}」在{feature_str}上有相似之处——这给了我一个新的理解角度"
            novelty = 0.5 + random.uniform(0, 0.25)
            coherence = 0.6 + random.uniform(0, 0.2)
        else:
            content = f"如果仔细观察，{topic}的内在结构其实像某种自然规律——也许可以从这个角度深入思考"
            novelty = 0.5 + random.uniform(0, 0.15)
            coherence = 0.4 + random.uniform(0, 0.15)

        return CreativeIdea(
            content=content,
            strategy="analogy_mapping",
            novelty_score=round(min(1.0, novelty), 4),
            coherence_score=round(min(1.0, coherence), 4),
            utility_score=round(0.4 + random.uniform(0, 0.3), 4),
        )

    def _strategy_reversal(self, topic: str) -> CreativeIdea:
        """
        逆向思考策略："如果反过来呢？"

        通过反转假设，发现被忽略的可能性。
        伟大的创新常常来自质疑"理所当然"的事情。

        Args:
            topic: 主题

        Returns:
            CreativeIdea
        """
        reversal_prompts = [
            f"如果{topic}的反面才是正确的呢？",
            f"假设{topic}完全不成立，世界会怎样？",
            f"如果我们不做{topic}，而是做完全相反的事呢？",
            f"假设{topic}的核心假设是错的——那替代方案是什么？",
            f"如果{topic}的效果恰好相反呢？",
        ]
        content = random.choice(reversal_prompts)

        # 根据主题长度调整评分
        base_novelty = 0.65 if len(topic) > 2 else 0.55
        return CreativeIdea(
            content=content,
            strategy="reversal",
            novelty_score=round(min(1.0, base_novelty + random.uniform(0, 0.2)), 4),
            coherence_score=round(0.45 + random.uniform(0, 0.2), 4),
            utility_score=round(0.4 + random.uniform(0, 0.25), 4),
        )

    def _strategy_combination(self, topic: str) -> CreativeIdea:
        """
        组合创新策略："把A和B结合起来呢？"

        从不同领域各取一个元素，与主题组合产生新想法。
        组合创新是最常见的创新类型之一。

        Args:
            topic: 主题
        """
        topic_domain = self._concept_domains.get(topic, "")
        other_domains = [d for d in self._domain_concepts.keys() if d != topic_domain]

        if not other_domains:
            other_domains = list(self._domain_concepts.keys())

        # 从不同领域随机选一个概念
        other_domain = random.choice(other_domains)
        other_concept = random.choice(self._domain_concepts[other_domain])

        templates = [
            f"如果把{topic}和{other_concept}结合在一起，也许会产生意想不到的效果——就像跨界融合那样",
            f"{topic}×{other_concept}：这个组合以前没有人尝试过，也许值得探索",
            f"当{topic}遇到{other_concept}，会碰撞出什么火花？这是一个有趣的交叉点",
        ]

        content = random.choice(templates)
        return CreativeIdea(
            content=content,
            strategy="combination",
            novelty_score=round(0.6 + random.uniform(0, 0.25), 4),
            coherence_score=round(0.4 + random.uniform(0, 0.25), 4),
            utility_score=round(0.35 + random.uniform(0, 0.3), 4),
            source_concepts=[topic, other_concept],
        )

    def _strategy_scaling(self, topic: str) -> CreativeIdea:
        """
        尺度变化策略："放大/缩小10倍会怎样？"

        通过极端假设打破思维惯性，发现新的可能性。

        Args:
            topic: 主题
        """
        scale_prompts = [
            f"如果{topic}的规模扩大100倍会怎样？哪些问题会浮现？哪些机会会出现？",
            f"如果{topic}缩小到微观级别——只剩核心——那剩下的是什么？",
            f"想象{topic}的影响范围扩大到全球，会发生什么连锁反应？",
            f"如果{topic}的时间尺度从一天变成十年，长远的效应是什么？",
            f"把{topic}的复杂度降到最低——最简单的版本是什么？还成立吗？",
        ]
        content = random.choice(scale_prompts)

        return CreativeIdea(
            content=content,
            strategy="scaling",
            novelty_score=round(0.55 + random.uniform(0, 0.2), 4),
            coherence_score=round(0.5 + random.uniform(0, 0.2), 4),
            utility_score=round(0.4 + random.uniform(0, 0.25), 4),
        )

    def _strategy_perspective_shift(self, topic: str) -> CreativeIdea:
        """
        视角转换策略："孩子/专家/外星人怎么看？"

        通过切换认知视角，打破既有的思维框架。
        不同视角会带来截然不同的理解和解决方案。

        Args:
            topic: 主题
        """
        perspectives = [
            ("孩子", "一个5岁的孩子看到{topic}会怎么理解？也许答案比我们想象的更简单"),
            ("外星人", "如果有一个从未见过{topic}的外星人来观察，他们会注意到什么我们忽略的东西？"),
            ("古代人", "如果让一个古代人理解{topic}，你会怎么解释？这种简化本身就是一种洞察"),
            ("艺术家", "从艺术家的角度来看{topic}——美在哪里？韵律在哪里？"),
            ("工程师", "用工程师的思维拆解{topic}——组件是什么？输入输出是什么？瓶颈在哪里？"),
            ("未来人", "从一百年后的视角回望{topic}——什么会觉得可笑？什么会被珍视？"),
        ]
        role, template = random.choice(perspectives)
        content = template.format(topic=topic)

        return CreativeIdea(
            content=content,
            strategy=f"perspective_shift({role})",
            novelty_score=round(0.6 + random.uniform(0, 0.2), 4),
            coherence_score=round(0.5 + random.uniform(0, 0.2), 4),
            utility_score=round(0.35 + random.uniform(0, 0.25), 4),
        )

    def _find_bridge_concept(self, concept_a: str, concept_b: str) -> Optional[str]:
        """
        寻找两个概念之间的桥梁概念

        一个桥梁概念是与A和B都有一定关联的中间概念，
        就像"寒→冬天→冷"中，"冬天"连接了"寒"和"冷"。

        Args:
            concept_a: 概念A
            concept_b: 概念B

        Returns:
            桥梁概念，若未找到则返回 None
        """
        emb_a = self._get_or_create_embedding(concept_a)
        emb_b = self._get_or_create_embedding(concept_b)

        # 中间向量 = (A + B) / 2
        midpoint = (emb_a + emb_b) / 2.0
        midpoint = F.normalize(midpoint, p=2, dim=0)

        best_bridge = None
        best_score = 0.0

        for concept, emb in self._concept_embeddings.items():
            if concept in (concept_a, concept_b):
                continue
            # 桥梁概念应该与中间向量有较高的相似度
            # 同时与两个端点都有一定的相似度
            sim_mid = torch.dot(midpoint, emb).item()
            sim_a = torch.dot(emb_a, emb).item()
            sim_b = torch.dot(emb_b, emb).item()

            # 综合评分：与中点的相似度 + 与两端点相似度的最小值
            bridge_score = sim_mid * 0.5 + min(sim_a, sim_b) * 0.5

            if bridge_score > best_score:
                best_score = bridge_score
                best_bridge = concept

        # 只有当桥梁质量足够好时才返回
        if best_score > 0.3:
            return best_bridge
        return None

    def _find_shared_features(self, concept_a: str, concept_b: str) -> List[str]:
        """
        找到两个概念的共同结构特征

        通过分析概念之间的关联网络，发现它们共享的邻接概念。

        Args:
            concept_a: 概念A
            concept_b: 概念B

        Returns:
            共同特征列表
        """
        # 找到A和B各自的相关概念
        neighbors_a = set(self._find_associated_concepts(concept_a, top_k=10))
        neighbors_b = set(self._find_associated_concepts(concept_b, top_k=10))

        # 共享的邻居就是共同特征
        shared = neighbors_a & neighbors_b

        # 过滤掉弱关联，返回最强的几个
        shared_list = [(c, s) for c, s in shared if s > 0.3]
        shared_list.sort(key=lambda x: x[1], reverse=True)

        return [c for c, s in shared_list[:3]]

    # ==================== 3. 远程联想测试 (Remote Associates Test) ====================

    def find_remote_association(
        self,
        word_a: str,
        word_b: str,
        word_c: str,
        top_k: int = 3,
    ) -> List[RemoteAssociationResult]:
        """
        远程联想测试 - 寻找连接三个无关词语的桥梁词

        基于 Mednick (1962) 的远程联想理论：
        创造力高的人更擅长找到三个看似无关词语之间的共同关联。
        例如：给出"寒/光/故事"→ 答案"月"（月光寒、月光明、月亮的故事）

        算法：在概念空间中找到一个与三个输入词距离均衡的概念
        （minimax 优化：最小化到最远输入词的距离）

        Args:
            word_a: 第一个词语
            word_b: 第二个词语
            word_c: 第三个词语
            top_k: 返回前 k 个候选

        Returns:
            RemoteAssociationResult 列表，按关联强度排序
        """
        # 确保 embedding 存在
        emb_a = self._get_or_create_embedding(word_a)
        emb_b = self._get_or_create_embedding(word_b)
        emb_c = self._get_or_create_embedding(word_c)

        input_words = [word_a, word_b, word_c]
        input_embs = torch.stack([emb_a, emb_b, emb_c])  # [3, dim]

        results = []

        for concept, emb in self._concept_embeddings.items():
            if concept in input_words:
                continue

            # 计算到三个输入词的距离
            distances = []
            for input_emb in input_embs:
                dist = 1.0 - torch.dot(emb, input_emb).item()
                distances.append(max(0.0, dist))

            # 计算到各个输入词的相似度
            similarities = {}
            for word, dist in zip(input_words, distances):
                similarities[word] = round(1.0 - dist, 4)

            # Minimax 评分：最小化最大距离（找最均衡的关联词）
            max_dist = max(distances)
            avg_dist = sum(distances) / len(distances)
            # 综合评分：最大距离越小越好，平均距离也要小
            minimax_score = -max_dist * 0.6 - avg_dist * 0.4

            results.append(RemoteAssociationResult(
                associate_word=concept,
                connections=similarities,
                minimax_score=round(minimax_score, 4),
            ))

        # 按评分降序排列
        results.sort(key=lambda x: x.minimax_score, reverse=True)
        return results[:top_k]

    # ==================== 4. 洞察检测 ====================

    def check_insight(
        self,
        new_connection: str,
        connecting_concepts: List[str],
        current_task_context: str = "",
    ) -> Optional[InsightRecord]:
        """
        检测是否发生了"灵光一现"的洞察

        洞察（Insight）的判定标准：
        1. 连接了2个以上之前不相关的领域
        2. 新颖度高（不仅仅是回忆已有知识）
        3. 实用度高（对当前任务有帮助）

        Args:
            new_connection: 新建立的连接/想法
            connecting_concepts: 被连接的概念列表
            current_task_context: 当前任务上下文（用于评估实用性）

        Returns:
            InsightRecord 如果检测到洞察，否则 None
        """
        if len(connecting_concepts) < 2:
            return None

        # 条件1: 检查概念是否来自不同领域
        if not self._are_different_domains(connecting_concepts):
            return None

        # 条件2: 评估新颖度
        novelty = self._evaluate_novelty(new_connection, connecting_concepts)
        if novelty < self.INSIGHT_NOVELTY_THRESHOLD:
            return None

        # 条件3: 评估概念间的领域距离（距离越大→越"跨域"→越有洞察感）
        max_domain_distance = self._compute_max_domain_distance(connecting_concepts)
        if max_domain_distance < self.INSIGHT_DOMAIN_DISTANCE:
            return None

        # 条件4: 评估实用度
        utility = self._evaluate_utility(new_connection, current_task_context)

        # 计算孵化贡献（如果这个洞察与某个孵化问题相关）
        incubation_time = self._check_incubation_contribution(new_connection, connecting_concepts)

        # 创建洞察记录
        insight = InsightRecord(
            insight=new_connection,
            connecting_concepts=connecting_concepts,
            novelty=round(novelty, 4),
            utility=round(utility, 4),
            incubation_time=incubation_time,
            timestamp=time.time(),
        )

        self._insight_history.append(insight)
        self._total_insights += 1

        logger.info(
            f"[洞察] 检测到Aha时刻! 概念={connecting_concepts}, "
            f"新颖度={novelty:.3f}, 实用度={utility:.3f}, "
            f"孵化贡献={incubation_time:.1f}s"
        )

        return insight

    def _evaluate_novelty(self, connection: str, concepts: List[str]) -> float:
        """
        评估一个想法的新颖度

        新颖度取决于：
        - 连接的概念之间的距离（越远越新颖）
        - 与已有洞察的差异度
        - 概念组合的罕见程度

        Args:
            connection: 连接内容
            concepts: 涉及的概念

        Returns:
            新颖度 [0, 1]
        """
        if len(concepts) < 2:
            return 0.3

        # 1. 概念间距离贡献新颖度
        pairwise_distances = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                dist = self._compute_concept_distance(concepts[i], concepts[j])
                pairwise_distances.append(dist)

        if not pairwise_distances:
            return 0.3

        avg_distance = sum(pairwise_distances) / len(pairwise_distances)
        # 距离越大越新颖，归一化到 [0, 1]
        distance_novelty = min(1.0, avg_distance / 1.5)

        # 2. 与历史洞察的差异度
        history_novelty = 1.0  # 默认最高（没有历史记录时）
        if self._insight_history:
            # 检查是否与最近的洞察过于相似
            recent_concepts = set()
            for record in list(self._insight_history)[-10:]:
                recent_concepts.update(record.connecting_concepts)

            overlap = len(set(concepts) & recent_concepts)
            history_novelty = 1.0 - (overlap / max(len(concepts), 1)) * 0.5

        # 3. 跨领域加分
        domain_bonus = 0.1 if self._are_different_domains(concepts) else 0.0

        novelty = distance_novelty * 0.5 + history_novelty * 0.4 + domain_bonus
        return min(1.0, max(0.0, novelty))

    def _evaluate_utility(self, connection: str, task_context: str) -> float:
        """
        评估一个想法对当前任务的实用度

        Args:
            connection: 想法内容
            task_context: 任务上下文

        Returns:
            实用度 [0, 1]
        """
        if not task_context:
            return 0.5  # 没有上下文时返回中等实用度

        # 基于概念重叠的简单实用度评估
        connection_keywords = set(self._extract_keywords(connection))
        context_keywords = set(self._extract_keywords(task_context))

        if not connection_keywords or not context_keywords:
            return 0.4

        overlap = connection_keywords & context_keywords
        # 有一定重叠但不完全一致 → 最有用
        overlap_ratio = len(overlap) / max(len(connection_keywords), 1)

        # 最佳状态：30%-70% 的重叠（既有相关性又有新意）
        if 0.3 <= overlap_ratio <= 0.7:
            utility = 0.6 + overlap_ratio * 0.3
        elif overlap_ratio > 0:
            utility = 0.3 + overlap_ratio * 0.3
        else:
            utility = 0.2

        return min(1.0, max(0.0, utility))

    def _compute_max_domain_distance(self, concepts: List[str]) -> float:
        """
        计算一组概念中最大的领域间距离

        Args:
            concepts: 概念列表

        Returns:
            最大领域距离 [0, 2]
        """
        if len(concepts) < 2:
            return 0.0

        max_dist = 0.0
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                dist = self._compute_concept_distance(concepts[i], concepts[j])
                max_dist = max(max_dist, dist)

        return max_dist

    def _check_incubation_contribution(
        self, insight: str, concepts: List[str]
    ) -> float:
        """
        检查一个洞察是否帮助解决了某个孵化中的问题

        如果洞察与某个孵化问题相关，记录其孵化时间贡献。

        Args:
            insight: 洞察内容
            concepts: 相关概念

        Returns:
            孵化时间（秒），如果无关则返回 0
        """
        if not self._incubation_problems:
            return 0.0

        insight_keywords = set(self._extract_keywords(insight))
        insight_concept_set = set(concepts)

        for problem in self._incubation_problems:
            if problem.resolved:
                continue

            problem_keywords = set(self._extract_keywords(problem.problem))
            problem_context_keywords = set(self._extract_keywords(problem.context))

            # 计算与问题的关联度
            overlap_problem = insight_keywords & problem_keywords
            overlap_context = insight_keywords & problem_context_keywords
            concept_overlap = insight_concept_set & problem_keywords

            total_relevance = len(overlap_problem) + len(overlap_context) + len(concept_overlap)

            if total_relevance >= 2:
                # 洞察解决了孵化问题！
                problem.solution = insight
                problem.resolved = True
                self._incubation_resolution_count += 1

                incubation_time = problem.incubation_elapsed()
                logger.info(
                    f"[孵化] 问题「{problem.problem[:30]}」在孵化 "
                    f"{incubation_time:.0f}s 后获得解答!"
                )
                return incubation_time

        return 0.0

    # ==================== 5. 创意组合引擎 ====================

    def combine_concepts(
        self,
        concept_a: str,
        concept_b: str,
        templates: Optional[List[CombinationTemplate]] = None,
    ) -> List[CreativeIdea]:
        """
        创意组合 - 将两个概念融合为新颖组合

        使用多种模板进行概念融合：
        - 应用型: "A使用B"
        - 类比型: "A像B"
        - 融合型: "A遇上B"
        - 改良型: "A但带有B"
        - 透视型: "通过B的视角看A"

        每个组合都会评估新颖性和连贯性。

        Args:
            concept_a: 概念A
            concept_b: 概念B
            templates: 指定使用的组合模板，None 则使用全部

        Returns:
            CreativeIdea 列表，按综合评分降序
        """
        self._combination_count += 1

        if templates is None:
            templates = list(CombinationTemplate)

        # 确保 embedding 存在
        self._get_or_create_embedding(concept_a)
        self._get_or_create_embedding(concept_b)

        # 计算概念间的距离和相似度
        similarity = self._compute_similarity(concept_a, concept_b)
        distance = self._compute_concept_distance(concept_a, concept_b)

        # 跨域组合更新关联
        is_cross_domain = self._are_different_domains([concept_a, concept_b])

        combinations = []
        for template in templates:
            idea = self._apply_combination_template(
                concept_a, concept_b, template, similarity, is_cross_domain
            )
            idea.source_concepts = [concept_a, concept_b]
            idea.timestamp = time.time()
            combinations.append(idea)

        # 检查是否产生洞察
        self.check_insight(
            new_connection=f"{concept_a} + {concept_b} 的创意组合",
            connecting_concepts=[concept_a, concept_b],
        )

        combinations.sort(key=lambda x: x.overall_score(), reverse=True)
        return combinations

    def _apply_combination_template(
        self,
        concept_a: str,
        concept_b: str,
        template: CombinationTemplate,
        similarity: float,
        is_cross_domain: bool,
    ) -> CreativeIdea:
        """
        应用单个组合模板

        Args:
            concept_a: 概念A
            concept_b: 概念B
            template: 组合模板
            similarity: 概念间相似度
            is_cross_domain: 是否跨域组合

        Returns:
            CreativeIdea
        """
        # 跨域组合的新颖度加成
        cross_domain_bonus = 0.15 if is_cross_domain else 0.0

        # 距离适中的组合更有创造性（太近=太普通，太远=不可理解）
        distance = 1.0 - similarity
        optimal_distance_bonus = 0.1 * math.exp(-(distance - 0.8) ** 2 / 0.2)

        if template == CombinationTemplate.APPLICATION:
            content = f"用{concept_b}的方式来增强{concept_a}——将{concept_b}的核心理念应用到{concept_a}的场景中，可能产生新的突破"
            base_novelty = 0.5
            base_coherence = 0.6

        elif template == CombinationTemplate.ANALOGY:
            content = f"{concept_a}的本质就像{concept_b}——如果我们理解了{concept_b}的运作方式，就能更深刻地理解{concept_a}"
            base_novelty = 0.45
            base_coherence = 0.65

        elif template == CombinationTemplate.FUSION:
            content = f"当{concept_a}遇上{concept_b}——两者的融合创造了一种全新的可能性，既有{concept_a}的深度，又有{concept_b}的广度"
            base_novelty = 0.6
            base_coherence = 0.5

        elif template == CombinationTemplate.IMPROVEMENT:
            content = f"{concept_a}的问题，也许可以用{concept_b}来解决——给{concept_a}注入{concept_b}的元素，就像给它升级一样"
            base_novelty = 0.55
            base_coherence = 0.55

        elif template == CombinationTemplate.PERSPECTIVE:
            content = f"通过{concept_b}的视角来审视{concept_a}——换一个完全不同的观察框架，{concept_a}展现出了我们从未注意到的面貌"
            base_novelty = 0.65
            base_coherence = 0.45

        else:
            content = f"{concept_a}和{concept_b}的某种结合也许值得探索"
            base_novelty = 0.4
            base_coherence = 0.4

        return CreativeIdea(
            content=content,
            strategy=f"combination_{template.value}",
            novelty_score=round(min(1.0, base_novelty + cross_domain_bonus + optimal_distance_bonus + random.uniform(0, 0.1)), 4),
            coherence_score=round(min(1.0, base_coherence + random.uniform(-0.05, 0.1)), 4),
            utility_score=round(min(1.0, 0.4 + random.uniform(0, 0.25) + cross_domain_bonus), 4),
        )

    # ==================== 6. 孵化系统 ====================

    def start_incubation(self, problem: str, context: str = "") -> bool:
        """
        将一个未解决的问题放入孵化队列

        模拟 Wallas (1926) 创造过程的"孵化"阶段：
        - 将问题搁置一段时间
        - 在后台（特别是"休息"状态下）继续处理
        - 创造性解决方案更可能在孵化后出现

        Args:
            problem: 问题描述
            context: 问题上下文

        Returns:
            是否成功加入孵化队列
        """
        # 检查是否已在队列中
        for existing in self._incubation_problems:
            if existing.problem == problem and not existing.resolved:
                logger.debug(f"[孵化] 问题已在孵化队列中: {problem[:30]}")
                return False

        # 检查队列是否已满
        active_problems = [p for p in self._incubation_problems if not p.resolved]
        if len(active_problems) >= self.max_incubation_problems:
            # 移除最旧的未解决问题
            oldest = min(active_problems, key=lambda p: p.start_time)
            oldest.resolved = True
            oldest.solution = "孵化超时，自动关闭"
            logger.info(f"[孵化] 队列已满，关闭最旧问题: {oldest.problem[:30]}")

        # 确保概念 embedding 存在
        keywords = self._extract_keywords(problem)
        for kw in keywords:
            self._get_or_create_embedding(kw)

        incubation = IncubationProblem(
            problem=problem,
            context=context,
            start_time=time.time(),
            attempts=0,
        )
        self._incubation_problems.append(incubation)

        logger.info(f"[孵化] 新问题加入孵化队列: {problem[:50]}, 当前活跃数: {len(active_problems) + 1}")
        return True

    def attempt_incubation_resolution(
        self, mind_state: Optional[str] = None
    ) -> List[CreativeIdea]:
        """
        尝试解决孵化中的问题

        在"休息"（RESTING）状态下，创造性解决方案更可能浮现。
        孵化效果随时间增长，但超过一定时间后递减。

        Args:
            mind_state: 当前的思维状态（若为 RESTING 则效果加倍）

        Returns:
            尝试产生的创意想法列表
        """
        if not self._incubation_problems:
            return []

        active_problems = [p for p in self._incubation_problems if not p.resolved]
        if not active_problems:
            return []

        ideas = []

        for problem in active_problems:
            elapsed = problem.incubation_elapsed()
            problem.attempts += 1

            # 计算孵化效果因子
            incubation_factor = self._compute_incubation_effectiveness(elapsed)

            # RESTING 状态的加成效果
            state_bonus = 1.0
            if mind_state == "resting":
                state_bonus = 1.5  # 休息状态孵化效率提升50%

            # 孵化效果太低时跳过（还没到"成熟"时间）
            if incubation_factor * state_bonus < 0.3:
                continue

            # 产生创意解决方案
            resolution_ideas = self._generate_incubation_ideas(problem, incubation_factor * state_bonus)
            ideas.extend(resolution_ideas)

        return ideas

    def _compute_incubation_effectiveness(self, elapsed_seconds: float) -> float:
        """
        计算孵化效果因子

        孵化效果曲线：
        - 0~optimal_time: 效果线性增长（准备阶段完成，潜意识在整合）
        - optimal_time~decay_start: 效果达到峰值平台期
        - decay_start之后: 效果缓慢下降（记忆衰退、上下文丢失）

        Args:
            elapsed_seconds: 已孵化时间（秒）

        Returns:
            效果因子 [0, 1]
        """
        if elapsed_seconds <= 0:
            return 0.1

        if elapsed_seconds < self.OPTIMAL_INCUBATION_TIME:
            # 快速上升期
            return 0.3 + 0.5 * (elapsed_seconds / self.OPTIMAL_INCUBATION_TIME)
        elif elapsed_seconds < self.INCUBATION_DECAY_START:
            # 平台期
            return 0.8 + 0.2 * random.random()
        else:
            # 缓慢衰减
            decay_ratio = (elapsed_seconds - self.INCUBATION_DECAY_START) / (
                self.MAX_INCUBATION_TIME - self.INCUBATION_DECAY_START
            )
            return max(0.2, 0.8 - decay_ratio * 0.6)

    def _generate_incubation_ideas(
        self, problem: IncubationProblem, effectiveness: float
    ) -> List[CreativeIdea]:
        """
        为孵化中的问题生成解决方案创意

        Args:
            problem: 孵化中的问题
            effectiveness: 孵化效果因子

        Returns:
            CreativeIdea 列表
        """
        ideas = []

        # 策略1: 用发散思维重新审视问题
        brainstorm_ideas = self.brainstorm(
            problem.problem,
            num_ideas=3,
            strategies=[
                ThinkingStrategy.RANDOM_ASSOCIATION,
                ThinkingStrategy.REVERSAL,
                ThinkingStrategy.PERSPECTIVE_SHIFT,
            ],
        )
        for idea in brainstorm_ideas:
            # 孵化效果提升质量
            idea.utility_score = min(1.0, idea.utility_score * effectiveness)
            ideas.append(idea)

        # 策略2: 尝试与上下文中的概念组合
        if problem.context:
            context_keywords = self._extract_keywords(problem.context)
            problem_keywords = self._extract_keywords(problem.problem)

            for pk in problem_keywords[:2]:
                for ck in context_keywords[:2]:
                    if pk != ck:
                        combos = self.combine_concepts(pk, ck, templates=[
                            CombinationTemplate.FUSION,
                            CombinationTemplate.IMPROVEMENT,
                        ])
                        for combo in combos:
                            combo.utility_score = min(1.0, combo.utility_score * effectiveness)
                            ideas.append(combo)

        return ideas

    def get_incubation_status(self) -> List[Dict[str, Any]]:
        """
        获取所有孵化问题的状态

        Returns:
            孵化状态列表
        """
        return [p.to_dict() for p in self._incubation_problems]

    # ==================== 7. 隐喻生成 ====================

    def generate_metaphor(
        self,
        source: str,
        target: str,
        style: str = "structural",
    ) -> CreativeIdea:
        """
        隐喻生成 - 在两个领域之间建立类比映射

        隐喻是人类最基本也最强大的认知工具之一。
        例如："记忆像大海"（记忆=source, 大海=target）
        - 海洋有深浅 → 记忆有深浅
        - 海洋有潮汐 → 记忆有起伏
        - 海洋有深处不可及之处 → 记忆有遗忘的角落

        Args:
            source: 源领域（被解释的对象）
            target: 目标领域（用作类比的对象）
            style: 隐喻风格 ("structural" | "functional" | "emotional")

        Returns:
            CreativeIdea 包含隐喻内容
        """
        self._metaphor_count += 1

        # 确保 embedding 存在
        self._get_or_create_embedding(source)
        self._get_or_create_embedding(target)

        # 生成解释性内容（基于两个领域的结构映射）
        explanation = self._generate_metaphor_explanation(source, target)

        # 确定源/目标领域的名称
        source_domain = self._concept_domains.get(source, source)
        target_domain = self._concept_domains.get(target, target)

        # 选择模板
        style_key = style if style in self._metaphor_templates else "structural"
        template = random.choice(self._metaphor_templates[style_key])

        # 填充模板
        content = template.format(
            source=source,
            target=target,
            explanation=explanation,
            domain_a=source_domain,
            domain_b=target_domain,
        )

        # 评估隐喻质量
        is_cross_domain = self._are_different_domains([source, target])
        novelty = 0.5 + (0.2 if is_cross_domain else 0.0) + random.uniform(0, 0.15)
        coherence = 0.5 + random.uniform(0, 0.25)

        return CreativeIdea(
            content=content,
            strategy=f"metaphor_{style}",
            novelty_score=round(min(1.0, novelty), 4),
            coherence_score=round(min(1.0, coherence), 4),
            utility_score=round(0.4 + random.uniform(0, 0.2), 4),
            source_concepts=[source, target],
            timestamp=time.time(),
        )

    def _generate_metaphor_explanation(self, source: str, target: str) -> str:
        """
        生成隐喻的解释性内容

        通过分析两个概念之间的结构相似性来产生解释。

        Args:
            source: 源概念
            target: 目标概念

        Returns:
            解释文本
        """
        # 找到桥梁概念作为解释基础
        bridge = self._find_bridge_concept(source, target)

        if bridge:
            # 找到源和目标各自与桥梁的关联
            source_bridge_sim = self._compute_similarity(source, bridge)
            target_bridge_sim = self._compute_similarity(target, bridge)

            explanations = [
                f"它们都有「{bridge}」的特质——{source}在这方面表现为内在结构，"
                f"{target}则体现为外在形态",
                f"「{bridge}」是理解这两者共通之处的关键——"
                f"就像{source}离不开{bridge}，{target}也是如此",
                f"从「{bridge}」的角度看，{source}和{target}遵循着相似的底层逻辑",
            ]
        else:
            # 没有明确的桥梁，使用距离特征
            distance = self._compute_concept_distance(source, target)
            explanations = [
                f"虽然看似遥远（距离{distance:.2f}），但内在有某种深层的同构关系",
                f"表面的差异之下隐藏着结构上的呼应——这正是跨领域洞察的来源",
                f"两者的核心逻辑在抽象层面上出奇地一致",
            ]

        return random.choice(explanations)

    def auto_generate_metaphors(
        self,
        concept: str,
        num_metaphors: int = 3,
    ) -> List[CreativeIdea]:
        """
        为一个概念自动生成多个隐喻

        从不同领域选取目标概念，为源概念生成多种隐喻。

        Args:
            concept: 源概念
            num_metaphors: 隐喻数量

        Returns:
            CreativeIdea 列表
        """
        concept_domain = self._concept_domains.get(concept, "")

        # 优先从不同领域选取目标
        other_domain_concepts = []
        for domain, concepts in self._domain_concepts.items():
            if domain != concept_domain:
                other_domain_concepts.extend(concepts)

        if not other_domain_concepts:
            other_domain_concepts = self._random_concept_pool[:]

        # 选取距离适中的目标概念
        candidates = []
        for target in other_domain_concepts:
            dist = self._compute_concept_distance(concept, target)
            if 0.4 < dist < 1.3:
                candidates.append((target, dist))

        if not candidates:
            # 回退：使用任意不同概念
            selected_targets = random.sample(
                other_domain_concepts, min(num_metaphors, len(other_domain_concepts))
            )
        else:
            # 从距离适中的候选中选取
            selected_targets = [c[0] for c in random.sample(
                candidates, min(num_metaphors, len(candidates))
            )]

        metaphors = []
        styles = ["structural", "functional", "emotional"]

        for i, target in enumerate(selected_targets):
            style = styles[i % len(styles)]
            metaphor = self.generate_metaphor(concept, target, style=style)
            metaphors.append(metaphor)

        metaphors.sort(key=lambda x: x.overall_score(), reverse=True)
        return metaphors

    # ==================== 8. InnerThoughtEngine 集成 ====================

    def set_mind_state(self, mind_state):
        """
        设置当前思维状态（由 InnerThoughtEngine 注入）

        Args:
            mind_state: MindState 枚举值或字符串
        """
        if mind_state is not None:
            self._mind_state = mind_state.value if hasattr(mind_state, 'value') else str(mind_state)

    def get_creative_spark(self) -> Optional[str]:
        """
        获取创造性火花 - 在 WANDERING 状态下提供灵感

        当 InnerThoughtEngine 处于漫游状态时，
        随机提供一个创造性想法，注入到思维流中。

        Returns:
            创造性想法字符串，或 None（如果没有合适的灵感）
        """
        if self._mind_state != "wandering":
            return None

        # 30% 的概率在漫游时提供创造性火花
        if random.random() > 0.3:
            return None

        spark_strategies = [
            self._spark_random_combination,
            self._spark_distant_association,
            self._spark_incubation_hint,
            self._spark_metaphor_flash,
        ]

        strategy = random.choice(spark_strategies)
        return strategy()

    def get_creative_alternatives(self, current_thought: str) -> List[str]:
        """
        获取创造性替代方案 - 在 REFLECTING 状态下提供建议

        当 InnerThoughtEngine 处于反思状态时，
        分析当前思维内容，提供替代视角。

        Args:
            current_thought: 当前的思维内容

        Returns:
            替代方案列表
        """
        if self._mind_state != "reflecting":
            return []

        alternatives = []

        # 从当前思维中提取关键词
        keywords = self._extract_keywords(current_thought)
        if not keywords:
            return []

        # 为关键词寻找不同的联想方向
        for kw in keywords[:3]:
            associated = self._find_associated_concepts(kw, top_k=2)
            for assoc_concept, sim in associated:
                if sim < 0.7:  # 选择有一定距离的关联（太近的不够有创意）
                    alt = f"换个角度想：关于「{kw}」，也许还可以从「{assoc_concept}」的角度来理解？"
                    alternatives.append(alt)

        return alternatives[:3]

    def check_association_chain_insight(self, association_chain: List[str]) -> Optional[InsightRecord]:
        """
        检查联想链是否产生了远距离连接（触发洞察）

        当 InnerThoughtEngine 的联想链连接了两个本不相关的概念时，
        可能产生了有价值的洞察。

        Args:
            association_chain: InnerThoughtEngine 的联想链

        Returns:
            InsightRecord 如果检测到远距离连接，否则 None
        """
        if len(association_chain) < 3:
            return None

        # 检查链首和链尾之间的距离
        start = association_chain[0]
        end = association_chain[-1]

        distance = self._compute_concept_distance(start, end)

        # 如果首尾距离很远（> 1.0），说明联想跨越了较大的概念空间
        if distance > 1.0:
            # 生成洞察描述
            middle_concepts = association_chain[1:-1] if len(association_chain) > 2 else []
            middle_str = " → ".join(middle_concepts[-3:]) if middle_concepts else "直接跳跃"

            insight_text = (
                f"联想链跨越了较大距离：{start} → ...({middle_str})... → {end}。"
                f"这种远距离联想可能蕴含创造性连接。"
            )

            insight = self.check_insight(
                new_connection=insight_text,
                connecting_concepts=[start, end] + middle_concepts[:2],
            )
            return insight

        return None

    # ---------- 创造性火花生成（内部方法）----------

    def _spark_random_combination(self) -> Optional[str]:
        """随机组合火花"""
        try:
            # 从两个不同领域各选一个概念进行组合
            domains = list(self._domain_concepts.keys())
            if len(domains) < 2:
                return None

            domain_a, domain_b = random.sample(domains, 2)
            concept_a = random.choice(self._domain_concepts[domain_a])
            concept_b = random.choice(self._domain_concepts[domain_b])

            templates = [
                f"突然想到：{concept_a}和{concept_b}之间是不是有某种联系？",
                f"有意思……如果把{concept_a}和{concept_b}放在一起思考……",
                f"不知为何，{concept_a}让我想到了{concept_b}……",
            ]
            return random.choice(templates)
        except Exception:
            return None

    def _spark_distant_association(self) -> Optional[str]:
        """远距离联想火花"""
        try:
            concept = random.choice(self._random_concept_pool)
            distant = self._find_associated_concepts(concept, top_k=20)

            # 找距离最远的几个
            far_concepts = [(c, s) for c, s in distant if s < 0.3]
            if not far_concepts:
                return None

            far_c, far_s = random.choice(far_concepts)
            return f"脑海里突然闪过：{concept}和{far_c}……这两个东西有关联吗？"
        except Exception:
            return None

    def _spark_incubation_hint(self) -> Optional[str]:
        """孵化提示火花"""
        active = [p for p in self._incubation_problems if not p.resolved]
        if not active:
            return None

        problem = random.choice(active)
        elapsed = problem.incubation_elapsed()

        # 只在有足够孵化时间后提示
        if elapsed < 60:
            return None

        elapsed_str = f"{elapsed / 60:.0f}分钟" if elapsed < 3600 else f"{elapsed / 3600:.1f}小时"
        hints = [
            f"（关于「{problem.problem[:20]}」这个问题，已经想了{elapsed_str}了，也许答案就在某个意想不到的地方……）",
            f"（潜意识似乎还在处理「{problem.problem[:20]}」……不急，让它在后台继续发酵）",
        ]
        return random.choice(hints)

    def _spark_metaphor_flash(self) -> Optional[str]:
        """隐喻闪现"""
        try:
            concept = random.choice(self._random_concept_pool)
            metaphors = self.auto_generate_metaphors(concept, num_metaphors=1)
            if metaphors:
                return f"（脑海中闪过一个比喻：{metaphors[0].content[:60]}……）"
        except Exception:
            pass
        return None

    # ==================== 9. 序列化与统计 ====================

    def get_state(self) -> Dict[str, Any]:
        """
        获取引擎完整状态（用于序列化保存）

        Returns:
            状态字典
        """
        # 序列化概念 embedding
        concept_embeddings_serialized = {}
        for concept, emb in self._concept_embeddings.items():
            concept_embeddings_serialized[concept] = (
                emb.detach().cpu().float().numpy().tolist()
            )

        return {
            "embedding_dim": self.embedding_dim,
            "concept_embeddings": concept_embeddings_serialized,
            "concept_domains": dict(self._concept_domains),
            "insight_history": [r.to_dict() for r in self._insight_history],
            "total_insights": self._total_insights,
            "incubation_problems": [p.to_dict() for p in self._incubation_problems],
            "association_links": {
                k: [(c, s) for c, s in v]
                for k, v in self._association_links.items()
            },
            "brainstorm_count": self._brainstorm_count,
            "combination_count": self._combination_count,
            "metaphor_count": self._metaphor_count,
            "incubation_resolution_count": self._incubation_resolution_count,
        }

    def set_state(self, state: Dict[str, Any]):
        """
        从状态字典恢复引擎（用于反序列化加载）

        Args:
            state: 状态字典（来自 get_state()）
        """
        # 恢复概念 embedding
        self._concept_embeddings = {}
        for concept, emb_list in state.get("concept_embeddings", {}).items():
            if isinstance(emb_list, list):
                self._concept_embeddings[concept] = torch.tensor(
                    emb_list, dtype=torch.float32, device=self.device
                )
            elif isinstance(emb_list, torch.Tensor):
                self._concept_embeddings[concept] = emb_list.to(self.device)

        # 恢复概念领域映射
        self._concept_domains = state.get("concept_domains", {})

        # 恢复洞察历史
        self._insight_history = deque(maxlen=self._insight_history.maxlen)
        for record_dict in state.get("insight_history", []):
            record = InsightRecord(
                insight=record_dict.get("insight", ""),
                connecting_concepts=record_dict.get("connecting_concepts", []),
                novelty=record_dict.get("novelty", 0.0),
                utility=record_dict.get("utility", 0.0),
                incubation_time=record_dict.get("incubation_time", 0.0),
                timestamp=record_dict.get("timestamp", 0.0),
            )
            self._insight_history.append(record)

        self._total_insights = state.get("total_insights", 0)

        # 恢复孵化问题
        self._incubation_problems = []
        for prob_dict in state.get("incubation_problems", []):
            problem = IncubationProblem(
                problem=prob_dict.get("problem", ""),
                context=prob_dict.get("context", ""),
                start_time=prob_dict.get("start_time", 0.0),
                attempts=prob_dict.get("attempts", 0),
                solution=prob_dict.get("solution"),
                resolved=prob_dict.get("resolved", False),
            )
            self._incubation_problems.append(problem)

        # 恢复关联链接
        self._association_links = {}
        for concept, links in state.get("association_links", {}).items():
            self._association_links[concept] = [(c, s) for c, s in links]

        # 恢复计数器
        self._brainstorm_count = state.get("brainstorm_count", 0)
        self._combination_count = state.get("combination_count", 0)
        self._metaphor_count = state.get("metaphor_count", 0)
        self._incubation_resolution_count = state.get("incubation_resolution_count", 0)

    def get_stats(self) -> Dict[str, Any]:
        """
        获取引擎统计信息

        Returns:
            统计信息字典
        """
        active_incubation = [p for p in self._incubation_problems if not p.resolved]
        resolved_incubation = [p for p in self._incubation_problems if p.resolved]

        # 最近洞察
        recent_insights = [r.to_dict() for r in list(self._insight_history)[-5:]]

        # 概念空间统计
        num_concepts = len(self._concept_embeddings)
        num_domains = len(set(self._concept_domains.values()))

        return {
            "brainstorm_count": self._brainstorm_count,
            "combination_count": self._combination_count,
            "metaphor_count": self._metaphor_count,
            "total_insights": self._total_insights,
            "recent_insights": recent_insights,
            "active_incubation_count": len(active_incubation),
            "resolved_incubation_count": len(resolved_incubation),
            "incubation_resolution_count": self._incubation_resolution_count,
            "concept_space_size": num_concepts,
            "domain_count": num_domains,
            "mind_state": self._mind_state,
            "incubation_problems": [p.to_dict() for p in active_incubation[:5]],
        }

    # ==================== 10. 前向传播（兼容 nn.Module）====================

    def forward(
        self,
        topic: str,
        mode: str = "brainstorm",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        前向传播接口（兼容 nn.Module）

        Args:
            topic: 输入主题
            mode: 操作模式 ("brainstorm" | "combine" | "metaphor" | "remote_association")
            **kwargs: 额外参数

        Returns:
            结果字典列表
        """
        if mode == "brainstorm":
            ideas = self.brainstorm(topic, **kwargs)
            return [idea.to_dict() for idea in ideas]

        elif mode == "combine":
            concept_b = kwargs.get("concept_b", "")
            if not concept_b:
                return []
            ideas = self.combine_concepts(topic, concept_b)
            return [idea.to_dict() for idea in ideas]

        elif mode == "metaphor":
            target = kwargs.get("target", "")
            if not target:
                # 自动生成
                metaphors = self.auto_generate_metaphors(topic)
            else:
                metaphors = [self.generate_metaphor(topic, target)]
            return [m.to_dict() for m in metaphors]

        elif mode == "remote_association":
            word_b = kwargs.get("word_b", "")
            word_c = kwargs.get("word_c", "")
            if not word_b or not word_c:
                return []
            results = self.find_remote_association(topic, word_b, word_c)
            return [
                {
                    "associate_word": r.associate_word,
                    "connections": r.connections,
                    "minimax_score": r.minimax_score,
                }
                for r in results
            ]

        else:
            logger.warning(f"[CreativeInsight] 未知模式: {mode}")
            return []
