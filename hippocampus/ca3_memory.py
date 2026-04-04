"""
CA3 区 - 情景记忆库 + 模式补全单元

功能:
- 以「记忆 ID+10ms 级时间戳 + 时序骨架 + 语义指针 + 因果关联」格式存储情景记忆
- 仅存指针不存完整文本 (节省内存)
- 基于部分线索完成完整记忆链条召回
- 每个周期输出 1-2 个最相关记忆锚点

[优化] 人类记忆增强:
- 语义摘要: 存储语义摘要/关键实体，而非仅池化特征
- Embedding 召回: 使用模型自身 embedding 做语义匹配，替代正则关键词
- 记忆分层: 短期→中期→长期，像人脑那样固化重要记忆
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
from collections import OrderedDict
import logging

from .memory_layers import MemoryTier, MemoryConsolidationManager, TierConfig
from .semantic_engine import SemanticSummarizer

logger = logging.getLogger(__name__)


@dataclass
class EpisodicMemory:
    """情景记忆数据结构 - 增强版（人类记忆模拟）"""
    memory_id: str                    # 唯一记忆 ID(DG 正交化生成)
    timestamp: int                    # 10ms 级时间戳
    temporal_skeleton: str            # 时序骨架 (前后 token 关系)
    semantic_pointer: str             # 语义指针 (不存完整文本)
    causal_links: List[str] = field(default_factory=list)  # 因果关联列表
    activation_strength: float = 1.0  # 激活强度 (STDP 权重)
    is_core: bool = False             # 核心记忆标记
    content: str = ""                 # 完整内容
    dg_features: Optional[torch.Tensor] = None  # DG 分离后的特征
    # 用于窄带宽注意力的 KV 特征
    key_features: Optional[torch.Tensor] = None    # [num_heads, head_dim]
    value_features: Optional[torch.Tensor] = None  # [num_heads, head_dim]
    # ===== [新增] 人类记忆增强字段 =====
    semantic_summary: str = ""        # 语义摘要（由模型生成，非原始文本）
    key_entities: str = ""            # 关键实体（人名/地名/数字，管道符分隔）
    emotion_tag: str = "中性"          # 情感标签（正面/负面/中性）
    semantic_embedding: Optional[torch.Tensor] = None  # 语义 embedding 向量（用于语义召回）
    tier: MemoryTier = MemoryTier.SHORT_TERM  # 记忆层级（短期/中期/长期）
    recall_count: int = 0             # 被成功召回的次数
    consecutive_misses: int = 0        # 连续未被召回的次数（用于降级判定）
    
    def to_dict(self) -> dict:
        """转换为字典 - 避免序列化torch.Tensor"""
        result = {
            'memory_id': self.memory_id,
            'timestamp': self.timestamp,
            'temporal_skeleton': self.temporal_skeleton,
            'semantic_pointer': self.semantic_pointer,
            'causal_links': self.causal_links,
            'activation_strength': self.activation_strength,
            'is_core': self.is_core,
            'content': self.content,
            # 人类记忆增强字段
            'semantic_summary': self.semantic_summary,
            'key_entities': self.key_entities,
            'emotion_tag': self.emotion_tag,
            'tier': int(self.tier),
            'recall_count': self.recall_count,
            'consecutive_misses': self.consecutive_misses,
        }
        
        # 安全序列化tensor：detach, 转float32, 再转numpy（避免BFloat16）
        if self.dg_features is not None:
            result['dg_features'] = self.dg_features.detach().cpu().float().numpy().tolist()
        
        if self.key_features is not None:
            result['key_features'] = self.key_features.detach().cpu().float().numpy().tolist()
        
        if self.value_features is not None:
            result['value_features'] = self.value_features.detach().cpu().float().numpy().tolist()
        
        # 序列化语义 embedding
        if self.semantic_embedding is not None:
            result['semantic_embedding'] = self.semantic_embedding.detach().cpu().float().numpy().tolist()
        
        return result


class CA3EpisodicMemory(nn.Module):
    """
    CA3 情景记忆库（增强版）
    
    使用循环缓存管理记忆存储，支持:
    - 快速存储与检索
    - 模式补全 (基于部分线索召回完整记忆)
    - 记忆修剪 (遗忘弱记忆)
    - [新增] Embedding 语义匹配召回
    - [新增] 记忆分层（短期/中期/长期）
    - [新增] 语义摘要存储
    """
    
    def __init__(
        self,
        max_capacity: int = 10000,
        feature_dim: int = 128,
        timestamp_precision_ms: int = 10,
        recall_threshold: float = 0.7,
        decay_rate: float = 0.999,
        semantic_engine: Optional[SemanticSummarizer] = None,
        tier_config: Optional[TierConfig] = None,
    ):
        super().__init__()
        self.max_capacity = max_capacity
        self.feature_dim = feature_dim
        self.timestamp_precision_ms = timestamp_precision_ms
        self.recall_threshold = recall_threshold
        self.decay_rate = decay_rate
        
        # ========== 记忆存储 (有序字典实现循环缓存) ==========
        self.memories: OrderedDict[str, EpisodicMemory] = OrderedDict()
        
        # ========== 索引结构 ==========
        self.time_index: Dict[int, List[str]] = {}
        self.semantic_index: Dict[str, str] = {}
        
        # ========== [新增] 语义引擎和分层管理器 ==========
        self.semantic_engine = semantic_engine
        self.consolidation_manager = MemoryConsolidationManager(tier_config)
        
        # ========== 当前时间戳 ==========
        self.current_timestamp = 0
        self.last_recall_trace: List[Dict] = []
        self.recall_count: int = 0
        self.last_recall_time: float = 0.0
    
    def store(
        self,
        memory_id: str,
        timestamp: int,
        semantic_pointer: str,
        temporal_skeleton: str = "",
        causal_links: List[str] = None,
        dg_features: Optional[torch.Tensor] = None,
        is_core: bool = False,
        content: str = "",
        key_features: Optional[torch.Tensor] = None,
        value_features: Optional[torch.Tensor] = None,
        # [新增] 人类记忆增强参数
        semantic_summary: str = "",
        key_entities: str = "",
        emotion_tag: str = "中性",
        user_input: str = "",
        ai_response: str = "",
    ) -> EpisodicMemory:
        """存储情景记忆（增强版：含语义摘要、embedding、记忆分层）"""
        # 0. 生成语义摘要（如果语义引擎可用）
        if not semantic_summary and self.semantic_engine and user_input:
            try:
                summary_result = self.semantic_engine.generate_semantic_summary(
                    user_input=user_input,
                    ai_response=ai_response,
                    is_core=is_core
                )
                semantic_summary = summary_result['semantic_summary']
                key_entities = summary_result['key_entities']
                emotion_tag = summary_result['emotion_tag']
            except Exception as e:
                logger.debug(f"[CA3] 语义摘要生成失败: {e}")
        
        # 0.5 生成语义 embedding（如果语义引擎可用，带重试逻辑）
        semantic_embedding = None
        if self.semantic_engine:
            embed_text = key_entities if key_entities else (semantic_summary if semantic_summary else content)
            if embed_text:
                # Fix 2: Retry up to 3 times with progressively shorter text
                truncation_lengths = [None, 100, 50, 20]
                for trunc_len in truncation_lengths:
                    try:
                        _text = embed_text[:trunc_len] if trunc_len else embed_text
                        semantic_embedding = self.semantic_engine.get_text_embedding(_text)
                        if semantic_embedding is not None:
                            break
                    except Exception as e:
                        logger.debug(f"[CA3] 语义embedding生成失败(trunc={trunc_len}): {e}")

                # Fix 2: Fallback to hash-based manual embedding placeholder
                if semantic_embedding is None:
                    logger.warning(f"[CA3] 所有embedding重试失败，使用hash占位向量 (is_core={is_core})")
                    # Build a deterministic hash-based vector as fallback
                    _text_for_hash = embed_text[:200]
                    _hash = hash(_text_for_hash)
                    embed_dim = self.feature_dim
                    manual_vec = [(((_hash * (i + 1) * 2654435761) >> 16) % 10000) / 10000.0 for i in range(embed_dim)]
                    # Normalize to unit vector
                    _norm = sum(v * v for v in manual_vec) ** 0.5
                    if _norm > 0:
                        manual_vec = [v / _norm for v in manual_vec]
                    semantic_embedding = torch.tensor(manual_vec, dtype=torch.float32)
        
        # 1. 创建记忆对象（含增强字段）
        memory = EpisodicMemory(
            memory_id=memory_id,
            timestamp=timestamp,
            temporal_skeleton=temporal_skeleton,
            semantic_pointer=semantic_pointer,
            causal_links=causal_links or [],
            dg_features=dg_features,
            is_core=is_core,
            content=content,
            activation_strength=10.0 if is_core else 1.0,
            key_features=key_features,
            value_features=value_features,
            # 人类记忆增强字段
            semantic_summary=semantic_summary,
            key_entities=key_entities,
            emotion_tag=emotion_tag,
            semantic_embedding=semantic_embedding,
            tier=MemoryTier.SHORT_TERM,
            recall_count=0,
            consecutive_misses=0,
        )
        
        # 2. 检查容量，优先淘汰短期记忆中激活度最低的
        if len(self.memories) >= self.max_capacity:
            tier_cands = [(mid, m) for mid, m in self.memories.items() if m.tier == MemoryTier.SHORT_TERM]
            if not tier_cands:
                tier_cands = list(self.memories.items())
            tier_cands.sort(key=lambda x: x[1].activation_strength)
            oldest_id = tier_cands[0][0]
            self._remove_memory(oldest_id)
        
        # 3. 存储记忆
        self.memories[memory_id] = memory
        
        # 4. 更新索引
        ts_key = timestamp // self.timestamp_precision_ms
        if ts_key not in self.time_index:
            self.time_index[ts_key] = []
        self.time_index[ts_key].append(memory_id)
        self.semantic_index[semantic_pointer] = memory_id
        
        # 5. 更新时间戳
        self.current_timestamp = max(self.current_timestamp, timestamp)
        
        return memory

    # ==================== 召回方法（保留原有辅助方法）====================

    @staticmethod
    def _chinese_tokenize(text: str) -> List[str]:
        """中文文本简易分词"""
        import re
        segments = re.split(r'[，。？！、；：\u201c\u201d\u2018\u2019（）【】\s,.\-!?;:()\[\]{}<>\\/|@#$%^&*+=~`]', text)
        tokens = set()
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            if len(seg) <= 8:
                tokens.add(seg.lower())
            chinese_chars = re.findall(r'[\u4e00-\u9fff]+', seg)
            for chunk in chinese_chars:
                # Fix 5: len >= 1 for Chinese characters (preserve single-char tokens like names)
                if len(chunk) >= 1:
                    tokens.add(chunk.lower())
                for i in range(len(chunk) - 1):
                    tokens.add(chunk[i:i+2].lower())
                for i in range(len(chunk) - 2):
                    tokens.add(chunk[i:i+3].lower())
            numbers = re.findall(r'\d+(?:\.\d+)?', seg)
            for num in numbers:
                tokens.add(num)
        return [t for t in tokens if len(t) >= 1]
    
    @staticmethod
    def _extract_recall_keywords(query: str) -> List[str]:
        """从查询中提取记忆召回关键词（增强版，保留作为辅助）"""
        import re
        keywords = []
        
        name_intent = re.search(r'叫什么|名字|姓名|称呼|如何称呼|我是谁|你知道我是', query)
        if name_intent:
            keywords.append('名字')
            keywords.append('用户名字')
            keywords.append('name')
        
        loc_intent = re.search(r'哪里|哪里人|来自|住在|住哪|城市|地方|地址|位置|在哪', query)
        if loc_intent:
            keywords.append('地点')
            keywords.append('城市')
            keywords.append('location')
        
        job_intent = re.search(r'职业|工作|做什么|干什么|专业|上班|职位|岗位|行业', query)
        if job_intent:
            keywords.append('职业')
            keywords.append('job')
        
        age_intent = re.search(r'多大|年龄|几岁|生日|出生|岁数|年级', query)
        if age_intent:
            keywords.append('年龄')
            keywords.append('age')
        
        hobby_intent = re.search(r'喜欢|爱好|兴趣|爱做什么|平时做|擅长|业余', query)
        if hobby_intent:
            keywords.append('爱好')
            keywords.append('hobby')
        
        remember_intent = re.search(r'记得|回忆|想起|忘记|忘了|记不记得|有没有告诉你|我说过', query)
        if remember_intent:
            keywords.append('记得')
        
        event_intent = re.search(r'之前说|上次|上次聊|上次提到|之前告诉|曾经说|说过什么', query)
        if event_intent:
            keywords.append('事件')
        
        # BUG FIX: 增加更多查询意图关键词（覆盖更多用户表达方式）
        contact_intent = re.search(r'电话|手机|邮箱|联系|怎么找|怎么联系', query)
        if contact_intent:
            keywords.append('联系方式')
            keywords.append('phone')
            keywords.append('email')
        
        school_intent = re.search(r'学校|大学|毕业|上学|就读', query)
        if school_intent:
            keywords.append('学校')
        
        company_intent = re.search(r'公司|在哪工作|什么单位|哪个公司', query)
        if company_intent:
            keywords.append('公司')
            keywords.append('职业')
        
        stop_words = set('的了是在有不也这那他她你我都它们可以会要能和与或但是如果因为所以虽然然而而且以及对于关于通过没有不是已经还又再更最被把给让向到得地着过吗呢吧啊哦哈呀嗯么个什么怎么怎样多少几哪请告诉知道想希望应该可能需要觉得认为说一样一个就是那个这个自己'.encode('utf-8').decode('utf-8'))
        
        entities = re.findall(r'[\u4e00-\u9fff]{2,4}', query)
        for ent in entities:
            if ent not in stop_words and len(ent) >= 2:
                keywords.append(ent)
        
        english_names = re.findall(r'[A-Z][a-z]{1,10}', query)
        keywords.extend(english_names)
        
        return keywords

    # ==================== 核心 Recall（增强版 - Embedding + 分层）====================
    
    def recall(
        self,
        query_features: Optional[torch.Tensor] = None,
        query_semantic: Optional[str] = None,
        query_timestamp: Optional[int] = None,
        topk: int = 2,
        return_all: bool = False,
    ) -> List[EpisodicMemory]:
        """
        记忆召回（增强版 - Embedding 语义匹配 + 记忆分层）
        
        三重召回策略:
        1. Embedding 语义匹配（主力）: 使用模型自身 embedding 做语义相似度
        2. 关键词匹配（辅助）: 保留正则关键词作为补充
        3. 分层加权: 长期记忆 > 中期记忆 > 短期记忆
        
        Args:
            return_all: 当为 True 时，跳过 _embedding_recall 中的层级阈值过滤，
                         返回所有候选记忆及其相似度分数（供 HippocampusSystem 使用，
                         避免在上层按 activation_strength 错误过滤）。
        """
        self.recall_count += 1
        self.last_recall_time = time.time()
        candidates = []
        recall_trace: List[Dict] = []

        # ========== Fix 6: 记忆查询意图关键词加速 ==========
        memory_query_boosters = [
            "记得", "记住", "名字", "电话", "手机", "邮箱", "职业",
            "年龄", "城市", "爱好", "喜欢", "来自", "工作", "学校"
        ]
        is_memory_query = any(bw in (query_semantic or "") for bw in memory_query_boosters)
        if is_memory_query:
            topk = max(topk, 10)  # recall more candidates for memory queries
        
        # ========== 1. Embedding 语义匹配（主力召回方式）==========
        if query_semantic and self.semantic_engine is not None:
            embedding_scores = self._embedding_recall(query_semantic, topk=topk * 3, return_all=return_all)
            for score_info in embedding_scores:
                mid = score_info['memory_id']
                if mid in self.memories:
                    memory = self.memories[mid]
                    memory._embedding_score = score_info['similarity']
                    candidates.append(memory)
                    recall_trace.append({
                        "memory_id": mid,
                        "rank": score_info['rank'],
                        "similarity": score_info['similarity'],
                        "embedding_score": score_info['similarity'],
                        "semantic_score": score_info['similarity'],
                        "final_score": score_info['similarity'],
                        "method": "embedding",
                        "selected": True,
                        "is_core": bool(memory.is_core),
                    })
        
        # ========== 2. 关键词匹配（辅助召回）==========
        recall_keywords = set()
        if query_semantic:
            intent_keywords = self._extract_recall_keywords(query_semantic)
            recall_keywords.update(intent_keywords)
            chinese_tokens = self._chinese_tokenize(query_semantic)
            recall_keywords.update(chinese_tokens)
        
        existing_ids = {c.memory_id for c in candidates}
        # BUG FIX: 关键词匹配应该搜索所有记忆，而不仅是核心记忆
        # 之前只搜索core_memories导致大量非核心记忆无法通过关键词召回
        all_searchable = [m for m in self.memories.values()]
        
        if recall_keywords:
            for memory in all_searchable:
                if memory.memory_id in existing_ids:
                    continue
                # 匹配语义摘要 + 关键实体（新增的富语义信息）
                semantic_text = (
                    memory.semantic_pointer + " " + 
                    memory.content + " " +
                    memory.semantic_summary + " " +
                    memory.key_entities
                ).lower()
                matched = sum(1 for kw in recall_keywords if kw in semantic_text)
                if matched > 0:
                    memory._recall_keyword_score = matched
                    candidates.append(memory)
        
        # ========== 3. DG 特征相似度匹配（兜底）==========
        if query_features is not None and len(candidates) < topk * 2:
            existing_ids = {c.memory_id for c in candidates}
            all_features = []
            all_ids = []
            
            for mid, memory in self.memories.items():
                if mid not in existing_ids and memory.dg_features is not None:
                    all_features.append(memory.dg_features)
                    all_ids.append(mid)
            
            if all_features:
                all_features = torch.stack(all_features)
                if all_features.numel() > 0:
                    if query_features.dim() == 1:
                        query_features = query_features.unsqueeze(0)
                    query_norm = F.normalize(query_features, p=2, dim=-1)
                    all_features_norm = F.normalize(all_features, p=2, dim=-1)
                    
                    if query_norm.dim() == 2 and all_features_norm.dim() == 2:
                        similarities = torch.mm(query_norm, all_features_norm.t()).squeeze(0)
                        k_val = min(topk * 2, len(all_ids))
                        if k_val > 0:
                            top_sim, top_indices = torch.topk(similarities, k=k_val)
                            for i, sim in enumerate(top_sim):
                                idx = top_indices[i].item() if hasattr(top_indices[i], "item") else int(top_indices[i])
                                memory_id = all_ids[idx]
                                memory = self.memories[memory_id]
                                sim_val = sim.item() if hasattr(sim, "item") else float(sim)
                                if sim_val > 0.20:  # 降低阈值，更多 DG 匹配进入候选
                                    candidates.append(memory)
                                    recall_trace.append({
                                        "memory_id": memory_id,
                                        "rank": int(i),
                                        "similarity": float(sim_val),
                                        "semantic_score": 0.0,
                                        "final_score": float(sim_val),
                                        "method": "dg_features",
                                        "selected": True,
                                        "is_core": bool(memory.is_core),
                                    })
        
        # ========== 4. 补充核心记忆（确保不遗漏重要信息）==========
        # BUG FIX: 先用all_searchable中的非核心记忆补充，再用核心记忆兜底
        if len(candidates) < topk:
            for memory in all_searchable:
                if memory.memory_id not in {c.memory_id for c in candidates}:
                    candidates.append(memory)
                    if len(candidates) >= topk:
                        break
        
        # ========== 5. 去重 ==========
        seen_ids = set()
        unique_candidates = []
        for cand in candidates:
            if cand.memory_id not in seen_ids:
                seen_ids.add(cand.memory_id)
                unique_candidates.append(cand)
        
        # ========== 6. 分层加权排序 ==========
        # Fix 4: Use composite scoring with similarity, recall frequency, and core bonus
        def tier_sort_key(m):
            tier_weight = (int(m.tier) + 1) * 0.3  # tier contribution
            sim_score = getattr(m, '_embedding_score', 0.0)  # embedding similarity
            recall_bonus = min(getattr(m, 'recall_count', 0) * 0.05, 0.3)  # recall frequency bonus
            core_bonus = 0.5 if getattr(m, 'is_core', False) else 0.0  # core memory priority
            return tier_weight + sim_score + recall_bonus + core_bonus
        
        self.last_recall_trace = recall_trace
        
        # 更新被召回记忆的统计
        for cand in unique_candidates[:topk]:
            cand.recall_count = getattr(cand, 'recall_count', 0) + 1
            cand.consecutive_misses = 0
            cand.activation_strength = min(cand.activation_strength * 1.05, 2.0)
        
        # 更新未被召回记忆的连续未命中计数
        for mid, mem in self.memories.items():
            if mid not in {c.memory_id for c in unique_candidates[:topk]}:
                mem.consecutive_misses = getattr(mem, 'consecutive_misses', 0) + 1
        
        unique_candidates.sort(key=tier_sort_key, reverse=True)
        return unique_candidates[:topk]
    
    def _embedding_recall(self, query_text: str, topk: int = 6, return_all: bool = False) -> List[Dict]:
        """
        使用 Embedding 做语义召回（核心优化）
        
        用模型自身的 embedding 层将查询和记忆文本编码为向量，
        通过余弦相似度找到语义最相关的记忆。
        能够理解"你记得我的名字吗"和"我叫张三"之间的语义关联。
        
        Args:
            return_all: 当为 True 时，跳过层级阈值过滤，返回所有候选及其相似度。
        """
        if not query_text or not self.semantic_engine:
            return []
        
        # Fix 3: Lazy embedding generation for memories with None semantic_embedding
        for mid, mem in list(self.memories.items()):
            if mem.semantic_embedding is None and self.semantic_engine is not None:
                embed_source = mem.key_entities or mem.semantic_summary or mem.content or mem.semantic_pointer
                if embed_source:
                    try:
                        generated_emb = self.semantic_engine.get_text_embedding(embed_source[:100])
                        if generated_emb is not None:
                            mem.semantic_embedding = generated_emb
                            logger.debug(f"[CA3] 惰性生成embedding成功: {mid}")
                    except Exception as e:
                        logger.debug(f"[CA3] 惰性生成embedding失败: {mid}, {e}")

        valid_memories = []
        for mid, mem in self.memories.items():
            if mem.semantic_embedding is not None:
                valid_memories.append((mid, mem))
        
        if not valid_memories:
            return []
        
        query_emb = self.semantic_engine.get_text_embedding(query_text)
        if query_emb is None:
            return []
        
        # 批量计算相似度
        memory_embs = [mem.semantic_embedding for _, mem in valid_memories]
        memory_ids = [mid for mid, _ in valid_memories]
        
        similarities = self.semantic_engine.batch_compute_similarities(query_text, memory_embs)
        
        results = []
        for i, (mid, sim) in enumerate(zip(memory_ids, similarities)):
            mem = self.memories[mid]
            tier = getattr(mem, 'tier', MemoryTier.SHORT_TERM)
            threshold = self.consolidation_manager.get_recall_threshold(tier)
            
            # 长期记忆的 embedding 匹配给额外加分
            tier_bonus = int(tier) * 0.05
            adjusted_sim = sim + tier_bonus
            
            # Fix 1: When return_all=True, skip the tier threshold filtering
            if return_all or adjusted_sim > threshold:
                results.append({
                    'memory_id': mid,
                    'similarity': adjusted_sim,
                    'raw_similarity': sim,
                    'tier': int(tier),
                    'rank': i,
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:topk]

    # ==================== 状态管理 ====================
    
    def get_state(self) -> dict:
        """获取 CA3 模块的完整状态"""
        memories_dict = {}
        for mem_id, memory in self.memories.items():
            memories_dict[mem_id] = memory.to_dict()
        
        return {
            'memories': memories_dict,
            'time_index': self.time_index,
            'semantic_index': self.semantic_index,
            'current_timestamp': self.current_timestamp,
            'recall_count': self.recall_count,
            'last_recall_time': self.last_recall_time,
        }
    
    def set_state(self, state: dict):
        """从状态字典恢复 CA3 模块 - 支持人类记忆增强字段"""
        memories_dict = state.get('memories', {})
        
        self.memories = OrderedDict()
        for mem_id, mem_dict in memories_dict.items():
            dg_features = None
            if 'dg_features' in mem_dict and mem_dict['dg_features'] is not None:
                if isinstance(mem_dict['dg_features'], list):
                    dg_features = torch.tensor(mem_dict['dg_features'], dtype=torch.float32)
                elif isinstance(mem_dict['dg_features'], torch.Tensor):
                    dg_features = mem_dict['dg_features']
            
            semantic_embedding = None
            if 'semantic_embedding' in mem_dict and mem_dict['semantic_embedding'] is not None:
                if isinstance(mem_dict['semantic_embedding'], list):
                    semantic_embedding = torch.tensor(mem_dict['semantic_embedding'], dtype=torch.float32)
                elif isinstance(mem_dict['semantic_embedding'], torch.Tensor):
                    semantic_embedding = mem_dict['semantic_embedding']
            
            # 修复: 恢复 key_features 和 value_features（之前遗漏导致 save/load 后 KV 特征丢失）
            key_features = None
            if 'key_features' in mem_dict and mem_dict['key_features'] is not None:
                if isinstance(mem_dict['key_features'], list):
                    key_features = torch.tensor(mem_dict['key_features'], dtype=torch.float32)
                elif isinstance(mem_dict['key_features'], torch.Tensor):
                    key_features = mem_dict['key_features']
            
            value_features = None
            if 'value_features' in mem_dict and mem_dict['value_features'] is not None:
                if isinstance(mem_dict['value_features'], list):
                    value_features = torch.tensor(mem_dict['value_features'], dtype=torch.float32)
                elif isinstance(mem_dict['value_features'], torch.Tensor):
                    value_features = mem_dict['value_features']
            
            memory = EpisodicMemory(
                memory_id=mem_dict.get('memory_id', mem_id),
                timestamp=mem_dict.get('timestamp', 0),
                temporal_skeleton=mem_dict.get('temporal_skeleton', ''),
                semantic_pointer=mem_dict.get('semantic_pointer', ''),
                causal_links=mem_dict.get('causal_links', []),
                activation_strength=mem_dict.get('activation_strength', 1.0),
                is_core=mem_dict.get('is_core', False),
                content=mem_dict.get('content', ''),
                dg_features=dg_features,
                semantic_summary=mem_dict.get('semantic_summary', ''),
                key_entities=mem_dict.get('key_entities', ''),
                emotion_tag=mem_dict.get('emotion_tag', '中性'),
                semantic_embedding=semantic_embedding,
                key_features=key_features,
                value_features=value_features,
                tier=MemoryTier(mem_dict.get('tier', 0)),
                recall_count=mem_dict.get('recall_count', 0),
                consecutive_misses=mem_dict.get('consecutive_misses', 0),
            )
            self.memories[mem_id] = memory
        
        self.time_index = state.get('time_index', {})
        self.semantic_index = state.get('semantic_index', {})
        self.current_timestamp = state.get('current_timestamp', 0)
        self.recall_count = int(state.get('recall_count', 0))
        self.last_recall_time = float(state.get('last_recall_time', 0.0))
    
    def consolidate_and_decay(self) -> dict:
        """
        执行记忆固化和衰减
        
        模拟人脑睡眠期间的记忆巩固:
        - 短期→中期→长期的层级转换
        - 基于层级的不同衰减率
        """
        return self.consolidation_manager.consolidate_memories(self.memories)
    
    # ==================== 原有辅助方法（保留兼容）====================
    
    def complete_pattern(
        self,
        partial_cue: dict,
        topk: int = 2
    ) -> List[EpisodicMemory]:
        """模式补全 - 基于部分线索召回完整记忆链条"""
        query_semantic = partial_cue.get('semantic')
        query_timestamp = partial_cue.get('timestamp')
        query_features = partial_cue.get('features')
        return self.recall(
            query_features=query_features,
            query_semantic=query_semantic,
            query_timestamp=query_timestamp,
            topk=topk
        )
    
    def update_memory_strength(self, memory_id: str, delta: float):
        """更新记忆激活强度 (STDP 更新接口)"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            time_elapsed = (self.current_timestamp - memory.timestamp) / 1000.0
            time_decay = 1.0 / (1.0 + time_elapsed * 0.001)
            effective_delta = delta * time_decay
            memory.activation_strength += effective_delta
            memory.activation_strength = max(0.1, min(2.0, memory.activation_strength))
    
    def prune_weak_memories(self, threshold: float = 0.3):
        """修剪弱记忆"""
        to_remove = []
        for memory_id, memory in self.memories.items():
            if memory.activation_strength < threshold:
                to_remove.append(memory_id)
        for memory_id in to_remove:
            self._remove_memory(memory_id)
        return len(to_remove)
    
    def _remove_memory(self, memory_id: str):
        """删除记忆及其索引"""
        if memory_id not in self.memories:
            return
        memory = self.memories[memory_id]
        ts_key = memory.timestamp // self.timestamp_precision_ms
        if ts_key in self.time_index and memory_id in self.time_index[ts_key]:
            self.time_index[ts_key].remove(memory_id)
        if memory.semantic_pointer in self.semantic_index:
            del self.semantic_index[memory.semantic_pointer]
        del self.memories[memory_id]
    
    def _compute_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """计算特征相似度 (余弦相似度)"""
        feat1_flat = feat1.flatten()
        feat2_flat = feat2.flatten()
        if len(feat1_flat) != len(feat2_flat):
            return 0.0
        sim = torch.nn.functional.cosine_similarity(
            feat1_flat.unsqueeze(0), feat2_flat.unsqueeze(0), dim=1
        ).item()
        return sim
    
    def get_stats(self) -> dict:
        """获取记忆库统计信息（含分层统计）"""
        if not self.memories:
            return {
                'num_memories': 0,
                'capacity_usage': 0.0,
                'avg_activation': 0.0,
                'tier_stats': self.consolidation_manager.get_tier_stats(self.memories),
            }
        
        activations = [m.activation_strength for m in self.memories.values()]
        
        return {
            'num_memories': len(self.memories),
            'capacity_usage': len(self.memories) / self.max_capacity,
            'avg_activation': sum(activations) / len(activations),
            'max_activation': max(activations),
            'min_activation': min(activations),
            'core_memory_count': sum(1 for m in self.memories.values() if m.is_core),
            'recall_count': self.recall_count,
            'last_recall_time': self.last_recall_time,
            'tier_stats': self.consolidation_manager.get_tier_stats(self.memories),
            'semantic_cache': (self.semantic_engine.get_cache_stats() if self.semantic_engine else {}),
        }
    
    def get_last_recall_trace(self) -> List[Dict]:
        """返回最近一次 recall 的评分明细"""
        return list(self.last_recall_trace)
    
    def get_memory_strength(self, memory_id: str) -> float:
        """获取记忆激活强度"""
        if memory_id in self.memories:
            return self.memories[memory_id].activation_strength
        return 0.0
    
    def update_link(self, from_memory_id: str, to_memory_id: str, strength: float):
        """更新记忆间关联（用于 SWR 序列学习）"""
        if from_memory_id in self.memories:
            link = f"linked_to_{to_memory_id}_{strength:.2f}"
            if link not in self.memories[from_memory_id].causal_links:
                self.memories[from_memory_id].causal_links.append(link)
                if len(self.memories[from_memory_id].causal_links) > 50:
                    self.memories[from_memory_id].causal_links.pop(0)
    
    def forward(
        self,
        query_features: torch.Tensor,
        topk: int = 2
    ) -> List[dict]:
        """前向传播 - 记忆召回"""
        memories = self.recall(query_features=query_features, topk=topk)
        return [mem.to_dict() for mem in memories]
