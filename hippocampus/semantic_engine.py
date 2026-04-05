"""
语义引擎 - Semantic Engine

提供两大核心能力:
1. 语义摘要生成: 利用模型自身对记忆内容生成压缩摘要（而非仅存原始token特征）
2. Embedding 语义匹配: 使用模型自身的 embedding 层计算文本的语义向量，用于语义召回

设计理念:
- 摘要不存完整对话文本，而是提取核心语义信息（"谁、什么、何时、何地"）
- 召回不依赖正则/关键词，而是用 embedding 向量做真正的语义相似度匹配
- 轻量级实现，不引入额外模型依赖，复用已有的 Qwen 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import time
import logging
import re
import hashlib
import struct

logger = logging.getLogger(__name__)

# Default embedding dimension for hash-based fallback
_DEFAULT_EMBEDDING_DIM = 768


class SemanticSummarizer:
    """
    语义摘要生成器
    
    利用模型自身能力，为每条记忆生成结构化语义摘要。
    摘要包含:
    - 核心语义 (semantic_summary): 1-2句概括
    - 关键实体 (key_entities): 提取的人名/地名/数字等
    - 情感标签 (emotion_tag): 正面/负面/中性
    """
    
    # 实体提取模式（复用但优化版）
    ENTITY_PATTERNS = {
        'name': [
            (r"我叫([\u4e00-\u9fa5a-zA-Z]{2,4})", 1),
            (r"我的名字(是|叫)([\u4e00-\u9fa5a-zA-Z]{2,4})", 2),
            (r"我是([\u4e00-\u9fa5a-zA-Z]{2,10}?)(?=[，。,.\s]|$)", 1),
            # Fix 3: additional name patterns
            (r"我姓([\u4e00-\u9fa5]{1})[，,]?(?:叫|名)([\u4e00-\u9fa5a-zA-Z]{1,4})", (1, 2)),
            (r"大家叫我([\u4e00-\u9fa5a-zA-Z]{2,4})", 1),
            (r"叫我([\u4e00-\u9fa5a-zA-Z]{2,4})就行", 1),
            (r"我的全名是([\u4e00-\u9fa5a-zA-Z]{2,6})", 1),
        ],
        'age': [
            (r"我今年(\d+)", 1),
            (r"我(\d+)岁", 1),
            (r"年龄[是:：](\d+)", 1),
            # Fix 3: additional age patterns
            (r"我(\d+)了", 1),
            (r"今年(\d+)了", 1),
            (r"(\d+)岁", 1),
            (r"芳龄(\d+)", 1),
        ],
        'location': [
            (r"来自([\u4e00-\u9fa5a-zA-Z]{2,10})", 1),
            (r"住在([\u4e00-\u9fa5a-zA-Z]{2,10})", 1),
            (r"在([\u4e00-\u9fa5a-zA-Z]{2,10}?)(工作|生活|上班)", 1),
        ],
        'job': [
            (r"(?:是|当)(.{0,10}?)(工程师|医生|老师|学生|设计师|程序员|律师|会计|经理|总监|分析师|研究员)", 1),
            (r"职业[是:：](.{2,10})", 1),
            # Fix 3: additional job patterns
            (r"当(?:一个)?([\u4e00-\u9fa5]{2,8}?)(?:工程师|医生|老师|学生|设计师|程序员|律师|会计|经理|总监|分析师|研究员|教授)", 1),
            (r"是一名([\u4e00-\u9fa5]{2,10}?)(?=[，。,.\s]|$)", 1),
        ],
        'phone': [
            (r"(\d{11})", 1),
            (r"电话[：:](\d{11})", 1),
        ],
        'email': [
            (r"([\w.-]+@[\w.-]+\.\w+)", 1),
        ],
        'hobby': [
            (r"喜欢([\u4e00-\u9fa5a-zA-Z]{2,20})", 1),
            (r"爱好([\u4e00-\u9fa5a-zA-Z]{2,20})", 1),
            # Fix 3: additional hobby patterns
            (r"平时(?:喜欢|爱)([\u4e00-\u9fa5a-zA-Z]{2,20})", 1),
            (r"业余(?:喜欢|爱)([\u4e00-\u9fa5a-zA-Z]{2,20})", 1),
        ],
        'money': [
            (r"(\d+(?:\.\d+)?)\s*(元|块钱|万)", 0),
            (r"(房租|押金|费用|租金)[：:]?(\d+(?:\.\d+)?)", 2),
        ],
        'date': [
            (r"(\d{4}年\d{1,2}月\d{1,2}[日号]?)", 1),
            (r"(\d{1,2}月\d{1,2}[日号]?)", 1),
        ],
    }
    
    # 情感关键词
    POSITIVE_WORDS = set("开心 高兴 喜欢 爱 美好 幸福 满意 期待 精彩 优秀 成功 感谢".split())
    NEGATIVE_WORDS = set("难过 悲伤 焦虑 压力 恐惧 遗憾 愤怒 讨厌 痛苦 失望 烦恼 无聊 紧张".split())

    # Fix 4: common filler words to remove during normalization
    FILLER_WORDS = set("的 了 是 在 有 和 就 不 都 也 而 与 或 但 却 又 很 把 被 让 给 向 从 到".split())
    
    def __init__(self, model_interface=None, device: str = 'cpu'):
        """
        Args:
            model_interface: QwenInterface 实例，用于获取 embedding
            device: 计算设备
        """
        self.model = model_interface
        self.device = device
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        self._cache_max_size = 500
        self._detected_embedding_dim: Optional[int] = None

    def _normalize_text(self, text: str) -> str:
        """
        Fix 4: 文本归一化 - 在生成 embedding 之前对文本进行预处理

        - 去除多余空白
        - 全角数字/字母转半角
        - 去除常见虚词/填充词

        Args:
            text: 原始文本
            
        Returns:
            归一化后的文本
        """
        if not text:
            return text
        
        # 1. 全角数字/字母转半角
        result = []
        for char in text:
            code = ord(char)
            # Full-width range: 0xFF01-0xFF5E maps to 0x0021-0x007E
            if 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            # Full-width space
            elif code == 0x3000:
                result.append(' ')
            else:
                result.append(char)
        text = ''.join(result)
        
        # 2. 去除多余空白（保留单个空格）
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 3. 去除常见虚词/填充词（仅去除作为独立字符出现的，保留其组成词的语义）
        #    使用字符级过滤：逐字检查，如果该字是虚词且前后都不是虚词，则移除
        normalized_chars = []
        text_len = len(text)
        for i, char in enumerate(text):
            if char in self.FILLER_WORDS:
                # Keep the filler if removing it would merge two characters
                # that shouldn't be merged (i.e., it's acting as a separator)
                prev_char = text[i - 1] if i > 0 else ''
                next_char = text[i + 1] if i < text_len - 1 else ''
                # Remove the filler if both neighbors are content characters
                if prev_char and next_char and prev_char not in self.FILLER_WORDS and next_char not in self.FILLER_WORDS:
                    continue
            normalized_chars.append(char)
        text = ''.join(normalized_chars)
        
        return text.strip()
    
    def _get_embedding_dim(self) -> int:
        """
        探测模型的 embedding 维度。如果无法探测则返回默认值。
        """
        if self._detected_embedding_dim is not None:
            return self._detected_embedding_dim
        
        try:
            if self.model is not None:
                embedding_layer = getattr(self.model, 'embeddings', None)
                if embedding_layer is not None:
                    # Try to get embedding dimension from the layer
                    if hasattr(embedding_layer, 'embedding_dim'):
                        self._detected_embedding_dim = embedding_layer.embedding_dim
                        return self._detected_embedding_dim
                    elif hasattr(embedding_layer, 'weight'):
                        self._detected_embedding_dim = embedding_layer.weight.shape[-1]
                        return self._detected_embedding_dim
        except Exception:
            pass
        
        self._detected_embedding_dim = _DEFAULT_EMBEDDING_DIM
        return self._detected_embedding_dim

    def _char_ngram_hash_embedding(self, text: str, dim: int = None) -> torch.Tensor:
        """
        Fix 1: 字符 n-gram 哈希嵌入 - 作为最后的 fallback 手段

        通过对文本的字符 n-gram 进行哈希，生成一个固定维度的确定性向量。
        虽然语义能力有限，但保证了非空文本一定能产生向量。

        Args:
            text: 输入文本
            dim: 输出维度（默认使用模型维度或 768）
            
        Returns:
            L2 归一化的 embedding 向量
        """
        if dim is None:
            dim = self._get_embedding_dim()
        
        vec = torch.zeros(dim, dtype=torch.float32)
        text_lower = text.lower()
        text_len = len(text_lower)
        
        if text_len == 0:
            vec[0] = 1.0
            return F.normalize(vec, p=2, dim=-1)
        
        # Extract character n-grams (unigrams, bigrams, trigrams)
        ngram_counts: Dict[str, int] = {}
        for n in range(1, 4):
            for i in range(max(0, text_len - n + 1)):
                ngram = text_lower[i:i + n]
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        
        # Hash each n-gram into the vector using multiple hash functions
        for ngram, count in ngram_counts.items():
            # Use multiple hash seeds to reduce collisions
            for seed in range(3):
                h = hashlib.md5(f"{seed}:{ngram}".encode('utf-8')).digest()
                # Convert first 8 bytes to a float index in [0, dim)
                idx = struct.unpack('<Q', h[:8])[0] % dim
                # Convert next 4 bytes to a sign value
                sign = 1.0 if (h[8] & 0x01) == 0 else -1.0
                vec[idx] += sign * count / (ngram_counts.__len__() or 1)
        
        # L2 normalize
        vec = F.normalize(vec, p=2, dim=-1)
        return vec

    def generate_semantic_summary(
        self, 
        user_input: str, 
        ai_response: str,
        is_core: bool = False
    ) -> Dict[str, str]:
        """
        为对话生成结构化语义摘要
        
        Args:
            user_input: 用户输入
            ai_response: AI 回复
            is_core: 是否核心记忆
        
        Returns:
            dict: {
                'semantic_summary': str,  # 语义摘要
                'key_entities': str,      # 关键实体（管道符分隔）
                'emotion_tag': str,       # 情感标签
                'structured': str,        # 结构化表示
            }
        """
        # 1. 提取关键实体
        entities = self._extract_entities(user_input)
        
        # 2. 生成情感标签
        emotion = self._detect_emotion(user_input + " " + ai_response)
        
        # 3. 生成语义摘要
        if is_core and entities:
            # 核心记忆：使用结构化实体摘要（精确、易于召回）
            semantic_summary = self._generate_structured_summary(entities, user_input)
        else:
            # 普通记忆：使用对话压缩摘要
            semantic_summary = self._generate_compressed_summary(user_input, ai_response)
        
        # 4. 组装结构化表示
        entity_str = " | ".join(entities) if entities else ""
        structured = entity_str
        if emotion != "中性":
            structured = f"[{emotion}] {structured}"
        
        return {
            'semantic_summary': semantic_summary,
            'key_entities': entity_str,
            'emotion_tag': emotion,
            'structured': structured,
        }
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        从文本中提取关键实体
        
        Fix 3: 支持多组捕获 (group_idx 为 tuple 时，拼接各组)
        """
        entities = []
        seen = set()
        
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern, group_idx in patterns:
                try:
                    match = re.search(pattern, text)
                    if match:
                        # Fix 3: handle tuple group indices (combine multiple groups)
                        if isinstance(group_idx, tuple):
                            value = ''.join(g.strip() for g in match.groups() if g is not None)
                        else:
                            value = match.group(group_idx).strip()
                        if value and value not in seen and len(value) < 50:
                            entities.append(f"{entity_type}:{value}")
                            seen.add(value)
                except Exception:
                    pass
        
        return entities
    
    def _detect_emotion(self, text: str) -> str:
        """检测文本情感"""
        positive_count = sum(1 for w in self.POSITIVE_WORDS if w in text)
        negative_count = sum(1 for w in self.NEGATIVE_WORDS if w in text)
        
        if positive_count > negative_count:
            return "正面"
        elif negative_count > positive_count:
            return "负面"
        return "中性"
    
    def _generate_structured_summary(self, entities: List[str], original_text: str) -> str:
        """生成结构化摘要（核心记忆用）"""
        summary_parts = []
        for entity in entities:
            summary_parts.append(entity)
        
        # 如果实体不够丰富，从原文中提取关键片段
        if len(summary_parts) < 2:
            # 提取原文中的核心短语
            sentences = re.split(r'[，。！？；\n]', original_text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) >= 3 and len(sent) <= 30:
                    summary_parts.append(sent)
                    if len(summary_parts) >= 3:
                        break
        
        return " | ".join(summary_parts)
    
    def _generate_compressed_summary(self, user_input: str, ai_response: str) -> str:
        """生成压缩摘要（普通记忆用）"""
        # 截取核心部分
        user_core = user_input[:60].strip()
        response_core = ai_response[:40].strip()
        
        if user_core and response_core:
            return f"问:{user_core} 答:{response_core}"
        elif user_core:
            return f"问:{user_core}"
        else:
            return ""
    
    def get_text_embedding(self, text: str) -> Optional[torch.Tensor]:
        """
        获取文本的 embedding 向量（增强版 - 健壮性修复）
        
        Fix 1: 实现完整 fallback 链，确保对非空文本永远不返回 None:
        1. 优先使用深层hidden states（包含上下文语义）
        2. 回退到 input embeddings + CLS 加权池化
        3. 最终回退到字符 n-gram 哈希嵌入
        
        Fix 4: 在生成 embedding 之前对文本进行归一化
        
        Args:
            text: 输入文本
            
        Returns:
            embedding: [hidden_size] 维度的向量，或 None（仅空文本时）
        """
        if not text or not text.strip():
            return None
        
        # Fix 4: 对文本进行归一化
        text = self._normalize_text(text)
        if not text or not text.strip():
            return None
        
        # 检查缓存
        cache_key = text[:200]  # 缓存前200字符
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # Fix 1: 如果模型不可用，直接使用哈希 fallback（不再返回 None）
        if self.model is None:
            logger.debug("[SemanticEngine] 模型不可用，使用字符 n-gram 哈希 fallback 生成 embedding")
            result = self._char_ngram_hash_embedding(text)
            if len(self._embedding_cache) < self._cache_max_size:
                self._embedding_cache[cache_key] = result
            return result
        
        embedding_result = None
        
        try:
            # 使用模型的 tokenizer 和 embedding 层
            tokenizer = self.model.tokenizer
            embedding_layer = self.model.embeddings  # QwenInterface.embeddings 属性
            
            # Tokenize
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                # tokenizer 返回空列表时使用哈希 fallback
                logger.debug("[SemanticEngine] tokenizer 对文本返回空结果，使用字符 n-gram 哈希 fallback")
                embedding_result = self._char_ngram_hash_embedding(text)
            else:
                # 限制长度以控制计算量
                max_tokens = 64
                if len(token_ids) > max_tokens:
                    token_ids = token_ids[:max_tokens]
                
                input_tensor = torch.tensor([token_ids], device=self.device)
                
                with torch.no_grad():
                    # ===== 策略 1: 深层 hidden states =====
                    deep_embedding = None
                    
                    base_model = getattr(self.model, 'model', None)
                    if base_model is not None and hasattr(base_model, 'base_model'):
                        try:
                            model_core = base_model.base_model
                            with torch.inference_mode():
                                outputs = model_core(
                                    input_tensor,
                                    output_hidden_states=True,
                                    return_dict=True
                                )
                            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                                # 取最后2层hidden states的平均
                                last_layers = outputs.hidden_states[-2:]
                                stacked = torch.stack(last_layers)  # [2, 1, seq_len, hidden_size]
                                deep_embedding = stacked.mean(dim=0)  # [1, seq_len, hidden_size]
                        except Exception as deep_err:
                            logger.debug(f"[SemanticEngine] 深层embedding提取失败，尝试 CLS 加权 fallback: {deep_err}")
                    
                    if deep_embedding is not None:
                        # 加权mean pooling：首尾token权重更高
                        seq_len = deep_embedding.shape[1]
                        weights = torch.ones(seq_len, device=deep_embedding.device)
                        weights[0] = 1.5
                        if seq_len > 1:
                            weights[-1] = 2.0
                        mid_sum = weights.sum() - weights[0] - (weights[-1] if seq_len > 1 else 0)
                        if mid_sum > 0 and seq_len > 2:
                            mid_count = seq_len - (1 if seq_len > 1 else 0) - 1
                            weights[1:-1 if seq_len > 1 else seq_len] = 0.5
                        
                        weights = weights / weights.sum()
                        embedding_result = (deep_embedding * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1).squeeze(0)
                    else:
                        # ===== Fix 1 策略 2: CLS 加权 input embedding =====
                        try:
                            embeddings = embedding_layer(input_tensor)  # [1, seq_len, hidden_size]
                            seq_len = embeddings.shape[1]
                            
                            # CLS-weighted pooling: first token weighted 3x, rest 1x
                            weights = torch.ones(seq_len, device=embeddings.device)
                            weights[0] = 3.0  # CLS token (first token) gets 3x weight
                            weights = weights / weights.sum()
                            
                            embedding_result = (embeddings * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1).squeeze(0)
                            logger.debug("[SemanticEngine] 使用 CLS 加权 input embedding fallback")
                        except Exception as cls_err:
                            logger.debug(f"[SemanticEngine] CLS 加权 embedding 也失败，尝试字符 n-gram 哈希 fallback: {cls_err}")
                    
                    # ===== Fix 1 策略 3: 字符 n-gram 哈希（最终 fallback）=====
                    if embedding_result is None:
                        logger.debug("[SemanticEngine] 所有模型 embedding 方法失败，使用字符 n-gram 哈希 fallback")
                        embedding_result = self._char_ngram_hash_embedding(text)
                
                # L2 归一化
                if embedding_result is not None:
                    embedding_result = F.normalize(embedding_result.float(), p=2, dim=-1)
        
        except Exception as e:
            logger.debug(f"[SemanticEngine] embedding 计算全部异常，使用字符 n-gram 哈希 fallback: {e}")
            embedding_result = self._char_ngram_hash_embedding(text)
        
        # Fix 1: 最终保障 - 对于非空文本，embedding_result 绝不为 None
        if embedding_result is None:
            logger.warning("[SemanticEngine] embedding_result 仍然为 None（不应发生），使用哈希 fallback")
            embedding_result = self._char_ngram_hash_embedding(text)
        
        # 更新缓存
        if len(self._embedding_cache) < self._cache_max_size:
            self._embedding_cache[cache_key] = embedding_result
        
        return embedding_result
    
    def compute_semantic_similarity(
        self, 
        query_text: str, 
        memory_texts: List[str]
    ) -> List[float]:
        """
        计算查询文本与多条记忆文本之间的语义相似度
        
        使用模型 embedding 做真正的语义匹配，替代正则关键词匹配。
        
        Args:
            query_text: 查询文本
            memory_texts: 记忆文本列表
            
        Returns:
            similarities: 相似度列表 [0.0, 1.0]
        """
        if not query_text or not memory_texts:
            return [0.0] * len(memory_texts)
        
        # 计算查询 embedding
        query_emb = self.get_text_embedding(query_text)
        if query_emb is None:
            return [0.0] * len(memory_texts)
        
        query_emb = query_emb.to(self.device)
        
        similarities = []
        for mem_text in memory_texts:
            if not mem_text or not mem_text.strip():
                similarities.append(0.0)
                continue
            
            mem_emb = self.get_text_embedding(mem_text)
            if mem_emb is None:
                similarities.append(0.0)
                continue
            
            mem_emb = mem_emb.to(self.device)
            
            # Fix 2: handle dimension mismatch between embeddings
            if query_emb.shape[-1] != mem_emb.shape[-1]:
                min_dim = min(query_emb.shape[-1], mem_emb.shape[-1])
                query_slice = query_emb[..., :min_dim]
                mem_slice = mem_emb[..., :min_dim]
                sim = F.cosine_similarity(query_slice.unsqueeze(0), mem_slice.unsqueeze(0)).item()
            else:
                # 余弦相似度（embedding 已经 L2 归一化）
                sim = F.cosine_similarity(query_emb.unsqueeze(0), mem_emb.unsqueeze(0)).item()
            similarities.append(max(0.0, min(1.0, sim)))
        
        return similarities
    
    def batch_compute_similarities(
        self,
        query_text: str,
        memory_embeddings: List[torch.Tensor]
    ) -> List[float]:
        """
        批量计算查询与预存 embedding 之间的相似度（高性能版本）
        
        Fix 2: 增加维度不一致时的 fallback 处理
        
        Args:
            query_text: 查询文本
            memory_embeddings: 预存的记忆 embedding 列表
            
        Returns:
            similarities: 相似度列表
        """
        if not query_text or not memory_embeddings:
            return [0.0] * len(memory_embeddings)
        
        query_emb = self.get_text_embedding(query_text)
        if query_emb is None:
            return [0.0] * len(memory_embeddings)
        
        query_emb = query_emb.to(self.device)
        query_dim = query_emb.shape[-1]
        
        # 批量计算
        _batch_mismatch_sims = {}
        valid_embeddings = []
        valid_indices = []
        for i, emb in enumerate(memory_embeddings):
            if emb is not None:
                emb = emb.to(self.device)
                # Fix 2: normalize to same dimension if needed
                if emb.shape[-1] != query_dim:
                    min_dim = min(emb.shape[-1], query_dim)
                    emb = emb[..., :min_dim]
                    emb = F.normalize(emb.float(), p=2, dim=-1)
                    query_slice = F.normalize(query_emb[..., :min_dim].float(), p=2, dim=-1)
                    # Compute similarity individually for mismatched dims
                    sim = torch.mm(emb.unsqueeze(0), query_slice.unsqueeze(1)).squeeze(1).item()
                    # Will be placed directly below
                    valid_embeddings.append(None)  # placeholder, handled separately
                    valid_indices.append(i)
                    # Store the pre-computed sim for this index
                    # BUG FIX: 使用局部变量确保线程安全
                    _batch_mismatch_sims[i] = max(0.0, min(1.0, sim))
                else:
                    valid_embeddings.append(emb)
                    valid_indices.append(i)
        
        if not valid_indices:
            return [0.0] * len(memory_embeddings)
        
        # Fix 2: 使用局部变量（线程安全）
        mismatch_sims = _batch_mismatch_sims
        
        # Build result with defaults
        result = [0.0] * len(memory_embeddings)
        
        # Place mismatched-dimension results first
        for idx in valid_indices:
            if idx in mismatch_sims:
                result[idx] = mismatch_sims[idx]
        
        # Batch compute for same-dimension embeddings
        same_dim_embeddings = []
        same_dim_indices = []
        for j, idx in enumerate(valid_indices):
            if valid_embeddings[j] is not None:
                same_dim_embeddings.append(valid_embeddings[j])
                same_dim_indices.append(idx)
        
        if same_dim_embeddings:
            # Use the query slice matching the embedding dimension
            emb_matrix = torch.stack(same_dim_embeddings).float()  # [N, hidden_size]
            actual_dim = emb_matrix.shape[-1]
            query_emb_f = F.normalize(query_emb[..., :actual_dim].float(), p=2, dim=-1)
            sims = torch.mm(emb_matrix, query_emb_f.unsqueeze(1)).squeeze(1)  # [N]
            sims = sims.tolist()
            
            for idx, sim in zip(same_dim_indices, sims):
                result[idx] = max(0.0, min(1.0, sim))
        
        return result
    
    def clear_cache(self):
        """清空 embedding 缓存"""
        self._embedding_cache.clear()
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计"""
        return {
            'cache_size': len(self._embedding_cache),
            'cache_max_size': self._cache_max_size,
        }
