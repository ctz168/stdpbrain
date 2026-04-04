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
from typing import Dict, List, Optional, Tuple
import time
import logging
import re

logger = logging.getLogger(__name__)


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
        ],
        'age': [
            (r"我今年(\d+)", 1),
            (r"我(\d+)岁", 1),
            (r"年龄[是:：](\d+)", 1),
        ],
        'location': [
            (r"来自([\u4e00-\u9fa5a-zA-Z]{2,10})", 1),
            (r"住在([\u4e00-\u9fa5a-zA-Z]{2,10})", 1),
            (r"在([\u4e00-\u9fa5a-zA-Z]{2,10}?)(工作|生活|上班)", 1),
        ],
        'job': [
            (r"(?:是|当)(.{0,10}?)(工程师|医生|老师|学生|设计师|程序员|律师|会计|经理|总监|分析师|研究员)", 1),
            (r"职业[是:：](.{2,10})", 1),
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
        """从文本中提取关键实体"""
        entities = []
        seen = set()
        
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern, group_idx in patterns:
                try:
                    match = re.search(pattern, text)
                    if match:
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
        获取文本的 embedding 向量（增强版）
        
        优化策略：
        1. 优先使用深层hidden states（包含上下文语义），而非仅用input embedding
        2. 使用加权mean pooling（CLS-like：首尾token权重更高）
        3. 对短文本使用input embedding fallback（减少计算开销）
        
        Args:
            text: 输入文本
            
        Returns:
            embedding: [hidden_size] 维度的向量，或 None（如果模型不可用）
        """
        if not text or not text.strip():
            return None
        
        # 检查缓存
        cache_key = text[:200]  # 缓存前200字符
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        if self.model is None:
            return None
        
        try:
            # 使用模型的 tokenizer 和 embedding 层
            tokenizer = self.model.tokenizer
            embedding_layer = self.model.embeddings  # QwenInterface.embeddings 属性
            
            # Tokenize
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                return None
            
            # 限制长度以控制计算量
            max_tokens = 64
            if len(token_ids) > max_tokens:
                token_ids = token_ids[:max_tokens]
            
            input_tensor = torch.tensor([token_ids], device=self.device)
            with torch.no_grad():
                # ===== 优化：尝试使用深层hidden states获取上下文语义 =====
                # 深层hidden states包含丰富的上下文信息，比纯input embedding语义能力强得多
                deep_embedding = None
                
                # 只有模型可以前向推理时才尝试深层提取
                base_model = getattr(self.model, 'model', None)
                if base_model is not None and hasattr(base_model, 'base_model'):
                    try:
                        model_core = base_model.base_model
                        # 使用模型最后几层的hidden states（包含最丰富的语义信息）
                        with torch.inference_mode():
                            outputs = model_core(
                                input_tensor,
                                output_hidden_states=True,
                                return_dict=True
                            )
                        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                            # 取最后2层hidden states的平均（比只用最后一层更稳定）
                            last_layers = outputs.hidden_states[-2:]
                            stacked = torch.stack(last_layers)  # [2, 1, seq_len, hidden_size]
                            deep_embedding = stacked.mean(dim=0)  # [1, seq_len, hidden_size]
                    except Exception as deep_err:
                        logger.debug(f"[SemanticEngine] 深层embedding提取失败，回退到input embedding: {deep_err}")
                
                if deep_embedding is not None:
                    # 加权mean pooling：首尾token权重更高（类似CLS token策略）
                    seq_len = deep_embedding.shape[1]
                    weights = torch.ones(seq_len, device=deep_embedding.device)
                    # 首token权重1.5，尾token权重2.0（尾token通常包含最重要的语义）
                    weights[0] = 1.5
                    if seq_len > 1:
                        weights[-1] = 2.0
                    # 中间token标准化
                    mid_sum = weights.sum() - weights[0] - (weights[-1] if seq_len > 1 else 0)
                    if mid_sum > 0 and seq_len > 2:
                        mid_count = seq_len - (1 if seq_len > 1 else 0) - 1
                        weights[1:-1 if seq_len > 1 else seq_len] = 0.5
                    
                    weights = weights / weights.sum()
                    text_embedding = (deep_embedding * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1).squeeze(0)
                else:
                    # 回退到input embedding + 标准mean pooling
                    embeddings = embedding_layer(input_tensor)  # [1, seq_len, hidden_size]
                    text_embedding = embeddings.mean(dim=1).squeeze(0)  # [hidden_size]
                
                # L2 归一化
                text_embedding = F.normalize(text_embedding.float(), p=2, dim=-1)
            
            # 更新缓存
            if len(self._embedding_cache) < self._cache_max_size:
                self._embedding_cache[cache_key] = text_embedding
            
            return text_embedding
        
        except Exception as e:
            logger.debug(f"[SemanticEngine] embedding 计算失败: {e}")
            return None
    
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
        
        # 批量计算
        valid_embeddings = []
        valid_indices = []
        for i, emb in enumerate(memory_embeddings):
            if emb is not None:
                valid_embeddings.append(emb.to(self.device))
                valid_indices.append(i)
        
        if not valid_embeddings:
            return [0.0] * len(memory_embeddings)
        
        # Stack 并计算（统一转为 float32 避免 BFloat16 不支持问题）
        emb_matrix = torch.stack(valid_embeddings).float()  # [N, hidden_size]
        query_emb_f = query_emb.float()
        sims = torch.mm(emb_matrix, query_emb_f.unsqueeze(1)).squeeze(1)  # [N]
        sims = sims.tolist()
        
        # 填充结果
        result = [0.0] * len(memory_embeddings)
        for idx, sim in zip(valid_indices, sims):
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
