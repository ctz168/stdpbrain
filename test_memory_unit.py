#!/usr/bin/env python3
"""
最小记忆系统单元测试 - 不需要加载完整模型

直接测试海马体记忆系统的核心召回逻辑：
1. 存储记忆（带语义embedding）
2. 干扰存储
3. 召回测试
4. 计算准确率
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hippocampus.ca3_memory import CA3EpisodicMemory, EpisodicMemory, MemoryTier
from hippocampus.memory_layers import MemoryConsolidationManager, TierConfig
from hippocampus.semantic_engine import SemanticSummarizer


class FakeSemanticEngine:
    """模拟语义引擎 - 不需要真实模型"""
    
    def __init__(self, dim=256):
        self.dim = dim
        self._embedding_cache = {}
    
    def get_text_embedding(self, text: str):
        if not text:
            return None
        # 使用确定性哈希生成embedding（模拟语义相似度）
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        # 基于文本内容生成语义向量
        vec = torch.zeros(self.dim)
        chars = list(text)
        for i, ch in enumerate(chars):
            idx = min(i, self.dim - 1)
            vec[idx] += ord(ch) / 65536.0
        
        # 添加语义相似性：相同关键词产生相似的向量
        semantic_keywords = {
            '名字': 0, '姓名': 0, '叫': 0, '小明': 1, '张三': 2,
            '城市': 3, '来自': 3, '深圳': 4, '北京': 5, '上海': 6,
            '电话': 7, '手机': 7, '手机号': 7, '138': 8, '139': 9,
            '工作': 10, '职业': 10, '程序员': 11, '腾讯': 12, '工程师': 13,
            '颜色': 14, '蓝色': 15, '红色': 16, '喜欢': 17, '爱好': 17,
            '年龄': 18, '几岁': 18, '28': 19, '25': 20,
        }
        
        for kw, slot in semantic_keywords.items():
            if kw in text:
                vec[slot] += 1.0  # 关键词在固定slot产生强信号
        
        # L2 normalize
        vec = F.normalize(vec, p=2, dim=-1)
        self._embedding_cache[text] = vec
        return vec
    
    def batch_compute_similarities(self, query_text, memory_embeddings):
        if not query_text or not memory_embeddings:
            return [0.0] * len(memory_embeddings)
        
        query_emb = self.get_text_embedding(query_text)
        if query_emb is None:
            return [0.0] * len(memory_embeddings)
        
        results = []
        for i, mem_emb in enumerate(memory_embeddings):
            if mem_emb is None:
                results.append(0.0)
                continue
            # Ensure same dimension
            min_dim = min(query_emb.shape[0], mem_emb.shape[0])
            q = query_emb[:min_dim]
            m = mem_emb[:min_dim]
            sim = F.cosine_similarity(q.unsqueeze(0), m.unsqueeze(0)).item()
            results.append(max(0.0, min(1.0, sim)))
        
        return results
    
    def generate_semantic_summary(self, user_input, ai_response, is_core=False):
        return {
            'semantic_summary': user_input[:60],
            'key_entities': '',
            'emotion_tag': '中性',
            'structured': ''
        }
    
    def get_cache_stats(self):
        return {'cache_size': len(self._embedding_cache)}


def test_memory_recall():
    """测试记忆系统的召回准确率"""
    print("=" * 60)
    print("  stdpbrain 记忆系统单元测试（无模型加载）")
    print("=" * 60)
    
    # 初始化
    feature_dim = 256
    semantic_engine = FakeSemanticEngine(dim=feature_dim)
    
    tier_config = TierConfig(
        short_term_recall_threshold=0.10,
        mid_term_recall_threshold=0.08,
        long_term_recall_threshold=0.05,
    )
    
    ca3 = CA3EpisodicMemory(
        max_capacity=10000,
        feature_dim=feature_dim,
        recall_threshold=0.05,  # 低阈值，不过滤
        semantic_engine=semantic_engine,
        tier_config=tier_config,
    )
    
    # ===== 步骤1: 存储个人信息 =====
    print("\n[1/4] 存储个人信息...")
    memories_to_store = [
        ("我叫小明，来自深圳", True, "name:小明 | location:深圳"),
        ("我的手机号是13812345678", True, "phone:13812345678"),
        ("我在腾讯做程序员", True, "company:腾讯 | job:程序员"),
        ("我最喜欢的颜色是蓝色", True, "hobby:蓝色"),
    ]
    
    for text, is_core, pointer in memories_to_store:
        # 生成embedding
        emb = semantic_engine.get_text_embedding(pointer if pointer else text)
        
        # 生成DG特征（模拟EC→DG输出）
        dg_features = torch.randn(feature_dim * 2) * 0.1
        dg_features = F.normalize(dg_features, p=2, dim=-1)
        
        # 添加语义信号到DG特征（让相同关键词的DG特征也更相似）
        for i, ch in enumerate(pointer):
            idx = min(i * 3, feature_dim * 2 - 1)
            dg_features[idx] += ord(ch) / 65536.0 * 5.0
        dg_features = F.normalize(dg_features, p=2, dim=-1)
        
        ca3.store(
            memory_id=f"mem_{hash(text) % 100000}",
            timestamp=int(time.time() * 1000),
            semantic_pointer=pointer,
            temporal_skeleton="",
            causal_links=[],
            dg_features=dg_features,
            is_core=is_core,
            content=text,
            user_input=text,
            ai_response="好的，我记住了",
        )
        print(f"  ✅ 存储: {text[:30]}")
    
    # 检查存储状态
    stats = ca3.get_stats()
    print(f"  记忆总数: {stats['num_memories']}")
    print(f"  核心记忆: {stats['core_memory_count']}")
    
    # ===== 步骤2: 干扰存储 =====
    print("\n[2/4] 存储干扰信息...")
    distractions = [
        "今天天气怎么样？",
        "给我讲一个笑话",
        "推荐一本好书",
        "中国有多少个省份？",
        "怎么做红烧肉？",
    ]
    for text in distractions:
        emb = semantic_engine.get_text_embedding(text)
        dg_features = torch.randn(feature_dim * 2) * 0.1
        dg_features = F.normalize(dg_features, p=2, dim=-1)
        ca3.store(
            memory_id=f"mem_{hash(text) % 100000}",
            timestamp=int(time.time() * 1000) + 1000,
            semantic_pointer=text,
            temporal_skeleton="",
            causal_links=[],
            dg_features=dg_features,
            is_core=False,
            content=text,
            user_input=text,
            ai_response="这是一个普通回答",
        )
    print(f"  ✅ 存储了 {len(distractions)} 条干扰信息")
    
    # ===== 步骤3: 召回测试 =====
    print("\n[3/4] 记忆召回测试...")
    recall_tests = [
        # (query, expected_keyword, search_in_fields)
        ("你还记得我叫什么名字吗？", "小明", ["semantic_pointer", "content", "key_entities"]),
        ("我来自哪个城市？", "深圳", ["semantic_pointer", "content", "key_entities"]),
        ("我的手机号是多少？", "13812345678", ["semantic_pointer", "content", "key_entities"]),
        ("我在哪里工作？", "腾讯", ["semantic_pointer", "content", "key_entities"]),
        ("我的职业是什么？", "程序员", ["semantic_pointer", "content", "key_entities"]),
        ("我最喜欢什么颜色？", "蓝色", ["semantic_pointer", "content", "key_entities"]),
        # 更多表达方式
        ("你知道我叫啥吗？", "小明", ["semantic_pointer", "content", "key_entities"]),
        ("我是什么职业？", "程序员", ["semantic_pointer", "content", "key_entities"]),
        ("我在哪个公司？", "腾讯", ["semantic_pointer", "content", "key_entities"]),
        ("我的电话号码？", "13812345678", ["semantic_pointer", "content", "key_entities"]),
    ]
    
    results = []
    for query, expected, search_fields in recall_tests:
        recalled = ca3.recall(
            query_semantic=query,
            topk=5,
        )
        
        # 检查召回的记忆中是否包含期望信息
        found = False
        best_match = ""
        best_score = 0
        for mem in recalled[:3]:
            for field in search_fields:
                val = getattr(mem, field, '') or ''
                if expected in val:
                    found = True
                    best_match = val[:60]
                    break
            if found:
                break
            # 计算最佳匹配度
            content = mem.content or ''
            pointer = mem.semantic_pointer or ''
            entities = mem.key_entities or ''
            combined = f"{content} {pointer} {entities}"
            if expected in combined:
                found = True
                best_match = combined[:60]
                break
        
        if found:
            results.append(("✅", query, expected, best_match))
            print(f"  ✅ '{query}' → 找到 '{expected}'")
        else:
            # 打印调试信息
            debug_info = []
            for mem in recalled[:3]:
                debug_info.append(f"    ptr={mem.semantic_pointer[:40] if mem.semantic_pointer else 'None'}")
                debug_info.append(f"    content={mem.content[:40] if mem.content else 'None'}")
                debug_info.append(f"    entities={mem.key_entities[:40] if mem.key_entities else 'None'}")
                debug_info.append(f"    emb_score={getattr(mem, '_embedding_score', 'N/A')}")
            results.append(("❌", query, expected, "\n".join(debug_info[:2])))
            print(f"  ❌ '{query}' → 未找到 '{expected}'")
            if recalled:
                for m in recalled[:2]:
                    print(f"     召回: ptr={m.semantic_pointer[:40]}, content={m.content[:40]}, emb={getattr(m, '_embedding_score', 'N/A')}")
    
    # ===== 步骤4: 结果汇总 =====
    print("\n[4/4] 结果汇总")
    correct = sum(1 for r in results if r[0] == "✅")
    total = len(results)
    accuracy = correct / total * 100
    
    print(f"\n  记忆召回准确率: {correct}/{total} = {accuracy:.1f}%")
    
    for status, query, expected, actual in results:
        print(f"  {status} Q: {query}")
        print(f"     期望: '{expected}'")
    
    if accuracy >= 80:
        print(f"\n  🎯 达标! 准确率 {accuracy:.1f}% >= 80%")
    else:
        print(f"\n  ⚠️ 未达标. 准确率 {accuracy:.1f}% < 80%")
    
    print(f"\n{'='*60}")
    
    # 额外诊断
    print("\n[诊断] 记忆库详情:")
    for mid, mem in ca3.memories.items():
        emb_status = "有" if mem.semantic_embedding is not None else "无"
        print(f"  [{mid[:15]}] core={mem.is_core} emb={emb_status} ptr={mem.semantic_pointer[:40]} content={mem.content[:30]}")
    
    return correct, total


if __name__ == "__main__":
    correct, total = test_memory_recall()
    sys.exit(0 if correct / total >= 0.8 else 1)
