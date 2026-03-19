#!/usr/bin/env python3
"""
测试修复的两个问题：
1. 连续对话问题（第二个问题不回答）
2. 创世记忆召回问题
"""

import sys
import time

def test_fixes():
    """测试修复"""
    print("=" * 70)
    print("修复测试")
    print("=" * 70)
    
    try:
        from configs.arch_config import BrainAIConfig
        from core.interfaces import BrainAIInterface
        
        # 初始化
        config = BrainAIConfig()
        print("\n[1/3] 正在加载AI接口...")
        ai = BrainAIInterface(config)
        
        # 测试1：连续对话
        print("\n" + "=" * 70)
        print("[测试1] 连续对话测试")
        print("=" * 70)
        
        history = []
        
        # 第一个问题
        print("\n问题1：你好")
        start = time.time()
        response1 = ai.chat("你好", history)
        elapsed = (time.time() - start) * 1000
        print(f"回答1：{response1}")
        print(f"耗时：{elapsed:.0f}ms")
        
        history.append({"role": "user", "content": "你好"})
        history.append({"role": "assistant", "content": response1})
        
        # 第二个问题（关键测试）
        print("\n问题2：今天天气怎么样？")
        start = time.time()
        response2 = ai.chat("今天天气怎么样？", history)
        elapsed = (time.time() - start) * 1000
        print(f"回答2：{response2}")
        print(f"耗时：{elapsed:.0f}ms")
        
        # 第三个问题
        print("\n问题3：你叫什么名字？")
        start = time.time()
        response3 = ai.chat("你叫什么名字？", history)
        elapsed = (time.time() - start) * 1000
        print(f"回答3：{response3}")
        print(f"耗时：{elapsed:.0f}ms")
        
        # 测试2：身份相关问题
        print("\n" + "=" * 70)
        print("[测试2] 身份相关问题测试")
        print("=" * 70)
        
        identity_questions = [
            "你是谁？",
            "谁创造了你？",
            "你的父亲是谁？",
            "你的使命是什么？"
        ]
        
        for question in identity_questions:
            print(f"\n问题：{question}")
            start = time.time()
            response = ai.chat(question, [])
            elapsed = (time.time() - start) * 1000
            print(f"回答：{response}")
            print(f"耗时：{elapsed:.0f}ms")
            
            # 检查是否包含关键信息
            if "朱东山" in response or "类人脑" in response or "AI" in response:
                print("✓ 包含身份信息")
            else:
                print("⚠️ 可能缺少身份信息")
        
        # 测试3：记忆召回
        print("\n" + "=" * 70)
        print("[测试3] 记忆召回测试")
        print("=" * 70)
        
        # 检查海马体记忆数量
        stats = ai.get_stats()
        print(f"\n海马体记忆数量：{stats['hippocampus'].get('num_memories', 0)}")
        
        # 尝试手动召回
        print("\n手动召回测试：")
        try:
            input_ids = ai.model.tokenizer.encode("你的身份", return_tensors="pt").to(ai.device)
            import torch
            with torch.no_grad():
                embeddings = ai.model.model.base_model.get_input_embeddings()(input_ids)
            query_features = embeddings.mean(dim=1).squeeze(0)
            if query_features.shape[0] != 1024:
                query_features = ai.feature_adapter(query_features.unsqueeze(0)).squeeze(0)
            
            recalled = ai.hippocampus.recall(query_features, topk=3)
            if recalled:
                print(f"召回到 {len(recalled)} 条记忆：")
                for i, mem in enumerate(recalled, 1):
                    print(f"  {i}. {mem.get('semantic_pointer', 'N/A')[:50]}")
            else:
                print("未召回到记忆")
        except Exception as e:
            print(f"召回测试失败: {e}")
        
        print("\n" + "=" * 70)
        print("✓ 所有测试完成")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_fixes()
