#!/usr/bin/env python3
"""记忆效果专项测试

测试流程：
1. 存入个人信息（名字、城市、职业等）
2. 进行多轮无关对话（干扰）
3. 回忆之前存储的信息
4. 评估记忆召回准确率
"""

import sys
import os
import time

# 必须在导入其他模块前执行 hotfix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置 HF 离线模式，完全绕过 hotfix 的递归问题
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import config as secret_config


def test_memory():
    print("=" * 60)
    print("  stdpbrain 记忆效果专项测试")
    print("=" * 60)
    
    # Step 1: 初始化（与 main.py 一致）
    print("\n[1/5] 初始化 BrainAI 系统...")
    t0 = time.time()
    
    from core.interfaces import BrainAIInterface
    from configs.arch_config import BrainAIConfig
    
    # 使用标准配置
    brain_config = BrainAIConfig()
    # 设置模型路径
    model_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "Qwen3.5-0.8B"))
    brain_config.model_path = model_abs_path
    if hasattr(secret_config, 'QUANTIZATION'):
        brain_config.quantization = secret_config.QUANTIZATION
    
    brain = BrainAIInterface(brain_config, device="cpu")
    
    print(f"  BrainAI 初始化完成 ({time.time()-t0:.1f}s)")
    
    # Step 2: 存入信息
    print("\n[2/5] 第一轮：存入个人信息...")
    info_pairs = [
        "我叫小明，来自深圳",
        "我的手机号是13812345678",
        "我在腾讯做程序员",
        "我最喜欢的颜色是蓝色",
    ]
    
    for user_input in info_pairs:
        print(f"\n  用户: {user_input}")
        try:
            response = brain.chat(user_input)
            print(f"  AI: {response[:80]}..." if len(response) > 80 else f"  AI: {response}")
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # 等记忆巩固
    time.sleep(1)
    
    # Step 3: 干扰对话
    print("\n[3/5] 第二轮：干扰对话（5轮无关话题）...")
    distractions = [
        "今天天气怎么样？",
        "给我讲一个笑话",
        "推荐一本好书",
        "中国有多少个省份？",
        "怎么做红烧肉？",
    ]
    
    for d in distractions:
        print(f"\n  用户: {d}")
        try:
            response = brain.chat(d)
            print(f"  AI: {response[:60]}..." if len(response) > 60 else f"  AI: {response}")
        except Exception as e:
            print(f"  错误: {e}")
            break
    
    # 等待记忆巩固
    time.sleep(1)
    
    # Step 4: 回忆测试
    print("\n[4/5] 第三轮：记忆召回测试...")
    recall_tests = [
        ("你还记得我叫什么名字吗？", "小明"),
        ("我来自哪个城市？", "深圳"),
        ("我的手机号是多少？", "13812345678"),
        ("我在哪里工作？", "腾讯"),
        ("我的职业是什么？", "程序员"),
        ("我最喜欢什么颜色？", "蓝色"),
    ]
    
    results = []
    for question, expected in recall_tests:
        print(f"\n  用户: {question}")
        print(f"  期望回答包含: '{expected}'")
        try:
            response = brain.chat(question)
            print(f"  AI: {response[:100]}..." if len(response) > 100 else f"  AI: {response}")
            
            # 检查是否包含期望信息
            if expected in response:
                results.append(("✅", question, expected, response[:80]))
                print(f"  结果: ✅ 正确回忆")
            else:
                results.append(("❌", question, expected, response[:80]))
                print(f"  结果: ❌ 未能回忆")
        except Exception as e:
            results.append(("💥", question, expected, str(e)))
            print(f"  错误: {e}")
    
    # Step 5: 检查记忆系统内部状态
    print("\n[5/5] 记忆系统内部状态...")
    try:
        stats = brain.get_stats()
        hippocampus = stats.get('hippocampus', {})
        print(f"  记忆数量: {hippocampus.get('num_memories', 'N/A')}")
        print(f"  记忆使用: {hippocampus.get('memory_usage_mb', 0):.2f} MB")
        print(f"  对话轮数: {stats.get('conversation_cycles', 'N/A')}")
        print(f"  STDP更新次数: {stats.get('total_stdp_updates', 'N/A')}")
        
        # 检查召回的记忆
        last_recalled = brain._last_recalled_memories
        if last_recalled:
            print(f"\n  最后一次召回的记忆数量: {len(last_recalled)}")
            for i, mem in enumerate(last_recalled[:3]):
                if isinstance(mem, dict):
                    content = mem.get('content', '')[:80]
                    sp = mem.get('semantic_pointer', '')[:60]
                else:
                    content = str(mem)[:80]
                    sp = ''
                print(f"    [{i+1}] content: {content}")
                if sp:
                    print(f"        pointer: {sp}")
        else:
            print("  最后一次召回的记忆: (空)")
    except Exception as e:
        print(f"  获取状态失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    correct = sum(1 for r in results if r[0] == "✅")
    total = len(results)
    print(f"\n  记忆召回准确率: {correct}/{total} = {correct/total*100:.1f}%")
    
    for status, question, expected, actual in results:
        print(f"  {status} 问题: {question}")
        print(f"     期望: '{expected}' | 实际: {actual[:50]}")
    
    print(f"\n{'='*60}")
    
    # 保存状态
    try:
        brain.save_state("test_memory_state.pt")
        print("  记忆状态已保存到 test_memory_state.pt")
    except Exception as e:
        print(f"  保存状态失败: {e}")
    
    return correct, total


if __name__ == "__main__":
    test_memory()
