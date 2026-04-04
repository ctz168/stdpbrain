#!/usr/bin/env python3
"""记忆效果快速测试 v2（max_tokens=50加速版）

流程：2轮存入 + 2轮干扰 + 3轮召回 = 7轮对话
预计 CPU 时间: 2-3分钟
"""

import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import config as secret_config


def test_memory():
    print("=" * 60)
    print("  stdpbrain 记忆效果快速测试 v2")
    print("=" * 60)
    
    # Step 1: 初始化
    print("\n[1/4] 初始化 BrainAI 系统...", flush=True)
    t0 = time.time()
    
    from core.interfaces import BrainAIInterface
    from configs.arch_config import BrainAIConfig
    
    brain_config = BrainAIConfig()
    model_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "Qwen3.5-0.8B"))
    brain_config.model_path = model_abs_path
    if hasattr(secret_config, 'QUANTIZATION'):
        brain_config.quantization = secret_config.QUANTIZATION
    
    brain = BrainAIInterface(brain_config, device="cpu")
    print(f"  BrainAI 初始化完成 ({time.time()-t0:.1f}s)", flush=True)
    
    # Step 2: 存入关键信息（2轮）
    print("\n[2/4] 第一阶段：存入个人信息（2轮）...", flush=True)
    info_pairs = [
        "我叫小明，来自深圳，在腾讯做程序员",
        "我最喜欢的颜色是蓝色，手机号13812345678",
    ]
    
    for user_input in info_pairs:
        print(f"\n  用户: {user_input}", flush=True)
        try:
            t1 = time.time()
            response = brain.chat(user_input, max_tokens=50)
            elapsed = time.time() - t1
            short = response[:100] + "..." if len(response) > 100 else response
            print(f"  AI ({elapsed:.1f}s): {short}", flush=True)
        except Exception as e:
            print(f"  错误: {e}", flush=True)
            import traceback
            traceback.print_exc()
            break
    
    time.sleep(0.5)
    
    # Step 3: 干扰对话（2轮）
    print("\n[3/4] 第二阶段：干扰对话（2轮）...", flush=True)
    distractions = [
        "今天天气怎么样？",
        "推荐一本好书",
    ]
    
    for d in distractions:
        print(f"\n  用户: {d}", flush=True)
        try:
            t1 = time.time()
            response = brain.chat(d, max_tokens=50)
            elapsed = time.time() - t1
            short = response[:80] + "..." if len(response) > 80 else response
            print(f"  AI ({elapsed:.1f}s): {short}", flush=True)
        except Exception as e:
            print(f"  错误: {e}", flush=True)
            break
    
    time.sleep(0.5)
    
    # Step 4: 回忆测试（3轮）
    print("\n[4/4] 第三阶段：记忆召回测试（3轮）...", flush=True)
    recall_tests = [
        ("你还记得我叫什么名字吗？", "小明"),
        ("我来自哪个城市？", "深圳"),
        ("我在哪里工作？做什么的？", "腾讯"),
    ]
    
    results = []
    for question, expected in recall_tests:
        print(f"\n  用户: {question}", flush=True)
        print(f"  期望回答包含: '{expected}'", flush=True)
        try:
            t1 = time.time()
            response = brain.chat(question, max_tokens=50)
            elapsed = time.time() - t1
            short = response[:150] + "..." if len(response) > 150 else response
            print(f"  AI ({elapsed:.1f}s): {short}", flush=True)
            
            if expected in response:
                results.append(("PASS", question, expected, response[:100]))
                print(f"  结果: PASS - 正确回忆", flush=True)
            else:
                results.append(("FAIL", question, expected, response[:100]))
                print(f"  结果: FAIL - 未能回忆", flush=True)
        except Exception as e:
            results.append(("ERROR", question, expected, str(e)))
            print(f"  错误: {e}", flush=True)
    
    # 检查内部状态
    print("\n--- 记忆系统内部状态 ---", flush=True)
    try:
        stats = brain.get_stats()
        hippocampus = stats.get('hippocampus', {})
        print(f"  记忆数量: {hippocampus.get('num_memories', 'N/A')}", flush=True)
        print(f"  记忆使用: {hippocampus.get('memory_usage_mb', 0):.2f} MB", flush=True)
        
        last_recalled = getattr(brain, '_last_recalled_memories', [])
        if last_recalled:
            print(f"  最后一次召回的记忆数量: {len(last_recalled)}", flush=True)
            for i, mem in enumerate(last_recalled[:3]):
                if isinstance(mem, dict):
                    content = mem.get('content', '')[:80]
                else:
                    content = str(mem)[:80]
                print(f"    [{i+1}] {content}", flush=True)
        else:
            print("  最后一次召回的记忆: (空)", flush=True)
    except Exception as e:
        print(f"  获取状态失败: {e}", flush=True)
    
    # 汇总
    total_time = time.time() - t0
    print("\n" + "=" * 60, flush=True)
    print("  测试结果汇总", flush=True)
    print("=" * 60, flush=True)
    correct = sum(1 for r in results if r[0] == "PASS")
    total = len(results)
    print(f"\n  记忆召回准确率: {correct}/{total} = {correct/total*100:.1f}%", flush=True)
    print(f"  总耗时: {total_time:.1f}s", flush=True)
    
    for status, question, expected, actual in results:
        print(f"  [{status}] {question}", flush=True)
        print(f"     期望: '{expected}' | 实际: {actual[:60]}", flush=True)
    
    print(f"\n{'='*60}", flush=True)
    return correct, total


if __name__ == "__main__":
    test_memory()
