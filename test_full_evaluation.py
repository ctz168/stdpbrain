#!/usr/bin/env python3
"""
全面聊天评估脚本 - 3轮优化 + 5轮流式聊天（含独白流评估）

测试内容:
1. 3轮单次聊天（评估输出质量、截断、独白流）
2. 5轮流式聊天（评估流式输出、思维连续性、独白流自然度）
"""

import sys
import os
import time
import traceback

# 确保在项目根目录
os.chdir('/home/z/my-project/stdpbrain')
sys.path.insert(0, '/home/z/my-project/stdpbrain')

from configs.arch_config import BrainAIConfig


def print_separator(title, char="="):
    print(f"\n{char * 60}")
    print(f"  {title}")
    print(f"{char * 60}\n")


def evaluate_response(user_input, response, monologue="", elapsed_ms=0):
    """评估回复质量"""
    scores = {}
    
    # 1. 长度评估 (不能太短也不能太长)
    scores['length'] = min(1.0, len(response) / 50)  # 至少50字
    
    # 2. 内容质量 (检查是否有实质内容)
    has_substance = len(response) > 20 and len(set(response)) > 10
    scores['substance'] = 1.0 if has_substance else 0.0
    
    # 3. 无乱码 (检查是否包含模型标签残留)
    bad_tags = ['<|', '|>', '<think', '</think', 'speaker', '内心思维', '【', '】']
    has_garbage = any(tag in response for tag in bad_tags)
    scores['clean'] = 1.0 if not has_garbage else 0.0
    
    # 4. 语言自然度 (简单检查)
    chinese_chars = sum(1 for c in response if '\u4e00' <= c <= '\u9fff')
    total_chars = len(response)
    scores['natural'] = min(1.0, chinese_chars / max(total_chars * 0.5, 1))
    
    # 5. 独白流评估
    if monologue:
        scores['monologue_length'] = min(1.0, len(monologue) / 20)  # 至少20字
        scores['monologue_clean'] = 1.0 if not any(tag in monologue for tag in bad_tags) else 0.0
    else:
        scores['monologue_length'] = 0.0
        scores['monologue_clean'] = 0.0
    
    # 6. 性能
    scores['speed'] = min(1.0, 5000 / max(elapsed_ms, 1))  # 5秒内完成得满分
    
    # 综合分
    weights = {
        'substance': 0.3, 'clean': 0.25, 'natural': 0.2,
        'length': 0.1, 'monologue_length': 0.1, 'speed': 0.05
    }
    total = sum(scores[k] * weights[k] for k in weights)
    
    return {
        'scores': scores,
        'total': total,
        'has_garbage': has_garbage
    }


def run_single_chat_test(ai, user_input, history, round_num):
    """运行单次聊天测试"""
    print(f"  用户: {user_input}")
    print(f"  等待回复...", end="", flush=True)
    
    start = time.time()
    try:
        response = ai.chat(user_input, history=history, max_tokens=256)
        elapsed = (time.time() - start) * 1000
        
        # 获取独白
        monologue = ""
        if hasattr(ai, 'inner_thought_engine') and ai.inner_thought_engine:
            last_thought = ai.inner_thought_engine.last_thought
            if last_thought:
                monologue = last_thought[:100]
        
        eval_result = evaluate_response(user_input, response, monologue, elapsed)
        
        print(f"\r  AI: {response[:200]}{'...' if len(response) > 200 else ''}")
        if monologue:
            print(f"  💭 独白: {monologue[:80]}{'...' if len(monologue) > 80 else ''}")
        print(f"  ⏱️ {elapsed:.0f}ms | 综合分: {eval_result['total']:.2f} | 乱码: {'有' if eval_result['has_garbage'] else '无'}")
        
        return {
            'user_input': user_input,
            'response': response,
            'monologue': monologue,
            'elapsed_ms': elapsed,
            'eval': eval_result
        }
    except Exception as e:
        print(f"\r  ❌ 错误: {e}")
        traceback.print_exc()
        return None


def run_stream_chat_test(ai, user_input, history, round_num):
    """运行流式聊天测试"""
    import asyncio
    
    print(f"  用户: {user_input}")
    
    start = time.time()
    full_response = ""
    thinking_text = ""
    monologue_found = False
    urge_score = 0.0
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def stream_chat():
            nonlocal thinking_text, full_response, monologue_found, urge_score
            
            async for event in ai.chat_stream(user_input, history):
                if event["type"] == "thinking":
                    thinking_text += event["content"]
                    # 实时显示
                    display = thinking_text[-40:] if len(thinking_text) > 40 else thinking_text
                    print(f"\r  💭 [思考中] {display}    ", end="", flush=True)
                elif event["type"] == "thinking_done":
                    if thinking_text.strip():
                        print(f"\r  💭 [独白] {thinking_text.strip()[:100]}")
                        monologue_found = True
                    print(f"  AI: ", end="", flush=True)
                elif event["type"] == "chunk":
                    full_response += event["content"]
                    print(event["content"], end="", flush=True)
            
            return full_response
        
        loop.run_until_complete(stream_chat())
        loop.close()
        
        elapsed = (time.time() - start) * 1000
        print()
        
        # 获取独白引擎状态
        if hasattr(ai, 'inner_thought_engine') and ai.inner_thought_engine:
            urge_score = getattr(ai.inner_thought_engine, '_last_urge_to_speak', 0.0)
        
        eval_result = evaluate_response(user_input, full_response, thinking_text, elapsed)
        
        print(f"  ⏱️ {elapsed:.0f}ms | 综合分: {eval_result['total']:.2f} | 独白: {'有' if monologue_found else '无'} | urge: {urge_score:.2f}")
        
        # 检查隐藏状态连续性
        hs_ok = ai.current_thought_state is not None
        print(f"  🧠 隐藏状态: {'✓ 连续' if hs_ok else '✗ 断裂'}")
        
        return {
            'user_input': user_input,
            'response': full_response,
            'monologue': thinking_text,
            'elapsed_ms': elapsed,
            'eval': eval_result,
            'urge': urge_score,
            'hidden_state_ok': hs_ok
        }
    except Exception as e:
        print(f"\n  ❌ 错误: {e}")
        traceback.print_exc()
        return None


def main():
    print_separator("类人脑AI 全面评估 - 3轮优化 + 5轮流式聊天")
    
    # ========== 初始化 ==========
    print("正在加载模型和系统...")
    config = BrainAIConfig()
    
    from core.interfaces import BrainAIInterface
    ai = BrainAIInterface(config)
    print("✅ 系统初始化完成")
    
    # ========== 阶段1: 3轮单次聊天 ==========
    print_separator("阶段1: 3轮单次聊天测试")
    
    single_chat_inputs = [
        "你好，介绍一下你自己",
        "我叫张三，我来自北京，我喜欢编程",
        "你还记得我叫什么名字吗？",
    ]
    
    history = []
    single_results = []
    
    for i, user_input in enumerate(single_chat_inputs):
        print(f"\n--- 第 {i+1} 轮 ---")
        result = run_single_chat_test(ai, user_input, history, i+1)
        if result:
            single_results.append(result)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result['response']})
    
    # 阶段1评估
    if single_results:
        avg_total = sum(r['eval']['total'] for r in single_results) / len(single_results)
        avg_clean = sum(r['eval']['scores']['clean'] for r in single_results) / len(single_results)
        avg_natural = sum(r['eval']['scores']['natural'] for r in single_results) / len(single_results)
        avg_monologue = sum(r['eval']['scores']['monologue_length'] for r in single_results) / len(single_results)
        
        print_separator("阶段1 评估结果")
        print(f"  平均综合分: {avg_total:.2f}")
        print(f"  平均洁净度: {avg_clean:.2f}")
        print(f"  平均自然度: {avg_natural:.2f}")
        print(f"  平均独白分: {avg_monologue:.2f}")
        print(f"  记忆召回测试: {'✓ 通过' if '张三' in single_results[-1]['response'] else '✗ 未通过'}")
    
    # ========== 阶段2: 5轮流式聊天 ==========
    print_separator("阶段2: 5轮流式聊天测试（含独白流评估）")
    
    stream_chat_inputs = [
        "今天天气怎么样？",
        "3+5等于几？",
        "你觉得人工智能会有意识吗？",
        "帮我分析一下学习和工作的平衡",
        "谢谢你，再见",
    ]
    
    history2 = []
    stream_results = []
    
    for i, user_input in enumerate(stream_chat_inputs):
        print(f"\n--- 流式第 {i+1} 轮 ---")
        result = run_stream_chat_test(ai, user_input, history2, i+1)
        if result:
            stream_results.append(result)
            history2.append({"role": "user", "content": user_input})
            history2.append({"role": "assistant", "content": result['response']})
    
    # 阶段2评估
    if stream_results:
        avg_total = sum(r['eval']['total'] for r in stream_results) / len(stream_results)
        avg_clean = sum(r['eval']['scores']['clean'] for r in stream_results) / len(stream_results)
        avg_urge = sum(r.get('urge', 0) for r in stream_results) / len(stream_results)
        hs_continuous = all(r.get('hidden_state_ok', False) for r in stream_results)
        monologue_present = sum(1 for r in stream_results if r.get('monologue', '') and len(r['monologue']) > 10)
        
        print_separator("阶段2 评估结果")
        print(f"  平均综合分: {avg_total:.2f}")
        print(f"  平均洁净度: {avg_clean:.2f}")
        print(f"  平均 urge 分: {avg_urge:.2f}")
        print(f"  独白流出现: {monologue_present}/{len(stream_results)} 轮")
        print(f"  隐藏状态连续: {'✓ 全程连续' if hs_continuous else '✗ 存在断裂'}")
        print(f"  数学测试: {'✓ 通过' if stream_results[1] and ('8' in stream_results[1]['response']) else '✗ 未通过'}")
    
    # ========== 最终总结 ==========
    print_separator("最终评估总结")
    all_results = single_results + stream_results
    if all_results:
        overall_avg = sum(r['eval']['total'] for r in all_results) / len(all_results)
        overall_clean = sum(r['eval']['scores']['clean'] for r in all_results) / len(all_results)
        
        print(f"  总测试轮数: {len(all_results)}")
        print(f"  总体平均分: {overall_avg:.2f}")
        print(f"  总体洁净度: {overall_clean:.2f}")
        
        if overall_avg >= 0.7:
            print(f"\n  ✅ 评估通过 (>= 0.7)")
        elif overall_avg >= 0.5:
            print(f"\n  ⚠️ 评估基本通过 (0.5-0.7)，仍需优化")
        else:
            print(f"\n  ❌ 评估未通过 (< 0.5)，需要重大改进")
    
    # 保存状态
    ai.save_state("brain_state.pt")
    print("\n[状态已保存]")


if __name__ == "__main__":
    main()
