#!/usr/bin/env python3
"""
持续思维流观察测试 - 监测内心独白与讲话分离

核心目标:
1. 观察AI的内心独白流（思维流）是否自然连贯
2. 观察独白流与正式讲话是否正确分离
3. 不截断输出，让思维自然流动
4. 记录完整的思维过程用于评估

测试流程:
- 启动AI后，不发送任何输入，观察自发独白
- 发送简单问题，观察 思考过程 → 回复过程 的分离
- 多轮测试，观察思维连贯性
"""

import sys
import os
import time
import threading
from datetime import datetime

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_monologue_flow():
    """测试持续思维流 - 非交互式，自动运行3轮"""
    
    print("=" * 70)
    print("  STDPBrain 持续思维流观察测试")
    print("  监测内心独白与讲话分离")
    print("=" * 70)
    
    # ========== 1. 初始化AI ==========
    from configs.arch_config import BrainAIConfig
    import config as secret_config
    
    config = BrainAIConfig()
    config.model_path = secret_config.MODEL_PATH
    config.quantization = getattr(secret_config, 'QUANTIZATION', config.quantization)
    config.QUANTIZATION = config.quantization
    
    from core.interfaces import BrainAIInterface
    ai = BrainAIInterface(config, device=None)
    
    results = {
        "rounds": [],
        "monologues": [],
        "speeches": [],
        "mind_states": [],
        "urges": [],
        "timing": []
    }
    
    print("\n" + "=" * 70)
    print("[初始化完成] AI已加载，开始思维流观察")
    print("=" * 70)
    
    # ========== 2. 测试1：自发独白观察（无用户输入） ==========
    print("\n" + "=" * 70)
    print("[测试1] 自发独白观察 - 无用户输入时的自然思维流")
    print("=" * 70)
    
    for i in range(3):
        print(f"\n--- 自发独白 #{i+1} ---")
        t_start = time.time()
        monologue_text = ""
        
        try:
            if ai.inner_thought_engine:
                state_before = ai.inner_thought_engine.mind_state.value
                urge_before = ai.inner_thought_engine._last_urge_to_speak
                
                for char in ai.inner_thought_engine.generate_inner_thought(
                    external_stimulus="",
                    max_tokens=150
                ):
                    monologue_text += char
                    print(char, end="", flush=True)
                
                state_after = ai.inner_thought_engine.mind_state.value
                urge_after = ai.inner_thought_engine._last_urge_to_speak
                
                elapsed = time.time() - t_start
                print(f"\n  [状态] {state_before} → {state_after}")
                print(f"  [开口欲望] {urge_before:.2f} → {urge_after:.2f}")
                print(f"  [耗时] {elapsed:.1f}s")
                print(f"  [字数] {len(monologue_text)}")
                
                results["monologues"].append({
                    "round": f"自发#{i+1}",
                    "text": monologue_text,
                    "state_before": state_before,
                    "state_after": state_after,
                    "urge_before": urge_before,
                    "urge_after": urge_after,
                    "elapsed": elapsed,
                    "char_count": len(monologue_text)
                })
        except Exception as e:
            print(f"\n  [错误] {e}")
            import traceback
            traceback.print_exc()
    
    # ========== 3. 测试2：对话模式 - 观察思考与讲话分离 ==========
    test_questions = [
        "你好，请介绍一下你自己。",
        "你觉得人类最伟大的发明是什么？为什么？",
        "如果让你给一个悲伤的人建议，你会说什么？"
    ]
    
    for round_idx, question in enumerate(test_questions):
        print("\n" + "=" * 70)
        print(f"[测试2-轮{round_idx+1}] 对话测试 - 观察思考→讲话分离")
        print("=" * 70)
        print(f"\n用户: {question}")
        
        t_start = time.time()
        
        # A. 观察思考过程（内心独白）
        print(f"\n💭 [内心独白开始]", flush=True)
        thinking_text = ""
        try:
            if ai.inner_thought_engine:
                state_before = ai.inner_thought_engine.mind_state.value
                urge_before = ai.inner_thought_engine._last_urge_to_speak
                
                for char in ai.inner_thought_engine.generate_inner_thought(
                    external_stimulus=question,
                    max_tokens=100
                ):
                    thinking_text += char
                    print(char, end="", flush=True)
                
                state_after = ai.inner_thought_engine.mind_state.value
                urge_after = ai.inner_thought_engine._last_urge_to_speak
                
                print(f"\n💭 [内心独白结束]")
                print(f"  [思维状态] {state_before} → {state_after}")
                print(f"  [开口欲望] {urge_before:.2f} → {urge_after:.2f}")
                print(f"  [独白字数] {len(thinking_text)}")
        except Exception as e:
            print(f"\n  [独白错误] {e}")
            thinking_text = ""
        
        t_think_done = time.time()
        
        # B. 生成正式回复（讲话）
        print(f"\n🗣️ [正式回复开始]", flush=True)
        response_text = ""
        try:
            response = ai.chat(question, history=[], max_tokens=200, thinking=False)
            response_text = response
            elapsed_speak = time.time() - t_think_done
            print(response)
            print(f"🗣️ [正式回复结束]")
            print(f"  [回复字数] {len(response_text)}")
            print(f"  [回复耗时] {elapsed_speak:.1f}s")
        except Exception as e:
            print(f"\n  [回复错误] {e}")
        
        t_total = time.time() - t_start
        
        # 记录结果
        results["rounds"].append({
            "round": f"对话轮{round_idx+1}",
            "question": question,
            "thinking": thinking_text,
            "response": response_text,
            "total_time": t_total,
            "think_time": t_think_done - t_start,
            "speak_time": t_total - (t_think_done - t_start)
        })
        
        print(f"\n  [总耗时] {t_total:.1f}s (思考{t_think_done-t_start:.1f}s + 回复{t_total-(t_think_done-t_start):.1f}s)")
    
    # ========== 4. 测试3：连续思维流 - 思维链测试 ==========
    print("\n" + "=" * 70)
    print("[测试3] 连续思维链 - 观察思维是否连贯流转")
    print("=" * 70)
    
    chain_topics = ["阳光", "时间", "记忆"]
    for i, topic in enumerate(chain_topics):
        print(f"\n--- 思维链节点{i+1}: {topic} ---")
        monologue = ""
        try:
            if ai.inner_thought_engine:
                for char in ai.inner_thought_engine.generate_inner_thought(
                    external_stimulus=topic,
                    max_tokens=120
                ):
                    monologue += char
                    print(char, end="", flush=True)
                print(f"\n  [字数] {len(monologue)}")
                results["monologues"].append({
                    "round": f"思维链#{i+1}_{topic}",
                    "text": monologue,
                    "char_count": len(monologue)
                })
        except Exception as e:
            print(f"  [错误] {e}")
    
    # ========== 5. 输出报告 ==========
    print("\n" + "=" * 70)
    print("  测试报告")
    print("=" * 70)
    
    total_monologue_chars = sum(m["char_count"] for m in results["monologues"])
    avg_monologue_chars = total_monologue_chars / max(len(results["monologues"]), 1)
    
    print(f"\n[独白统计]")
    print(f"  总独白轮数: {len(results['monologues'])}")
    print(f"  总独白字数: {total_monologue_chars}")
    print(f"  平均每轮字数: {avg_monologue_chars:.0f}")
    
    # 检查截断问题
    short_monologues = [m for m in results["monologues"] if m["char_count"] < 10]
    if short_monologues:
        print(f"  ⚠️ {len(short_monologues)} 轮独白过短（<10字），可能存在截断")
    else:
        print(f"  ✓ 所有独白长度正常（≥10字）")
    
    print(f"\n[对话统计]")
    for r in results["rounds"]:
        print(f"  {r['round']}: 思考{r['think_time']:.1f}s + 回复{r['speak_time']:.1f}s = 总{r['total_time']:.1f}s")
    
    # 保存结果到文件
    report_path = "/home/z/my-project/download/monologue_test_report.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("STDPBrain 持续思维流观察测试报告\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("【自发独白观察】\n")
        for m in results["monologues"]:
            if m["round"].startswith("自发"):
                f.write(f"  {m['round']}: [{m['char_count']}字] {m['text'][:200]}\n")
                f.write(f"    状态: {m.get('state_before', 'N/A')} → {m.get('state_after', 'N/A')}\n")
                f.write(f"    开口欲望: {m.get('urge_before', 0):.2f} → {m.get('urge_after', 0):.2f}\n\n")
        
        f.write("【对话测试 - 思考与讲话分离】\n")
        for r in results["rounds"]:
            f.write(f"  {r['round']}: {r['question']}\n")
            f.write(f"    思考({r['think_time']:.1f}s): {r['thinking'][:200]}\n")
            f.write(f"    回复({r['speak_time']:.1f}s): {r['response'][:200]}\n\n")
        
        f.write("【思维链测试】\n")
        for m in results["monologues"]:
            if m["round"].startswith("思维链"):
                f.write(f"  {m['round']}: [{m['char_count']}字] {m['text'][:200]}\n\n")
        
        f.write(f"\n【统计汇总】\n")
        f.write(f"  总独白轮数: {len(results['monologues'])}\n")
        f.write(f"  总独白字数: {total_monologue_chars}\n")
        f.write(f"  平均每轮字数: {avg_monologue_chars:.0f}\n")
    
    print(f"\n报告已保存: {report_path}")
    return results


if __name__ == "__main__":
    results = test_monologue_flow()
