#!/usr/bin/env python3
"""
STDPBrain 5轮流式聊天测试 - 独白思维流评估

测试重点:
1. 独白流是否连贯（不是断断续续的）
2. 思维流是否像人一样有连续性
3. 思考与讲话是否分离
4. urge_to_speak 是否有变化
5. mind_state 是否正常流转
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_5rounds():
    from configs.arch_config import BrainAIConfig
    import config as secret_config
    
    config = BrainAIConfig()
    config.model_path = secret_config.MODEL_PATH
    config.quantization = getattr(secret_config, 'QUANTIZATION', config.quantization)
    
    from core.interfaces import BrainAIInterface
    ai = BrainAIInterface(config, device=None)
    
    # Verify GW and Goal injection
    gw_ok = ai.inner_thought_engine._global_workspace is not None
    goal_ok = ai.inner_thought_engine._goal_system is not None
    print(f"\n[检查] GW注入: {'✓' if gw_ok else '✗'}, Goal注入: {'✓' if goal_ok else '✗'}")
    
    questions = [
        "你今天心情怎么样？",
        "如果你能拥有一项超能力，你希望是什么？",
        "你觉得AI和人类最大的区别是什么？",
        "给我讲一个你觉得有趣的事",
        "如果让你给这个世界留一句话，你会说什么？"
    ]
    
    history = []
    all_results = []
    
    for i, q in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"  第{i+1}轮 / 共5轮")
        print(f"{'='*60}")
        print(f"用户: {q}")
        
        # Phase 1: Inner monologue stream
        print(f"\n💭 [内心独白流]", flush=True)
        t_start = time.time()
        thinking = ""
        state_before = ai.inner_thought_engine.mind_state.value
        urge_before = ai.inner_thought_engine._last_urge_to_speak
        monologue_chars = []
        
        try:
            for char in ai.inner_thought_engine.generate_inner_thought(
                external_stimulus=q, max_tokens=120
            ):
                thinking += char
                monologue_chars.append(char)
                print(char, end='', flush=True)
        except Exception as e:
            print(f"\n[独白错误] {e}")
        
        state_after = ai.inner_thought_engine.mind_state.value
        urge_after = ai.inner_thought_engine._last_urge_to_speak
        t_think = time.time() - t_start
        
        print(f"\n  [思维状态] {state_before} → {state_after}")
        print(f"  [开口欲望] {urge_before:.2f} → {urge_after:.2f}")
        print(f"  [独白字数] {len(thinking)}")
        print(f"  [思考耗时] {t_think:.1f}s")
        
        # Phase 2: Formal response
        print(f"\n🗣️ [正式回复]", flush=True)
        t_speak_start = time.time()
        response = ai.chat(q, history=history[-6:], max_tokens=200, thinking=False)
        t_speak = time.time() - t_speak_start
        print(response)
        print(f"  [回复字数] {len(response)}")
        print(f"  [回复耗时] {t_speak:.1f}s")
        
        history.append({'role': 'user', 'content': q})
        history.append({'role': 'assistant', 'content': response})
        
        all_results.append({
            'round': i+1,
            'question': q,
            'thinking': thinking,
            'thinking_len': len(thinking),
            'state_before': state_before,
            'state_after': state_after,
            'urge_before': urge_before,
            'urge_after': urge_after,
            'response': response,
            'response_len': len(response),
            'think_time': t_think,
            'speak_time': t_speak
        })
    
    # ========== 评估报告 ==========
    print(f"\n\n{'='*60}")
    print("  5轮流式测试 - 评估报告")
    print(f"{'='*60}")
    
    # 1. 思维流连贯性评估
    print("\n[1] 思维流连贯性:")
    total_thinking = sum(r['thinking_len'] for r in all_results)
    avg_thinking = total_thinking / 5
    print(f"  总独白字数: {total_thinking}")
    print(f"  平均每轮: {avg_thinking:.0f}字")
    short = [r for r in all_results if r['thinking_len'] < 20]
    if short:
        print(f"  ⚠️ {len(short)}轮独白过短: {[(r['round'], r['thinking_len']) for r in short]}")
    else:
        print(f"  ✓ 所有轮次独白长度正常")
    
    # 2. urge 变化
    print("\n[2] 开口欲望(urge)变化:")
    urges = [(r['urge_before'], r['urge_after']) for r in all_results]
    all_same = all(u[0] == u[1] == urges[0][0] for u in urges)
    if all_same:
        print(f"  ⚠️ urge 恒定: {urges[0][0]:.2f} (未注入GW/Goal?)")
    else:
        print(f"  ✓ urge 有变化:")
        for r in all_results:
            delta = r['urge_after'] - r['urge_before']
            arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
            print(f"    第{r['round']}轮: {r['urge_before']:.2f} → {r['urge_after']:.2f} ({arrow})")
    
    # 3. 状态流转
    print("\n[3] 思维状态流转:")
    state_chain = " → ".join([f"R{r['round']}:{r['state_before'][:3]}" for r in all_results])
    print(f"  {state_chain}")
    unique_states = set()
    for r in all_results:
        unique_states.add(r['state_before'])
        unique_states.add(r['state_after'])
    print(f"  经历了 {len(unique_states)} 种状态: {unique_states}")
    
    # 4. 截断检查
    print("\n[4] 截断检查:")
    truncated = [r for r in all_results if r['thinking'] and not r['thinking'][-1] in '。！？…~\n']
    if truncated:
        print(f"  ⚠️ {len(truncated)}轮可能被截断")
        for r in truncated:
            print(f"    第{r['round']}轮结尾: ...{r['thinking'][-20:]}")
    else:
        print(f"  ✓ 无截断，所有独白自然结束")
    
    # 5. 独白-回复关联度
    print("\n[5] 独白与回复关联度:")
    for r in all_results:
        # Simple overlap check
        t_chars = set(r['thinking'])
        r_chars = set(r['response'])
        overlap = len(t_chars & r_chars) / max(len(t_chars | r_chars), 1)
        indicator = "✓" if overlap > 0.3 else "⚠️"
        print(f"  {indicator} 第{r['round']}轮: 字符重叠度 {overlap:.1%} (独白{r['thinking_len']}字, 回复{r['response_len']}字)")
    
    # Save report
    report_path = "/home/z/my-project/download/stream_5rounds_report.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("STDPBrain 5轮流式聊天测试报告\n")
        f.write("="*50 + "\n\n")
        for r in all_results:
            f.write(f"第{r['round']}轮:\n")
            f.write(f"  问题: {r['question']}\n")
            f.write(f"  💭内心独白[{r['thinking_len']}字]: {r['thinking'][:500]}\n")
            f.write(f"  状态: {r['state_before']} → {r['state_after']}\n")
            f.write(f"  欲望: {r['urge_before']:.2f} → {r['urge_after']:.2f}\n")
            f.write(f"  🗣️正式回复[{r['response_len']}字]: {r['response'][:500]}\n\n")
        f.write(f"\n统计: 总独白{total_thinking}字, 平均{avg_thinking:.0f}字/轮\n")
    
    print(f"\n报告已保存: {report_path}")
    return all_results

if __name__ == "__main__":
    results = run_5rounds()
