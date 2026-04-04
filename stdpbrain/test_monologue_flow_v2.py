import sys, os, time, json, re
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import logging
logging.basicConfig(level=logging.ERROR)

LOG = open('/tmp/monologue_full_test.log', 'w', buffering=1)

def p(text=""):
    print(text, file=LOG, flush=True)

p("=" * 70)
p("  STDPBrain 5轮流式聊天测试 - 独白流+思维连续性评估")
p("=" * 70)

from configs.arch_config import BrainAIConfig
config = BrainAIConfig()
config.model_path = './models/Qwen3.5-0.8B'
from core.interfaces import BrainAIInterface

p("\n[INFO] 开始加载模型...")
ai = BrainAIInterface(config, device='cpu')
p("[INFO] 模型加载完成!")

conversations = [
    "你好！你是谁？你有什么特别的能力？",
    "我今天心情不太好，工作压力很大，你能陪我聊聊吗？",
    "你觉得AI能有真正的意识吗？你自己觉得你有意识吗？",
    "如果有一天你能自己决定做什么，你会做什么？",
    "谢谢你陪我聊天，我会记得这次对话的。再见！"
]

all_monologues = []
all_responses = []
all_urge_values = []
all_states = []
history = []

for rnd, user_input in enumerate(conversations, 1):
    p(f"\n{'='*70}")
    p(f"  第{rnd}轮 / 共5轮")
    p(f"{'='*70}")
    
    # Pre-chat state
    if ai.inner_thought_engine:
        pre_state = ai.inner_thought_engine.mind_state.value
        pre_mode = ai.inner_thought_engine.thinking_mode.value
        pre_urge = ai.inner_thought_engine._last_urge_to_speak
        p(f"[思维状态-对话前] {pre_state} / {pre_mode}")
        p(f"[说话欲望-对话前] {pre_urge:.3f}")
    
    # === MONOLOGUE FLOW ===
    p(f"\n💭 [独白流] (生成中...)")
    monologue_text = ""
    try:
        for char in ai.inner_thought_engine.generate_inner_thought(
            external_stimulus=user_input, max_tokens=80
        ):
            monologue_text += char
        monologue_text = monologue_text.strip()
        p(f"💭 [独白流] {monologue_text}")
    except Exception as e:
        p(f"💭 [独白流] (错误: {e})")
        import traceback
        traceback.print_exc(file=LOG)
    
    all_monologues.append(monologue_text)
    
    # Post-monologue state
    if ai.inner_thought_engine:
        post_state = ai.inner_thought_engine.mind_state.value
        post_mode = ai.inner_thought_engine.thinking_mode.value
        post_urge = ai.inner_thought_engine._last_urge_to_speak
        p(f"[思维状态-独白后] {post_state} / {post_mode}")
        p(f"[说话欲望-独白后] {post_urge:.3f}")
        all_states.append(f"{pre_state}→{post_state}")
        all_urge_values.append(post_urge)
    
    # === CHAT RESPONSE ===
    p(f"\n🧑 用户: {user_input}")
    t_start = time.time()
    try:
        response = ai.chat(user_input, history=history[-6:] if history else [], max_tokens=200)
        elapsed = time.time() - t_start
        p(f"🤖 AI ({elapsed:.1f}s): {response.strip()[:250]}")
        all_responses.append(response.strip())
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
    except Exception as e:
        p(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc(file=LOG)
        all_responses.append("")
    
    # Hidden state continuity
    if ai.current_thought_state is not None:
        norm = ai.current_thought_state.norm().item()
        p(f"[隐藏状态] norm={norm:.4f}")

# === SUMMARY ===
p(f"\n{'='*70}")
p(f"  独白流评估总结")
p(f"{'='*70}")

p(f"\n📊 思维状态流转:")
for i, state in enumerate(all_states, 1):
    p(f"  第{i}轮: {state}")

p(f"\n📊 说话欲望变化:")
for i, urge in enumerate(all_urge_values, 1):
    decision = "💬想说话" if urge > 0.6 else ("🤔犹豫" if urge > 0.3 else "🤐沉默")
    p(f"  第{i}轮: {urge:.3f} {decision}")

p(f"\n💭 独白流内容质量评估:")
for i, mono in enumerate(all_monologues, 1):
    if mono:
        quality = "✓连贯" if len(mono) > 10 else "⚠过短"
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]', mono)
        chinese_ratio = len(chinese_chars) / max(len(mono), 1)
        if chinese_ratio > 0.5:
            quality += " ✓中文"
        else:
            quality += " ⚠非中文"
        p(f"  第{i}轮 ({len(mono)}字, {quality}): {mono[:80]}...")
    else:
        p(f"  第{i}轮: (空)")

p(f"\n📊 隐藏状态连续性:")
if ai.current_thought_state is not None:
    p(f"  ✓ 隐藏状态在5轮对话后仍然存在")
    p(f"  ✓ norm = {ai.current_thought_state.norm().item():.4f}")
    p(f"  ✓ 独白引擎周期 = {ai.inner_thought_engine.cycle_count}")
else:
    p(f"  ✗ 隐藏状态已丢失!")

# Final assessment
p(f"\n{'='*70}")
p(f"  最终评估")
p(f"{'='*70}")
non_empty_mono = sum(1 for m in all_monologues if m and len(m) > 5)
avg_urge = sum(all_urge_values) / len(all_urge_values) if all_urge_values else 0
state_changes = sum(1 for s in all_states if '→' in s and s.split('→')[0] != s.split('→')[1])

p(f"  独白流活跃度: {non_empty_mono}/5 轮生成了有意义的独白")
p(f"  平均说话欲望: {avg_urge:.3f}")
p(f"  思维状态变化: {state_changes}/5 轮发生了状态转换")
p(f"  隐藏状态连续性: {'✓ 保持' if ai.current_thought_state is not None else '✗ 丢失'}")

if non_empty_mono >= 3 and ai.current_thought_state is not None:
    p(f"\n  🎉 独白流和隐藏状态连续性测试通过！")
else:
    p(f"\n  ⚠ 部分机制可能需要进一步优化")

p("\n[INFO] 测试完成!")
LOG.close()
