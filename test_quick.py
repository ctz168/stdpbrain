#!/usr/bin/env python3
"""快速Agent能力测试 - 分步运行"""
import sys, os, time, traceback

# ===== HOTFIX (从main.py复制) =====
def _is_local_path(path):
    if not path or not isinstance(path, str): return False
    if path.startswith('/') or path.startswith('~') or path.startswith('.'): return os.path.isdir(path)
    if len(path.split('/')) > 2 and os.path.isdir(path): return True
    return False

try:
    from transformers.utils.hub import cached_file as _orig_cached_file
    from transformers.utils.hub import cached_files as _orig_cached_files
    def _patched_cached_file(path_or_repo_id, filename, **kwargs):
        if _is_local_path(path_or_repo_id):
            local_path = os.path.join(path_or_repo_id, filename)
            if os.path.isfile(local_path):
                return local_path
        return _orig_cached_file(path_or_repo_id, filename, **kwargs)
    def _patched_cached_files(path_or_repo_id, filenames, cache_dir=None, **kwargs):
        if _is_local_path(path_or_repo_id):
            results = []
            for fname in filenames:
                local_path = os.path.join(path_or_repo_id, fname)
                if os.path.isfile(local_path):
                    results.append(local_path)
            if results:
                return results
        return _orig_cached_files(path_or_repo_id, filenames, cache_dir=cache_dir, **kwargs)
    import transformers.utils.hub as _hf_hub
    _hf_hub.cached_file = _patched_cached_file
    _hf_hub.cached_files = _patched_cached_files
except Exception as _e:
    print(f"[HOTFIX] Warning: {_e}")

try:
    import huggingface_hub.utils._validators as _hf_validators
    _orig_validate = _hf_validators.validate_repo_id
    def _patched_validate_repo_id(repo_id, **kwargs):
        if _is_local_path(repo_id):
            return
        return _orig_validate(repo_id, **kwargs)
    _hf_validators.validate_repo_id = _patched_validate_repo_id
except Exception:
    pass

# 选择测试步骤: 1=基础, 2=记忆, 3=思维, 4=目标/自闭环, 5=规划, 6=全部统计
step = int(sys.argv[1]) if len(sys.argv) > 1 else 1
max_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 80

print(f"[测试步骤] step={step}, max_tokens={max_tok}")

import config as user_config
from configs.arch_config import BrainAIConfig
config = BrainAIConfig()
config.model_path = './models/Qwen3.5-0.8B'
config.quantization = getattr(user_config, 'QUANTIZATION', config.quantization)
config.QUANTIZATION = config.quantization

from core.interfaces import BrainAIInterface
ai = BrainAIInterface(config, device='cpu')
print('[OK] AI系统加载完成\n')

history = []
def chat(text, mt=None):
    t = time.time()
    r = ai.chat(text, history=history[-4:] if history else [], max_tokens=mt or max_tok)
    history.append({'role':'user','content':text})
    history.append({'role':'assistant','content':r})
    if len(history)>10: del history[:-10]
    return r, time.time()-t

if step == 1:
    print("="*60)
    print("测试1: 基础响应能力")
    print("="*60)
    for q in ['你好', '1+1等于几？', '今天天气如何？']:
        r, e = chat(q)
        print(f"Q: {q}")
        print(f"A: {r[:150]}")
        print(f"[{e:.1f}s]\n")

elif step == 2:
    print("="*60)
    print("测试2: 记忆注入+召回 (Agent核心能力)")
    print("="*60)
    print("--- 阶段1: 注入个人信息 ---")
    r, e = chat('我叫小明，手机号13800138000，住在北京，程序员')
    print(f"注入: {r[:120]}\n")

    print("--- 阶段2: 干扰对话 ---")
    r, e = chat('帮我翻译一下good morning')
    print(f"干扰: {r[:80]}\n")

    print("--- 阶段3: 记忆召回测试 ---")
    tests = [
        ('你还记得我叫什么名字吗？', '小明'),
        ('我的手机号是多少？', '13800138000'),
        ('我住在哪里？', '北京'),
        ('我的职业是什么？', '程序员'),
    ]
    for q, expected in tests:
        r, e = chat(q)
        ok = expected in r
        status = "✅" if ok else "❌"
        print(f"{status} Q: {q}")
        print(f"   A: {r[:120]}")
        print(f"   期望: {expected}\n")

    # 海马体统计
    s = ai.get_stats()
    print(f"海马体记忆数: {s['hippocampus']['num_memories']}")

elif step == 3:
    print("="*60)
    print("测试3: 持续思维流")
    print("="*60)
    for i in range(3):
        mono = ai._generate_spontaneous_monologue(max_tokens=60)
        print(f"独白{i+1}: {mono[:150]}")
        print(f"思维种子: {ai.thought_seed}\n")

elif step == 4:
    print("="*60)
    print("测试4: 目标系统 + 自闭环优化")
    print("="*60)
    # 先做一轮对话触发目标
    r, e = chat('请记住：我下周一要去上海出差', mt=60)
    print(f"输入: 请记住：我下周一要去上海出差")
    print(f"回复: {r[:100]}\n")

    # 目标系统
    if ai.goal_system and ai.goal_system.current_goal:
        g = ai.goal_system.current_goal
        print(f"当前目标类型: {g.goal_type.value}")
        print(f"目标描述: {g.description}")
        print(f"目标进度: {g.progress:.1%}")
    else:
        print("无活跃目标")

    # 自闭环
    if ai.self_loop:
        sl = ai.self_loop.get_stats()
        mode = ai.self_loop.decide_mode('证明根号2是无理数')
        print(f"\n自闭环周期: {sl['cycle_count']}")
        print(f"平均准确率: {sl['avg_accuracy']:.2f}")
        print(f"高难度任务模式: {mode}")

    # 自我编码器
    if ai.self_encoder:
        desc = ai.self_encoder.interpret()
        print(f"\n自我描述: {desc[:80]}")

elif step == 5:
    print("="*60)
    print("测试5: Agent长任务 - 多步规划")
    print("="*60)
    r, e = chat('请帮我规划一个3天的北京旅游行程，包括景点和美食推荐', mt=200)
    print(f"回复 ({len(r)}字, {e:.1f}s):\n{r}")

elif step == 6:
    print("="*60)
    print("测试6: 完整系统统计")
    print("="*60)
    s = ai.get_stats()
    for k, v in s.items():
        if isinstance(v, dict):
            print(f"\n[{k}]")
            for k2, v2 in v.items():
                print(f"  {k2}: {v2}")
        else:
            print(f"{k}: {v}")
