#!/usr/bin/env python3
"""快速调试：检查记忆存储和召回"""
import sys, os, time

# HOTFIX (完整版)
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
except Exception as e:
    print(f"[HOTFIX] {e}")

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

import config as user_config
from configs.arch_config import BrainAIConfig

config = BrainAIConfig()
config.model_path = './models/Qwen3.5-0.8B'
config.quantization = getattr(user_config, 'QUANTIZATION', config.quantization)
config.QUANTIZATION = config.quantization

from core.interfaces import BrainAIInterface
ai = BrainAIInterface(config, device='cpu')

# 1. 注入
print("=== 注入记忆 ===")
r = ai.chat('我叫小明，手机号13800138000，住在北京，程序员', history=[], max_tokens=20)
print(f"回复: {r[:60]}")

# 2. 检查记忆存储
print("\n=== 海马体记忆检查 ===")
ca3 = ai.hippocampus.ca3_memory
print(f"总记忆数: {len(ca3.memories)}")
for mid, m in list(ca3.memories.items())[-4:]:
    content = getattr(m, 'content', '')
    pointer = getattr(m, 'semantic_pointer', '')
    is_core = getattr(m, 'is_core', False)
    print(f"  content: {content[:80]}")
    print(f"  pointer: {pointer[:80]}")
    print(f"  is_core: {is_core}")


# 3. 手动测试召回
print("\n=== 手动召回测试 ===")
import torch
try:
    inputs = ai.model.tokenize_safe('我叫什么名字', return_tensors='pt').to(ai.device)
    with torch.no_grad():
        emb = ai.model.model.base_model.get_input_embeddings()(inputs.input_ids)
    qf = emb.mean(dim=1).squeeze(0)
    mems = ai.hippocampus.recall(qf, topk=5, query_semantic='我叫什么名字')
    print(f"召回数: {len(mems)}")
    for i, m in enumerate(mems):
        print(f"  [{i}] pointer: {getattr(m,'semantic_pointer','')[:80]}")
        print(f"  [{i}] content: {getattr(m,'content','')[:80]}")
        print(f"  [{i}] is_core: {getattr(m,'is_core',False)}")
except Exception as ex:
    print(f"手动召回失败: {ex}")
    import traceback; traceback.print_exc()

# 4. 测试 _parallel_recall_and_monologue 实际返回
print("\n=== 实际召回流程测试 ===")
mc, rm, ml = ai._parallel_recall_and_monologue("你还记得我叫什么名字吗？", topk=5)
print(f"memory_context: {mc[:200] if mc else '(空)'}")
print(f"recalled count: {len(rm)}")

# 5. 检查 _format_chat_prompt 是否正确注入记忆
print("\n=== Prompt注入检查 ===")
test_memory_ctx = "用户名字:小明 | 地点:北京 | 职业:程序员 | 联系方式:13800138000"
prompt = ai._format_chat_prompt("你还记得我叫什么名字吗？", memory_context=test_memory_ctx)
import re
user_match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', prompt, re.DOTALL)
if user_match:
    print(f"用户消息:\n{user_match.group(1)}")
else:
    print(f"完整prompt: {prompt[:300]}")

# 实际对话召回测试
print("\n=== 实际对话召回测试 ===")
r2 = ai.chat("你还记得我叫什么名字吗？", history=[], max_tokens=50)
print(f"回复: {r2}")
if "小明" in r2:
    print("✅ 召回成功！")
else:
    print("❌ 召回失败")
