#!/usr/bin/env python3
"""分层诊断脚本 - 找到输出乱码的根因"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, time, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

MODEL_PATH = "./models/Qwen3.5-0.8B"
QUESTION = "你好，用一句话介绍你自己。"

print("="*60)
print("分层诊断 - 找到输出乱码的根因")
print("="*60)

# ========== 层0: 原始模型 ==========
print("\n[层0] 原始模型（无补丁，无双权重）")
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_raw = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32, trust_remote_code=True, low_cpu_mem_usage=True)
model_raw.eval()

messages = [
    {"role": "system", "content": "你是一个智能AI助手，请用中文回答。"},
    {"role": "user", "content": QUESTION}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"  Prompt: {prompt[:100]}...")

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
t0 = time.time()
with torch.no_grad():
    outputs = model_raw.generate(
        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
        max_new_tokens=40, temperature=0.6, do_sample=True, top_k=20,
        pad_token_id=tokenizer.eos_token_id
    )
resp0 = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"  回复: {resp0[:200]}")
print(f"  耗时: {time.time()-t0:.1f}s")

del model_raw
import gc; gc.collect()

# ========== 层1: 双权重层（无注意力补丁）==========
print("\n[层1] 双权重层（无注意力补丁）")
# 临时禁用注意力补丁
import core.qwen_narrow_band_patch as patch_module
orig_patch = patch_module.patch_qwen_attention
patch_module.patch_qwen_attention = lambda: False

from core.qwen_interface import QwenInterface, QwenModelWrapper
import config as cfg
from configs.arch_config import BrainAIConfig
config = BrainAIConfig()
config.model_path = MODEL_PATH
config.quantization = "FP32"

wrapper = QwenModelWrapper(model_path=MODEL_PATH, config=config, device="cpu", quantization="FP32")
wrapper.eval()

messages = [
    {"role": "system", "content": "你是一个智能AI助手，请用中文回答。"},
    {"role": "user", "content": QUESTION}
]
prompt = wrapper.apply_chat_template_safe(messages, tokenize=False, add_generation_prompt=True)
inputs = wrapper.tokenize_safe(prompt, return_tensors="pt", truncation=True, max_length=512)

t0 = time.time()
with torch.no_grad():
    gen_ids = wrapper.base_model.generate(
        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
        max_new_tokens=40, temperature=0.6, do_sample=True, top_k=20,
        pad_token_id=wrapper.tokenizer.eos_token_id
    )
resp1 = wrapper.decode_safe(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"  回复: {resp1[:200]}")
print(f"  耗时: {time.time()-t0:.1f}s")

del wrapper
gc.collect()

# ========== 层2: 完整系统（BrainAIInterface）==========
print("\n[层2] 完整系统 BrainAIInterface")
# 恢复注意力补丁
patch_module.patch_qwen_attention = orig_patch
# 需要重新import以应用补丁
import importlib
import core.qwen_interface
importlib.reload(core.qwen_interface)

from core.interfaces import BrainAIInterface
config2 = BrainAIConfig()
config2.model_path = MODEL_PATH
config2.quantization = "FP32"

ai = BrainAIInterface(config2, device='cpu')
t0 = time.time()
resp2 = ai.chat(QUESTION, history=[], max_tokens=40, thinking=False)
print(f"  回复: {resp2[:200]}")
print(f"  耗时: {time.time()-t0:.1f}s")

# ========== 汇总 ==========
print("\n" + "="*60)
print("诊断汇总")
print("="*60)

garbage_words = ['painsWatch','becks','Progr','ilde','watch','andering','disposto','Bomb']
for i, (label, resp) in enumerate([("层0-原始", resp0), ("层1-双权重", resp1), ("层2-完整", resp2)]):
    gc = sum(1 for w in garbage_words if w in resp)
    has_chinese = sum(1 for c in resp if '\u4e00' <= c <= '\u9fff')
    status = "✅正常" if gc == 0 and has_chinese > 5 else ("⚠️可疑" if gc < 3 else "❌乱码")
    print(f"  {label}: {status} (乱码词:{gc}, 中文字:{has_chinese})")
    print(f"    回复前50字: {resp[:50]}")
