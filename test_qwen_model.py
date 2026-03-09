#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试真实的 Qwen 模型加载和生成"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_qwen_model():
    """测试 Qwen 模型加载和生成"""
    model_path = "./models/Qwen3.5-0.8B-Base"
    
    print("=" * 60)
    print("测试 Qwen3.5-0.8B 模型")
    print("=" * 60)
    
    # 1. 加载 tokenizer
    print("\n[1/3] 正在加载 Qwen tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✓ Tokenizer 加载成功!")
        print(f"  - vocab_size: {len(tokenizer)}")
        print(f"  - model_max_length: {tokenizer.model_max_length}")
    except Exception as e:
        print(f"✗ Tokenizer 加载失败：{e}")
        return False
    
    # 2. 加载模型
    print("\n[2/3] 正在加载 Qwen 模型...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # CPU 使用 float32
            device_map="cpu",
            trust_remote_code=True
        )
        print(f"✓ 模型加载成功!")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  - 参数量：{param_count / 1e6:.2f}M ({param_count:,})")
        print(f"  - 设备：{next(model.parameters()).device}")
    except Exception as e:
        print(f"✗ 模型加载失败：{e}")
        return False
    
    # 3. 测试生成
    print("\n[3/3] 测试文本生成...")
    test_prompts = [
        "你好，",
        "类人脑 AI 架构的核心是",
        "Python 是一种"
    ]
    
    for prompt in test_prompts:
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n  输入：{prompt}")
            print(f"  输出：{response[:100]}...")
        except Exception as e:
            print(f"  ✗ 生成失败：{e}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ Qwen 模型测试全部通过!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_qwen_model()
    sys.exit(0 if success else 1)
