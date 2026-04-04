#!/usr/bin/env python3
"""简单聊天测试脚本 - 验证系统基本功能"""
import sys
import os
import time

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("STDPBrain 聊天测试 - Qwen2.5-0.5B-Instruct")
    print("=" * 60)
    
    # Step 1: Check model exists
    model_path = "./models/Qwen2.5-0.5B-Instruct"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at: {model_path}")
        return
    
    print(f"[OK] Model found: {model_path}")
    
    # Step 2: Try to load system
    try:
        from configs.arch_config import BrainAIConfig
        config = BrainAIConfig()
        config.model_path = model_path
        config.QUANTIZATION = "FP16"
        config.quantization = "FP16"
        
        # Disable heavy features for faster testing
        config.stdp.enabled = False
        # Disable KV hippocampus integration (dimension mismatch)
        config.hard_constraints.ENABLE_KV_HIPPOCAMPUS_INTEGRATION = False
        
        from core.interfaces import BrainAIInterface
        
        print("[Loading model...]")
        start = time.time()
        ai = BrainAIInterface(config, device="cpu")
        print(f"[OK] Model loaded in {time.time()-start:.1f}s")
    except Exception as e:
        print(f"[ERROR] Failed to load: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Test basic generation
    test_inputs = [
        "你好",
        "什么是人工智能？",
    ]
    
    history = []
    for user_input in test_inputs:
        print(f"\n{'='*40}")
        print(f"用户: {user_input}")
        print(f"AI: ", end="", flush=True)
        
        try:
            start = time.time()
            response = ai.chat(user_input, history=history, max_tokens=80, thinking=False)
            elapsed = time.time() - start
            # Clean response - remove prompt echo
            if response:
                print(response[:500])
            print(f"\n[耗时: {elapsed:.1f}s, 长度: {len(response)}字]")
            
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            # Keep history short
            if len(history) > 6:
                history = history[-6:]
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("测试完成!")
    
    # Print stats
    try:
        stats = ai.get_stats()
        print(f"\n[系统统计]")
        print(f"  海马体记忆数: {stats['hippocampus']['num_memories']}")
        print(f"  STDP 周期数: {stats['stdp']['cycle_count']}")
        print(f"  思维状态: {ai.inner_thought_engine.mind_state.value if ai.inner_thought_engine else 'N/A'}")
    except:
        pass

if __name__ == "__main__":
    main()
