#!/usr/bin/env python3
"""
持续独白模式测试脚本

测试后台持续独白生成功能
"""

import sys
import time

def test_continuous_monologue():
    """测试持续独白生成"""
    print("=" * 70)
    print("持续独白模式测试")
    print("=" * 70)
    
    try:
        from configs.arch_config import BrainAIConfig
        from core.interfaces import BrainAIInterface
        
        # 初始化
        config = BrainAIConfig()
        print("\n[1/2] 正在加载AI接口...")
        ai = BrainAIInterface(config)
        
        print("\n[2/2] 测试持续独白生成...")
        print("\n生成10次独白（模拟持续模式）:\n")
        
        for i in range(10):
            start = time.time()
            
            # 模拟持续独白生成
            if hasattr(ai, 'think'):
                result = ai.think()
                monologue = result.get('monologue', '思考中...')
            elif hasattr(ai, '_generate_spontaneous_monologue'):
                monologue = ai._generate_spontaneous_monologue(max_tokens=25, temperature=0.85)
            else:
                monologue = "简化独白..."
            
            elapsed = (time.time() - start) * 1000
            
            # 显示
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"💭 [{timestamp}] {monologue} ({elapsed:.0f}ms)")
            
            # 模拟间隔
            if i < 9:
                time.sleep(2)
        
        print("\n" + "=" * 70)
        print("✓ 测试完成")
        print("\n提示: 运行 '持续独白观察.bat' 体验完整的持续独白模式")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_continuous_monologue()
