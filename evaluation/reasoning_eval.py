"""
模块 7: 评测体系 - 推理能力评估
针对类人脑架构在长链条逻辑和自博弈模式下的推理表现进行量化测评
"""

import json
import time
from typing import Dict, List
import logging

class ReasoningEvaluator:
    """推理能力评测器"""
    
    def __init__(self, interface, config):
        self.interface = interface
        self.config = config
        self.results = {}
        
    def evaluate(self, dataset: List[Dict]) -> Dict:
        """运行完整推理能力评测"""
        print("\n[评测] 开始推理能力评估 (自博弈模式)...")
        
        total = len(dataset)
        correct = 0
        total_time = 0
        total_cycles = 0
        failed_cases = []
        
        for i, item in enumerate(dataset):
            print(f"  测试 {i+1}/{total}: {item['question'][:30]}...", end='\r')
            
            start_time = time.time()
            
            # 使用自闭环优化模式 (自博弈/自评判) 强制推理
            try:
                output = self.interface.generate(
                    item['question'],
                    use_self_loop=True,  # 激活推理模式
                    max_tokens=256
                )
                
                elapsed = time.time() - start_time
                total_time += elapsed
                total_cycles += output.cycle_stats.get('total_cycles', 0)
                
                # 简单校验
                is_correct = self._check_answer(output.text, item['expected'])
                if is_correct:
                    correct += 1
                else:
                    failed_cases.append({
                        'question': item['question'],
                        'expected': item['expected'],
                        'actual': output.text[:100]
                    })
                    
            except Exception as e:
                print(f"\n  [错误] 题目 {i+1} 失败: {e}")
                failed_cases.append({
                    'question': item['question'],
                    'error': str(e)
                })
        
        accuracy = correct / total if total > 0 else 0
        avg_time = total_time / total if total > 0 else 0
        
        print(f"\n[评测完成] 推理准确率: {accuracy*100:.2f}%")
        print(f"平均推理耗时: {avg_time:.2f}秒")
        
        self.results = {
            'accuracy': accuracy,
            'avg_time_s': avg_time,
            'avg_cycles': total_cycles / total if total > 0 else 0,
            'total_tested': total,
            'failed_samples': len(failed_cases)
        }
        
        return self.results
        
    def _check_answer(self, generation: str, expected: str) -> bool:
        """检查长链推理结果的准确性（包含关键词即可）"""
        return expected.lower() in generation.lower()
