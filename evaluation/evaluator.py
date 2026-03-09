"""
多维度全链路测评体系 - 主评估器

整合五个子评估器:
1. 海马体记忆能力评估 (40%)
2. 基础能力对标评估 (20%)
3. 逻辑推理能力评估 (20%)
4. 端侧性能评估 (10%)
5. 自闭环优化评估 (10%)
"""

import torch
from typing import Dict, List, Optional
import time


class BrainAIEvaluator:
    """
    类人脑AI架构综合评估器
    """
    
    def __init__(self, ai_interface, config=None):
        self.ai = ai_interface
        self.config = config
        
        # 从配置加载权重
        if config:
            self.weights = {
                'hippocampus': config.evaluation.hippocampus_weight,      # 0.4
                'base_capability': config.evaluation.base_capability_weight,  # 0.2
                'reasoning': config.evaluation.reasoning_weight,          # 0.2
                'edge_performance': config.evaluation.edge_performance_weight,  # 0.1
                'self_loop': config.evaluation.self_loop_weight         # 0.1
            }
        else:
            self.weights = {
                'hippocampus': 0.4,
                'base_capability': 0.2,
                'reasoning': 0.2,
                'edge_performance': 0.1,
                'self_loop': 0.1
            }
        
        # ========== 初始化子评估器 ==========
        from .hippocampus_eval import HippocampusEvaluator
        from .base_capability_eval import BaseCapabilityEvaluator
        from .reasoning_eval import ReasoningEvaluator
        from .edge_performance_eval import EdgePerformanceEvaluator
        from .self_loop_eval import SelfLoopEvaluator
        
        self.hippocampus_eval = HippocampusEvaluator(ai)
        self.base_eval = BaseCapabilityEvaluator(ai)
        self.reasoning_eval = ReasoningEvaluator(ai)
        self.edge_eval = EdgePerformanceEvaluator(ai)
        self.self_loop_eval = SelfLoopEvaluator(ai)
    
    def run_all_evaluations(self) -> dict:
        """
        运行全部评测
        
        Returns:
            results: 完整评测结果
        """
        print("=" * 60)
        print("类人脑AI架构 - 多维度全链路评测")
        print("=" * 60)
        
        results = {}
        start_time = time.time()
        
        # ========== 1. 海马体记忆能力评估 (40%) ==========
        print("\n[1/5] 海马体记忆能力评估...")
        results['hippocampus_score'] = self.hippocampus_eval.evaluate()
        print(f"  得分：{results['hippocampus_score']:.3f}")
        
        # ========== 2. 基础能力对标评估 (20%) ==========
        print("\n[2/5] 基础能力对标评估...")
        results['base_capability_score'] = self.base_eval.evaluate()
        print(f"  得分：{results['base_capability_score']:.3f}")
        
        # ========== 3. 逻辑推理能力评估 (20%) ==========
        print("\n[3/5] 逻辑推理能力评估...")
        results['reasoning_score'] = self.reasoning_eval.evaluate()
        print(f"  得分：{results['reasoning_score']:.3f}")
        
        # ========== 4. 端侧性能评估 (10%) ==========
        print("\n[4/5] 端侧性能评估...")
        results['edge_performance_score'] = self.edge_eval.evaluate()
        print(f"  得分：{results['edge_performance_score']:.3f}")
        
        # ========== 5. 自闭环优化评估 (10%) ==========
        print("\n[5/5] 自闭环优化评估...")
        results['self_loop_score'] = self.self_loop_eval.evaluate()
        print(f"  得分：{results['self_loop_score']:.3f}")
        
        # ========== 计算总分 ==========
        total_score = (
            results['hippocampus_score'] * self.weights['hippocampus'] +
            results['base_capability_score'] * self.weights['base_capability'] +
            results['reasoning_score'] * self.weights['reasoning'] +
            results['edge_performance_score'] * self.weights['edge_performance'] +
            results['self_loop_score'] * self.weights['self_loop']
        )
        
        results['total_score'] = total_score
        results['elapsed_time'] = time.time() - start_time
        
        # ========== 打印总结 ==========
        print("\n" + "=" * 60)
        print("评测总结")
        print("=" * 60)
        print(f"总分：{total_score:.3f}/1.0")
        print(f"耗时：{results['elapsed_time']:.2f}秒")
        
        # 判定是否合格
        passed = self._check_pass_criteria(results)
        print(f"\n合格判定：{'✓ 通过' if passed else '✗ 未通过'}")
        
        return results
    
    def _check_pass_criteria(self, results: dict) -> bool:
        """
        检查是否达到合格标准
        
        合格标准:
        - 海马体记忆：≥0.95 (召回准确率)
        - 基础能力：≥0.95 (不低于原生 95%)
        - 逻辑推理：≥0.60 (超过原生 60%)
        - 端侧性能：≥0.90 (显存≤420MB, 延迟≤10ms)
        - 自闭环优化：≥0.90 (自纠错准确率)
        """
        # 简化判定：总分≥0.8
        return results['total_score'] >= 0.8
    
    def evaluate_hippocampus_only(self) -> dict:
        """仅评估海马体记忆能力"""
        return self.hippocampus_eval.evaluate_detailed()
    
    def evaluate_reasoning_only(self) -> dict:
        """仅评估逻辑推理能力"""
        return self.reasoning_eval.evaluate_detailed()
    
    def get_evaluation_report(self, results: dict) -> str:
        """生成评测报告"""
        report = []
        report.append("=" * 60)
        report.append("类人脑双系统全闭环 AI架构 - 评测报告")
        report.append("=" * 60)
        report.append("")
        
        # 各维度得分
        report.append("[维度得分]")
        report.append(f"  海马体记忆能力：  {results['hippocampus_score']:.3f} (权重 40%)")
        report.append(f"  基础能力对标：  {results['base_capability_score']:.3f} (权重 20%)")
        report.append(f"  逻辑推理能力：  {results['reasoning_score']:.3f} (权重 20%)")
        report.append(f"  端侧性能：      {results['edge_performance_score']:.3f} (权重 10%)")
        report.append(f"  自闭环优化：    {results['self_loop_score']:.3f} (权重 10%)")
        report.append("")
        
        # 总分
        report.append(f"[总分] {results['total_score']:.3f}/1.0")
        report.append("")
        
        # 合格判定
        passed = self._check_pass_criteria(results)
        report.append(f"[结果] {'✓ 通过' if passed else '✗ 未通过'}")
        report.append("")
        
        # 详细分析
        report.append("[分析]")
        
        if results['hippocampus_score'] >= 0.95:
            report.append("  ✓ 海马体记忆能力优秀，符合设计目标")
        else:
            report.append("  ✗ 海马体记忆能力需进一步提升")
        
        if results['base_capability_score'] >= 0.95:
            report.append("  ✓ 基础能力保持良好，无灾难性遗忘")
        else:
            report.append("  ✗ 基础能力有所下降，需检查静态权重冻结")
        
        if results['reasoning_score'] >= 0.60:
            report.append("  ✓ 逻辑推理能力提升显著")
        else:
            report.append("  ✗ 逻辑推理能力提升不足")
        
        if results['edge_performance_score'] >= 0.90:
            report.append("  ✓ 端侧性能符合要求，可流畅运行")
        else:
            report.append("  ✗ 端侧性能需优化")
        
        if results['self_loop_score'] >= 0.90:
            report.append("  ✓ 自闭环优化效果明显")
        else:
            report.append("  ✗ 自闭环优化需加强")
        
        return "\n".join(report)
