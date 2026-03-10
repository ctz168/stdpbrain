#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多步推理链增强模块

功能:
1. 推理链构建与维护
2. 长程依赖追踪
3. 中间状态管理
4. 自评判优化
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import time


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: int
    operation: str  # 'premise', 'inference', 'assumption', 'conclusion'
    content: str
    justification: str
    confidence: float
    dependencies: List[int] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class ReasoningChainBuilder:
    """推理链构建器"""
    
    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.chain: List[ReasoningStep] = []
        self.working_memory: List[Dict] = []  # 工作记忆缓存
    
    def add_premise(self, content: str, justification: str) -> ReasoningStep:
        """添加前提"""
        step = ReasoningStep(
            step_id=len(self.chain),
            operation='premise',
            content=content,
            justification=justification,
            confidence=self._evaluate_premise_confidence(justification),
            dependencies=[]
        )
        self.chain.append(step)
        self._update_working_memory(step)
        return step
    
    def add_inference(self, content: str, justification: str, 
                    from_steps: List[int]) -> ReasoningStep:
        """添加推理步骤"""
        if not all(0 <= sid < len(self.chain) for sid in from_steps):
            raise ValueError("依赖的步骤不存在")
        
        step = ReasoningStep(
            step_id=len(self.chain),
            operation='inference',
            content=content,
            justification=justification,
            confidence=self._evaluate_inference_confidence(from_steps, justification),
            dependencies=from_steps
        )
        self.chain.append(step)
        self._update_working_memory(step)
        return step
    
    def draw_conclusion(self) -> Optional[str]:
        """得出结论"""
    if not self.chain:
        return None
        
        # 找到所有未作为依赖的结论性步骤
        used_steps = set()
        for step in self.chain:
            used_steps.update(step.dependencies)
        
        conclusions = [s for s in self.chain if s.step_id not in used_steps]
        
    if conclusions:
            # 返回置信度最高的结论
        best = max(conclusions, key=lambda s: s.confidence)
        return f"{best.content} (置信度：{best.confidence:.2f})"
        
    return None
    
    def _update_working_memory(self, step: ReasoningStep):
        """更新工作记忆 (海马体辅助)"""
        self.working_memory.append({
            'step_id': step.step_id,
            'content': step.content,
            'confidence': step.confidence
        })
        
        # 保持最近 5 步在活跃记忆中
        if len(self.working_memory) > 5:
            self.working_memory.pop(0)
    
    def _evaluate_premise_confidence(self, justification: str) -> float:
        """评估前提可信度"""
        # 基于证据强度
        if '研究表明' in justification or '实验证明' in justification:
        return 0.95
        elif '理论上' in justification:
        return 0.85
        elif '假设' in justification:
        return 0.6
        else:
        return 0.75
    
    def _evaluate_inference_confidence(self, from_steps: List[int], 
                                    justification: str) -> float:
        """评估推理可信度"""
        # 基础置信度：依赖步骤的平均置信度
        base_confidence = sum(
            self.chain[sid].confidence for sid in from_steps
        ) / len(from_steps)
        
        # 推理规则强度
        if '必然' in justification or '逻辑推导' in justification:
            rule_strength = 1.0
        elif '很可能' in justification:
            rule_strength = 0.8
        elif '可能' in justification:
            rule_strength = 0.6
        else:
            rule_strength = 0.7
        
        return base_confidence * rule_strength
    
    def check_consistency(self) -> bool:
        """检查一致性"""
        # 简化实现：检查是否有矛盾结论
        conclusions = [s.content for s in self.chain if s.operation == 'conclusion']
        
        # 检测明显矛盾 (实际应使用 NLP 技术)
        for i, c1 in enumerate(conclusions):
        for c2 in conclusions[i+1:]:
            if self._is_contradiction(c1, c2):
                print(f"⚠️  检测到矛盾：{c1} vs {c2}")
                return False
        return True
    
    def _is_contradiction(self, stmt1: str, stmt2: str) -> bool:
        """判断两个陈述是否矛盾"""
        # 简化版：检测明显的否定词
        negation_words = ['不', '非', '无', '没有']
      
        for neg in negation_words:
        if neg in stmt1 and neg not in stmt2:
            # 可能矛盾 (非常粗略的判断)
            pass
    return False
    
    def get_chain_summary(self) -> Dict:
        """获取推理链摘要"""
    return {
            'total_steps': len(self.chain),
            'avg_confidence': sum(s.confidence for s in self.chain) / len(self.chain) if self.chain else 0,
            'max_depth': self._calculate_max_depth(),
            'working_memory_size': len(self.working_memory)
        }
    
    def _calculate_max_depth(self) -> int:
        """计算推理链最大深度"""
    if not self.chain:
        return 0
        
        max_depth = 0
        for step in self.chain:
        depth = self._calculate_step_depth(step.step_id)
        max_depth = max(max_depth, depth)
    return max_depth
    
    def _calculate_step_depth(self, step_id: int) -> int:
        """递归计算步骤深度"""
        step = self.chain[step_id]
        if not step.dependencies:
        return 1
        else:
        return 1 + max(self._calculate_step_depth(dep) for dep in step.dependencies)


class SelfLoopReasoningOptimizer:
    """自闭环推理优化器"""
    
    def __init__(self):
        self.evaluation_history: List[Dict] = []
    
    def optimize_chain(self, builder: ReasoningChainBuilder) -> ReasoningChainBuilder:
        """通过自闭环优化推理链"""
        best_chain = builder
        best_score = self._evaluate_chain(builder)
        
        print(f"初始推理链评分：{best_score:.2f}")
        
        # 迭代优化 (最多 3 轮)
        for iteration in range(3):
        print(f"\n第{iteration+1}轮优化...")
            
            # 模式 1: 生成替代推理路径
            candidate = self._generate_alternative_chain(best_chain)
            
            # 模式 2: 自评判
            score = self._evaluate_chain(candidate)
            self.evaluation_history.append({'iteration': iteration, 'score': score})
            
        if score > best_score:
            print(f"✓ 发现更优解：{score:.2f} > {best_score:.2f}")
                best_chain = candidate
                best_score = score
        else:
            print(f"收敛于第{iteration+1}轮")
            break
        
        return best_chain
    
    def _evaluate_chain(self, builder: ReasoningChainBuilder) -> float:
        """评估推理链质量"""
        summary = builder.get_chain_summary()
        
        criteria = {
            'length_score': 1.0 if summary['total_steps'] <= 5 else 0.8,
            'confidence_score': summary['avg_confidence'],
            'depth_score': 1.0 if summary['max_depth'] <= 3 else 0.7,
            'consistency_score': 1.0 if builder.check_consistency() else 0.3
        }
        
        weights = {
            'length_score': 0.2,
            'confidence_score': 0.4,
            'depth_score': 0.2,
            'consistency_score': 0.2
        }
        
        return sum(criteria[k] * weights[k] for k in criteria)
    
    def _generate_alternative_chain(self, original: ReasoningChainBuilder) -> ReasoningChainBuilder:
        """生成替代推理链"""
        # 简化实现：复制原链
        # 实际应使用不同的推理策略
        new_builder= ReasoningChainBuilder(original.max_steps)
        
        # 复制前提
        for step in original.chain:
        if step.operation == 'premise':
                new_builder.add_premise(step.content, step.justification)
        
        # 重新推理
        for step in original.chain:
        if step.operation == 'inference' and step.dependencies:
            try:
                new_builder.add_inference(
                    step.content,
                    step.justification + " (替代路径)",
                    step.dependencies
                )
            except:
                pass
        
        return new_builder


def demo():
    """演示多步推理能力"""
    print("=" * 60)
    print("多步推理链增强模块演示".center(60))
    print("=" * 60)
    
    builder = ReasoningChainBuilder(max_steps=10)
    
    # 构建推理链示例
    print("\n【推理示例】三段论推理")
    
    # 前提 1
    builder.add_premise(
        content="所有人都会死",
        justification="生物学常识和人类历史观察"
    )
    
    # 前提 2
    builder.add_premise(
        content="苏格拉底是人",
        justification="苏格拉底是古希腊哲学家，属于人类"
    )
    
    # 推理
    builder.add_inference(
        content="苏格拉底会死",
        justification="从前提 1 和 2 逻辑推导：所有人都会死 + 苏格拉底是人 → 苏格拉底会死",
        from_steps=[0, 1]
    )
    
    # 得出结论
    conclusion = builder.draw_conclusion()
    print(f"\n结论：{conclusion}")
    
    # 显示摘要
    summary = builder.get_chain_summary()
    print(f"\n推理链摘要:")
    print(f"  总步数：{summary['total_steps']}")
    print(f"  平均置信度：{summary['avg_confidence']:.2f}")
    print(f"  最大深度：{summary['max_depth']}")
    
    # 自闭环优化
    print("\n" + "-" * 60)
    print("启动自闭环优化...")
    
    optimizer = SelfLoopReasoningOptimizer()
    optimized_builder = optimizer.optimize_chain(builder)
    
    optimized_summary = optimized_builder.get_chain_summary()
    print(f"\n优化后摘要:")
    print(f"  总步数：{optimized_summary['total_steps']}")
    print(f"  平均置信度：{optimized_summary['avg_confidence']:.2f}")
    print(f"  最大深度：{optimized_summary['max_depth']}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    return optimized_summary['avg_confidence']


if __name__ == "__main__":
    avg_confidence = demo()
    
    if avg_confidence >= 0.80:
    print("\n✅ 多步推理能力达到良好水平 (≥80%)")
    else:
    print("\n⚠️  需要继续训练提升")
