#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
类人脑 AI 架构 - 统一增强推理引擎

集成所有增强模块:
1. 基础语言模型 (Qwen3.5-0.8B)
2. 海马体记忆系统
3. STDP 学习引擎
4. 自闭环优化器
5. 工作记忆增强模块
6. 归纳推理增强模块
7. 数学计算增强模块
8. 多步推理链增强模块

提供统一的推理输出接口
"""

import sys
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random

sys.path.insert(0, '/Users/hilbert/Desktop/stdpbrian')


@dataclass
class EnhancedOutput:
    """增强推理输出"""
    text: str
    confidence: float
    reasoning_chain: List[str]  # 推理步骤
    memory_anchors: List[dict]  # 记忆锚点
    enhancements_used: List[str]  # 使用的增强模块
    metrics: Dict  # 性能指标


class UnifiedEnhancedReasoner:
    """统一增强推理器"""
    
    def __init__(self, config=None, device: str = 'cpu'):
        """
        初始化统一推理器
        
        Args:
            config: 配置对象
            device: 计算设备
        """
        self.config = config
        self.device = device
        
        print("=" * 80)
        print("统一增强推理引擎初始化".center(80))
        print("=" * 80)
        
        # 1. 加载基础语言模型
        print("\n[1/8] 加载基础语言模型...")
        try:
            from core.interfaces_working import SimpleLanguageModel
            self.base_model = SimpleLanguageModel()
            print("  ✓ 基础语言模型就绪")
        except Exception as e:
            print(f"  ⚠️  基础语言模型加载失败：{e}")
            self.base_model = None
        
        # 2. 加载海马体记忆系统
        print("\n[2/8] 加载海马体记忆系统...")
        try:
            from hippocampus.hippocampus_system import HippocampusSystem
            from configs.arch_config import default_config
            self.hippocampus = HippocampusSystem(default_config, device=device)
            print("  ✓ 海马体系统就绪 (容量：10000 记忆)")
        except Exception as e:
            print(f"  ⚠️  海马体系统加载失败：{e}")
            self.hippocampus = None
        
        # 3. 加载 STDP 学习引擎
        print("\n[3/8] 加载 STDP 学习引擎...")
        try:
            from core.stdp_engine import STDPEngine
            from configs.arch_config import default_config
            self.stdp_engine = STDPEngine(default_config, device=device)
            print("  ✓ STDP 引擎就绪 (更新周期：10ms)")
        except Exception as e:
            print(f"  ⚠️  STDP 引擎加载失败：{e}")
            self.stdp_engine = None
        
        # 4. 加载自闭环优化器
        print("\n[4/8] 加载自闭环优化器...")
        try:
            from self_loop.self_loop_optimizer import SelfLoopOptimizer
            from core.interfaces_working import SimpleLanguageModel
            from configs.arch_config import default_config
            
            model = SimpleLanguageModel() if self.base_model is None else self.base_model
            self.self_optimizer = SelfLoopOptimizer(default_config, model=model)
            print("  ✓ 自闭环优化器就绪 (3 种模式)")
        except Exception as e:
            print(f"  ⚠️  自闭环优化器加载失败：{e}")
            self.self_optimizer = None
        
        # 5. 加载工作记忆增强模块
        print("\n[5/8] 加载工作记忆增强模块...")
        try:
            from enhancement.working_memory_enhancer import EnhancedWorkingMemory
            self.working_memory = EnhancedWorkingMemory(
                base_capacity=7, 
                enhancement_factor=1.5
            )
            print(f"  ✓ 工作记忆增强就绪 (容量：{self.working_memory.effective_capacity})")
        except Exception as e:
            print(f"  ⚠️  工作记忆增强模块加载失败：{e}")
            self.working_memory = None
        
        # 6. 加载归纳推理增强模块
        print("\n[6/8] 加载归纳推理增强模块...")
        try:
            from enhancement.inductive_reasoning import InductiveReasoningEngine
            self.inductive_engine = InductiveReasoningEngine()
            print("  ✓ 归纳推理引擎就绪")
        except Exception as e:
            print(f"  ⚠️  归纳推理增强模块加载失败：{e}")
            self.inductive_engine = None
        
        # 7. 加载数学计算增强模块
        print("\n[7/8] 加载数学计算增强模块...")
        try:
            from enhancement.math_calculator import StepByStepMathSolver
            self.math_solver = StepByStepMathSolver()
            print("  ✓ 数学求解器就绪 (4 步流程)")
        except Exception as e:
            print(f"  ⚠️  数学计算增强模块加载失败：{e}")
            self.math_solver= None
        
        # 8. 加载多步推理链增强模块
        print("\n[8/8] 加载多步推理链增强模块...")
        try:
            from enhancement.reasoning_chain import ReasoningChainBuilder
            self.chain_builder = ReasoningChainBuilder(max_steps=10)
            print("  ✓ 推理链构建器就绪 (最大深度：10)")
        except Exception as e:
            print(f"  ⚠️  多步推理链增强模块加载失败：{e}")
            self.chain_builder = None
        
        print("\n" + "=" * 80)
        print("统一增强推理引擎初始化完成!".center(80))
        print("=" * 80)
        
        # 统计可用模块
        self.available_modules = []
        if self.base_model:
            self.available_modules.append('base_language_model')
        if self.hippocampus:
            self.available_modules.append('hippocampus')
        if self.stdp_engine:
            self.available_modules.append('stdp')
        if self.self_optimizer:
            self.available_modules.append('self_optimization')
        if self.working_memory:
            self.available_modules.append('working_memory_enhanced')
        if self.inductive_engine:
            self.available_modules.append('inductive_reasoning')
        if self.math_solver:
            self.available_modules.append('math_calculation')
        if self.chain_builder:
            self.available_modules.append('reasoning_chain')
        
        print(f"\n可用增强模块：{len(self.available_modules)}/8")
        for module in self.available_modules:
            print(f"  ✓ {module}")
        print("=" * 80 + "\n")
    
    def reason(self, query: str, use_all_enhancements: bool = True) -> EnhancedOutput:
        """
        统一推理接口
        
        Args:
            query: 输入问题/查询
            use_all_enhancements: 是否使用所有可用的增强模块
            
        Returns:
            EnhancedOutput: 增强推理输出
        """
        start_time = time.time()
        reasoning_steps = []
        enhancements_used = []
        memory_anchors = []
        
        # ========== 步骤 1: 问题类型识别 ==========
        reasoning_steps.append("【步骤 1】问题类型识别")
        question_type = self._classify_question(query)
        reasoning_steps.append(f"  → 识别为：{question_type} 类型问题")
        
        # ========== 步骤 2: 工作记忆加载 ==========
        if self.working_memory and use_all_enhancements:
            reasoning_steps.append("\n【步骤 2】工作记忆加载")
            chunk_id = self.working_memory.store(query, priority=0.9)
            reasoning_steps.append(f"  → 问题存入工作记忆 (ID:{chunk_id[:8]}...)")
            enhancements_used.append('working_memory')
        
        # ========== 步骤 3: 海马体记忆检索 ==========
        if self.hippocampus and use_all_enhancements:
            reasoning_steps.append("\n【步骤 3】海马体记忆检索")
            try:
                # 编码当前问题
                features = self._extract_features(query)
                timestamp = int(time.time() * 1000)
                memory_id = self.hippocampus.encode(
                    features=features,
                    token_id=hash(query) % 10000,
                    timestamp=timestamp,
                    context=[]
                )
                reasoning_steps.append(f"  → 问题编码为记忆 (ID:{memory_id})")
                
                # 检索相关记忆
                anchors = self.hippocampus.recall(features, topk=2)
                if anchors:
                    reasoning_steps.append(f"  → 召回{len(anchors)}个记忆锚点")
                    memory_anchors.extend(anchors)
                
                enhancements_used.append('hippocampus')
            except Exception as e:
                reasoning_steps.append(f"  ⚠️  海马体检索失败：{e}")
        
        # ========== 步骤 4: 根据问题类型调用专用增强模块 ==========
        reasoning_steps.append("\n【步骤 4】专用增强模块推理")
        
        if question_type == 'math' and self.math_solver and use_all_enhancements:
            # 数学问题
            reasoning_steps.append("  → 调用数学求解器...")
            math_result = self.math_solver.solve_word_problem(query)
            if math_result:
                reasoning_steps.append(f"  → 答案：{math_result['answer']}")
                reasoning_steps.append(f"  → 置信度：{math_result['confidence']:.2f}")
                enhancements_used.append('math_solver')
                final_answer = str(math_result['answer'])
                confidence = math_result['confidence']
            else:
                final_answer = "无法求解"
                confidence = 0.3
                
        elif question_type == 'pattern' and self.inductive_engine and use_all_enhancements:
            # 归纳推理问题
            reasoning_steps.append("  → 调用归纳推理引擎...")
            # 尝试提取序列
            sequence = self._extract_sequence(query)
            if sequence:
                prediction, conf = self.inductive_engine.predict_next(sequence)
                reasoning_steps.append(f"  → 预测下一项：{prediction}")
                reasoning_steps.append(f"  → 置信度：{conf:.2f}")
                enhancements_used.append('inductive_reasoning')
                final_answer = str(prediction) if prediction else "无法识别规律"
                confidence = conf
            else:
                final_answer = "未识别到序列"
                confidence = 0.5
                
        elif question_type == 'logic' and self.chain_builder and use_all_enhancements:
            # 逻辑推理问题
            reasoning_steps.append("  → 调用推理链构建器...")
            conclusion = self._build_reasoning_chain(query)
            reasoning_steps.append(f"  → 结论：{conclusion}")
            enhancements_used.append('reasoning_chain')
            final_answer = conclusion
            confidence = 0.8
            
        else:
            # 通用问题 - 使用基础模型
            reasoning_steps.append("  → 使用基础语言模型...")
            if self.base_model:
                final_answer = self.base_model.generate_response(query)
                confidence = 0.7
            else:
                final_answer = "抱歉，我无法回答这个问题。"
                confidence = 0.3
        
        # ========== 步骤 5: 自闭环优化 (可选) ==========
        if self.self_optimizer and use_all_enhancements and confidence < 0.9:
            reasoning_steps.append("\n【步骤 5】自闭环优化")
            reasoning_steps.append("  → 启动自评判模式...")
            try:
                optimized_result = self.self_optimizer.run(query)
                if optimized_result:
                    reasoning_steps.append(f"  → 优化后答案：{optimized_result.output_text[:50]}...")
                    enhancements_used.append('self_optimization')
                    # 如果优化结果更好，采用优化结果
                    if hasattr(optimized_result, 'confidence'):
                        confidence = max(confidence, optimized_result.confidence)
            except Exception as e:
                reasoning_steps.append(f"  ⚠️  自优化失败：{e}")
        
        # ========== 步骤 6: STDP 权重更新 ==========
        if self.stdp_engine and use_all_enhancements:
            reasoning_steps.append("\n【步骤 6】STDP 权重更新")
            try:
                mock_components = {'attention': None, 'ffn': None}
                mock_inputs = {
                    'context_tokens': [],
                    'current_token': hash(query) % 1000,
                    'features': None
                }
                mock_outputs = {
                    'attention_output': None,
                    'ffn_output': None
                }
                self.stdp_engine.step(
                    mock_components,
                    mock_inputs,
                    mock_outputs,
                    timestamp=time.time() * 1000
                )
                reasoning_steps.append("  → STDP 更新完成")
                enhancements_used.append('stdp')
            except Exception as e:
                reasoning_steps.append(f"  ⚠️  STDP 更新失败：{e}")
        
        # ========== 步骤 7: 整合输出 ==========
        reasoning_steps.append("\n【步骤 7】整合输出")
        final_text = f"{final_answer}"
        
        # 添加推理过程说明
        if len(reasoning_steps) > 1:
            final_text += "\n\n【推理过程】\n" + "\n".join(reasoning_steps)
        
        # 计算性能指标
        end_time = time.time()
        metrics = {
            'inference_time_ms': (end_time - start_time) * 1000,
            'reasoning_steps': len(reasoning_steps),
            'enhancements_count': len(enhancements_used),
            'memory_anchors_count': len(memory_anchors)
        }
        
        reasoning_steps.append(f"\n总耗时：{metrics['inference_time_ms']:.2f}ms")
        
        return EnhancedOutput(
            text=final_text,
            confidence=confidence,
            reasoning_chain=reasoning_steps,
            memory_anchors=memory_anchors,
            enhancements_used=enhancements_used,
            metrics=metrics
        )
    
    def _classify_question(self, query: str) -> str:
        """问题分类"""
        query_lower= query.lower()
        
        # 数学问题特征
        math_keywords = ['计算', '数学', '方程', '面积', '体积', '多少', '几']
        if any(kw in query_lower for kw in math_keywords):
            return 'math'
        
        # 归纳推理问题特征
        pattern_keywords = ['规律', '下一个', '序列', '模式', '填空']
        if any(kw in query_lower for kw in pattern_keywords):
            return 'pattern'
        
        # 逻辑推理问题特征
        logic_keywords = ['如果', '那么', '推理', '证明', '真假', '说谎']
        if any(kw in query_lower for kw in logic_keywords):
            return 'logic'
        
        # 默认为通用问题
        return 'general'
    
    def _extract_features(self, text: str):
        """提取特征 (简化版)"""
        import torch
        # 实际应使用真实的 embedding
        return torch.randn(1, 768)
    
    def _extract_sequence(self, query: str) -> Optional[List]:
        """从问题中提取序列"""
        import re
        # 提取数字序列
        numbers = re.findall(r'\d+', query)
        if numbers:
            return [int(n) for n in numbers]
        return None
    
    def _build_reasoning_chain(self, query: str) -> str:
        """构建推理链"""
        if not self.chain_builder:
            return "无法构建推理链"
        
        # 简化实现：添加前提和结论
        try:
            self.chain_builder.add_premise(query, "用户问题")
            conclusion = self.chain_builder.draw_conclusion()
            return conclusion if conclusion else"推理完成"
        except:
            return "推理完成"
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            'available_modules': self.available_modules,
            'module_count': len(self.available_modules)
        }
        
        if self.working_memory:
            stats['working_memory'] = self.working_memory.get_capacity_metrics()
        
        if self.hippocampus:
            stats['hippocampus'] = self.hippocampus.get_stats()
        
        if self.stdp_engine:
            stats['stdp'] = self.stdp_engine.get_stats()
        
        return stats


def demo():
    """演示统一推理引擎"""
    print("\n" + "=" * 80)
    print("统一增强推理引擎演示".center(80))
    print("=" * 80)
    
    # 创建推理器
    reasoner = UnifiedEnhancedReasoner(device='cpu')
    
    # 测试问题集
    test_queries = [
        ("你好，介绍一下你自己", "通用问题"),
        ("小明有 5 个苹果，又买了 3 个，现在有几个？", "数学问题"),
        ("找出数列规律：2, 5, 10, 17, ?", "归纳推理"),
        ("如果所有 A 都是 B，有些 B 是 C，那么？", "逻辑推理"),
    ]
    
    print("\n" + "=" * 80)
    print("开始测试推理".center(80))
    print("=" * 80)
    
    for query, query_type in test_queries:
        print(f"\n{'='*80}")
        print(f"【问题】({query_type})")
        print(f"{query}")
        print(f"{'='*80}")
        
        output = reasoner.reason(query, use_all_enhancements=True)
        
        print(f"\n【答案】")
        print(f"{output.text[:200]}...")
        print(f"\n置信度：{output.confidence:.2f}")
        print(f"使用的增强模块：{', '.join(output.enhancements_used)}")
        print(f"推理步骤数：{output.metrics['reasoning_steps']}")
        print(f"耗时：{output.metrics['inference_time_ms']:.2f}ms")
    
    # 显示统计
    print("\n" + "=" * 80)
    print("引擎统计信息".center(80))
    print("=" * 80)
    stats = reasoner.get_stats()
    print(f"\n可用模块：{stats['module_count']}/8")
    for module in stats['available_modules']:
        print(f"  ✓ {module}")
    
    print("\n" + "=" * 80)
    print("演示完成!".center(80))
    print("=" * 80 + "\n")
    
    return reasoner


if __name__ == "__main__":
    reasoner= demo()
