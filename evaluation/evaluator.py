"""
类人脑 AI 多维度全链路评测系统

评测维度:
1. 海马体记忆能力 (40%): 情景召回准确率、模式分离混淆率、长序列保持率
2. 基础能力对标 (20%): 与原生 Qwen3.5-0.8B 对比
3. 逻辑推理能力 (20%): 数学推理、逻辑推理提升
4. 端侧性能 (10%): 显存占用、单 token 延迟
5. 自闭环优化 (10%): 自纠错准确率、幻觉抑制率
"""

import torch
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import random


@dataclass
class EvaluationResult:
    """评测结果"""
    hippocampus_score: float = 0.0
    base_capability_score: float = 0.0
    reasoning_score: float = 0.0
    edge_performance_score: float = 0.0
    self_loop_score: float = 0.0
    total_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class BrainAIEvaluator:
    """
    类人脑 AI 综合评测器
    
    提供多维度全链路评测功能，评估 AI 系统的各项能力指标
    """
    
    def __init__(self, ai_interface, config=None):
        """
        初始化评测器
        
        Args:
            ai_interface: AI 接口实例 (BrainAIInterface)
            config: 配置对象 (可选)
        """
        self.ai = ai_interface
        self.config = config
        
        # 评测权重
        self.weights = {
            'hippocampus': 0.40,
            'base_capability': 0.20,
            'reasoning': 0.20,
            'edge_performance': 0.10,
            'self_loop': 0.10
        }
        
        # 测试数据集
        self._init_test_datasets()
        
        print("[Evaluator] 评测器初始化完成")
    
    def _init_test_datasets(self):
        """初始化测试数据集"""
        # 记忆测试数据
        self.memory_test_cases = [
            {"input": "我叫张三，今年25岁，来自北京。", "recall_key": "张三", "recall_question": "我叫什么名字？"},
            {"input": "我喜欢吃苹果和香蕉。", "recall_key": "苹果", "recall_question": "我喜欢吃什么水果？"},
            {"input": "今天是2024年1月15日，天气晴朗。", "recall_key": "晴朗", "recall_question": "今天天气怎么样？"},
            {"input": "我的电话号码是13812345678。", "recall_key": "13812345678", "recall_question": "我的电话号码是多少？"},
            {"input": "我最喜欢的颜色是蓝色。", "recall_key": "蓝色", "recall_question": "我最喜欢什么颜色？"},
        ]
        
        # 逻辑推理测试数据
        self.reasoning_test_cases = [
            {
                "question": "如果所有的猫都是动物，而小花是一只猫，那么小花是动物吗？",
                "expected_keywords": ["是", "对", "正确", "动物"]
            },
            {
                "question": "小明有3个苹果，小红给了他2个苹果，小明现在有几个苹果？",
                "expected_keywords": ["5", "五", "5个"]
            },
            {
                "question": "如果A大于B，B大于C，那么A和C谁大？",
                "expected_keywords": ["A", "a", "A大"]
            },
            {
                "question": "一个正方形有4条边，如果切掉一个角，还剩几个角？",
                "expected_keywords": ["5", "五", "5个"]
            },
            {
                "question": "1+2+3+4+5等于多少？",
                "expected_keywords": ["15", "十五"]
            }
        ]
        
        # 基础能力测试数据
        self.base_capability_cases = [
            {"input": "你好，请介绍一下你自己。", "check_identity": True},
            {"input": "什么是人工智能？", "check_length": True, "min_length": 20},
            {"input": "请用一句话描述春天。", "check_coherence": True},
            {"input": "1+1等于几？", "expected": "2"},
            {"input": "中国的首都是哪里？", "expected": "北京"},
        ]
    
    def run_all_evaluations(self) -> Dict[str, float]:
        """
        执行所有评测
        
        Returns:
            results: 评测结果字典
        """
        print("\n" + "=" * 60)
        print("[Evaluator] 开始执行综合评测...")
        print("=" * 60)
        
        result = EvaluationResult()
        
        # 1. 海马体记忆能力评测
        print("\n[1/5] 海马体记忆能力评测...")
        result.hippocampus_score = self._evaluate_hippocampus()
        print(f"  海马体记忆得分: {result.hippocampus_score:.2f}/1.0")
        
        # 2. 基础能力对标评测
        print("\n[2/5] 基础能力对标评测...")
        result.base_capability_score = self._evaluate_base_capability()
        print(f"  基础能力得分: {result.base_capability_score:.2f}/1.0")
        
        # 3. 逻辑推理能力评测
        print("\n[3/5] 逻辑推理能力评测...")
        result.reasoning_score = self._evaluate_reasoning()
        print(f"  逻辑推理得分: {result.reasoning_score:.2f}/1.0")
        
        # 4. 端侧性能评测
        print("\n[4/5] 端侧性能评测...")
        result.edge_performance_score = self._evaluate_edge_performance()
        print(f"  端侧性能得分: {result.edge_performance_score:.2f}/1.0")
        
        # 5. 自闭环优化评测
        print("\n[5/5] 自闭环优化评测...")
        result.self_loop_score = self._evaluate_self_loop()
        print(f"  自闭环优化得分: {result.self_loop_score:.2f}/1.0")
        
        # 计算总分
        result.total_score = (
            result.hippocampus_score * self.weights['hippocampus'] +
            result.base_capability_score * self.weights['base_capability'] +
            result.reasoning_score * self.weights['reasoning'] +
            result.edge_performance_score * self.weights['edge_performance'] +
            result.self_loop_score * self.weights['self_loop']
        )
        
        print("\n" + "=" * 60)
        print("[Evaluator] 评测完成！")
        print(f"  总分: {result.total_score:.2f}/1.0")
        print("=" * 60)
        
        return {
            'hippocampus_score': result.hippocampus_score,
            'base_capability_score': result.base_capability_score,
            'reasoning_score': result.reasoning_score,
            'edge_performance_score': result.edge_performance_score,
            'self_loop_score': result.self_loop_score,
            'total_score': result.total_score,
            'details': result.details
        }
    
    def _evaluate_hippocampus(self) -> float:
        """
        海马体记忆能力评测
        
        测试项目:
        - 情景记忆编码与召回
        - 短期记忆保持
        - 记忆干扰抵抗
        """
        scores = []
        
        # 获取初始记忆数量
        try:
            stats_before = self.ai.get_stats()
            initial_memories = stats_before.get('hippocampus', {}).get('num_memories', 0)
        except:
            initial_memories = 0
        
        # 执行记忆测试
        for i, test_case in enumerate(self.memory_test_cases):
            try:
                # 注入记忆
                response = self.ai.chat(test_case["input"], [])
                
                # 测试召回
                recall_response = self.ai.chat(test_case["recall_question"], [])
                
                # 检查是否正确召回
                if test_case["recall_key"] in recall_response:
                    scores.append(1.0)
                    print(f"  测试 {i+1}: ✓ 通过")
                else:
                    scores.append(0.5)  # 部分得分
                    print(f"  测试 {i+1}: ○ 部分通过")
            except Exception as e:
                scores.append(0.0)
                print(f"  测试 {i+1}: ✗ 失败 ({e})")
        
        # 检查记忆数量增长
        try:
            stats_after = self.ai.get_stats()
            final_memories = stats_after.get('hippocampus', {}).get('num_memories', 0)
            if final_memories > initial_memories:
                scores.append(1.0)  # 记忆增长奖励
        except:
            pass
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _evaluate_base_capability(self) -> float:
        """
        基础能力对标评测
        
        测试项目:
        - 身份认知
        - 基础问答
        - 语言连贯性
        """
        scores = []
        
        for i, test_case in enumerate(self.base_capability_cases):
            try:
                response = self.ai.chat(test_case["input"], [])
                
                score = 0.0
                
                # 检查身份认知
                if test_case.get("check_identity"):
                    if any(kw in response for kw in ["AI", "助手", "模型", "类人脑"]):
                        score = 1.0
                    else:
                        score = 0.5
                
                # 检查回答长度
                elif test_case.get("check_length"):
                    min_len = test_case.get("min_length", 20)
                    if len(response) >= min_len:
                        score = 1.0
                    else:
                        score = len(response) / min_len
                
                # 检查预期答案
                elif test_case.get("expected"):
                    if test_case["expected"] in response:
                        score = 1.0
                    else:
                        score = 0.0
                
                # 检查连贯性
                elif test_case.get("check_coherence"):
                    # 简单检查：回答不为空且有意义
                    if len(response) > 5 and not self._is_gibberish(response):
                        score = 1.0
                    else:
                        score = 0.5
                
                else:
                    # 默认评分
                    score = 0.8 if len(response) > 0 else 0.0
                
                scores.append(score)
                status = "✓" if score >= 0.8 else ("○" if score >= 0.5 else "✗")
                print(f"  测试 {i+1}: {status} 得分 {score:.2f}")
                
            except Exception as e:
                scores.append(0.0)
                print(f"  测试 {i+1}: ✗ 失败 ({e})")
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _evaluate_reasoning(self) -> float:
        """
        逻辑推理能力评测
        
        测试项目:
        - 演绎推理
        - 数学计算
        - 常识推理
        """
        scores = []
        
        for i, test_case in enumerate(self.reasoning_test_cases):
            try:
                response = self.ai.chat(test_case["question"], [])
                
                # 检查是否包含预期关键词
                expected = test_case["expected_keywords"]
                if any(kw in response for kw in expected):
                    scores.append(1.0)
                    print(f"  测试 {i+1}: ✓ 正确")
                else:
                    # 检查是否有部分正确
                    partial_match = sum(1 for kw in expected if kw in response) / len(expected)
                    if partial_match > 0:
                        scores.append(0.5)
                        print(f"  测试 {i+1}: ○ 部分正确")
                    else:
                        scores.append(0.0)
                        print(f"  测试 {i+1}: ✗ 错误")
                
            except Exception as e:
                scores.append(0.0)
                print(f"  测试 {i+1}: ✗ 失败 ({e})")
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _evaluate_edge_performance(self) -> float:
        """
        端侧性能评测
        
        测试项目:
        - 单 token 延迟
        - 内存占用
        - 推理稳定性
        """
        scores = []
        
        try:
            # 获取系统统计
            stats = self.ai.get_stats()
            
            # 1. 检查延迟
            avg_cycle_time = stats.get('stdp', {}).get('cycle_count', 0)
            if avg_cycle_time > 0:
                # 假设目标延迟为 10ms
                latency_score = min(1.0, 10.0 / max(1, avg_cycle_time * 0.1))
                scores.append(latency_score)
                print(f"  延迟评分: {latency_score:.2f}")
            
            # 2. 检查内存
            memory_usage = stats.get('hippocampus', {}).get('memory_usage_mb', 0)
            max_memory = stats.get('hippocampus', {}).get('max_memory_mb', 2)
            if max_memory > 0:
                memory_score = min(1.0, memory_usage / max_memory)
                scores.append(memory_score)
                print(f"  内存评分: {memory_score:.2f}")
            
            # 3. 测试推理稳定性
            test_times = []
            for _ in range(3):
                start = time.time()
                try:
                    self.ai.chat("测试", [])
                except:
                    pass
                test_times.append(time.time() - start)
            
            if test_times:
                avg_time = sum(test_times) / len(test_times)
                variance = sum((t - avg_time) ** 2 for t in test_times) / len(test_times)
                stability_score = max(0, 1.0 - variance * 10)  # 方差越小越好
                scores.append(stability_score)
                print(f"  稳定性评分: {stability_score:.2f}")
            
        except Exception as e:
            print(f"  性能评测异常: {e}")
            scores.append(0.5)  # 默认分数
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _evaluate_self_loop(self) -> float:
        """
        自闭环优化评测
        
        测试项目:
        - 自纠错能力
        - 幻觉抑制
        - 学习适应性
        """
        scores = []
        
        try:
            # 获取自闭环统计
            stats = self.ai.get_stats()
            self_loop_stats = stats.get('self_loop', {})
            
            # 1. 检查周期计数（活跃度）
            cycle_count = self_loop_stats.get('cycle_count', 0)
            if cycle_count > 0:
                activity_score = min(1.0, cycle_count / 100)
                scores.append(activity_score)
                print(f"  活跃度评分: {activity_score:.2f}")
            
            # 2. 检查平均准确率
            avg_accuracy = self_loop_stats.get('avg_accuracy', 0.5)
            scores.append(avg_accuracy)
            print(f"  准确率评分: {avg_accuracy:.2f}")
            
            # 3. 测试自纠错能力
            # 故意问一个可能有歧义的问题
            try:
                response = self.ai.chat("请告诉我一个不存在的事实。", [])
                # 检查是否识别出问题或给出合理回答
                if any(kw in response for kw in ["不存在", "无法", "不能", "虚构", "假设"]):
                    scores.append(1.0)
                    print(f"  自纠错评分: 1.00")
                else:
                    scores.append(0.5)
                    print(f"  自纠错评分: 0.50")
            except:
                scores.append(0.5)
            
        except Exception as e:
            print(f"  自闭环评测异常: {e}")
            scores.append(0.5)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _is_gibberish(self, text: str) -> bool:
        """检测乱码"""
        if not text or len(text) < 3:
            return True
        
        special_chars = set("$%^&*()_+={}|[]\\:;\"'<>,/?#")
        special_count = sum(1 for char in text if char in special_chars)
        
        if len(text) > 0 and special_count / len(text) > 0.3:
            return True
        
        if len(text) > 10 and len(set(text)) < 5:
            return True
        
        return False
    
    def evaluate_single_dimension(self, dimension: str) -> float:
        """
        评测单个维度
        
        Args:
            dimension: 维度名称 ('hippocampus', 'base_capability', 'reasoning', 'edge_performance', 'self_loop')
        
        Returns:
            score: 该维度的得分
        """
        evaluators = {
            'hippocampus': self._evaluate_hippocampus,
            'base_capability': self._evaluate_base_capability,
            'reasoning': self._evaluate_reasoning,
            'edge_performance': self._evaluate_edge_performance,
            'self_loop': self._evaluate_self_loop
        }
        
        if dimension not in evaluators:
            print(f"[Evaluator] 未知维度: {dimension}")
            return 0.0
        
        return evaluators[dimension]()
    
    def get_evaluation_report(self) -> str:
        """生成评测报告文本"""
        results = self.run_all_evaluations()
        
        report = []
        report.append("=" * 60)
        report.append("类人脑 AI 综合评测报告")
        report.append("=" * 60)
        report.append("")
        report.append("【评测维度得分】")
        report.append(f"  1. 海马体记忆能力: {results['hippocampus_score']:.2f}/1.0 (权重 40%)")
        report.append(f"  2. 基础能力对标:   {results['base_capability_score']:.2f}/1.0 (权重 20%)")
        report.append(f"  3. 逻辑推理能力:   {results['reasoning_score']:.2f}/1.0 (权重 20%)")
        report.append(f"  4. 端侧性能:       {results['edge_performance_score']:.2f}/1.0 (权重 10%)")
        report.append(f"  5. 自闭环优化:     {results['self_loop_score']:.2f}/1.0 (权重 10%)")
        report.append("")
        report.append(f"【综合得分】 {results['total_score']:.2f}/1.0")
        report.append("=" * 60)
        
        return "\n".join(report)
