"""
模块 4: 单智体自生成 - 自博弈 - 自评判闭环优化系统

核心功能:
- 模式 1: 自生成组合输出 (默认基础模式)
- 模式 2: 自博弈竞争优化 (推理增强模式)
- 模式 3: 自双输出 + 自评判选优 (决策增强模式)
- 模式自动切换机制
"""

import torch
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class GenerationResult:
    """生成结果"""
    output_text: str
    candidates: List[str]
    scores: Optional[Dict[str, float]]
    mode_used: str
    confidence: float


class SelfLoopOptimizer:
    """
    自闭环优化器
    
    根据输入任务难度自动选择执行模式:
    - 简单任务 → 模式 1: 自组合
    - 高难度推理 → 模式 2: 自博弈
    - 高准确性要求 → 模式 3: 自评判
    """
    
    def __init__(self, config, model=None):
        self.config = config
        self.model = model
        self.sl_config = config.self_loop
        
        # 周期计数器
        self.cycle_count = 0
        
        # 历史准确率记录 (用于模式 1 加权)
        self.accuracy_history: List[float] = []
        self.accuracy_window = self.sl_config.mode1_accuracy_window
        
        # 角色状态 (用于模式 2)
        self.current_role = "proposer"  # "proposer" | "verifier"
        
        # 关键词配置
        self.high_difficulty_keywords = set(self.sl_config.high_difficulty_keywords)
        self.high_accuracy_keywords = set(self.sl_config.high_accuracy_keywords)
    
    def decide_mode(self, input_text: str) -> str:
        """
        自动判断执行模式
        
        Args:
            input_text: 输入文本
        
        Returns:
            mode: "self_combine" | "self_game" | "self_eval"
        """
        input_lower = input_text.lower()
        
        # ========== 检查高难度关键词 ==========
        has_difficulty_keyword = any(
            kw in input_lower for kw in self.high_difficulty_keywords
        )
        
        # ========== 检查高准确性关键词 ==========
        has_accuracy_keyword = any(
            kw in input_lower for kw in self.high_accuracy_keywords
        )
        
        # ========== 决策逻辑 ==========
        if has_difficulty_keyword:
            # 数学、逻辑、代码等高难度任务 → 自博弈
            return "self_game"
        
        elif has_accuracy_keyword:
            # 方案、决策、专业问答等 → 自评判
            return "self_eval"
        
        else:
            # 通用对话、简单问答 → 自组合
            return "self_combine"
    
    def run(
        self,
        input_text: str,
        context: Optional[List[str]] = None
    ) -> GenerationResult:
        """
        执行自闭环优化
        
        Args:
            input_text: 输入文本
            context: 上下文列表
        
        Returns:
            result: 生成结果
        """
        self.cycle_count += 1
        
        # ========== 1. 决定执行模式 ==========
        mode = self.decide_mode(input_text)
        
        # ========== 2. 执行对应模式 ==========
        if mode == "self_combine":
            result = self._run_self_combine(input_text, context)
        
        elif mode == "self_game":
            result = self._run_self_game(input_text, context)
        
        elif mode == "self_eval":
            result = self._run_self_evaluation(input_text, context)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # ========== 3. 更新准确率历史 ==========
        self._update_accuracy_history(result.confidence)
        
        return result
    
    def _run_self_combine(
        self,
        input_text: str,
        context: Optional[List[str]] = None
    ) -> GenerationResult:
        """
        模式 1: 自生成组合输出
        
        执行逻辑:
        1. 并行生成 2 个同源候选 token/结果
        2. STDP 加权一致性投票融合
        3. 输出最终结果
        """
        temperature_range = self.sl_config.mode1_temperature_range
        num_candidates = self.sl_config.mode1_num_candidates
        
        # ========== 1. 生成多个候选 ==========
        candidates = []
        for i in range(num_candidates):
            # 使用不同随机种子
            temperature = random.uniform(*temperature_range)
            
            candidate = self._generate_with_temperature(
                input_text, 
                temperature=temperature,
                context=context
            )
            candidates.append(candidate)
        
        # ========== 2. STDP 加权一致性投票 ==========
        weights = self._compute_candidate_weights(candidates)
        
        # ========== 3. 融合输出 ==========
        if len(candidates) == 1:
            final_output = candidates[0]
        else:
            # 选择权重最高的候选
            best_idx = weights.index(max(weights))
            final_output = candidates[best_idx]
        
        # ========== 4. 计算置信度 ==========
        confidence = max(weights) / sum(weights) if weights else 0.5
        
        return GenerationResult(
            output_text=final_output,
            candidates=candidates,
            scores={'weights': weights},
            mode_used="self_combine",
            confidence=confidence
        )
    
    def _run_self_game(
        self,
        input_text: str,
        context: Optional[List[str]] = None
    ) -> GenerationResult:
        """
        模式 2: 自博弈竞争优化
        
        执行逻辑:
        1. 奇数周期：提案角色生成推理结果
        2. 偶数周期：验证角色校验纠错
        3. 循环迭代直到收敛
        """
        max_iterations = self.sl_config.mode2_max_iterations
        convergence_threshold = self.sl_config.mode2_convergence_threshold
        
        # ========== 1. 初始化 ==========
        proposal = input_text
        verification_result = None
        iteration = 0
        
        # ========== 2. 自博弈迭代 ==========
        while iteration < max_iterations:
            iteration += 1
            
            # 切换角色
            if iteration % 2 == 1:
                # 奇数：提案角色
                proposal = self._generate_proposal(proposal, context)
            else:
                # 偶数：验证角色
                verification_result = self._verify_proposal(proposal)
                
                # 检查是否收敛
                if verification_result['is_valid']:
                    break
                
                # 应用修正
                if verification_result['corrections']:
                    proposal = self._apply_corrections(
                        proposal, 
                        verification_result['corrections']
                    )
        
        # ========== 3. 返回最终结果 ==========
        confidence = verification_result['confidence'] if verification_result else 0.7
        
        return GenerationResult(
            output_text=proposal,
            candidates=[proposal],
            scores={'iterations': iteration, 'verification': verification_result},
            mode_used="self_game",
            confidence=confidence
        )
    
    def _run_self_evaluation(
        self,
        input_text: str,
        context: Optional[List[str]] = None
    ) -> GenerationResult:
        """
        模式 3: 自双输出 + 自评判选优
        
        执行逻辑:
        1. 并行输出 2 个完整候选结果
        2. 切换为评判角色打分
        3. 输出得分最高的最优结果
        """
        eval_period = self.sl_config.mode3_eval_period
        
        # ========== 1. 生成两个候选 ==========
        candidate_a = self._generate_with_temperature(
            input_text, 
            temperature=0.7,
            context=context,
            seed=42
        )
        
        candidate_b = self._generate_with_temperature(
            input_text,
            temperature=0.9,
            context=context,
            seed=43
        )
        
        candidates = [candidate_a, candidate_b]
        
        # ========== 2. 评判角色打分 (每 10 周期执行一次) ==========
        if self.cycle_count % eval_period == 0:
            scores = self._evaluate_candidates(candidates, input_text)
            
            # ========== 3. 选择最优结果 ==========
            best_idx = scores.index(max(scores))
            final_output = candidates[best_idx]
            
            # 归一化分数
            total_score = sum(scores)
            confidence = max(scores) / total_score if total_score > 0 else 0.5
            
        else:
            # 非评判周期，直接返回第一个候选
            final_output = candidate_a
            scores = [0.5, 0.5]
            confidence = 0.5
        
        return GenerationResult(
            output_text=final_output,
            candidates=candidates,
            scores={'eval_scores': scores},
            mode_used="self_eval",
            confidence=confidence
        )
    
    # ========== 辅助方法 ==========
    
    def _generate_with_temperature(
        self,
        input_text: str,
        temperature: float = 0.7,
        context: Optional[List[str]] = None,
        seed: Optional[int] = None
    ) -> str:
        """带温度参数的生成"""
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        # TODO: 调用模型生成
        # 简化处理：返回输入 + 温度
        return f"[T={temperature:.2f}] {input_text}"
    
    def _compute_candidate_weights(self, candidates: List[str]) -> List[float]:
        """计算候选权重 (基于历史准确率和一致性)"""
        if not candidates:
            return []
        
        # ========== 1. 计算候选间一致性 ==========
        consistency_scores = []
        for i, cand in enumerate(candidates):
            consistency = 1.0
            for j, other in enumerate(candidates):
                if i != j:
                    # 简化：文本相似度
                    similarity = self._text_similarity(cand, other)
                    consistency *= similarity
            consistency_scores.append(consistency)
        
        # ========== 2. 结合历史准确率 ==========
        recent_accuracy = (
            sum(self.accuracy_history[-self.accuracy_window:])
            / len(self.accuracy_history[-self.accuracy_window:])
            if self.accuracy_history else 0.5
        )
        
        # ========== 3. 计算最终权重 ==========
        weights = [
            c * recent_accuracy for c in consistency_scores
        ]
        
        return weights
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度 (简化版)"""
        # 实际应使用更复杂的语义相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _generate_proposal(
        self,
        input_text: str,
        context: Optional[List[str]] = None
    ) -> str:
        """生成提案 (模式 2)"""
        # TODO: 调用模型生成提案
        return f"[Proposal] {input_text}"
    
    def _verify_proposal(self, proposal: str) -> dict:
        """
        验证提案 (模式 2)
        
        Returns:
            dict: {
                'is_valid': bool,
                'confidence': float,
                'corrections': List[str]
            }
        """
        # TODO: 调用模型验证
        # 简化：假设有效
        return {
            'is_valid': True,
            'confidence': 0.8,
            'corrections': []
        }
    
    def _apply_corrections(self, proposal: str, corrections: List[str]) -> str:
        """应用修正"""
        if not corrections:
            return proposal
        
        # 简单拼接修正
        corrected = proposal + " [Corrected: " + ", ".join(corrections) + "]"
        return corrected
    
    def _evaluate_candidates(
        self,
        candidates: List[str],
        input_text: str
    ) -> List[float]:
        """
        评判候选结果
        
        评判维度:
        - 事实准确性 (0-10)
        - 逻辑完整性 (0-10)
        - 语义连贯性 (0-10)
        - 指令遵循度 (0-10)
        
        Returns:
            scores: 每个候选的总分
        """
        scores = []
        
        for cand in candidates:
            # TODO: 调用评判模型
            # 简化：随机分数
            total = sum([random.uniform(6, 10) for _ in range(4)])
            scores.append(total)
        
        return scores
    
    def _update_accuracy_history(self, confidence: float):
        """更新准确率历史"""
        self.accuracy_history.append(confidence)
        
        # 保持窗口大小
        if len(self.accuracy_history) > self.accuracy_window * 2:
            self.accuracy_history = self.accuracy_history[-self.accuracy_window:]
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'cycle_count': self.cycle_count,
            'current_role': self.current_role,
            'avg_accuracy': (
                sum(self.accuracy_history) / len(self.accuracy_history)
                if self.accuracy_history else 0.5
            ),
            'accuracy_window_size': len(self.accuracy_history)
        }
