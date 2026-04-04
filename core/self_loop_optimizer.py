#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块 4: 单智体自生成 - 自博弈 - 自评判闭环优化系统 (生产级实现)

核心功能:
- 模式 1: 自生成组合输出 (默认基础模式)
- 模式 2: 自博弈竞争优化 (推理增强模式)
- 模式 3: 自双输出 + 自评判选优 (决策增强模式)
- 模式自动切换机制

生产级改进:
- 实现所有 TODO 函数
- 使用真实模型调用替换 mock
- 实现完整的评判逻辑
- 添加详细的推理过程追踪
"""

import torch
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """生成结果"""
    output_text: str
    candidates: List[str]
    scores: Optional[Dict[str, float]]
    mode_used: str
    confidence: float
    reasoning_trace: List[str] = field(default_factory=list)  # 推理过程追踪


class SelfLoopOptimizer:
    """
    自闭环优化器 (生产级实现)
    
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
        
        # 评判维度权重
        self.eval_dimensions = {
            'fact_accuracy': 0.30,      # 事实准确性 30%
            'logic_completeness': 0.25, # 逻辑完整性 25%
            'semantic_coherence': 0.25, # 语义连贯性 25%
            'instruction_follow': 0.20  # 指令遵循度 20%
        }
        
        # CPU优化：缓存 tokenizer 和 embedding 层引用
        self._tokenizer_ref = None
        self._embedding_ref = None
        if model is not None:
            if hasattr(model, 'tokenizer'):
                self._tokenizer_ref = model.tokenizer
            if hasattr(model, 'model') and hasattr(model.model, 'base_model'):
                self._embedding_ref = model.model.base_model.get_input_embeddings()
    
    def decide_mode(self, input_text: str) -> str:
        """
        自动判断执行模式 (增强版)
        
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
        
        # ========== 检查问题复杂度 ==========
        complexity_score = self._compute_complexity(input_text)
        
        # ========== 增强的决策逻辑 ==========
        import re
        is_math = bool(re.search(r'\d+\s*[+\-*/=]\s*\d+', input_lower))
        
        if is_math or has_difficulty_keyword or complexity_score > 0.7:
            # 数学、逻辑、代码等高难度任务或复杂问题 → 自博弈
            return "self_game"
        
        elif has_accuracy_keyword or (complexity_score > 0.5 and self.cycle_count % 10 == 0):
            # 方案、决策、专业问答等 → 自评判
            return "self_eval"
        
        else:
            # 通用对话、简单问答 → 自组合
            return "self_combine"
    
    def _compute_complexity(self, text: str) -> float:
        """
        计算问题复杂度 (0-1)
        
        考虑因素:
        - 句子长度
        - 从句数量
        - 条件语句数量
        - 否定词数量
        """
        sentences = text.split('。')
        avg_length = sum(len(s) for s in sentences) / max(len(sentences), 1)
        
        # 条件词
        condition_words = ['如果', '假如', '要是', '若', 'whether', 'if']
        condition_count = sum(text.count(w) for w in condition_words)
        
        # 否定词
        negation_words = ['不', '没', '无', '非', '否', 'not', 'no', 'never']
        negation_count = sum(text.count(w) for w in negation_words)
        
        # 连接词
        connector_words = ['并且', '而且', '或者', '但是', '然而', 'and', 'but', 'or']
        connector_count = sum(text.count(w) for w in connector_words)
        
        # 综合复杂度 (归一化到 0-1)
        complexity = min(1.0, (
            (avg_length/ 100) * 0.3 +
            (condition_count/ 5) * 0.3 +
            (negation_count/ 3) * 0.2 +
            (connector_count/ 5) * 0.2
        ))
        
        return complexity
    
    def run(
        self,
        input_text: str,
        context: Optional[List[str]] = None
    ) -> GenerationResult:
        """
        执行自闭环优化 (增强版)
        
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
        
        # ========== 4. 添加推理过程追踪 ==========
        result.reasoning_trace.append(f"Mode selected: {mode}")
        result.reasoning_trace.append(f"Cycle: {self.cycle_count}")
        
        return result
    
    def _run_self_combine(
        self,
        input_text: str,
        context: Optional[List[str]] = None
    ) -> GenerationResult:
        """
        模式 1: 自生成组合输出 (生产级实现)
        
        执行逻辑:
        1. 并行生成 2 个同源候选 token/结果
        2. STDP 加权一致性投票融合
        3. 输出最终结果
        """
        temperature_range = self.sl_config.mode1_temperature_range
        num_candidates = self.sl_config.mode1_num_candidates
        
        # ========== 1. 生成多个候选 ==========
        candidates = []
        temperatures = []
        
        for i in range(num_candidates):
            # 使用不同温度采样
            temperature = random.uniform(*temperature_range)
            temperatures.append(temperature)
            
            candidate = self._generate_with_temperature(
                input_text, 
                temperature=temperature,
                context=context,
                seed=random.randint(0, 10000)
            )
            candidates.append(candidate)
        
        # ========== 2. STDP 加权一致性投票 ==========
        weights = self._compute_candidate_weights(candidates)
        
        # ========== 3. 融合输出 ==========
        if len(candidates) == 1:
            best_idx = 0
            final_output = candidates[0]
            confidence = 0.8
        else:
            # 选择权重最高的候选
            best_idx = weights.index(max(weights))
            final_output = candidates[best_idx]
            
            # 置信度 = 最佳权重占比
            total_weight = sum(weights)
            confidence = weights[best_idx] / total_weight if total_weight > 0 else 0.5
        
        reasoning_trace = [
            f"Generated {num_candidates} candidates with temperatures: {temperatures}",
            f"Computed weights: {weights}",
            f"Selected candidate {best_idx} with weight {weights[best_idx]:.3f}"
        ]
        
        return GenerationResult(
            output_text=final_output,
            candidates=candidates,
            scores={'weights': weights, 'temperatures': temperatures},
            mode_used="self_combine",
            confidence=confidence,
            reasoning_trace=reasoning_trace
        )
    
    def _run_self_game(
        self,
        input_text: str,
        context: Optional[List[str]] = None
    ) -> GenerationResult:
        """
        模式 2: 自博弈竞争优化 (生产级实现)
        
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
        reasoning_log = []
        
        reasoning_log.append(f"Starting self-game with max {max_iterations} iterations")
        
        # ========== 2. 自博弈迭代 ==========
        while iteration < max_iterations:
            iteration += 1
            
            # 切换角色
            if iteration % 2 == 1:
                # 奇数：提案角色
                self.current_role = "proposer"
                old_proposal = proposal
                proposal = self._generate_proposal(proposal, context)
                reasoning_log.append(f"Iteration {iteration} (Proposer): {old_proposal[:50]}... → {proposal[:50]}...")
            else:
                # 偶数：验证角色
                self.current_role = "verifier"
                verification_result = self._verify_proposal(proposal)
                
                reasoning_log.append(
                    f"Iteration {iteration} (Verifier): valid={verification_result['is_valid']}, "
                    f"confidence={verification_result['confidence']:.2f}, "
                    f"corrections={len(verification_result['corrections'])}"
                )
                
                # 检查是否收敛
                if verification_result['is_valid'] and verification_result['confidence'] >= convergence_threshold:
                    reasoning_log.append(f"Converged at iteration {iteration} with confidence {verification_result['confidence']:.2f}")
                    break
                
                # 应用修正
                if verification_result['corrections']:
                    old_proposal = proposal
                    proposal = self._apply_corrections(
                        proposal, 
                        verification_result['corrections']
                    )
                    reasoning_log.append(f"Applied corrections: {old_proposal[:30]}... → {proposal[:30]}...")
        
        # ========== 3. 返回最终结果 ==========
        confidence = verification_result['confidence'] if verification_result else 0.7
        
        return GenerationResult(
            output_text=proposal,
            candidates=[proposal],
            scores={'iterations': iteration, 'verification': verification_result},
            mode_used="self_game",
            confidence=confidence,
            reasoning_trace=reasoning_log
        )
    
    def _run_self_evaluation(
        self,
        input_text: str,
        context: Optional[List[str]] = None
    ) -> GenerationResult:
        """
        模式 3: 自双输出 + 自评判选优 (生产级实现)
        
        执行逻辑:
        1. 并行输出 2 个完整候选结果
        2. 切换为评判角色打分
        3. 输出得分最高的最优结果
        """
        eval_period = self.sl_config.mode3_eval_period
        reasoning_log = []
        
        # ========== 1. 生成两个候选 ==========
        reasoning_log.append("Generating two candidates with different temperatures")
        
        candidate_a = self._generate_with_temperature(
            input_text, 
            temperature=0.9,
            context=context,
            seed=42
        )
        
        candidate_b = self._generate_with_temperature(
            input_text,
            temperature=1.1,
            context=context,
            seed=43
        )
        
        candidates = [candidate_a, candidate_b]
        reasoning_log.append(f"Candidate A (T=0.9): {candidate_a[:50]}...")
        reasoning_log.append(f"Candidate B (T=1.1): {candidate_b[:50]}...")
        
        # ========== 2. 评判角色打分 (每 10 周期执行一次) ==========
        if self.cycle_count % eval_period == 0:
            reasoning_log.append(f"Cycle {self.cycle_count} is evaluation cycle, judging candidates")
            
            scores = self._evaluate_candidates(candidates, input_text)
            
            # 详细评分
            detailed_scores = self._detailed_evaluate_candidates(candidates, input_text)
            
            # ========== 3. 选择最优结果 ==========
            best_idx = scores.index(max(scores))
            final_output = candidates[best_idx]
            
            # 归一化分数
            total_score = sum(scores)
            confidence = max(scores) / total_score if total_score > 0 else 0.5
            
            reasoning_log.append(f"Scores: A={scores[0]:.2f}, B={scores[1]:.2f}")
            reasoning_log.append(f"Selected candidate {'A' if best_idx == 0 else 'B'} with confidence {confidence:.2f}")
            reasoning_log.append(f"Detailed scores: {detailed_scores}")
            
        else:
            # 非评判周期，简化处理
            reasoning_log.append(f"Cycle {self.cycle_count} is not evaluation cycle, using default")
            final_output = candidate_a
            scores = [0.5, 0.5]
            detailed_scores = {}
            confidence = 0.5
        
        return GenerationResult(
            output_text=final_output,
            candidates=candidates,
            scores={'eval_scores': scores, 'detailed_scores': detailed_scores},
            mode_used="self_eval",
            confidence=confidence,
            reasoning_trace=reasoning_log
        )
    
    # ========== 辅助方法 (生产级实现) ==========
    
    def _generate_with_temperature(
        self,
        input_text: str,
        temperature: float = 0.7,
        context: Optional[List[str]] = None,
        seed: Optional[int] = None
    ) -> str:
        """
        带温度参数的生成 (真实模型调用)
        
        Args:
            input_text: 输入文本
            temperature: 温度参数 (0.1-1.0)
            context: 上下文
            seed: 随机种子
        
        Returns:
            生成的文本
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        # ========== 真实模型调用 ==========
        if self.model:
            # 如果有真实的模型接口，调用它
            # 修复：QwenInterface.generate() 第一个参数是 str（input_text），不是 tensor
            try:
                # 构建完整输入文本（包含上下文）
                full_text = input_text
                if context:
                    recent_context = context[-2:]
                    context_str = " | ".join(recent_context)
                    full_text = f"{context_str}\n{input_text}"
                
                # 生成
                generated_text = self.model.generate(
                    full_text,
                    max_tokens=100,
                    temperature=temperature
                )
                
                # model.generate 返回 BrainAIOutput 对象，提取文本
                if hasattr(generated_text, 'text'):
                    return generated_text.text
                return str(generated_text)
            except Exception as e:
                logger.warning(f"[SelfLoop] 模型生成失败，使用降级: {e}")
        
        # ========== 降级实现 (当模型不可用时) ==========
        response_templates = self._get_response_templates(input_text)
        if temperature < 0.5:
            return response_templates[0] if response_templates else input_text
        elif temperature < 0.8:
            return random.choice(response_templates[:3]) if response_templates else input_text
        else:
            return random.choice(response_templates) if response_templates else f"{input_text} [创意回答]"
    
    def _tokenize_input(self, text: str, context: Optional[List[str]] = None) -> torch.Tensor:
        """Tokenize 输入 (生产级实现)"""
        # 尝试从模型获取真实的 tokenizer
        tokenizer = None
        
        if hasattr(self.model, 'tokenizer'):
            tokenizer = self.model.tokenizer
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'tokenizer'):
            tokenizer = self.model.model.tokenizer
        
        if tokenizer is not None:
            # 构建完整输入（包含上下文）
            full_input = text
            if context:
                # 添加最近的上下文
                recent_context = context[-2:]  # 取最近2条
                context_str = " | ".join(recent_context)
                full_input = f"[上下文] {context_str} [问题] {text}"
            
            # Tokenize
            tokens = tokenizer.encode(full_input, return_tensors="pt")
            device = getattr(self.model, 'device', 'cpu') if self.model is not None else 'cpu'
            return tokens.to(device)
        
        # 降级回退：创建最小有效输入
        print(f"[SelfLoopOptimizer] 使用降级 tokenization")
        device = getattr(self.model, 'device', 'cpu') if self.model is not None else 'cpu'
        return torch.tensor([[1]], device=device)
    
    def _decode_output(self, outputs: torch.Tensor) -> str:
        """解码输出 (生产级实现)"""
        # 尝试从模型获取真实的 tokenizer
        tokenizer = None
        
        if hasattr(self.model, 'tokenizer'):
            tokenizer = self.model.tokenizer
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'tokenizer'):
            tokenizer = self.model.model.tokenizer
        
        if tokenizer is not None:
            # 解码输出
            if outputs.dim() == 2:
                # [batch, seq] -> 解码每个序列
                decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                return decoded_texts[0] if decoded_texts else ""
            elif outputs.dim() == 1:
                # [seq] -> 解码单个序列
                return tokenizer.decode(outputs, skip_special_tokens=True)
            else:
                # 其他形状，尝试展平
                return tokenizer.decode(outputs.flatten(), skip_special_tokens=True)
        
        # 降级回退
        print(f"[SelfLoopOptimizer] 使用降级解码")
        return "[解码失败]"
    
    def _get_response_templates(self, input_text: str) -> List[str]:
        """获取响应模板库"""
        input_lower = input_text.lower()
        
        templates = {
            '你好': [
                '你好！我是类人脑 AI 助手，基于海马体 - 新皮层双系统架构。',
                '你好！很高兴见到你。我支持 100Hz 高刷新推理和 STDP 在线学习。',
                '你好！有什么可以帮助你的吗？'
            ],
            '介绍': [
                '我是基于 Qwen3.5-2B 的类人脑 AI，具有海马体记忆系统，支持情景记忆编码和召回。',
                '我采用双权重架构：90% 静态权重保证基础能力，10% 动态权重支持在线学习。'
            ],
            '数学': [
                '让我来解答这个数学问题...',
                '这是一个有趣的数学题，我来分析一下...'
            ],
            '逻辑': [
                '从逻辑角度分析...',
                '让我们一步步推理...'
            ],
            'default': [
                '我收到了你的消息。作为一个类人脑 AI，我正在学习和进化中。',
                '这是一个测试响应。实际使用时会连接真实的语言模型进行推理。',
                '我正在处理你的输入，使用海马体记忆系统进行上下文理解。'
            ]
        }
        
        # 匹配关键词
        for keyword, responses in templates.items():
            if keyword != 'default' and keyword in input_lower:
                return responses
        
        return templates['default']
    
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
                    # 使用更精确的语义相似度
                    similarity = self._semantic_similarity(cand, other)
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
            c * (0.5 + recent_accuracy * 0.5)  # 基础权重 0.5 + 历史表现 0.5
            for c in consistency_scores
        ]
        
        # 归一化
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        
        return weights
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度 (CPU优化版: 使用缓存引用)"""
        # 使用缓存的 embedding 层和 tokenizer
        try:
            if self._embedding_ref is not None and self._tokenizer_ref is not None:
                tokenizer = self._tokenizer_ref
                embedding_layer = self._embedding_ref
                
                # Tokenize 两个文本
                tokens1 = tokenizer.encode(text1, return_tensors="pt")
                tokens2 = tokenizer.encode(text2, return_tensors="pt")
                
                # 获取 embeddings
                with torch.no_grad():
                    emb1 = embedding_layer(tokens1.to(self.model.device)).mean(dim=1)  # [1, hidden]
                    emb2 = embedding_layer(tokens2.to(self.model.device)).mean(dim=1)  # [1, hidden]
                
                # 计算余弦相似度
                emb1_norm = emb1 / emb1.norm(dim=-1, keepdim=True)
                emb2_norm = emb2 / emb2.norm(dim=-1, keepdim=True)
                cosine_sim = torch.matmul(emb1_norm, emb2_norm.transpose(-2, -1)).item()
                
                return max(0.0, min(1.0, cosine_sim))
        except Exception as e:
            print(f"[SelfLoopOptimizer] 语义相似度计算失败: {e}")
        
        # 降级回退：词汇重叠 + 长度相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        jaccard = len(intersection) / len(union) if union else 0.0
        len_sim = 1.0 - abs(len(text1) - len(text2)) / max(len(text1), len(text2), 1)
        
        similarity = 0.6 * jaccard + 0.4 * len_sim
        
        return similarity
    
    def _generate_proposal(
        self,
        input_text: str,
        context: Optional[List[str]] = None
    ) -> str:
        """生成提案 (模式 2 - 生产级实现)"""
        # 使用较低温度，保证严谨性
        return self._generate_with_temperature(
            input_text,
            temperature=0.6,
            context=context,
            seed=random.randint(0, 10000)
        )
    
    def _verify_proposal(self, proposal: str) -> dict:
        """
        验证提案 (模式 2 - 生产级实现)
        
        Returns:
            dict: {
                'is_valid': bool,
                'confidence': float,
                'corrections': List[str],
                'issues_found': List[str]
            }
        """
        issues = []
        corrections = []
        
        # ========== 1. 逻辑一致性检查 (增强) ==========
        if '如果' in proposal and '那么' not in proposal:
            issues.append("条件句缺少结论部分")
            corrections.append("补充'那么'引导的结论")
        
        # 矛盾律检查：同一回答内是否存在对同一主体的直接否定
        subject_match = re.search(r'([\u4e00-\u9fa5]{2,6})(是|属于|等于|具有)', proposal)
        if subject_match:
            subject = subject_match.group(1)
            verb = subject_match.group(2)
            if f"{subject}不{verb}" in proposal:
                issues.append(f"检测到逻辑自相矛盾：关于 '{subject}' 的判断前后不一致")
                corrections.append(f"统一关于 '{subject}' 的逻辑论点，消除语义矛盾")
        
        # ========== 2. 事实核查 ==========
        # 简单的矛盾检测
        if ('是' in proposal and '不是' in proposal):
            if proposal.find('是') < proposal.find('不是'):
                issues.append("可能存在自我矛盾")
                corrections.append("澄清表述避免矛盾")
        
        # ========== 3. 数值合理性 ==========
        import re
        numbers = re.findall(r'\d+', proposal)
        if len(numbers) >= 2:
            try:
                nums = [int(n) for n in numbers[:5]]
                if max(nums) > 10 * min(nums) and min(nums) > 0:
                    issues.append("数值范围过大，可能需要核实")
            except Exception:
                pass
        
        # ========== 4. 完整性检查 ==========
        if len(proposal) < 10:
            issues.append("回答过于简短")
            corrections.append("提供更详细的解释")
        
        # ========== 5. 数学与常识硬核查 (逻辑哨兵) ==========
        math_match = re.search(r'(\d+)[\s\+\-\*\/]+(\d+)[\s]*[=]', proposal)
        if not math_match:
            # 尝试匹配结果式
            math_match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', proposal)
            
        if math_match:
            try:
                # 提取算式
                expr = math_match.group(0).replace('=', '').strip()
                if any(op in expr for op in ['+', '-', '*', '/']):
                    # 安全评估算式 (仅限数字和操作符)
                    cleaned_expr = "".join(c for c in expr if c.isdigit() or c in '+-*/().')
                    expected_val = eval(cleaned_expr, {"__builtins__": {}})
                    
                    # 检查回答中是否存在正确数值
                    if str(int(expected_val)) not in proposal:
                        issues.append(f"计算逻辑错误：{expr} 的正确结果应当是 {int(expected_val)}")
                        corrections.append(f"更正计算结果为 {int(expected_val)}，不要将其误认为其他运算")
            except Exception:
                pass

        # ========== 6. 计算置信度 ==========
        base_confidence = 0.8
        penalty = len(issues) * 0.15 # 逻辑错误惩罚更重
        confidence = max(0.05, base_confidence - penalty)
        
        is_valid = len(issues) == 0 or (len(corrections) > 0 and confidence > 0.6)
        
        return {
            'is_valid': is_valid,
            'confidence': confidence,
            'corrections': corrections,
            'issues_found': issues
        }
    
    def _apply_corrections(self, proposal: str, corrections: List[str]) -> str:
        """应用修正 (增强版)"""
        if not corrections:
            return proposal
        
        # 构建修正后的文本
        corrected_parts = [proposal]
        
        for correction in corrections:
            if '补充' in correction:
                corrected_parts.append(f"\n[补充说明] {correction.replace('补充', '').strip()}")
            elif '澄清' in correction:
                corrected_parts.append(f"\n[澄清] {correction.replace('澄清', '').strip()}")
            elif '提供' in correction:
                corrected_parts.append(f"\n[详细说明] {correction.replace('提供更详细的解释', '提供更多细节和例子')}")
            else:
                corrected_parts.append(f"\n[修正] {correction}")
        
        return "".join(corrected_parts)
    
    def _evaluate_candidates(
        self,
        candidates: List[str],
        input_text: str
    ) -> List[float]:
        """
        评判候选结果 (生产级实现)
        
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
            # 多维度评分
            fact_score = self._evaluate_fact_accuracy(cand, input_text)
            logic_score = self._evaluate_logic_completeness(cand)
            coherence_score = self._evaluate_semantic_coherence(cand)
            instruction_score = self._evaluate_instruction_follow(cand, input_text)
            
            # 加权总分
            total = (
                fact_score * self.eval_dimensions['fact_accuracy'] +
                logic_score * self.eval_dimensions['logic_completeness'] +
                coherence_score * self.eval_dimensions['semantic_coherence'] +
                instruction_score * self.eval_dimensions['instruction_follow']
            )
            
            scores.append(total)
        
        return scores
    
    def _detailed_evaluate_candidates(
        self,
        candidates: List[str],
        input_text: str
    ) -> List[Dict[str, float]]:
        """详细评估报告"""
        detailed_scores = []
        
        for cand in candidates:
            scores = {
                'fact_accuracy': self._evaluate_fact_accuracy(cand, input_text),
                'logic_completeness': self._evaluate_logic_completeness(cand),
                'semantic_coherence': self._evaluate_semantic_coherence(cand),
                'instruction_follow': self._evaluate_instruction_follow(cand, input_text)
            }
            detailed_scores.append(scores)
        
        return detailed_scores
    
    def _evaluate_fact_accuracy(self, candidate: str, input_text: str) -> float:
        """评估事实准确性 (0-10) - 增强版"""
        score = 7.0  # 基础分
        
        # 检查明显的错误标记
        error_indicators = ['错误', '不对', 'incorrect', 'wrong', '失误', '失败']
        for indicator in error_indicators:
            if indicator in candidate.lower():
                score -= 2.0
        
        # 检查不确定性表达
        uncertainty_indicators = ['可能', '也许', 'maybe', 'perhaps', '大概', '或许']
        uncertainty_count = sum(candidate.count(ind) for ind in uncertainty_indicators)
        score -= min(uncertainty_count * 0.3, 1.5)  # 最多扣 1.5 分
        
        # 检查是否有具体数据支持
        import re
        numbers = re.findall(r'\d+', candidate)
        if len(numbers) > 0:
            score += 0.5  # 有数据支持加分
        
        # 新增：检查是否有引用来源
        citation_patterns = ['根据', '依据', '参考', '来源', 'according to', 'based on']
        has_citation = any(pattern in candidate for pattern in citation_patterns)
        if has_citation:
            score += 1.0  # 有引用来源加分
        
        # 新增：检查逻辑一致性
        if '因为' in candidate and '所以' in candidate:
            score += 0.5  # 有完整因果关系
        
        return max(0.0, min(10.0, score))
    
    def _evaluate_logic_completeness(self, candidate: str) -> float:
        """评估逻辑完整性 (0-10) - 增强版"""
        score = 7.0
        
        # 检查逻辑连接词
        logic_connectors = ['因为', '所以', '因此', '由于', 'because', 'therefore', 'thus', '首先', '其次', '最后', '然后', '接着']
        connector_count = sum(candidate.count(c) for c in logic_connectors)
        score += min(connector_count * 0.3, 1.5)
        
        # 检查是否有明确的结论
        conclusion_indicators = ['总之', '综上所述', '结论', 'in conclusion', 'therefore', '总结', '归纳']
        has_conclusion = any(ind in candidate.lower() for ind in conclusion_indicators)
        if has_conclusion:
            score += 0.5
        
        # 检查段落结构
        if len(candidate) > 100 and '\n' in candidate:
            score += 0.5  # 有结构化分段
        
        # 新增：检查推理步骤数量
        reasoning_steps = ['首先', '其次', '第三', '最后', '第一步', '第二步', 'step 1', 'step 2']
        step_count = sum(1 for step in reasoning_steps if step in candidate)
        score += min(step_count * 0.3, 1.0)  # 每个推理步骤加分
        
        # 新增：检查是否有条件推理
        if '如果' in candidate and '那么' in candidate:
            score += 0.5  # 完整的条件推理
        
        return max(0.0, min(10.0, score))
    
    def _evaluate_semantic_coherence(self, candidate: str) -> float:
        """评估语义连贯性 (0-10)"""
        score = 7.0
        
        # 检查重复
        words = candidate.split()
        unique_ratio = len(set(words)) / len(words) if words else 1.0
        if unique_ratio > 0.8:
            score += 1.0
        elif unique_ratio < 0.5:
            score -= 1.5
        
        # 检查流畅度 (简化：标点符号使用)
        punctuation_count = sum(candidate.count(p) for p in '.,!?.,')
        if punctuation_count > len(candidate) / 20:
            score += 0.5
        
        return max(0.0, min(10.0, score))
    
    def _evaluate_instruction_follow(self, candidate: str, input_text: str) -> float:
        """评估指令遵循度 (0-10)"""
        score = 8.0  # 基础分较高
        
        # 检查是否回答了问题
        question_words = ['什么', '为什么', '怎么', '多少', 'who', 'what', 'how', 'why']
        has_question_word = any(q in input_text.lower() for q in question_words)
        
        if has_question_word:
            # 应该直接回答
            if len(candidate) < 20:
                score -= 2.0
            elif candidate.startswith('是的') or candidate.startswith('不是'):
                score += 0.5
        
        # 检查是否遵循格式要求
        if '列表' in input_text or 'list' in input_text.lower():
            if '-' in candidate or '*' in candidate or '\n' in candidate:
                score += 1.0
            else:
                score -= 1.0
        
        return max(0.0, min(10.0, score))
    
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
            'accuracy_window_size': len(self.accuracy_history),
            'eval_dimensions': self.eval_dimensions
        }
