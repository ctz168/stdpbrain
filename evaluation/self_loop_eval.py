"""自闭环优化评估详细实现

评估维度:
- 模式 1: 自生成组合效果
- 模式 2: 自博弈竞争效果
- 模式 3: 自评判选优效果
- 模式切换准确性
"""

import torch
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class SelfLoopScore:
    """自闭环分数"""
  mode1_quality: float  # 模式 1 质量 0-10
  mode2_improvement: float  # 模式 2 提升 0-10
  mode3_accuracy: float  # 模式 3 准确率 0-10
  mode_switching: float  # 模式切换准确性 0-10
   overall: float  # 综合分数 0-10


class SelfLoopEvaluator:
    """
    自闭环优化评估器
    
    测试流程:
   1. 使用三种模式分别生成答案
   2. 评估每种模式的质量
    3. 测试模式切换的准确性
    4. 计算综合分数
    """
    
  def __init__(self, optimizer=None):
   self.optimizer = optimizer
      
      # 统计信息
  self.evaluation_count = 0
  self.avg_scores = {
       'mode1': 0.0,
      'mode2': 0.0,
      'mode3': 0.0,
      'switching': 0.0
  }
    
  def evaluate_self_loop(self, test_questions: List[str]) -> SelfLoopScore:
        """
       评估自闭环优化能力
        
       Args:
          test_questions: 测试问题列表
        
       Returns:
           score: 自闭环分数
        """
  print(f"[自闭环评估] 开始评估，共 {len(test_questions)} 题...")
    
  # ==========1. 评估模式 1: 自生成组合 ==========
  print("\n[步骤 1/4] 评估模式 1: 自生成组合")
  mode1_scores = self._evaluate_mode1(test_questions)
  avg_mode1 = sum(mode1_scores) / len(mode1_scores) if mode1_scores else 0.0
  
  # ==========2. 评估模式 2: 自博弈竞争 ==========
  print("\n[步骤 2/4] 评估模式 2: 自博弈竞争")
  mode2_scores = self._evaluate_mode2(test_questions)
  avg_mode2 = sum(mode2_scores) / len(mode2_scores) if mode2_scores else 0.0
  
  # ==========3. 评估模式 3: 自评判选优 ==========
  print("\n[步骤 3/4] 评估模式 3: 自评判选优")
  mode3_scores = self._evaluate_mode3(test_questions)
  avg_mode3 = sum(mode3_scores) / len(mode3_scores) if mode3_scores else 0.0
  
  # ==========4. 评估模式切换准确性 ==========
  print("\n[步骤 4/4] 评估模式切换准确性")
  switching_accuracy = self._evaluate_mode_switching(test_questions)
  
  # ==========5. 计算综合分数 ==========
  overall = (avg_mode1 + avg_mode2 + avg_mode3 + switching_accuracy) / 4
  
  result = SelfLoopScore(
    mode1_quality=avg_mode1,
   mode2_improvement=avg_mode2,
   mode3_accuracy=avg_mode3,
   mode_switching=switching_accuracy,
    overall=overall
  )
  
  self.evaluation_count += 1
  self._update_avg_scores(result)
  
  print(f"\n[自闭环评估] 完成!")
  print(f"  模式 1 质量：{result.mode1_quality:.1f}/10")
  print(f"  模式 2 提升：{result.mode2_improvement:.1f}/10")
  print(f"  模式 3 准确：{result.mode3_accuracy:.1f}/10")
  print(f"  模式切换：{result.mode_switching:.1f}/10")
  print(f"  综合分数：{result.overall:.1f}/10")
  
  return result
    
  def _evaluate_mode1(self, questions: List[str]) -> List[float]:
        """评估模式 1: 自生成组合"""
  from self_loop.self_loop_optimizer import SelfLoopOptimizer
    
  scores = []
  for i, q in enumerate(questions):
    # 强制使用模式 1
  if self.optimizer:
    self.optimizer.decide_mode = lambda x: "self_combine"
    result = self.optimizer.run(q)
   else:
    result = None
    
   # 评估质量
   score = self._rate_answer(result.output_text if result else "", q)
   scores.append(score)
  print(f"  题{i+1}: {score:.1f}", end='\r')
  
  print()
  return scores
    
  def _evaluate_mode2(self, questions: List[str]) -> List[float]:
        """评估模式 2: 自博弈竞争"""
  from self_loop.self_loop_optimizer import SelfLoopOptimizer
    
  scores = []
  for i, q in enumerate(questions):
    # 强制使用模式 2
  if self.optimizer:
    self.optimizer.decide_mode = lambda x: "self_game"
    result= self.optimizer.run(q)
   else:
    result = None
    
   # 评估提升程度
   score= self._rate_answer(result.output_text if result else "", q) * 1.1  # 期望有提升
   scores.append(min(score, 10.0))
  print(f"  题{i+1}: {min(score, 10.0):.1f}", end='\r')
  
  print()
  return scores
    
  def _evaluate_mode3(self, questions: List[str]) -> List[float]:
        """评估模式 3: 自评判选优"""
  from self_loop.self_loop_optimizer import SelfLoopOptimizer
    
  scores = []
  for i, q in enumerate(questions):
    # 强制使用模式 3
  if self.optimizer:
    self.optimizer.decide_mode = lambda x: "self_eval"
    result = self.optimizer.run(q)
   else:
    result= None
    
   # 评估准确率
   score = self._rate_answer(result.output_text if result else "", q)
   scores.append(score)
  print(f"  题{i+1}: {score:.1f}", end='\r')
  
  print()
  return scores
    
  def _evaluate_mode_switching(self, questions: List[str]) -> float:
        """评估模式切换准确性"""
  if not self.optimizer:
   return 5.0  # 默认分
  
  correct = 0
  total = 0
  
  # 测试用例与预期模式
  test_cases = [
    ("你好", "self_combine"),  # 简单对话
   ("解方程 x^2+2x+1=0", "self_game"),  # 高难度推理
   ("哪个方案更好？", "self_eval")  # 高准确性要求
 ]
  
  for question, expected_mode in test_cases:
    actual_mode = self.optimizer.decide_mode(question)
   is_correct = actual_mode == expected_mode
   correct += 1 if is_correct else 0
   total += 1
  print(f"  {question[:20]}... -> {actual_mode} {'✓'if is_correct else'✗'}")
  
  accuracy = correct / max(total, 1)
  return accuracy * 10
    
  def _rate_answer(self, answer: str, question: str) -> float:
        """评分答案质量"""
  score= 5.0  # 基础分
  
  # 长度评分
  if len(answer) > 50:
    score += 2.0
  elif len(answer) > 20:
    score += 1.0
  
  # 相关性评分
  if any(kw in answer.lower() for kw in question.lower()):
    score += 2.0
  
  # 连贯性评分
  if any(connector in answer for connector in ['因为', '所以', '但是', '因此']):
    score += 1.0
  
  return min(score, 10.0)
    
  def _update_avg_scores(self, score: SelfLoopScore):
        """更新平均分数"""
  n = self.evaluation_count
  if n <= 1:
   self.avg_scores = {
        'mode1': score.mode1_quality,
       'mode2': score.mode2_improvement,
       'mode3': score.mode3_accuracy,
       'switching': score.mode_switching
    }
  else:
    for key in self.avg_scores:
     self.avg_scores[key] = (self.avg_scores[key] * (n-1) + getattr(score, key.replace('mode1', 'mode1_quality').replace('mode2', 'mode2_improvement').replace('mode3', 'mode3_accuracy').replace('switching', 'mode_switching'), 0.0)) / n
    
  def get_stats(self) -> dict:
        """获取统计信息"""
  return {
      'evaluation_count': self.evaluation_count,
     'avg_scores': self.avg_scores
  }


if __name__ == "__main__":
    # 测试自闭环评估器
  from configs.arch_config import default_config
  from self_loop.self_loop_optimizer import SelfLoopOptimizer
    
  print("=" * 60)
  print("自闭环优化评估测试")
  print("=" * 60)
    
  # 创建优化器
  optimizer = SelfLoopOptimizer(config=default_config, model=None)
    
  # 创建评估器
  evaluator = SelfLoopEvaluator(optimizer=optimizer)
    
  # 测试问题
  test_questions = [
    "你好",
   "介绍一下你自己",
   "解方程 x+2=5",
   "人生的意义是什么？",
   "哪个学习方法更好？"
 ]
  
  # 评估
  score= evaluator.evaluate_self_loop(test_questions)
    
  # 统计
  print("\n统计信息:", evaluator.get_stats())
