#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
归纳推理增强模块

功能:
1. 模式识别与补全
2. 类比映射
3. 规则归纳
4. 序列预测
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Pattern:
    """模式数据结构"""
   id: str
    rule: str
   examples: List[str]
    confidence: float
    support_count: int = 1


class InductiveReasoningEngine:
    """归纳推理引擎"""
    
  def __init__(self):
        self.pattern_library: Dict[str, Pattern] = {}
        self.similarity_threshold = 0.8
        self.min_support = 2  # 最小支持数
        
  def identify_pattern(self, sequence: List) -> Optional[Pattern]:
        """
        识别序列中的模式
        
        Args:
            sequence: 输入序列 (数字、字母、符号等)
            
       Returns:
            识别出的模式，如果无法识别则返回 None
        """
       # 1. 提取特征
        features = self._extract_features(sequence)
        
       # 2. 与已知模式匹配
       matches = self._match_patterns(features)
        
      if matches:
            # 找到匹配模式，使用最佳匹配
           best_match = max(matches, key=lambda p: p.confidence)
            best_match.support_count += 1
          return best_match
        else:
            # 未找到匹配，归纳新模式
           new_pattern = self._induce_new_pattern(features, sequence)
            self.pattern_library[new_pattern.id] = new_pattern
          return new_pattern
    
  def _extract_features(self, sequence: List) -> Dict:
        """提取序列特征"""
      if len(sequence) < 2:
          return {'type': 'single'}
        
       # 转换为数值数组 (如果是数字序列)
       try:
            nums = np.array([float(x) for x in sequence])
            
            features = {
                'type': 'numeric',
               'first_diff': np.diff(nums).tolist(),  # 一阶差分
               'second_diff': np.diff(nums, 2).tolist() if len(nums) > 2 else [],
               'ratio': (nums[1:] / nums[:-1]).tolist() if np.all(nums[:-1] != 0) else [],
               'is_arithmetic': self._is_arithmetic_sequence(nums),
               'is_geometric': self._is_geometric_sequence(nums),
               'periodicity': self._detect_periodicity(nums)
            }
          return features
      except (ValueError, TypeError):
           # 非数值序列 (字母、符号等)
          return {
               'type': 'symbolic',
               'sequence': sequence,
               'length': len(sequence)
           }
    
  def _is_arithmetic_sequence(self, nums: np.ndarray) -> bool:
        """判断是否为等差数列"""
      if len(nums) < 3:
          return False
        diffs = np.diff(nums)
      return np.allclose(diffs, diffs[0])
    
  def _is_geometric_sequence(self, nums: np.ndarray) -> bool:
        """判断是否为等比数列"""
      if len(nums) < 3 or np.any(nums == 0):
          return False
        ratios = nums[1:] / nums[:-1]
      return np.allclose(ratios, ratios[0])
    
  def _detect_periodicity(self, nums: np.ndarray, max_period=5) -> int:
        """检测周期性"""
      for period in range(2, min(max_period + 1, len(nums) // 2)):
           is_periodic = True
          for i in range(len(nums) - period):
             if nums[i] != nums[i + period]:
                    is_periodic = False
                   break
          if is_periodic:
             return period
      return 0
    
  def _match_patterns(self, features: Dict) -> List[Pattern]:
        """匹配已知模式"""
       matches = []
        
      for pattern_id, pattern in self.pattern_library.items():
            similarity = self._calculate_similarity(features, pattern)
          if similarity >= self.similarity_threshold:
                pattern.confidence = similarity
               matches.append(pattern)
        
      return matches
    
  def _calculate_similarity(self, features: Dict, pattern: Pattern) -> float:
        """计算特征与模式的相似度"""
       # 简化实现：基于规则字符串匹配
       # 实际应使用更复杂的相似度计算
        feature_str = str(features)
      if pattern.rule in feature_str:
          return 0.9
      return 0.5
    
  def _induce_new_pattern(self, features: Dict, sequence: List) -> Pattern:
        """归纳新模式"""
        pattern_id = f"pattern_{len(self.pattern_library)}"
        
      if features['type'] == 'numeric':
           rule = self._induce_numeric_rule(features, sequence)
       else:
            rule = self._induce_symbolic_rule(features, sequence)
        
      return Pattern(
           id=pattern_id,
            rule=rule,
           examples=[str(sequence)],
            confidence=0.7,  # 新模式初始置信度
            support_count=1
        )
    
  def _induce_numeric_rule(self, features: Dict, sequence: List) -> str:
        """归纳数值规则"""
      if features.get('is_arithmetic'):
            diff = features['first_diff'][0]
          return f"arithmetic: a[n] = a[0] + n*{diff}"
        
      if features.get('is_geometric'):
            ratio = features['ratio'][0]
          return f"geometric: a[n] = a[0] * {ratio}^n"
        
      if features.get('periodicity', 0) > 0:
           period = features['periodicity']
          return f"periodic: period={period}"
        
       # 默认：记录差分规律
      if features['first_diff']:
          return f"custom_diff: {features['first_diff']}"
        
      return "no_clear_pattern"
    
  def _induce_symbolic_rule(self, features: Dict, sequence: List) -> str:
        """归纳符号规则"""
       # 字母序列检测
      if all(isinstance(x, str) and len(x) == 1 for x in sequence):
           ords = [ord(x) for x in sequence]
           diffs = np.diff(ords)
         if np.all(diffs == diffs[0]):
             return f"alphabetical_step: {diffs[0]}"
        
      return f"symbolic_sequence: len={len(sequence)}"
    
  def predict_next(self, sequence: List) -> Tuple[Optional[str], float]:
        """
        预测序列的下一个元素
        
        Args:
            sequence: 输入序列
            
       Returns:
            (预测值，置信度)
        """
        pattern = self.identify_pattern(sequence)
        
      if not pattern:
          return None, 0.0
        
       # 基于模式进行预测
      if 'arithmetic' in pattern.rule:
           # 等差数列
           diff = float(pattern.rule.split('*')[1])
           next_val = sequence[-1] + diff
         return str(next_val if diff == int(diff) else next_val), pattern.confidence
        
      if 'geometric' in pattern.rule:
           # 等比数列
           ratio = float(pattern.rule.split('*')[1])
           next_val = sequence[-1] * ratio
         return str(next_val), pattern.confidence
        
      if 'periodic' in pattern.rule:
           # 周期序列
           period = int(pattern.rule.split('=')[1])
           next_val = sequence[-period]
         return next_val, pattern.confidence
        
      return None, 0.0
    
  def train_on_examples(self, examples: List[Tuple[List, str]]):
        """
        从示例中学习模式
        
        Args:
           examples: [(输入序列，正确答案), ...]
        """
      for sequence, answer in examples:
            pattern = self.identify_pattern(sequence)
          if pattern:
                # 验证预测是否正确
               predicted, _ = self.predict_next(sequence)
              if predicted == answer:
                    pattern.confidence = min(1.0, pattern.confidence + 0.1)
                    pattern.support_count += 1
                else:
                    pattern.confidence = max(0.0, pattern.confidence - 0.1)


def demo():
    """演示归纳推理能力"""
   print("=" * 60)
   print("归纳推理引擎演示".center(60))
   print("=" * 60)
    
    engine = InductiveReasoningEngine()
    
    # 测试用例
    test_cases = [
        ([1, 3, 5, 7], "9", "等差数列"),
        ([2, 4, 8, 16], "32", "等比数列"),
        ([1, 2, 3, 1, 2, 3], "1", "周期序列"),
        (['A', 'C', 'E', 'G'], "I", "字母序列"),
    ]
    
    correct_count = 0
    
  for i, (seq, expected, desc) in enumerate(test_cases, 1):
      print(f"\n测试{i}: {desc}")
      print(f"  输入：{seq}")
        
        pattern = engine.identify_pattern(seq)
      print(f"  识别模式：{pattern.rule if pattern else 'None'}")
        
       prediction, confidence = engine.predict_next(seq)
      print(f"  预测：{prediction} (置信度：{confidence:.2f})")
      print(f"  期望：{expected}")
        
      if prediction == expected:
          print(f"  ✓ 正确")
            correct_count += 1
       else:
          print(f"  ✗ 错误")
    
  print("\n" + "=" * 60)
  print(f"测试结果：{correct_count}/{len(test_cases)} 正确")
  print(f"准确率：{correct_count/len(test_cases)*100:.1f}%")
  print("=" * 60)
    
  return correct_count / len(test_cases)


if __name__ == "__main__":
    accuracy = demo()
    
  if accuracy >= 0.75:
      print("\n✅ 归纳推理能力达到良好水平 (≥75%)")
   else:
      print("\n⚠️  需要继续训练提升")
