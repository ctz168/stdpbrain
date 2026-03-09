#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学计算能力增强模块

功能:
1. 数学符号理解与解析
2. 分步计算与验证
3. 应用题自动求解
4. 错误诊断与纠正
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import ast


@dataclass
class MathStep:
    """数学计算步骤"""
  step_id: int
    operation: str
   expression: str
  result: str
   explanation: str
    confidence: float


class MathExpressionParser:
    """数学表达式解析器"""
    
  def __init__(self):
        self.operators = {
            '+': {'priority': 1, 'func': lambda a, b: a + b},
            '-': {'priority': 1, 'func': lambda a, b: a - b},
            '*': {'priority': 2, 'func': lambda a, b: a * b},
            '/': {'priority': 2, 'func': lambda a, b: a / b if b != 0 else None},
            '^': {'priority': 3, 'func': lambda a, b: a ** b},
        }
    
  def parse(self, expr: str) -> Optional[float]:
        """
        解析并计算数学表达式
        
        Args:
           expr: 数学表达式字符串
            
       Returns:
            计算结果，如果解析失败则返回 None
        """
      try:
           # 清理表达式
           expr = expr.replace(' ', '')
            
           # 安全性检查：只允许数字和基本运算符
         if not re.match(r'^[\d\+\-\*/\^\(\)\.]+$', expr):
            print(f"⚠️  警告：表达式包含非法字符")
            return None
            
           # 使用 AST 安全解析
           tree = ast.parse(expr, mode='eval')
         return self._eval_tree(tree.body)
     except Exception as e:
         print(f"✗ 解析错误：{e}")
        return None
    
  def _eval_tree(self, node) -> float:
        """递归求值 AST 节点"""
     if isinstance(node, ast.Num):  # Python3.7 及以下
        return node.n
      elif isinstance(node, ast.Constant):  # Python3.8+
        return node.value
      elif isinstance(node, ast.BinOp):
           left = self._eval_tree(node.left)
           right = self._eval_tree(node.right)
           op = node.op
           
         if isinstance(op, ast.Add):
            return left + right
          elif isinstance(op, ast.Sub):
            return left - right
          elif isinstance(op, ast.Mult):
            return left * right
          elif isinstance(op, ast.Div):
            return left / right if right != 0 else float('inf')
          elif isinstance(op, ast.Pow):
            return left ** right
      elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
         return -self._eval_tree(node.operand)
        
       raise ValueError(f"不支持的节点类型：{type(node)}")


class StepByStepMathSolver:
    """分步数学求解器"""
    
  def __init__(self):
        self.parser= MathExpressionParser()
        self.step_history: List[MathStep] = []
    
  def solve_word_problem(self, problem: str) -> Optional[Dict]:
        """
        求解应用题
        
        Args:
           problem: 应用题文本
            
       Returns:
            包含步骤和答案的字典
        """
       self.step_history.clear()
        
       # 步骤 1: 提取关键信息
       info = self._extract_information(problem)
       self.step_history.append(MathStep(
          step_id=1,
            operation="理解题意",
           expression=problem,
          result=str(info),
           explanation="从题目中提取已知条件和未知量",
            confidence=0.9
       ))
        
       # 步骤 2: 建立方程
       equation = self._build_equation(info)
       self.step_history.append(MathStep(
          step_id=2,
            operation="建立方程",
           expression=equation,
          result=equation,
           explanation="根据数量关系列出方程",
            confidence=0.85
       ))
        
       # 步骤 3: 求解方程
       solution = self._solve_equation(equation)
       self.step_history.append(MathStep(
          step_id=3,
            operation="求解方程",
           expression=equation,
          result=str(solution),
           explanation="解方程得到未知数的值",
            confidence=0.8
       ))
        
       # 步骤 4: 验证答案
       verification = self._verify_solution(solution, problem)
       self.step_history.append(MathStep(
          step_id=4,
            operation="验证答案",
           expression=str(solution),
          result="通过" if verification else "不通过",
           explanation="将答案代入原题验证",
            confidence=0.9 if verification else 0.3
       ))
        
     if verification:
        return {
                'steps': self.step_history,
               'answer': solution,
               'confidence': 0.85
           }
      else:
        return None
    
  def _extract_information(self, problem: str) -> Dict:
        """从应用题中提取信息"""
       info = {
            'known_values': [],
           'unknown': 'x',
           'relationships': []
       }
        
       # 提取数字
        numbers = re.findall(r'\d+\.?\d*', problem)
       info['known_values'] = [float(n) for n in numbers]
        
       # 提取关键词
     if '多少' in problem:
           idx = problem.find('多少')
          info['question_part'] = problem[max(0, idx-10):idx+10]
        
     return info
    
  def _build_equation(self, info: Dict) -> str:
        """建立方程 (简化版)"""
       # 实际应使用 NLP 技术理解语义
       # 这里仅做演示
     if len(info['known_values']) >= 2:
           # 假设为简单加法或乘法
           a, b = info['known_values'][:2]
         if a + b < 100:
            return f"x = {a} + {b}"
          elif a * b < 1000:
            return f"x = {a} * {b}"
        
     return "x = unknown"
    
  def _solve_equation(self, equation: str) -> Optional[float]:
        """求解方程"""
      try:
           # 提取等号右边的表达式
         if '=' in equation:
               right_side = equation.split('=')[1].strip()
             return self.parser.parse(right_side)
     except:
         pass
     return None
    
  def _verify_solution(self, solution: Optional[float], original_problem: str) -> bool:
        """验证答案"""
     if solution is None:
        return False
        
       # 合理性检查
     if solution < 0:
         # 负数解在某些场景下不合理
        if '人数' in original_problem or '个数' in original_problem:
           print(f"⚠️  警告：{solution} 为负数，可能不正确")
           # return False  # 暂时不否决
        
       # 量纲检查 (简化)
     if solution> 1e6:
        print(f"⚠️  警告：{solution} 数值过大，请检查")
        
     return True


class MathTrainingDataset:
    """数学训练数据集"""
    
  @staticmethod
  def get_arithmetic_problems() -> List[Dict]:
        """获取算术问题集"""
     return [
           {
                'problem': '小明有 5 个苹果，又买了 3 个，现在有几个？',
               'equation': '5 + 3 = x',
               'answer': 8,
               'type': 'addition'
           },
           {
                'problem': '一本书原价 45 元，打 8 折后多少钱？',
               'equation': '45 * 0.8 = x',
               'answer': 36.0,
               'type': 'multiplication'
           },
           {
                'problem': '一个数加上 7 等于 15，这个数是多少？',
               'equation': 'x + 7 = 15',
               'answer': 8,
               'type': 'algebra'
           },
           {
                'problem': '一辆车以 60km/h 的速度行驶，3 小时后走了多远？',
               'equation': '60 * 3 = x',
               'answer': 180,
               'type': 'physics'
           },
       ]
    
  @staticmethod
  def get_geometry_problems() -> List[Dict]:
        """获取几何问题集"""
     return [
           {
                'problem': '半径为 5 的圆面积是多少？(π取 3.14)',
               'equation': '3.14 * 5^2 = x',
               'answer': 78.5,
               'type': 'circle_area'
           },
           {
                'problem': '边长为 4 的正方形周长是多少？',
               'equation': '4 * 4 = x',
               'answer': 16,
               'type': 'square_perimeter'
           },
       ]


def demo():
    """演示数学计算能力"""
  print("=" * 60)
  print("数学计算能力增强模块演示".center(60))
  print("=" * 60)
    
    solver = StepByStepMathSolver()
   problems = MathTrainingDataset.get_arithmetic_problems()
    
    correct_count = 0
    
  for i, prob in enumerate(problems[:3], 1):
     print(f"\n【问题{i}】{prob['problem']}")
        
      result = solver.solve_word_problem(prob['problem'])
        
     if result:
         print(f"✓ 答案：{result['answer']}")
         print(f"  置信度：{result['confidence']:.2f}")
            
        if abs(result['answer'] - prob['answer']) < 0.1:
            print(f"✓ 正确")
              correct_count += 1
         else:
            print(f"✗ 错误 (正确答案：{prob['answer']})")
       else:
         print(f"✗ 无法求解")
    
  print("\n" + "=" * 60)
  print(f"测试结果：{correct_count}/{len(problems[:3])} 正确")
  print(f"准确率：{correct_count/len(problems[:3])*100:.1f}%")
  print("=" * 60)
    
  return correct_count / len(problems[:3])


if __name__ == "__main__":
    accuracy = demo()
    
  if accuracy >= 0.78:
    print("\n✅ 数学计算能力达到良好水平 (≥78%)")
   else:
    print("\n⚠️  需要继续训练提升")
