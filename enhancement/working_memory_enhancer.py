#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作记忆容量增强模块

功能:
1. 扩展工作记忆容量 (从 5±2 提升到 9±2)
2. 组块化策略 (chunking)
3. 注意控制机制
4. 双重任务协调
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class MemoryChunk:
    """记忆组块"""
  id: str
    content: List[str]  # 组块内的信息单元
    chunk_type: str  # 'visual', 'phonological', 'semantic'
   priority: float  # 优先级 (0-1)
   decay_rate: float  # 衰减速率
    last_refresh: float = field(default_factory=time.time)
    associations: List[str] = field(default_factory=list)  # 与其他组块的关联


class EnhancedWorkingMemory:
    """增强型工作记忆系统"""
    
  def __init__(self, base_capacity: int = 7, enhancement_factor: float = 1.5):
        """
        初始化增强型工作记忆
        
        Args:
           base_capacity: 基础容量 (Miller's magic number: 7±2)
            enhancement_factor: 增强因子 (1.0=正常，1.5=增强 50%)
        """
       self.base_capacity = base_capacity
        self.enhancement_factor = enhancement_factor
       self.effect ive_capacity = int(base_capacity * enhancement_factor)
        
        # 记忆存储
       self.chunks: Dict[str, MemoryChunk] = {}
       self.active_set: List[str] = []  # 当前活跃的记忆组块
       
        # 注意控制
        self.attention_focus: Optional[str] = None  # 注意焦点
        self.executive_control = True  # 执行控制启用
        
        # 双存储系统
       self.phonological_loop: List[str] = []  # 语音回路
       self.visuospatial_sketchpad: List[str] = []  # 视觉空间模板
       self.episodic_buffer: Dict[str, any] = {}  # 情景缓冲区
        
     print(f"✓ 工作记忆系统初始化完成")
    print(f"  基础容量：{base_capacity}")
    print(f"  增强因子：{enhancement_factor}x")
    print(f"  有效容量：{self.effective_capacity}")
    
  def store(self, content: str, chunk_type: str = 'semantic', 
         priority: float = 0.8) -> str:
        """
        存储信息到工作记忆
        
        Args:
           content: 要存储的信息内容
           chunk_type: 记忆类型 ('visual', 'phonological', 'semantic')
          priority: 优先级 (高优先级的不易被替换)
            
       Returns:
            chunk_id: 记忆组块的 ID
        """
      import uuid
       chunk_id = str(uuid.uuid4())[:8]
        
       # 创建记忆组块
       chunk = MemoryChunk(
         id=chunk_id,
            content=[content],
            chunk_type=chunk_type,
           priority=priority,
           decay_rate=0.1 if priority > 0.7 else 0.2
       )
        
       # 存入存储系统
       self.chunks[chunk_id] = chunk
       self.active_set.append(chunk_id)
        
       # 根据类型存入子系统
     if chunk_type == 'phonological':
         self.phonological_loop.append(content)
      elif chunk_type == 'visual':
         self.visuospatial_sketchpad.append(content)
      else:
         self.episodic_buffer[chunk_id] = content
        
       # 容量管理：如果超出容量，淘汰最低优先级的
     if len(self.active_set) > self.effective_capacity:
         self._evict_low_priority()
        
    return chunk_id
    
  def retrieve(self, chunk_id: str) -> Optional[str]:
        """检索记忆组块"""
    if chunk_id not in self.chunks:
      return None
        
       chunk = self.chunks[chunk_id]
        
       # 刷新最后访问时间
       chunk.last_refresh = time.time()
        
       # 将注意焦点移到该组块
       self.attention_focus = chunk_id
        
       # 返回内容
     if chunk.chunk_type == 'semantic':
      return self.episodic_buffer.get(chunk_id, chunk.content[0])
     else:
      return chunk.content[0]
    
  def bind_features(self, features: Dict[str, str]) -> str:
        """
        绑定多个特征为一个组块 (组块化策略)
        
        Args:
           features: 特征字典 {特征名：特征值}
            
       Returns:
            chunk_id: 绑定后的组块 ID
        """
       # 将多个特征绑定为单一组块
       bound_content = " | ".join([f"{k}:{v}" for k, v in features.items()])
       chunk_id = self.store(bound_content, chunk_type='semantic', priority=0.9)
        
       # 记录内部结构
      self.chunks[chunk_id].content = list(features.values())
      self.chunks[chunk_id].associations = list(features.keys())
        
    return chunk_id
    
  def manipulate(self, chunk_id: str, operation: str) -> Optional[str]:
        """
        心理操作 (工作记忆的核心功能)
        
        Args:
           chunk_id: 要操作的组块 ID
           operation: 操作类型 ('reverse', 'sort', 'combine', 'transform')
            
       Returns:
            操作结果
        """
    if chunk_id not in self.chunks:
      return None
        
       chunk = self.chunks[chunk_id]
        
    if operation == 'reverse':
         # 倒序操作
        chunk.content.reverse()
     elif operation == 'sort':
         # 排序操作
        chunk.content.sort()
     elif operation == 'combine':
         # 组合操作
        combined = ' '.join(chunk.content)
        chunk.content = [combined]
     elif operation == 'transform':
         # 转换操作 (例如大小写转换)
        chunk.content = [c.upper() for c in chunk.content]
        
       chunk.last_refresh = time.time()
    return ' '.join(chunk.content)
    
  def _evict_low_priority(self):
        """淘汰最低优先组的记忆组块"""
    if not self.active_set:
      return
        
       # 按优先级和最后刷新时间排序
      def chunk_score(chunk_id):
         chunk = self.chunks[chunk_id]
        recency = time.time() - chunk.last_refresh
       return chunk.priority - 0.01 * recency  # 新近性也有影响
        
       # 找到最低优先级的组块
       lowest_id = min(self.active_set, key=chunk_score)
        
       # 从活跃集移除
       self.active_set.remove(lowest_id)
        
       # 从各子系统移除
       chunk = self.chunks[lowest_id]
     if chunk.content[0] in self.phonological_loop:
         self.phonological_loop.remove(chunk.content[0])
     if chunk.content[0] in self.visuospatial_sketchpad:
         self.visuospatial_sketchpad.remove(chunk.content[0])
     if lowest_id in self.episodic_buffer:
       del self.episodic_buffer[lowest_id]
        
    print(f"  ↘ 淘汰低优先级组块：{lowest_id[:4]}...")
    
  def dual_task_coordination(self, task1_data: List, task2_data: List) -> Dict:
        """
        双重任务协调 (测试执行功能)
        
        Args:
           task1_data: 任务 1 的数据 (例如：数字)
           task2_data: 任务 2 的数据 (例如：字母)
            
       Returns:
            协调处理结果
        """
      results = {
           'task1_processed': [],
           'task2_processed': [],
           'switching_cost': 0,
           'interference': 0
       }
        
       # 模拟任务切换
       switch_count = 0
      start_time = time.time()
        
      for i in range(max(len(task1_data), len(task2_data))):
        if i < len(task1_data):
               # 处理任务 1 (使用语音回路)
              item1 = str(task1_data[i])
             self.phonological_loop.append(item1)
            results['task1_processed'].append(item1)
            
        if i < len(task2_data):
               # 处理任务 2 (使用视觉空间模板)
              item2 = str(task2_data[i])
             self.visuospatial_sketchpad.append(item2)
            results['task2_processed'].append(item2)
            
            # 任务切换
           switch_count += 1
        
       end_time = time.time()
        
       # 计算切换成本
      results['switching_cost'] = (end_time - start_time) / switch_count
       
       # 计算干扰程度 (基于错误率或延迟)
      results['interference'] = 0.1 * switch_count  # 简化估计
        
    return results
    
  def get_capacity_metrics(self) -> Dict:
        """获取容量指标"""
    return {
           'total_chunks': len(self.chunks),
           'active_chunks': len(self.active_set),
           'effective_capacity': self.effective_capacity,
           'capacity_usage': len(self.active_set) / self.effective_capacity,
           'phonological_load': len(self.phonological_loop),
           'visuospatial_load': len(self.visuospatial_sketchpad),
           'episodic_load': len(self.episodic_buffer),
           'attention_focus': self.attention_focus
       }
    
  def run_span_test(self, test_type: str = 'digit') -> int:
        """
        运行广度测试 (测量工作记忆容量)
        
        Args:
           test_type: 测试类型 ('digit', 'letter', 'word', 'complex')
            
       Returns:
            span: 工作记忆广度 (能正确回忆的最大项目数)
        """
      import random
        
      print(f"\n运行工作记忆广度测试 ({test_type})...")
        
       # 生成测试序列
     if test_type == 'digit':
         items = [str(random.randint(0, 9)) for _ in range(2, 15)]
      elif test_type == 'letter':
         items = [chr(random.randint(65, 90)) for _ in range(2, 15)]
      elif test_type == 'word':
         words = ['苹果', '香蕉', '桌子', '椅子', '红色', '蓝色', 
                 '汽车', '书本', '电脑', '手机', '树木', '河流']
         items = random.sample(words, 12)
      else:  # complex
         items = [f"{random.randint(1,9)}-{chr(random.randint(65,70))}" 
                for _ in range(2, 12)]
        
      max_span = 0
        
       # 逐步增加长度
     for length in range(2, len(items) + 1):
         sequence = items[:length]
            
         # 存储序列
        chunk_ids = []
       for item in sequence:
             chunk_id = self.store(item, chunk_type='phonological', priority=0.5)
             chunk_ids.append(chunk_id)
            
         # 立即回忆
        recalled = []
       for chunk_id in chunk_ids:
             content = self.retrieve(chunk_id)
           if content:
               recalled.append(content)
            
         # 评分
        correct = sum(1 for i, r in enumerate(recalled) 
                    if i < len(sequence) and r == sequence[i])
         accuracy = correct / length
            
       print(f"  长度{length}: 回忆{correct}/{length} (准确率{accuracy:.1%})")
            
       if accuracy >= 0.5:  # 50% 正确率阈值
            max_span = length
        else:
           break
        
     print(f"\n工作记忆广度：{max_span}")
    return max_span


def demo():
    """演示工作记忆增强效果"""
  print("=" * 80)
  print("工作记忆容量增强模块演示".center(80))
  print("=" * 80)
    
  # 创建增强型工作记忆
  print("\n【1】初始化增强型工作记忆")
  wm = EnhancedWorkingMemory(base_capacity=7, enhancement_factor=1.5)
    
  print("\n【2】测试基础存储功能")
  # 存储多个项目
  for i in range(10):
     chunk_id = wm.store(f"信息{i+1}", priority=0.5 + i*0.05)
   print(f"  存储：信息{i+1} → ID:{chunk_id}")
    
  print("\n【3】容量指标")
  metrics = wm.get_capacity_metrics()
  for key, value in metrics.items():
   print(f"  {key}: {value}")
    
  print("\n【4】组块化策略演示")
  # 将多个特征绑定为一个组块
  features = {
      '姓名': '张三',
      '年龄': '28',
      '职业': '工程师',
      '城市': '北京'
  }
  chunk_id = wm.bind_features(features)
  print(f"  绑定特征→组块 ID:{chunk_id}")
  print(f"  组块内容：{wm.retrieve(chunk_id)}")
    
  print("\n【5】心理操作演示")
  # 存储并操作
  numbers = ['3', '7', '1', '9', '5']
  chunk_ids = [wm.store(n, 'phonological') for n in numbers]
  
  # 倒序操作
  result = wm.manipulate(chunk_ids[0], 'reverse')
  print(f"  倒序操作：{numbers} → {result}")
    
  print("\n【6】双重任务协调")
  task1 = [1, 2, 3, 4, 5]  # 数字任务
  task2 = ['A', 'B', 'C', 'D', 'E']  # 字母任务
  coord_result = wm.dual_task_coordination(task1, task2)
  print(f"  任务 1 处理：{coord_result['task1_processed']}")
  print(f"  任务 2 处理：{coord_result['task2_processed']}")
  print(f"  切换成本：{coord_result['switching_cost']:.4f}s")
  print(f"  干扰程度：{coord_result['interference']:.2f}")
    
  print("\n【7】工作记忆广度测试")
  digit_span = wm.run_span_test('digit')
  letter_span = wm.run_span_test('letter')
    
  print("\n" + "=" * 80)
  print("增强效果总结".center(80))
  print("=" * 80)
  print(f"基础容量：7 ± 2")
  print(f"增强后容量：{wm.effective_capacity} ± 2")
  print(f"容量提升：{(wm.effective_capacity - 7) / 7 * 100:.1f}%")
  print(f"数字广度：{digit_span} (正常人平均 7)")
  print(f"字母广度：{letter_span} (正常人平均 7)")
  print("=" * 80 + "\n")
    
  return {
      'effective_capacity': wm.effective_capacity,
      'digit_span': digit_span,
      'letter_span': letter_span
  }


if __name__ == "__main__":
   results = demo()
