#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作记忆容量增强模块 - 正确缩进版本

功能:
1. 扩展工作记忆容量 (从 7±2 提升到 11±2)
2. 组块化策略 (chunking)
3. 注意控制机制
4. 双重任务协调
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time
import uuid


@dataclass
class MemoryChunk:
    """记忆组块"""
    id: str
    content: List[str]
    chunk_type: str
    priority: float
    decay_rate: float
    last_refresh: float = field(default_factory=time.time)
    associations: List[str] = field(default_factory=list)


class EnhancedWorkingMemory:
    """增强型工作记忆系统"""
    
    def __init__(self, base_capacity: int = 7, enhancement_factor: float = 1.5):
        self.base_capacity = base_capacity
        self.enhancement_factor = enhancement_factor
        self.effective_capacity = int(base_capacity * enhancement_factor)
        self.chunks: Dict[str, MemoryChunk] = {}
        self.active_set: List[str] = []
        self.attention_focus: Optional[str] = None
        self.executive_control = True
        self.phonological_loop: List[str] = []
        self.visuospatial_sketchpad: List[str] = []
        self.episodic_buffer: Dict[str, any] = {}
        
        print(f"✓ 工作记忆系统初始化完成")
        print(f"  基础容量：{base_capacity}")
        print(f"  增强因子：{enhancement_factor}x")
        print(f"  有效容量：{self.effective_capacity}")
    
    def store(self, content: str, chunk_type: str = 'semantic', 
                priority: float = 0.8) -> str:
        chunk_id = str(uuid.uuid4())[:8]
        chunk = MemoryChunk(
            id=chunk_id,
            content=[content],
            chunk_type=chunk_type,
            priority=priority,
            decay_rate=0.1 if priority > 0.7 else 0.2
        )
        self.chunks[chunk_id] = chunk
        self.active_set.append(chunk_id)
        
        if chunk_type == 'phonological':
            self.phonological_loop.append(content)
        elif chunk_type == 'visual':
            self.visuospatial_sketchpad.append(content)
        else:
            self.episodic_buffer[chunk_id] = content
        
        if len(self.active_set) > self.effective_capacity:
            self._evict_low_priority()
        
        return chunk_id
    
    def retrieve(self, chunk_id: str) -> Optional[str]:
        if chunk_id not in self.chunks:
            return None
        
        chunk = self.chunks[chunk_id]
        chunk.last_refresh = time.time()
        self.attention_focus = chunk_id
        
        if chunk.chunk_type == 'semantic':
            return self.episodic_buffer.get(chunk_id, chunk.content[0])
        else:
            return chunk.content[0]
    
    def bind_features(self, features: Dict[str, str]) -> str:
        bound_content = " | ".join([f"{k}:{v}" for k, v in features.items()])
        chunk_id = self.store(bound_content, chunk_type='semantic', priority=0.9)
        self.chunks[chunk_id].content = list(features.values())
        self.chunks[chunk_id].associations = list(features.keys())
        return chunk_id
    
    def manipulate(self, chunk_id: str, operation: str) -> Optional[str]:
        if chunk_id not in self.chunks:
            return None
        
        chunk = self.chunks[chunk_id]
        
        if operation == 'reverse':
            chunk.content.reverse()
        elif operation == 'sort':
            chunk.content.sort()
        elif operation == 'combine':
            combined = ' '.join(chunk.content)
            chunk.content = [combined]
        elif operation == 'transform':
            chunk.content = [c.upper() for c in chunk.content]
        
        chunk.last_refresh = time.time()
        return ' '.join(chunk.content)
    
    def _evict_low_priority(self):
        if not self.active_set:
            return
        
        def chunk_score(chunk_id):
            chunk = self.chunks[chunk_id]
            recency = time.time() - chunk.last_refresh
            return chunk.priority - 0.01 * recency
        
        lowest_id = min(self.active_set, key=chunk_score)
        self.active_set.remove(lowest_id)
        
        chunk = self.chunks[lowest_id]
        if chunk.content[0] in self.phonological_loop:
            self.phonological_loop.remove(chunk.content[0])
        if chunk.content[0] in self.visuospatial_sketchpad:
            self.visuospatial_sketchpad.remove(chunk.content[0])
        if lowest_id in self.episodic_buffer:
            del self.episodic_buffer[lowest_id]
        
        print(f"  ↘ 淘汰低优先级组块：{lowest_id[:4]}...")
    
    def dual_task_coordination(self, task1_data: List, task2_data: List) -> Dict:
        results = {
            'task1_processed': [],
            'task2_processed': [],
            'switching_cost': 0,
            'interference': 0
        }
        
        switch_count = 0
        start_time = time.time()
        
        for i in range(max(len(task1_data), len(task2_data))):
            if i < len(task1_data):
                item1 = str(task1_data[i])
                self.phonological_loop.append(item1)
                results['task1_processed'].append(item1)
            
            if i < len(task2_data):
                item2 = str(task2_data[i])
                self.visuospatial_sketchpad.append(item2)
                results['task2_processed'].append(item2)
                switch_count += 1
        
        end_time = time.time()
        results['switching_cost'] = (end_time - start_time) / max(switch_count, 1)
        results['interference'] = len(self.active_set) / self.effective_capacity
        
        return results
    
    def run_span_test(self, test_type: str = 'digit') -> int:
        import random
        span = 3
        correct_count = 0
        
        while span <= self.effective_capacity + 2:
            if test_type == 'digit':
                sequence = [str(random.randint(0, 9)) for _ in range(span)]
            else:
                sequence = [chr(ord('A') + random.randint(0, 25)) for _ in range(span)]
            
            chunk_id = self.store(','.join(sequence), priority=0.95)
            retrieved = self.retrieve(chunk_id)
            
            if retrieved and retrieved.replace(',', '') == ''.join(sequence):
                correct_count += 1
                span += 1
            else:
                break
        
        return span -1
    
    def get_capacity_metrics(self) -> Dict:
        return {
            'base_capacity': self.base_capacity,
            'effective_capacity': self.effective_capacity,
            'current_load': len(self.active_set),
            'utilization': len(self.active_set) / self.effective_capacity,
            'enhancement_factor': self.enhancement_factor
        }


if __name__ == "__main__":
    wm = EnhancedWorkingMemory(base_capacity=7, enhancement_factor=1.5)
    print(f"\n有效容量：{wm.effective_capacity}")
    
    chunk_id = wm.store("测试记忆内容", priority=0.8)
    print(f"存储成功，ID: {chunk_id}")
    
    retrieved = wm.retrieve(chunk_id)
    print(f"检索结果：{retrieved}")
    
    span = wm.run_span_test('digit')
    print(f"数字广度：{span}")
    
    metrics = wm.get_capacity_metrics()
    print(f"\n容量指标:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
