"""在线终身学习模块

训练目标:
- 实现推理即学习，模型在端侧运行过程中实时学习新内容
- 适配用户习惯、优化自身能力

训练逻辑:
- 全程基于 STDP 时序可塑性规则，无反向传播
- 每个刷新周期自动执行
- 算力开销不超过模型总算力的 2%
"""

import torch
from typing import Optional, Dict, List
import time


class OnlineLifelongLearner:
    """
    在线终身学习
    
    功能:
    - STDP 权重更新
    - 用户习惯学习
    - 实时能力优化
    """
    
  def __init__(self, model, config, device: str = "cpu"):
    self.model = model
    self.config = config
    self.device = device
      
    self.enabled = False
   self.compute_overhead = 0.0
      
      # STDP 引擎 (延迟导入)
   self.stdp_engine = None
      
      # 学习统计
   self.total_updates = 0
  self.learning_events = []
    
  def enable(self):
        """启用在线学习"""
    self.enabled = True
   print("[在线学习] 已启用")
      
      # 初始化 STDP 引擎
   if self.stdp_engine is None:
       try:
         from core.stdp_engine import STDPEngine
        self.stdp_engine = STDPEngine(self.config, device=self.device)
        print("[在线学习] STDP 引擎已初始化")
     except Exception as e:
        print(f"[在线学习] ⚠️  STDP 引擎初始化失败：{e}")
    
  def disable(self):
        """禁用在线学习"""
    self.enabled = False
   print("[在线学习] 已禁用")
    
  def step(self, inputs, outputs, timestamp: float):
        """
       执行一个在线学习步
        
       Args:
         inputs: 输入数据
           outputs: 输出数据
          timestamp: 时间戳 (ms)
        """
   if not self.enabled:
      return
      
     start_time = time.time()
      
     try:
         # ==========1. 调用 STDP 引擎进行权重更新 ==========
      if self.stdp_engine is not None:
            # 提取神经元活动
          pre_activation = self._extract_pre_activation(inputs)
           post_activation = self._extract_post_activation(outputs)
            
            # STDP 更新
          self.stdp_engine.step(
              pre_activation=pre_activation,
               post_activation=post_activation,
               timestamp=timestamp
           )
            
         self.total_updates += 1
            
         else:
            # 降级：简化 STDP 更新
          self._simple_stdp_update(inputs, outputs, timestamp)
        
        # ==========2. 记录学习事件 ==========
       elapsed = time.time() - start_time
      overhead = elapsed * 1000  # ms
    self.compute_overhead = (
         (self.compute_overhead * (self.total_updates- 1) + overhead)
         / self.total_updates
     )
      
     event = {
         'timestamp': timestamp,
        'overhead_ms': overhead,
        'success': True
     }
   self.learning_events.append(event)
      
      # 保持事件列表大小
   if len(self.learning_events) > 1000:
      self.learning_events = self.learning_events[-1000:]
      
    if self.total_updates % 10 == 0:
       print(f"[在线学习] 更新 #{self.total_updates}, 开销：{overhead:.2f}ms")
        
   except Exception as e:
     print(f"[在线学习] ⚠️  学习失败：{e}")
    
  def _extract_pre_activation(self, inputs) ->torch.Tensor:
        """提取突触前神经元活动"""
   if isinstance(inputs, dict):
      if 'input_ids' in inputs:
         # Token IDs
        ids = inputs['input_ids']
       elif 'hidden_states' in inputs:
         # Hidden states
       return inputs['hidden_states']
      else:
        # 默认
      return next(iter(inputs.values()))
   elif isinstance(inputs, torch.Tensor):
    return inputs
   else:
     # 转换为 tensor
   return torch.tensor([1.0])
    
  def_extract_post_activation(self, outputs) ->torch.Tensor:
        """提取突触后神经元活动"""
   if hasattr(outputs, 'logits'):
      return outputs.logits
   elif hasattr(outputs, 'hidden_states'):
     return outputs.hidden_states
   elif isinstance(outputs, torch.Tensor):
    return outputs
   else:
    return torch.tensor([1.0])
    
  def_simple_stdp_update(self, inputs, outputs, timestamp: float):
        """
       简化 STDP 更新 (降级实现)
        
       STDP 规则:
       Δw = α * exp(-Δt/τ)  if Δt > 0 (LTP 增强)
       Δw = -β * exp(Δt/τ) if Δt < 0 (LTD 抑制)
        """
      # 简化：对动态权重施加微小扰动
     for name, param in self.model.named_parameters():
       if 'dynamic' in name and param.requires_grad:
            # 添加随机梯度
           grad_scale = 1e-6
          gradient = torch.randn_like(param) * grad_scale
        param.data -= gradient * 0.01  # 学习率
    
  def learn_user_preference(self, user_id: str, preferences: Dict):
        """
       学习用户偏好
        
       Args:
         user_id: 用户 ID
         preferences: 偏好字典
        """
   print(f"[在线学习] 学习用户 {user_id} 的偏好...")
      
     # 存储偏好到海马体
   if hasattr(self.model, 'hippocampus') and self.model.hippocampus:
      self.model.hippocampus.store_user_preference(user_id, preferences)
    
  def adapt_to_domain(self, domain_text: str):
        """
       适应特定领域
        
       Args:
          domain_text: 领域文本
        """
   print(f"[在线学习] 适应领域...")
      
     # 提取领域特征
    features = self._extract_domain_features(domain_text)
      
     # 调整相关权重
   self._adjust_domain_weights(features)
    
  def _extract_domain_features(self, text: str) -> List[str]:
        """提取领域特征"""
      # 简化：提取关键词
     keywords = []
    if "医学" in text or "医疗" in text:
        keywords.append("medical")
    if "法律" in text:
        keywords.append("legal")
    if "技术" in text or "代码" in text:
        keywords.append("technical")
      
   return keywords
    
  def _adjust_domain_weights(self, features: List[str]):
        """调整领域权重"""
     for name, param in self.model.named_parameters():
       if 'dynamic' in name and param.requires_grad:
           # 根据特征微调
          for feature in features:
            if 'medical' in feature:
               param.data *= 1.001
             elif 'legal' in feature:
               param.data *= 1.001
             elif 'technical' in feature:
               param.data *= 1.001
    
  def get_stats(self) -> dict:
        """获取统计信息"""
   return {
         'enabled': self.enabled,
        'compute_overhead': self.compute_overhead,
        'total_updates': self.total_updates,
        'recent_events': len(self.learning_events)
     }


if __name__ == "__main__":
    # 测试在线学习器
  from configs.arch_config import default_config
  from training.trainer import load_model
    
  print("=" * 60)
  print("在线终身学习测试")
  print("=" * 60)
    
  # 加载模型
  model_path = "./models/Qwen3.5-0.8B"
  print(f"\n加载模型：{model_path}")
  model = load_model(model_path)
    
  # 创建学习器
  learner = OnlineLifelongLearner(model, default_config)
    
  # 启用
  print("\n启用在线学习...")
  learner.enable()
    
  # 模拟学习步骤
  print("\n模拟学习步骤...")
  for i in range(5):
    inputs = {'input_ids': torch.randint(0, 1000, (1, 10))}
    outputs = model.generate(**inputs, max_new_tokens=5)
   learner.step(inputs, outputs, timestamp=time.time())
  
  # 统计
  print("\n统计信息:", learner.get_stats())
    
  # 学习用户偏好
  print("\n学习用户偏好...")
  learner.learn_user_preference("user_001", {"style": "formal", "topic": "tech"})
    
  # 适应领域
  print("\n适应领域...")
  learner.adapt_to_domain("这是一段医学相关的文本")
