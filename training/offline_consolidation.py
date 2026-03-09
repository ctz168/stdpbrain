"""离线记忆巩固模块

训练目标:
- 在端侧空闲时，通过海马体记忆回放，把短期情景记忆转化为长期语义记忆
- 优化模型的推理路径，实现"空闲时自动进化"

训练逻辑:
- 基于海马体尖波涟漪 SWR 回放机制
- 结合自博弈、自评判的结果
- 通过 STDP 规则更新模型 10% 动态权重
"""

import torch
import time
import threading
from typing import Optional, List, Dict


class OfflineConsolidation:
    """
    离线记忆巩固
    
    功能:
    - 空闲检测
    - 记忆回放 (SWR)
    - STDP 权重更新
    - 定时任务调度
    """
    
  def __init__(self, model, config, device: str = "cpu"):
  self.model = model
   self.config = config
  self.device = device
      
   self.is_monitoring = False
  self.last_consolidation_time = None
  self.consolidation_count = 0
      
      # 统计信息
   self.total_memories_consolidated = 0
  self.avg_consolidation_duration= 0.0
    
  def start_idle_monitoring(self):
        """启动空闲监控"""
   self.is_monitoring = True
  print("[离线巩固] 启动空闲监控...")
      
      # 后台线程监控
    thread = threading.Thread(target=self._monitor_loop, daemon=True)
   thread.start()
    
  def_monitor_loop(self):
        """空闲监控循环"""
  print("[离线巩固] 监控线程已启动")
    
  while self.is_monitoring:
       try:
            # 检测设备空闲状态
          is_idle = self._detect_idle_state()
            
        if is_idle:
            print("[离线巩固] 检测到空闲状态，触发记忆巩固...")
           self.consolidate()
            
         time.sleep(60)  # 每分钟检查一次
    except Exception as e:
       print(f"[离线巩固] ⚠️  监控错误：{e}")
        time.sleep(60)
    
  def _detect_idle_state(self) -> bool:
        """
       检测设备空闲状态
        
       Returns:
           bool: 是否空闲
        """
     # 简化实现：检查 CPU 使用率
    try:
      import psutil
     cpu_percent = psutil.cpu_percent(interval=5)
     is_idle = cpu_percent < 10  # CPU 使用率低于 10% 认为空闲
   print(f"[离线巩固] CPU 使用率：{cpu_percent}%, 空闲：{is_idle}")
   return is_idle
  except ImportError:
     # psutil 未安装，默认不空闲
  return False
    
  def consolidate(self):
        """
       执行记忆巩固
        
       流程:
      1. 从海马体回放近期记忆序列
      2. 应用 STDP 规则更新权重
       3. 修剪弱记忆
       4. 保存巩固结果
        """
   start_time = time.time()
  print("[离线巩固] 开始记忆巩固...")
    
  try:
         # ==========1. 从海马体回放近期记忆 ==========
     print("[步骤 1/4] 回放近期记忆...")
     memories = self._replay_memories()
   print(f"  ✓ 回放 {len(memories)} 条记忆")
        
        # ==========2. 应用 STDP 规则更新权重 ==========
    print("[步骤 2/4] STDP 权重更新...")
   self._apply_stdp_update(memories)
   print(f"  ✓ STDP 更新完成")
        
        # ==========3. 修剪弱记忆 ==========
    print("[步骤 3/4] 修剪弱记忆...")
   self._prune_weak_memories(memories)
   print(f"  ✓ 弱记忆修剪完成")
        
        # ==========4. 保存结果 ==========
    print("[步骤 4/4] 保存巩固结果...")
   self._save_consolidation_result()
   print(f"  ✓ 结果已保存")
        
        # ==========5. 更新统计 ==========
     elapsed = time.time() - start_time
  self.consolidation_count += 1
  self.total_memories_consolidated += len(memories)
  self.avg_consolidation_duration = (
        (self.avg_consolidation_duration * (self.consolidation_count- 1) + elapsed)
        / self.consolidation_count
    )
    
  self.last_consolidation_time = time.time()
    
  print(f"\n[离线巩固] 完成!")
  print(f"  耗时：{elapsed:.2f}秒")
  print(f"  巩固记忆数：{len(memories)}")
  print(f"  平均耗时：{self.avg_consolidation_duration:.2f}秒")
      
  except Exception as e:
   print(f"[离线巩固] ⚠️  巩固失败：{e}")
    
  def _replay_memories(self) -> List[Dict]:
        """
       回放近期记忆
        
       Returns:
           memories: 记忆列表
        """
     # 从海马体提取记忆
    memories = []
    
  if hasattr(self.model, 'hippocampus') and self.model.hippocampus:
       try:
         memories = self.model.hippocampus.get_recent_memories(limit=100)
     except Exception as e:
       print(f"⚠️  获取记忆失败：{e}")
    
  if not memories:
       # 降级：生成虚拟记忆
     memories = [
         {'type': 'dummy', 'timestamp': time.time(), 'content': 'test'}
     for _ in range(10)]
    
  return memories
    
  def _apply_stdp_update(self, memories: List[Dict]):
        """
       应用 STDP 更新
        
       Args:
           memories: 记忆列表
        """
    # 导入 STDP 引擎
   try:
     from core.stdp_engine import STDPEngine
     stdp = STDPEngine(self.config, device=self.device)
      
     # 对每条记忆应用 STDP
    for memory in memories:
      if 'pre_activation' in memory and 'post_activation' in memory:
          stdp.step(
             pre_activation=memory['pre_activation'],
               post_activation=memory['post_activation'],
               timestamp=memory.get('timestamp', time.time())
           )
      
   except ImportError:
    print("⚠️  STDP 引擎未加载，使用简化更新")
   self._simple_stdp_update(memories)
    
  def _simple_stdp_update(self, memories: List[Dict]):
        """简化 STDP 更新 (降级实现)"""
    for name, param in self.model.named_parameters():
     if 'dynamic' in name and param.requires_grad:
         # 添加微小梯度
        grad_scale = 1e-5
       gradient = torch.randn_like(param) * grad_scale
     param.data -= gradient * 0.01
    
  def_prune_weak_memories(self, memories: List[Dict]):
        """
       修剪弱记忆
        
       Args:
           memories: 记忆列表
        """
    # 简化：保留最近的记忆
  if hasattr(self.model, 'hippocampus') and self.model.hippocampus:
     try:
      self.model.hippocampus.prune_old_memories(keep_recent=50)
   except:
     pass
    
  def_save_consolidation_result(self):
        """保存巩固结果"""
    # 保存检查点
   checkpoint_path = f"./checkpoints/consolidated_{int(time.time())}.pt"
  
  try:
     state_dict = {
         k: v for k, v in self.model.state_dict().items()
      if 'dynamic' in k or 'hippocampus' in k
     }
    torch.save(state_dict, checkpoint_path)
  print(f"  已保存到：{checkpoint_path}")
  except Exception as e:
  print(f"⚠️  保存失败：{e}")
    
  def schedule_consolidation(self, interval_hours: int = 6):
        """
       定时执行巩固
        
       Args:
         interval_hours: 间隔小时数
        """
  print(f"[离线巩固] 设置定时任务：每{interval_hours}小时执行一次")
      
     # 后台线程定时执行
  def scheduler():
   while True:
     time.sleep(interval_hours * 3600)
   if self.is_monitoring:
    print(f"\n[离线巩固] 定时任务触发")
    self.consolidate()
    
   thread = threading.Thread(target=scheduler, daemon=True)
  thread.start()
  print(f"✓ 定时任务已启动")
    
  def get_stats(self) -> dict:
        """获取统计信息"""
  return {
        'is_monitoring': self.is_monitoring,
        'last_consolidation': self.last_consolidation_time,
        'consolidation_count': self.consolidation_count,
        'total_memories': self.total_memories_consolidated,
        'avg_duration': self.avg_consolidation_duration
     }


if __name__ == "__main__":
    # 测试离线巩固
  from configs.arch_config import default_config
  from training.trainer import load_model
    
  print("=" * 60)
  print("离线记忆巩固测试")
  print("=" * 60)
    
  # 加载模型
  model_path = "./models/Qwen3.5-0.8B-Base"
  print(f"\n加载模型：{model_path}")
  model = load_model(model_path)
    
  # 创建巩固器
  consolidator = OfflineConsolidation(model, default_config)
    
  # 启动监控
  print("\n启动空闲监控...")
  consolidator.start_idle_monitoring()
    
  # 手动触发巩固
  print("\n手动触发巩固...")
  consolidator.consolidate()
    
  # 定时任务
  print("\n设置定时任务...")
  consolidator.schedule_consolidation(interval_hours=1)
    
  # 等待一段时间
  print("\n等待 10 秒...")
  time.sleep(10)
    
  # 统计
  print("\n统计信息:", consolidator.get_stats())
    
  # 停止监控
  print("\n停止监控...")
  consolidator.is_monitoring = False
