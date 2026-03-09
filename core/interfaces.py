"""
类人脑双系统全闭环 AI架构 - 核心接口定义

提供统一的 BrainAIInterface 接口，封装所有底层模块
"""

import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class BrainAIOutput:
    """输出数据结构"""
    text: str
    tokens: List[int]
    confidence: float
    memory_anchors: List[dict]
    stdp_stats: dict
    cycle_stats: dict


class BrainAIInterface:
    """
    类人脑AI架构统一接口
    
    封装以下核心模块:
    - 双权重 Qwen3.5-0.8B 模型
    - 100Hz 高刷新推理引擎
    - STDP 权重更新系统
    - 海马体记忆系统
    - 自闭环优化系统
    
    使用示例:
        config = BrainAIConfig()
        ai = BrainAIInterface(config)
        
        # 推理
        output = ai.generate("你好")
        
        # 获取统计
        stats = ai.get_stats()
    """
    
    def __init__(self, config, device: Optional[str] = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ========== 延迟导入 (避免循环依赖) ==========
        from core.dual_weight_layers import DualWeightLinear, DualWeightFFN
        from core.stdp_engine import STDPEngine
        from core.refresh_engine import RefreshCycleEngine
        from hippocampus.hippocampus_system import HippocampusSystem
        from self_loop.self_loop_optimizer import SelfLoopOptimizer
        
        # ========== 1. 初始化模型组件 ==========
        # TODO: 加载 Qwen3.5-0.8B 底座并转换为双权重架构
        self.model = self._build_dual_weight_model()
        
        # ========== 2. 初始化海马体系统 ==========
        self.hippocampus = HippocampusSystem(config, device=self.device)
        
        # ========== 3. 初始化 STDP 引擎 ==========
        self.stdp_engine = STDPEngine(config, device=self.device)
        
        # ========== 4. 初始化 100Hz 推理引擎 ==========
        self.refresh_engine = RefreshCycleEngine(
            model=self.model,
            hippocampus=self.hippocampus,
            stdp_engine=self.stdp_engine,
            period_ms=config.hard_constraints.REFRESH_PERIOD_MS,
            narrow_window_size=config.hard_constraints.NARROW_WINDOW_SIZE,
            device=self.device
        )
        
        # ========== 5. 初始化自闭环优化器 ==========
        self.self_loop = SelfLoopOptimizer(config, model=self.model)
        
        # ========== 6. 启动 SWR 监控 ==========
        self.hippocampus.start_swr_monitoring()
        
        # ========== 7. 设置回调函数 ==========
        self._setup_callbacks()
    
    def _build_dual_weight_model(self) -> torch.nn.Module:
        """构建双权重模型架构"""
        # TODO: 从 Qwen3.5-0.8B 官方权重加载并转换
        # 这里提供简化实现
        
        model = torch.nn.Module()
        
        # 添加双权重层作为示例
        from core.dual_weight_layers import DualWeightLinear
        
        # Embedding 层 (简化)
        model.embeddings = torch.nn.Embedding(
            num_embeddings=151936,  # Qwen vocab size
            embedding_dim=1024       # Qwen hidden size
        )
        
        # 示例双权重层
        model.example_layer = DualWeightLinear(
            in_features=1024,
            out_features=1024,
            static_weight=torch.randn(1024, 1024) * 0.9
        )
        
        return model.to(self.device)
    
    def _setup_callbacks(self):
        """设置各模块间的回调函数"""
        # SWR 巩固回调
        def stdp_update_callback(memory, reward):
            # 在 SWR 回放时更新 STDP 权重
            pass
        
        def memory_prune_callback(threshold):
            return self.hippocampus.prune_weak_memories(threshold)
        
        self.hippocampus.swr_consolidation.set_callbacks(
            stdp_update_fn=stdp_update_callback,
            memory_prune_fn=memory_prune_callback
        )
    
    def generate(
        self,
        input_text: str,
        max_tokens: int = 100,
        use_self_loop: bool = True
    ) -> BrainAIOutput:
        """
        生成回复
        
        Args:
            input_text: 输入文本
            max_tokens: 最大生成 token 数
            use_self_loop: 是否启用自闭环优化
        
        Returns:
            output: 生成结果
        """
        # ========== 1. 记录活动 (重置空闲计时器) ==========
        self.hippocampus.record_activity()
        
        # ========== 2. 使用自闭环优化器 (可选) ==========
        if use_self_loop:
            result = self.self_loop.run(input_text)
            optimized_input = result.output_text
        else:
            optimized_input = input_text
        
        # ========== 3. Tokenize 输入 ==========
        input_tokens = self._tokenize(optimized_input)
        
        # ========== 4. 执行 100Hz 刷新推理 ==========
        generated_tokens = []
        
        for i, token_id in enumerate(input_tokens[:max_tokens]):
            # 运行一个刷新周期
            cycle_result = self.refresh_engine.run_cycle(
                input_token=token_id,
                input_text=optimized_input
            )
            
            if cycle_result.success:
                generated_tokens.append(cycle_result.output_token)
            else:
                break
        
        # ========== 5. Detokenize 输出 ==========
        output_text = self._detokenize(generated_tokens)
        
        # ========== 6. 计算置信度 ==========
        confidence = self._compute_confidence(generated_tokens)
        
        # ========== 7. 获取记忆锚点 ==========
        memory_anchors = self._get_recent_memory_anchors()
        
        return BrainAIOutput(
            text=output_text,
            tokens=generated_tokens,
            confidence=confidence,
            memory_anchors=memory_anchors,
            stdp_stats=self.stdp_engine.get_stats(),
            cycle_stats=self.refresh_engine.get_stats()
        )
    
    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        对话接口
        
        Args:
            message: 用户消息
            history: 对话历史
        
        Returns:
            response: 回复文本
        """
        # 构建带上下文的输入
        if history:
            context = "\n".join([
                f"{h['role']}: {h['content']}" 
                for h in history[-5:]  # 最近 5 轮
            ])
            full_input = f"{context}\nUser: {message}\nAssistant:"
        else:
            full_input = f"User: {message}\nAssistant:"
        
        # 生成回复
        output = self.generate(full_input, max_tokens=200)
        
        return output.text
    
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize 输入"""
        # TODO: 使用 Qwen tokenizer
        # 简化：返回字符 ASCII 码
        return [ord(c) % 1000 for c in text[:100]]
    
    def _detokenize(self, tokens: List[int]) -> str:
        """Detokenize 输出"""
        # TODO: 使用 Qwen tokenizer
        # 简化：转回字符
        return "".join([chr(t + 65) for t in tokens[:50]])
    
    def _compute_confidence(self, tokens: List[int]) -> float:
        """计算置信度"""
        # 简化：基于生成长度
        return min(1.0, len(tokens) / 50.0 + 0.5)
    
    def _get_recent_memory_anchors(self) -> List[dict]:
        """获取最近的记忆锚点"""
        # 从海马体召回最近记忆
        dummy_features = torch.randn(1024, device=self.device)
        return self.hippocampus.recall(dummy_features, topk=2)
    
    def get_stats(self) -> dict:
        """获取完整统计信息"""
        return {
            'hippocampus': self.hippocampus.get_stats(),
            'stdp': self.stdp_engine.get_stats(),
            'refresh_engine': self.refresh_engine.get_stats(),
            'self_loop': self.self_loop.get_stats()
        }
    
    def reset(self):
        """重置所有状态"""
        self.stdp_engine.reset()
        self.refresh_engine.reset()
        self.hippocampus.reset()
        self.self_loop.cycle_count = 0
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'hippocampus_memories': [
                mem.to_dict() 
                for mem in self.hippocampus.ca3_memory.memories.values()
            ],
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # TODO: 恢复海马体记忆
        # for mem_data in checkpoint['hippocampus_memories']:
        #     ...
        
        print(f"Loaded checkpoint from {path}")
    
    def __del__(self):
        """析构函数"""
        try:
            self.hippocampus.stop_swr_monitoring()
        except:
            pass


# ==================== 快捷创建函数 ====================

def create_brain_ai(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    quantization: str = "INT4"
) -> BrainAIInterface:
    """
    快捷创建类人脑AI 实例
    
    Args:
        model_path: Qwen3.5-0.8B 模型路径
        device: 设备 ("cuda" | "cpu")
        quantization: 量化类型
    
    Returns:
        ai: BrainAIInterface 实例
    """
    from configs.arch_config import default_config
    
    if model_path:
        default_config.model_path = model_path
    
    return BrainAIInterface(default_config, device=device)
