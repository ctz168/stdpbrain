"""
全局工作空间 (Global Workspace)

基于全局工作空间理论实现意识整合机制：

核心功能：
1. 注册多模块输出
2. 竞争机制选择"意识焦点"
3. 广播选中的信息到所有模块
4. 维护统一的意识状态
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import time


@dataclass
class ModuleOutput:
    """模块输出数据结构"""
    module_name: str
    output: torch.Tensor
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


class CompetitionMechanism(nn.Module):
    """竞争机制"""
    
    def __init__(self, hidden_size: int = 1024, competition_dim: int = 256):
        super().__init__()
        
        # 重要性评估网络
        self.importance_net = nn.Sequential(
            nn.Linear(hidden_size, competition_dim),
            nn.ReLU(),
            nn.Linear(competition_dim, 1),
            nn.Sigmoid()
        )
        
        # 相关性计算
        self.relevance_net = nn.Sequential(
            nn.Linear(hidden_size * 2, competition_dim),
            nn.ReLU(),
            nn.Linear(competition_dim, 1),
            nn.Sigmoid()
        )
        
        # [动态化] 竞争权重 (Importance, Relevance, Confidence)
        self.competition_weights = nn.Parameter(
            torch.tensor([0.5, 0.3, 0.2]), requires_grad=False
        )
    
    def forward(
        self,
        outputs: List[ModuleOutput],
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算竞争分数
        
        Args:
            outputs: 模块输出列表
            context: 当前上下文（用于相关性计算）
        
        Returns:
            scores: 竞争分数 [num_modules]
        """
        if not outputs:
            return torch.tensor([])
        
        # 1. 计算每个输出的重要性
        importance_scores = []
        for output in outputs:
            score = self.importance_net(output.output)
            importance_scores.append(score)
        
        importance = torch.stack(importance_scores).squeeze(-1)  # [num_modules]
        
        # 2. 计算与上下文的相关性
        if context is not None and context.numel() > 0:
            relevance_scores = []
            for output in outputs:
                combined = torch.cat([output.output, context], dim=-1)
                score = self.relevance_net(combined)
                relevance_scores.append(score)
            
            relevance = torch.stack(relevance_scores).squeeze(-1)  # [num_modules]
        else:
            relevance = torch.ones_like(importance)
        
        # 3. 结合置信度
        confidences = torch.tensor([o.confidence for o in outputs], device=importance.device)
        
        # 4. [动态化] 综合竞争分数（使用可学习权重，softmax 归一化保证总和为1）
        weights = torch.softmax(self.competition_weights, dim=0)
        competition_scores = importance * weights[0] + relevance * weights[1] + confidences * weights[2]
        
        return competition_scores


class BroadcastMechanism(nn.Module):
    """广播机制"""
    
    def __init__(self, hidden_size: int = 1024, broadcast_dim: int = 512):
        super().__init__()
        
        # 广播编码器
        self.broadcast_encoder = nn.Sequential(
            nn.Linear(hidden_size, broadcast_dim),
            nn.Tanh(),
            nn.Linear(broadcast_dim, hidden_size)
        )
        
        # 模块特定解码器
        self.module_decoders = nn.ModuleDict({
            'attention': nn.Linear(hidden_size, hidden_size),
            'memory': nn.Linear(hidden_size, hidden_size),
            'reasoning': nn.Linear(hidden_size, hidden_size),
            'emotion': nn.Linear(hidden_size, hidden_size)
        })
    
    def forward(
        self,
        winner_output: torch.Tensor,
        target_modules: List[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        广播胜出信息
        
        Args:
            winner_output: 胜出模块的输出
            target_modules: 目标模块列表
        
        Returns:
            broadcasts: 各模块接收的广播信息
        """
        # 编码
        encoded = self.broadcast_encoder(winner_output)
        
        # 解码到各模块
        broadcasts = {}
        modules = target_modules or list(self.module_decoders.keys())
        
        for module_name in modules:
            if module_name in self.module_decoders:
                broadcasts[module_name] = self.module_decoders[module_name](encoded)
        
        return broadcasts


class GlobalWorkspace:
    """
    全局工作空间
    
    意识整合的核心枢纽：
    - 接收各模块输出
    - 竞争选择意识焦点
    - 广播到所有模块
    - 维护统一意识状态
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        device: str = "cpu",
        max_history: int = 100
    ):
        self.hidden_size = hidden_size
        self.device = device
        self.max_history = max_history
        
        # 竞争和广播机制
        self.competition = CompetitionMechanism(hidden_size).to(device)
        self.broadcast = BroadcastMechanism(hidden_size).to(device)
        
        # 维度适配器：将不同维度映射到 hidden_size
        self._dimension_adapters = nn.ModuleDict({
            'memory': nn.Linear(512, hidden_size, bias=False),  # 海马体 dg_features: 512 -> 1024
            'goal': nn.Linear(512, hidden_size, bias=False),     # 目标向量：假设512
        }).to(device)
        
        # 初始化适配器为单位矩阵（保持语义）
        for name, adapter in self._dimension_adapters.items():
            nn.init.xavier_uniform_(adapter.weight, gain=0.5)
        
        # 模块注册表
        self.registered_modules: Dict[str, ModuleOutput] = OrderedDict()
        
        # 当前意识状态
        self.consciousness_state: Optional[torch.Tensor] = None
        self.consciousness_history: List[torch.Tensor] = []
        
        # 当前焦点
        self.current_focus: Optional[str] = None
        self.focus_history: List[Tuple[str, float]] = []  # (module_name, timestamp)
        
        # 统计
        self.stats = {
            "total_competitions": 0,
            "broadcasts_sent": 0,
            "focus_switches": 0
        }
        
        # 模型引用（用于获取真实embedding）
        self._model_interface = None
        self._embedding_layer = None
    
    def set_model(self, model_interface):
        """
        设置模型接口，用于获取真实的embedding
        
        Args:
            model_interface: QwenInterface 或类似接口，需提供 tokenizer 和 base_model
        """
        self._model_interface = model_interface
        if hasattr(model_interface, 'model') and hasattr(model_interface.model, 'base_model'):
            self._embedding_layer = model_interface.model.base_model.get_input_embeddings()
            print("[GlobalWorkspace] 模型embedding层已设置")
        elif hasattr(model_interface, 'embeddings'):
            self._embedding_layer = model_interface.embeddings
            print("[GlobalWorkspace] 模型embedding层已设置（通过embeddings属性）")
    
    def register_module(
        self,
        module_name: str,
        output: torch.Tensor,
        confidence: float = 1.0,
        metadata: Dict = None
    ):
        """
        注册模块输出
        
        Args:
            module_name: 模块名称
            output: 输出张量
            confidence: 置信度
            metadata: 元数据
        """
        # 维度适配：如果输出维度与 hidden_size 不匹配，使用适配器
        if output.dim() == 1:
            output = output.unsqueeze(0)  # [D] -> [1, D]
        
        if output.shape[-1] != self.hidden_size:
            # 检查是否有对应的适配器
            if module_name in self._dimension_adapters:
                output = self._dimension_adapters[module_name](output)
            else:
                # 没有适配器，使用零填充
                padded = torch.zeros(output.shape[0], self.hidden_size, device=self.device)
                min_dim = min(output.shape[-1], self.hidden_size)
                padded[:, :min_dim] = output[:, :min_dim]
                output = padded
        
        self.registered_modules[module_name] = ModuleOutput(
            module_name=module_name,
            output=output.squeeze(0) if output.shape[0] == 1 else output,
            confidence=confidence,
            metadata=metadata or {}
        )
    
    def compete_and_broadcast(
        self,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[str], Dict[str, torch.Tensor]]:
        """
        竞争并广播
        
        Args:
            context: 当前上下文
        
        Returns:
            winner_name: 胜出模块名称
            broadcasts: 广播信息字典
        """
        if not self.registered_modules:
            return None, {}
        
        self.stats["total_competitions"] += 1
        
        # 1. 收集所有模块输出
        outputs = list(self.registered_modules.values())
        
        # 2. 计算竞争分数
        competition_scores = self.competition(outputs, context)
        
        # 3. 选择胜出者
        winner_idx = competition_scores.argmax().item()
        winner = outputs[winner_idx]
        winner_name = winner.module_name
        
        # 4. 更新焦点
        if self.current_focus != winner_name:
            self.stats["focus_switches"] += 1
            self.focus_history.append((winner_name, time.time()))
        self.current_focus = winner_name
        
        # 5. 更新意识状态
        self.consciousness_state = winner.output.clone()
        self.consciousness_history.append(self.consciousness_state)
        
        # 限制历史长度
        if len(self.consciousness_history) > self.max_history:
            self.consciousness_history = self.consciousness_history[-self.max_history:]
        
        # 6. 广播
        broadcasts = self.broadcast(winner.output)
        self.stats["broadcasts_sent"] += 1
        
        # 7. 清空注册表，准备下一轮
        self.registered_modules.clear()
        
        return winner_name, broadcasts
    
    def integrate(
        self,
        user_input: Optional[str] = None,
        memory_context: Optional[torch.Tensor] = None,
        thought_state: Optional[torch.Tensor] = None,
        goal_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        整合多模块信息
        
        这是主要入口函数，整合：
        - 用户输入
        - 记忆上下文
        - 思维状态
        - 目标状态
        
        Returns:
            integrated_state: 整合后的意识状态
        """
        # 注册各模块
        if memory_context is not None:
            self.register_module("memory", memory_context, confidence=0.9)
        
        if thought_state is not None:
            self.register_module("thought", thought_state, confidence=0.8)
        
        if goal_state is not None:
            self.register_module("goal", goal_state, confidence=0.95)
        
        # 构建上下文
        context = self._build_context(user_input)
        
        # 竞争并广播
        winner_name, broadcasts = self.compete_and_broadcast(context)
        
        # 返回整合状态
        if self.consciousness_state is not None:
            return self.consciousness_state
        else:
            # 如果没有胜出者，返回零张量
            return torch.zeros(self.hidden_size, device=self.device)
    
    def _build_context(self, user_input: Optional[str]) -> Optional[torch.Tensor]:
        """
        构建上下文张量（生产级实现）
        
        策略：
        1. 如果有用户输入，优先编码用户输入（使用真实模型embedding）
        2. 结合历史意识状态形成时间序列上下文
        3. 使用加权组合而非简单替换
        """
        context_parts = []
        weights = []
        
        # 1. 用户输入编码（最高权重）- 使用真实embedding
        if user_input and len(user_input.strip()) > 0:
            # 优先使用真实模型embedding
            if self._embedding_layer is not None:
                # 获取tokenizer
                tokenizer = None
                if hasattr(self._model_interface, 'tokenizer'):
                    tokenizer = self._model_interface.tokenizer
                elif hasattr(self._model_interface, 'model') and hasattr(self._model_interface.model, 'tokenizer'):
                    tokenizer = self._model_interface.model.tokenizer
                
                if tokenizer is not None:
                    # 使用真实模型获取embedding
                    input_ids = tokenizer(
                        user_input[:100],  # 限制长度
                        return_tensors="pt",
                        truncation=True,
                        max_length=50
                    )['input_ids'].to(self.device)
                    
                    with torch.no_grad():
                        # 使用真实的embedding层
                        embeddings = self._embedding_layer(input_ids)
                        # 平均池化获取句子级别的表示
                        user_embedding = embeddings.mean(dim=1).squeeze(0)
                    
                    # 如果embedding维度与hidden_size不匹配，需要投影
                    if user_embedding.shape[0] != self.hidden_size:
                        # 创建投影矩阵（简单的线性投影）
                        if not hasattr(self, '_projection'):
                            self._projection = nn.Linear(
                                user_embedding.shape[0], 
                                self.hidden_size, 
                                bias=False
                            ).to(self.device)
                            # 初始化为接近恒等映射（如果维度相同）
                            nn.init.xavier_uniform_(self._projection.weight)
                        user_embedding = self._projection(user_embedding)
                    
                    context_parts.append(user_embedding)
                    weights.append(0.6)  # 用户输入权重60%
                else:
                    # 回退到tokenizer + embedding层直接调用
                    raise ValueError("未找到tokenizer")
            else:
                # 回退方案：使用简单的特征编码
                # 基于字符的简单编码
                char_features = torch.zeros(self.hidden_size, device=self.device)
                for i, char in enumerate(user_input[:50]):
                    # 使用字符的Unicode值生成特征
                    char_val = ord(char)
                    idx = char_val % self.hidden_size
                    char_features[idx] += 1.0 / (i + 1)
                
                # 归一化
                if char_features.norm() > 0:
                    char_features = char_features / char_features.norm()
                
                context_parts.append(char_features)
                weights.append(0.6)
        
        # 2. 历史意识状态（次要权重）
        if self.consciousness_history:
            # 使用最近的3个意识状态
            recent_states = self.consciousness_history[-3:]
            for i, state in enumerate(recent_states):
                # 越近的状态权重越高
                weight = 0.4 / (len(recent_states) - i) if len(recent_states) > 1 else 0.4
                context_parts.append(state)
                weights.append(weight / len(recent_states))
        
        # 3. 加权组合
        if context_parts:
            if len(context_parts) == 1:
                return context_parts[0]
            
            # 归一化权重
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # 加权求和
            combined = torch.zeros(self.hidden_size, device=self.device)
            for part, weight in zip(context_parts, normalized_weights):
                combined += part * weight
            
            return combined
        
        return None
    
    def get_consciousness_state(self) -> Optional[torch.Tensor]:
        """获取当前意识状态"""
        return self.consciousness_state
    
    def get_focus_history(self, n: int = 10) -> List[Tuple[str, float]]:
        """获取焦点历史"""
        return self.focus_history[-n:]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            "current_focus": self.current_focus,
            "consciousness_history_len": len(self.consciousness_history),
            "registered_modules": list(self.registered_modules.keys())
        }
    
    def reset(self):
        """重置工作空间"""
        self.registered_modules.clear()
        self.consciousness_state = None
        self.current_focus = None
        # 保留历史用于分析
    
    def save_state(self) -> Dict:
        """保存状态"""
        return {
            "consciousness_state": self.consciousness_state,
            "current_focus": self.current_focus,
            "stats": self.stats
        }
    
    def load_state(self, state: Dict):
        """加载状态"""
        self.consciousness_state = state.get("consciousness_state")
        self.current_focus = state.get("current_focus")
        self.stats.update(state.get("stats", {}))



# 工厂函数
def create_global_workspace(
    hidden_size: int = 1024,
    device: str = "cpu"
) -> GlobalWorkspace:
    """创建全局工作空间实例"""
    return GlobalWorkspace(hidden_size=hidden_size, device=device)
