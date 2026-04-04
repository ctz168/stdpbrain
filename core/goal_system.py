"""
目标驱动系统 (Goal-Driven System)

核心功能：
1. 从用户输入推断当前目标
2. 分解复杂目标为子目标
3. 追踪目标完成进度
4. 生成内在奖励信号
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import re


class GoalType(Enum):
    """目标类型"""
    UNDERSTAND = "understand"       # 理解用户意图
    ANSWER = "answer"               # 回答问题
    SOLVE = "solve"                 # 解决问题
    REMEMBER = "remember"           # 记忆信息
    RECALL = "recall"               # 回忆/查询记忆
    GENERATE = "generate"           # 生成内容
    EXPLORE = "explore"             # 探索/学习
    SELF_REFLECT = "self_reflect"   # 自我反思


@dataclass
class Goal:
    """目标数据结构"""
    goal_id: str
    goal_type: GoalType
    description: str
    priority: float = 1.0
    progress: float = 0.0
    sub_goals: List['Goal'] = field(default_factory=list)
    parent_goal: Optional['Goal'] = None
    created_at: float = field(default_factory=time.time)
    completed: bool = False
    context: Dict = field(default_factory=dict)
    
    def is_complete(self) -> bool:
        """检查目标是否完成"""
        if self.sub_goals:
            return all(g.is_complete() for g in self.sub_goals)
        return self.progress >= 1.0
    
    def get_progress(self) -> float:
        """获取整体进度"""
        if self.sub_goals:
            return sum(g.get_progress() for g in self.sub_goals) / len(self.sub_goals)
        return self.progress


class GoalInference(nn.Module):
    """目标推断网络"""
    
    def __init__(self, hidden_size: int = 1024, goal_embedding_dim: int = 128):
        super().__init__()
        
        # 目标类型嵌入
        self.goal_type_embeddings = nn.Embedding(len(GoalType), goal_embedding_dim)
        
        # 目标推断网络
        self.inference_net = nn.Sequential(
            nn.Linear(hidden_size + goal_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(GoalType))
        )
        
        # 复杂度评估网络
        self.complexity_net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # 优先级计算网络
        self.priority_net = nn.Sequential(
            nn.Linear(hidden_size + 2, 128),  # hidden + progress + complexity
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推断目标类型和复杂度
        
        Args:
            hidden_state: 输入的隐藏状态 [batch, hidden_size]
        
        Returns:
            goal_type_logits: 目标类型logits [batch, num_goal_types]
            complexity: 复杂度分数 [batch, 1]
        """
        # dtype 对齐：确保输入与网络权重类型一致（兼容 BFloat16 模型）
        target_dtype = next(self.parameters()).dtype
        if hidden_state.dtype != target_dtype:
            hidden_state = hidden_state.to(target_dtype)
        
        # 目标类型推断
        goal_embeddings = self.goal_type_embeddings.weight.unsqueeze(0).expand(
            hidden_state.size(0), -1, -1
        )  # [batch, num_types, embed_dim]
        
        hidden_expanded = hidden_state.unsqueeze(1).expand(-1, goal_embeddings.size(1), -1)
        combined = torch.cat([hidden_expanded, goal_embeddings], dim=-1)
        
        goal_type_logits = self.inference_net(combined).mean(dim=1)  # [batch, num_types]
        
        # 复杂度评估
        complexity = self.complexity_net(hidden_state)
        
        return goal_type_logits, complexity


class GoalSystem:
    """
    目标驱动系统
    
    提供：
    - 目标推断与创建
    - 目标分解
    - 进度追踪
    - 内在奖励生成
    """
    
    def __init__(self, hidden_size: int = 1024, device: str = "cpu"):  # hidden_size 应由调用方动态传入
        self.hidden_size = hidden_size
        self.device = device
        
        # 目标推断网络
        self.inference = GoalInference(hidden_size).to(device)
        
        # 当前目标栈
        self.goal_stack: List[Goal] = []
        self.current_goal: Optional[Goal] = None
        self.goal_history: List[Goal] = []
        
        # 目标计数器
        self.goal_counter = 0
        
        # 当前目标向量（用于参与生成过程）
        self.current_goal_vector: Optional[torch.Tensor] = None
        
        # 内在奖励历史
        self.reward_history: List[float] = []
        
        # ========== [动态化] 自适应目标奖励权重（初始值与原硬编码一致）==========
        # 由用户正/负反馈驱动 LTP/LTD 更新，而非永远固定
        self.goal_type_reward_weights: Dict[GoalType, float] = {
            GoalType.REMEMBER:      1.2,
            GoalType.SOLVE:         1.0,
            GoalType.UNDERSTAND:    0.9,
            GoalType.ANSWER:        0.8,
            GoalType.GENERATE:      0.7,
            GoalType.EXPLORE:       0.6,
            GoalType.RECALL:        0.9,
            GoalType.SELF_REFLECT:  1.1,
        }
        # 权重上下限，防止极化
        self._weight_min = 0.3
        self._weight_max = 2.0
        # 最近一次完成的目标类型（用于反馈关联）
        self._last_completed_goal_type: Optional[GoalType] = None
        
        # 关键词模式
        self.patterns = {
            GoalType.UNDERSTAND: ["什么", "为什么", "如何", "怎么", "解释"],
            GoalType.ANSWER: ["是", "吗", "对不对", "是不是"],
            GoalType.SOLVE: ["解决", "处理", "修复", "完成", "帮我"],
            GoalType.REMEMBER: ["记住", "我叫", "我喜欢", "我的职业", "我的工作", "我住"],
            GoalType.RECALL: ["还记得", "记得吗", "你还记得", "记得我", "我叫什么"],
            GoalType.GENERATE: ["写", "生成", "创建", "编", "作"],
            GoalType.EXPLORE: ["有趣", "发现", "探索", "学习", "了解更多"],
            GoalType.SELF_REFLECT: ["你的想法", "你怎么看", "你觉得"]
        }
    
    def to(self, dtype=None, device=None):
        """
        将内部 nn.Module 子组件转换到指定 dtype/device。
        因为 GoalSystem 本身不是 nn.Module，需要手动转发。
        """
        if dtype is not None:
            self.inference = self.inference.to(dtype=dtype)
        if device is not None:
            self.inference = self.inference.to(device=device)
        return self
    
    def infer_goal(self, user_input: str, hidden_state: Optional[torch.Tensor] = None) -> Goal:
        """
        从用户输入推断目标
        
        Args:
            user_input: 用户输入文本
            hidden_state: 可选的隐藏状态张量
        
        Returns:
            goal: 推断出的目标
        """
        self.goal_counter += 1
        goal_id = f"goal_{self.goal_counter}_{int(time.time() * 1000)}"
        
        # 1. 关键词匹配推断
        goal_type = self._match_keywords(user_input)
        
        # 2. 如果有隐藏状态，使用网络推断
        complexity = 0.5
        if hidden_state is not None:
            with torch.no_grad():
                if hidden_state.dim() == 3:
                    hidden_state = hidden_state[:, -1, :]  # 取最后一个token
                if hidden_state.dim() == 2:
                    hidden_state = hidden_state.squeeze(0)
                
                # 推断目标类型
                goal_logits, complexity_tensor = self.inference(hidden_state.unsqueeze(0))
                predicted_type_idx = goal_logits.argmax(dim=-1).item()
                predicted_type = list(GoalType)[predicted_type_idx]
                
                # 结合关键词和网络推断
                if goal_type == GoalType.ANSWER and predicted_type != GoalType.ANSWER:
                    goal_type = predicted_type
                
                complexity = complexity_tensor.item()
        
        # 3. 创建目标描述
        description = self._create_goal_description(goal_type, user_input)
        
        # 4. 创建目标
        goal = Goal(
            goal_id=goal_id,
            goal_type=goal_type,
            description=description,
            priority=1.0 - complexity * 0.3,  # 复杂目标稍低优先级
            context={"user_input": user_input}
        )
        
        # 5. 如果是复杂目标，分解
        if complexity > 0.6:
            goal.sub_goals = self._decompose_goal(goal)
        
        # 6. 设置为当前目标
        self.current_goal = goal
        self.goal_stack.append(goal)
        
        # 7. 生成目标向量（用于参与生成过程）
        self.current_goal_vector = self._generate_goal_vector(goal)
        
        return goal
    
    def _match_keywords(self, text: str) -> GoalType:
        """基于关键词匹配目标类型"""
        text_lower = text.lower()
        
        scores = {}
        for goal_type, keywords in self.patterns.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[goal_type] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return GoalType.ANSWER  # 默认类型
    
    def _create_goal_description(self, goal_type: GoalType, user_input: str) -> str:
        """创建目标描述"""
        descriptions = {
            GoalType.UNDERSTAND: f"理解用户问题：{user_input[:30]}",
            GoalType.ANSWER: f"回答用户问题：{user_input[:30]}",
            GoalType.SOLVE: f"解决用户问题：{user_input[:30]}",
            GoalType.REMEMBER: f"记住用户提供的信息：{user_input[:30]}",
            GoalType.RECALL: f"回忆并回答用户的问题：{user_input[:30]}",
            GoalType.GENERATE: f"生成内容：{user_input[:30]}",
            GoalType.EXPLORE: f"探索话题：{user_input[:30]}",
            GoalType.SELF_REFLECT: f"自我反思：{user_input[:30]}"
        }
        return descriptions.get(goal_type, f"处理用户输入：{user_input[:30]}")
    
    def _decompose_goal(self, goal: Goal) -> List[Goal]:
        """分解复杂目标为子目标"""
        sub_goals = []
        
        if goal.goal_type == GoalType.SOLVE:
            # 解决问题的标准分解
            decompositions = [
                ("分析问题", GoalType.UNDERSTAND),
                ("制定方案", GoalType.GENERATE),
                ("执行方案", GoalType.SOLVE),
                ("验证结果", GoalType.SELF_REFLECT)
            ]
            
            for i, (desc, gtype) in enumerate(decompositions):
                sub_goal = Goal(
                    goal_id=f"{goal.goal_id}_sub{i}",
                    goal_type=gtype,
                    description=desc,
                    parent_goal=goal,
                    priority=goal.priority
                )
                sub_goals.append(sub_goal)
        
        elif goal.goal_type == GoalType.UNDERSTAND:
            # 理解问题的分解
            decompositions = [
                ("识别关键概念", GoalType.UNDERSTAND),
                ("关联已有知识", GoalType.REMEMBER),
                ("构建理解框架", GoalType.GENERATE)
            ]
            
            for i, (desc, gtype) in enumerate(decompositions):
                sub_goal = Goal(
                    goal_id=f"{goal.goal_id}_sub{i}",
                    goal_type=gtype,
                    description=desc,
                    parent_goal=goal,
                    priority=goal.priority
                )
                sub_goals.append(sub_goal)
        
        return sub_goals
    
    def update_progress(self, progress: float, goal_id: Optional[str] = None):
        """
        更新目标进度
        
        Args:
            progress: 进度值 (0.0 - 1.0)
            goal_id: 目标ID，默认为当前目标
        """
        goal = self.current_goal if goal_id is None else self._find_goal(goal_id)
        
        if goal:
            old_progress = goal.progress
            goal.progress = min(1.0, max(0.0, progress))
            
            print(f"🎯 [目标进度更新] {goal.goal_id}: {old_progress:.2f} → {goal.progress:.2f}", flush=True)
            
            # 检查是否完成
            if goal.is_complete():
                self._complete_goal(goal)
    
    def _find_goal(self, goal_id: str) -> Optional[Goal]:
        """查找目标"""
        for goal in self.goal_stack:
            if goal.goal_id == goal_id:
                return goal
            if goal.sub_goals:
                for sub in goal.sub_goals:
                    if sub.goal_id == goal_id:
                        return sub
        return None
    
    def _complete_goal(self, goal: Goal):
        """完成目标"""
        goal.completed = True
        goal.progress = 1.0
        
        print(f"🎯 [目标完成] ✓ {goal.goal_id}: {goal.description}", flush=True)
        print(f"🎯 [目标完成] 类型: {goal.goal_type.value}", flush=True)
        
        # 添加到历史
        self.goal_history.append(goal)
        
        # 从栈中移除
        if goal in self.goal_stack:
            self.goal_stack.remove(goal)
        
        # 更新父目标
        if goal.parent_goal:
            self._update_parent_progress(goal.parent_goal)
        
        # 如果是当前目标，切换到下一个
        if goal == self.current_goal:
            self.current_goal = self.goal_stack[-1] if self.goal_stack else None
            if self.current_goal:
                print(f"🎯 [目标切换] → 下一个目标: {self.current_goal.description}", flush=True)
            else:
                print(f"🎯 [目标栈空] 所有目标已完成", flush=True)
    
    def _update_parent_progress(self, parent: Goal):
        """更新父目标进度"""
        if parent.sub_goals:
            parent.progress = sum(g.progress for g in parent.sub_goals) / len(parent.sub_goals)
        # [FIX] 原来这段代码误放在 _generate_goal_vector 中作为不可达死代码。
        # 子目标全部完成后，应在此处触发父目标的完成检查。
        if parent.is_complete():
            self._complete_goal(parent)
    
    def _generate_goal_vector(self, goal: Goal) -> torch.Tensor:
        """
        生成目标向量（用于参与生成过程）
        
        Args:
            goal: 目标对象
        
        Returns:
            goal_vector: 目标向量 [hidden_size]
        """
        with torch.no_grad():
            # 1. 获取目标类型嵌入
            goal_type_idx = list(GoalType).index(goal.goal_type)
            goal_type_embedding = self.inference.goal_type_embeddings(
                torch.tensor([goal_type_idx], device=self.device)
            )  # [1, embed_dim]
            
            print(f"   └─ 目标类型嵌入维度: {goal_type_embedding.shape}", flush=True)
            
            # 2. 结合目标优先级和进度
            priority_weight = torch.tensor([goal.priority], device=self.device)
            progress_weight = torch.tensor([goal.progress], device=self.device)
            
            # 3. 扩展到hidden_size维度
            # 目标嵌入: embed_dim -> hidden_size
            goal_vector = torch.zeros(self.hidden_size, device=self.device)
            
            # 将目标嵌入填充到向量的前部分
            embed_dim = goal_type_embedding.shape[-1]
            goal_vector[:embed_dim] = goal_type_embedding.squeeze(0)
            
            # 在特定位置编码优先级和进度
            if embed_dim < self.hidden_size - 2:
                goal_vector[embed_dim] = priority_weight
                goal_vector[embed_dim + 1] = progress_weight
            
            print(f"   └─ 优先级编码位置: {embed_dim}, 值: {priority_weight.item():.2f}", flush=True)
            print(f"   └─ 进度编码位置: {embed_dim+1}, 值: {progress_weight.item():.2f}", flush=True)
            
            # 4. 如果有子目标，编码子目标信息
            if goal.sub_goals:
                # 子目标数量编码
                num_subgoals = min(len(goal.sub_goals), 10)  # 最多10个子目标
                if embed_dim + 2 < self.hidden_size:
                    goal_vector[embed_dim + 2] = torch.tensor([num_subgoals], device=self.device, dtype=torch.float)
                print(f"   └─ 子目标数量: {num_subgoals}", flush=True)
            
            return goal_vector
    
    def get_reward_signal(self) -> float:
        """
        获取内在奖励信号（使用自适应权重而非固定系数）
        
        Returns:
            reward: 奖励值 (0.0 - 1.0)
        """
        if not self.current_goal:
            return 0.5
        
        # 基于进度
        progress_reward = self.current_goal.get_progress()
        
        # [动态化] 基于目标类型：使用自适应权重（而非固定字典）
        type_factor = self.goal_type_reward_weights.get(
            self.current_goal.goal_type, 1.0
        )
        # 记录本次目标类型，用于后续反馈关联
        self._last_completed_goal_type = self.current_goal.goal_type
        
        # 基于优先级
        priority_factor = self.current_goal.priority
        
        # 综合奖励
        reward = progress_reward * type_factor * priority_factor
        reward = min(1.0, max(0.0, reward))
        
        self.reward_history.append(reward)
        
        return reward
    
    def update_reward_from_feedback(self, positive: bool):
        """
        [新增] 根据用户反馈动态调整当前目标类型的奖励权重。
        正向反馈 → LTP（权重上涨5%）
        负向反馈 → LTD（权重下降5%）
        应在对话结束时由 interfaces.py 调用。
        
        Args:
            positive: True=用户正向互动（继续、感谢等），False=负向（沉默、打断等）
        """
        goal_type = self._last_completed_goal_type
        if goal_type is None:
            return
        
        current_weight = self.goal_type_reward_weights.get(goal_type, 1.0)
        if positive:
            new_weight = min(self._weight_max, current_weight * 1.05)  # LTP
        else:
            new_weight = max(self._weight_min, current_weight * 0.95)  # LTD
        
        self.goal_type_reward_weights[goal_type] = new_weight
    
    def get_current_goal_info(self) -> Dict:
        """获取当前目标信息"""
        if not self.current_goal:
            return {"status": "no_goal"}
        
        return {
            "goal_id": self.current_goal.goal_id,
            "type": self.current_goal.goal_type.value,
            "description": self.current_goal.description,
            "progress": self.current_goal.get_progress(),
            "priority": self.current_goal.priority,
            "sub_goals": [
                {
                    "description": g.description,
                    "progress": g.progress,
                    "completed": g.completed
                }
                for g in self.current_goal.sub_goals
            ],
            "completed": self.current_goal.completed
        }
    
    def get_stats(self) -> Dict:
        """获取系统统计"""
        return {
            "total_goals": self.goal_counter,
            "active_goals": len(self.goal_stack),
            "completed_goals": len(self.goal_history),
            "avg_reward": sum(self.reward_history[-100:]) / max(1, len(self.reward_history[-100:]))
        }


# 工具函数
def create_goal_system(hidden_size: int = 1024, device: str = "cpu") -> GoalSystem:  # hidden_size 应由调用方动态传入
    """创建目标系统实例"""
    return GoalSystem(hidden_size=hidden_size, device=device)
