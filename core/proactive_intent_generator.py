"""
主动意图生成器

决定系统何时、为何、输出什么内容（不依赖用户输入）

核心设计：
- 多源触发器：时间、目标、记忆显著性、思维状态
- 意图分类：分享/提问/反思/沉默
- 节流机制：防止骚扰
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
import time
import random
from dataclasses import dataclass
from enum import Enum

class ProactiveIntent(Enum):
    """主动意图类型"""
    SHARE_THOUGHT = "share_thought"    # 分享一个想法
    ASK_QUESTION = "ask_question"      # 提问（非澄清，而是好奇）
    REFLECT_SHARE = "reflect_share"    # 分享反思
    REMIND = "remind"                  # 提醒（基于目标）
    SILENCE = "silence"                # 沉默（默认）

@dataclass
class ProactiveContext:
    """主动输出的上下文"""
    time_silence_seconds: float = 0.0  # 距上次用户输入的时间
    time_since_output: float = 0.0     # 距上次输出的时间
    current_thought: str = ""          # 当前思维内容
    mind_state: str = "RESTING"
    goal_context: Optional[str] = None
    memory_salience: float = 0.0       # 最近记忆的显著性
    recent_clarifications: int = 0     # 近期澄清次数
    conversation_turns: int = 0        # 对话轮次

class ProactiveIntentGenerator(nn.Module):
    """主动意图生成器"""
    
    def __init__(
        self,
        hidden_size: int,
        intent_hidden: int = 128,
        min_interval_seconds: int = 300,  # 最小主动间隔（5分钟）
        max_daily_count: int = 10          # 每日主动上限
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.min_interval = min_interval_seconds
        self.max_daily = max_daily_count
        
        # 意图分类器
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size + 5, intent_hidden),  # hidden + 5个时间特征
            nn.LayerNorm(intent_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intent_hidden, len(ProactiveIntent)),
            nn.Softmax(dim=-1)
        )
        
        # 意图质量评估器（避免无意义输出）
        self.quality_scorer = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 统计
        self.daily_count = 0
        self.last_proactive_time = 0.0
        self.last_user_input_time = time.time()
        
    def forward(
        self, 
        current_state: torch.Tensor,
        context: ProactiveContext
    ) -> Tuple[ProactiveIntent, float, Dict]:
        """
        决定是否主动输出
        
        Returns:
            (intent, confidence, debug_info)
        """
        debug_info = {}
        
        # ========== 1. 节流检查（第一道防线）==========
        time_silence = context.time_silence_seconds
        time_since_output = context.time_since_output
        
        # 1.1 最小间隔检查
        if time_since_output < self.min_interval:
            debug_info["throttle_reason"] = f"interval too short ({time_since_output:.0f}s < {self.min_interval}s)"
            return ProactiveIntent.SILENCE, 0.0, debug_info
        
        # 1.2 每日上限检查
        if self.daily_count >= self.max_daily:
            debug_info["throttle_reason"] = f"daily limit reached ({self.daily_count}/{self.max_daily})"
            return ProactiveIntent.SILENCE, 0.0, debug_info
        
        # 1.3 近期澄清过多检查（避免主动+澄清双重打扰）
        if context.recent_clarifications > 2:
            debug_info["throttle_reason"] = f"too many clarifications ({context.recent_clarifications})"
            return ProactiveIntent.SILENCE, 0.0, debug_info
        
        # ========== 2. 构建分类器输入 ==========
        # 时间特征（归一化）
        time_features = torch.tensor([
            min(time_silence / 3600.0, 1.0),      # 沉默时长（小时，最大1小时）
            min(time_since_output / 3600.0, 1.0),
            1.0 if context.mind_state == "REFLECTING" else 0.0,  # 是否在反思
            1.0 if context.goal_context else 0.0,               # 是否有目标
            context.memory_salience  # 记忆显著性
        ], device=current_state.device).float()
        
        classifier_input = torch.cat([current_state.mean(dim=-1), time_features])
        
        # ========== 3. 意图分类 ==========
        intent_probs = self.intent_classifier(classifier_input.unsqueeze(0)).squeeze(0)
        
        # 获取最高概率的意图
        top_intent_idx = torch.argmax(intent_probs).item()
        top_intent = ProactiveIntent(top_intent_idx)
        top_confidence = intent_probs[top_intent_idx].item()
        
        # ========== 4. 质量评估 ==========
        quality_score = self.quality_scorer(current_state.mean(dim=-1).unsqueeze(0)).item()
        
        debug_info.update({
            "intent_probs": {intent.value: prob.item() for intent, prob in zip(ProactiveIntent, intent_probs)},
            "quality_score": quality_score,
            "time_silence": time_silence,
            "mind_state": context.mind_state
        })
        
        # ========== 5. 决策 ==========
        # 需要同时满足：意图置信度高 + 质量评分高 + 时间条件
        confidence_threshold = 0.4  # 意图置信度阈值
        quality_threshold = 0.6     # 质量阈值
        
        if (top_confidence > confidence_threshold and 
            quality_score > quality_threshold and
            top_intent != ProactiveIntent.SILENCE):
            
            # 通过！更新统计
            self.daily_count += 1
            self.last_proactive_time = time.time()
            
            debug_info["decision"] = f"proactive_output: {top_intent.value}"
            return top_intent, top_confidence * quality_score, debug_info
        
        debug_info["decision"] = "below_thresholds"
        return ProactiveIntent.SILENCE, 0.0, debug_info
    
    def reset_daily_count(self):
        """重置每日计数（午夜调用）"""
        self.daily_count = 0
    
    def record_user_input(self):
        """记录用户输入时间（更新沉默计时器）"""
        self.last_user_input_time = time.time()
