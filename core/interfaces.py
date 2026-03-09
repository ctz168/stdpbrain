"""
类人脑双系统全闭环 AI架构 - 核心接口定义

提供统一的 BrainAIInterface 接口，封装所有底层模块

注意：当前为简化实现版本，用于测试和演示。
完整版本需要加载真实的 Qwen3.5-0.8B 模型权重。
"""

import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# 使用可运行版本
from .interfaces_working import BrainAIInterface, BrainAIOutput, create_brain_ai

__all__ = [
    'BrainAIInterface',
    'BrainAIOutput',
    'create_brain_ai'
]
