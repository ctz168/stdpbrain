"""
类人脑双系统全闭环 AI架构 - 核心模块
"""

from .dual_weight_layers import DualWeightLinear, DualWeightAttention
from .stdp_engine import STDPEngine, FullLinkSTDP
from .refresh_engine import RefreshCycleEngine
# SelfLoopOptimizer is in self_loop package, commented out to avoid import error
# from .self_loop import SelfLoopOptimizer
from .interfaces import BrainAIInterface

__all__ = [
    'DualWeightLinear',
    'DualWeightAttention', 
    'STDPEngine',
    'FullLinkSTDP',
    'RefreshCycleEngine',
    # 'SelfLoopOptimizer',
    'BrainAIInterface'
]
