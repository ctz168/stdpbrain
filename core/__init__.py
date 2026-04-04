"""
类人脑双系统全闭环 AI架构 - 核心模块
"""

from .dual_weight_layers import DualWeightLinear
from .stdp_engine import STDPEngine, FullLinkSTDP
from .refresh_engine import RefreshCycleEngine
# SelfLoopOptimizer is in self_loop package, commented out to avoid import error
# from .self_loop import SelfLoopOptimizer
try:
    from .interfaces import BrainAIInterface
except ImportError:
    BrainAIInterface = None

__all__ = [
    'DualWeightLinear', 
    'STDPEngine',
    'FullLinkSTDP',
    'RefreshCycleEngine',
    # 'SelfLoopOptimizer',
    'BrainAIInterface'
]
