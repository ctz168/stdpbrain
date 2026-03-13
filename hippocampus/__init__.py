"""
海马体记忆系统模块

严格基于人脑海马体 - 新皮层双系统神经科学原理开发
是整个架构的记忆中枢与推理导航仪
"""

from .hippocampus_system import HippocampusSystem
from .ec_encoder import EntorhinalEncoder
from .dg_separator import DentateGyrusSeparator
from .ca3_memory import CA3EpisodicMemory
from .ca1_gate import CA1AttentionGate
from .swr_consolidation import SWRConsolidation

__all__ = [
    'HippocampusSystem',
    'EntorhinalEncoder',
    'DentateGyrusSeparator',
    'CA3EpisodicMemory',
    'CA1AttentionGate',
    'SWRConsolidation'
]
