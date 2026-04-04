"""
海马体记忆系统模块

严格基于人脑海马体 - 新皮层双系统神经科学原理开发
是整个架构的记忆中枢与推理导航仪

[优化] 人类记忆增强:
- SemanticSummarizer: 语义摘要生成 + Embedding 语义匹配
- MemoryConsolidationManager: 记忆分层（短期/中期/长期）
"""

from .hippocampus_system import HippocampusSystem
from .ec_encoder import EntorhinalEncoder
from .dg_separator import DentateGyrusSeparator
from .ca3_memory import CA3EpisodicMemory, EpisodicMemory
from .ca1_gate import CA1AttentionGate
from .swr_consolidation import SWRConsolidation
from .semantic_engine import SemanticSummarizer
from .memory_layers import MemoryTier, MemoryConsolidationManager, TierConfig

__all__ = [
    'HippocampusSystem',
    'EntorhinalEncoder',
    'DentateGyrusSeparator',
    'CA3EpisodicMemory',
    'EpisodicMemory',
    'CA1AttentionGate',
    'SWRConsolidation',
    # 人类记忆增强
    'SemanticSummarizer',
    'MemoryTier',
    'MemoryConsolidationManager',
    'TierConfig',
]
