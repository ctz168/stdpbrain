"""
专项全流程训练模块

三个子模块:
1. 底座预适配微调 (部署前一次性执行)
2. 在线终身学习 (推理时实时执行)
3. 离线记忆巩固 (空闲时执行)
"""

from .trainer import BrainAITrainer
from .pretrain_adapter import PretrainAdapter
from .online_learner import OnlineLifelongLearner
from .offline_consolidation import OfflineConsolidation

__all__ = [
    'BrainAITrainer',
    'PretrainAdapter',
    'OnlineLifelongLearner',
    'OfflineConsolidation'
]
