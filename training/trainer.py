"""
专项全流程训练模块 - 主训练器

整合三个子训练模块:
1. PretrainAdapter: 底座预适配微调
2. OnlineLifelongLearner: 在线终身学习
3. OfflineConsolidation: 离线记忆巩固
"""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """训练配置"""
    learning_rate: float = 1e-5
    batch_size: int = 8
    epochs: int = 3
    quantization: str = "INT4"


class BrainAITrainer:
    """
    类脑架构训练器
    
    三阶段训练流程:
    1. 底座预适配微调 (部署前一次性)
    2. 在线终身学习 (推理时实时)
    3. 离线记忆巩固 (空闲时执行)
    """
    
    def __init__(self, model, config, device: Optional[str] = None):
        self.model = model
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化子训练器
        from .pretrain_adapter import PretrainAdapter
        from .online_learner import OnlineLifelongLearner
        from .offline_consolidation import OfflineConsolidation
        
        self.pretrain_adapter = PretrainAdapter(model, config, device=self.device)
        self.online_learner = OnlineLifelongLearner(model, config, device=self.device)
        self.offline_consolidation = OfflineConsolidation(model, config, device=self.device)
    
    def pretrain(self, datasets: List[str], output_path: str) -> dict:
        """
        子模块 1: 底座预适配微调
        
        Args:
            datasets: 数据集列表
            output_path: 输出路径
        
        Returns:
            metrics: 训练指标
        """
        print("=" * 60)
        print("阶段 1: 底座预适配微调")
        print("=" * 60)
        
        # 冻结 90% 静态权重
        self.pretrain_adapter.freeze_static_weights()
        
        # 获取可训练参数 (仅 10% 动态权重 + 海马体连接)
        trainable_params = self.pretrain_adapter.get_trainable_parameters()
        
        print(f"\n可训练参数：{trainable_params:,}")
        print(f"冻结参数：{sum(p.numel() for p in self.model.parameters() if not p.requires_grad):,}")
        
        # 执行训练
        metrics = self.pretrain_adapter.train(
            datasets=datasets,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs
        )
        
        # 保存预适配权重
        self.pretrain_adapter.save(output_path)
        
        print(f"\n✓ 预适配完成，权重已保存至 {output_path}")
        
        return metrics
    
    def train_online(self, enable: bool = True):
        """
        子模块 2: 启用/禁用在线终身学习
        
        Args:
            enable: 是否启用
        """
        if enable:
            print("[训练] 启用在线终身学习")
            self.online_learner.enable()
        else:
            print("[训练] 禁用在线终身学习")
            self.online_learner.disable()
    
    def consolidate_offline(self, trigger: str = "idle"):
        """
        子模块 3: 触发离线记忆巩固
        
        Args:
            trigger: 触发条件 ("idle" | "manual" | "scheduled")
        """
        if trigger == "idle":
            # 自动检测空闲状态
            self.offline_consolidation.start_idle_monitoring()
        
        elif trigger == "manual":
            # 手动触发
            print("[训练] 手动触发离线记忆巩固...")
            self.offline_consolidation.consolidate()
        
        elif trigger == "scheduled":
            # 定时触发
            self.offline_consolidation.schedule_consolidation()
    
    def get_training_stats(self) -> dict:
        """获取训练统计信息"""
        return {
            'pretrain': self.pretrain_adapter.get_stats(),
            'online': self.online_learner.get_stats(),
            'offline': self.offline_consolidation.get_stats()
        }
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """从检查点恢复"""
        print(f"[训练] 从检查点恢复：{checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        
        # 恢复模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复训练状态
        if 'pretrain_metrics' in checkpoint:
            print(f"  预适配指标：{checkpoint['pretrain_metrics']}")
        
        print("✓ 恢复完成")


def run_training(
    model_path: str,
    output_path: str,
    datasets: Optional[List[str]] = None
):
    """
    运行完整训练流程
    
    Args:
        model_path: 模型路径
        output_path: 输出路径
        datasets: 数据集列表
    """
    from configs.arch_config import default_config, TrainingConfig
    
    # 加载模型
    print(f"[训练] 加载模型：{model_path}")
    model = load_model(model_path)  # TODO: 实现模型加载
    
    # 创建训练器
    trainer = BrainAITrainer(model, default_config)
    
    # 执行预训练
    if datasets is None:
        datasets = default_config.training.datasets
    
    metrics = trainer.pretrain(datasets, output_path)
    
    print("\n[训练完成]")
    print(f"  损失：{metrics['loss']:.4f}")
    print(f"  准确率：{metrics['accuracy']:.4f}")
    
    return trainer


# TODO: 实现模型加载函数
def load_model(path: str):
    """加载模型"""
    pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="类脑 AI 训练脚本")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="./checkpoints/pretrained.pt")
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    
    args = parser.parse_args()
    
    trainer = run_training(args.model_path, args.output_path, args.datasets)
