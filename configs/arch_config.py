"""
类人脑双系统全闭环 AI架构 - 核心配置文件
基于Qwen3.5-0.8B 底座模型
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch


# ==================== 核心刚性约束配置 ====================

@dataclass
class HardConstraints:
    """不可突破的硬性约束配置"""
    # 权重安全约束
    STATIC_WEIGHT_RATIO: float = 0.9  # 90% 静态基础权重比例
    DYNAMIC_WEIGHT_RATIO: float = 0.1  # 10% STDP动态增量权重比例
    
    # 端侧算力约束
    MAX_MEMORY_MB: int = 420  # INT4 量化后最大显存占用 (MB)
    REFRESH_PERIOD_MS: int = 10  # 10ms 刷新周期
    MAX_COMPUTE_OVERHEAD: float = 0.1  # 单周期算力开销不超过原生 10%
    
    # 窄窗口约束
    NARROW_WINDOW_SIZE: int = 2  # 每个周期处理 1-2 个 token
    ATTENTION_COMPLEXITY: str = "O(1)"  # 固定 O(1) 复杂度
    
    # 海马体内存约束
    HIPPOCAMPUS_MAX_MEMORY_MB: int = 2  # 情景记忆库最大 2MB


# ==================== STDP超参数配置 ====================

@dataclass
class STDPConfig:
    """STDP 时序可塑性权重更新配置"""
    # 学习率
    alpha_LTP: float = 0.01  # 权重增强学习率 (Long-Term Potentiation)
    beta_LTD: float = 0.008  # 权重减弱学习率 (Long-Term Depression)
    
    # 阈值
    update_threshold: float = 0.001  # 最小更新阈值
    weight_min: float = -1.0  # 权重下界
    weight_max: float = 1.0  # 权重上界
    
    # 时间窗口
    time_window_ms: int = 20  # STDP 时间窗口 (毫秒)
    decay_rate: float = 0.99  # 权重衰减率
    
    # 更新节点
    update_attention: bool = True  # 注意力层 STDP 更新
    update_ffn: bool = True  # FFN 层 STDP 更新
    update_self_eval: bool = True  # 自评判 STDP 更新
    update_hippocampus_gate: bool = True  # 海马体门控 STDP 更新


# ==================== 海马体系统配置 ====================

@dataclass
class HippocampusConfig:
    """海马体记忆系统配置"""
    # 特征编码
    EC_feature_dim: int = 64  # 内嗅皮层编码维度 (64 维低维特征)
    DG_sparsity: float = 0.9  # 齿状回稀疏度
    DG_orthogonalization: bool = True  # 模式分离正交化
    
    # 情景记忆存储
    CA3_memory_format: str = "ID+timestamp+skeleton+semantic_pointer+causal"
    CA3_max_capacity: int = 10000  # 最大记忆容量
    CA3_timestamp_precision_ms: int = 10  # 时间戳精度 10ms
    
    # 时序编码与门控
    CA1_temporal_encoding: bool = True
    CA1_attention_gate: bool = True
    recall_topk: int = 2  # 每个周期召回 1-2 个记忆锚点
    
    # 离线回放巩固
    SWR_enabled: bool = True
    SWR_idle_threshold_s: int = 300  # 空闲 5 分钟触发
    SWR_replay_frequency: float = 0.1  # 回放频率
    
    # 内存约束
    max_memory_bytes: int = 2 * 1024 * 1024  # 2MB
    use_cycle_buffer: bool = True  # 循环缓存


# ==================== 自闭环优化系统配置 ====================

@dataclass
class SelfLoopConfig:
    """单智体自闭环优化系统配置"""
    # 模式 1: 自生成组合输出
    mode1_temperature_range: tuple = (0.7, 0.9)
    mode1_num_candidates: int = 2
    mode1_accuracy_window: int = 10  # 基于过往 10 个周期准确率
    
    # 模式 2: 自博弈竞争优化
    mode2_role_switch_period: int = 1  # 每周期切换角色
    mode2_max_iterations: int = 5  # 最大迭代次数
    mode2_convergence_threshold: float = 0.95
    
    # 模式 3: 自双输出 + 自评判选优
    mode3_eval_period: int = 10  # 每 10 个周期执行一次
    mode3_eval_dimensions: List[str] = field(default_factory=lambda: [
        "fact_accuracy",      # 事实准确性 0-10 分
        "logic_completeness", # 逻辑完整性 0-10 分
        "semantic_coherence", # 语义连贯性 0-10 分
        "instruction_follow"  # 指令遵循度 0-10 分
    ])
    mode3_max_score: int = 40  # 总分 40 分
    
    # 自动触发条件
    high_difficulty_keywords: List[str] = field(default_factory=lambda: [
        "数学", "计算", "逻辑推理", "代码", "编程", 
        "事实性", "证明", "推导", "算法"
    ])
    high_accuracy_keywords: List[str] = field(default_factory=lambda: [
        "方案", "决策", "建议", "专业", "准确", "精确"
    ])


# ==================== 训练配置 ====================

@dataclass
class TrainingConfig:
    """专项全流程训练配置"""
    # 子模块 1: 底座预适配微调
    pretrain_learning_rate: float = 1e-5
    pretrain_batch_size: int = 8
    pretrain_epochs: int = 3
    pretrain_optimizer: str = "AdamW"
    pretrain_quantization: str = "INT4"
    
    # 训练数据集
    datasets: List[str] = field(default_factory=lambda: [
        "Alpaca_zh_lite",      # Alpaca 中文轻量化版
        "ShareGPT_filtered",   # ShareGPT 小额过滤版
        "GSM8K_lite",          # GSM8K 小额版
        "TIMEDIAL_lite",       # TIMEDIAL 时序推理
        "EpisodicMemory_lite"  # 情景记忆召回
    ])
    
    # 子模块 2: 在线终身学习
    online_learning_enabled: bool = True
    online_compute_overhead: float = 0.02  # 不超过 2%
    
    # 子模块 3: 离线记忆巩固
    offline_consolidation_enabled: bool = True
    offline_idle_threshold_s: int = 300
    offline_max_duration_s: int = 600


# ==================== 测评配置 ====================

@dataclass
class EvaluationConfig:
    """多维度全链路测评配置"""
    # 权重占比
    hippocampus_weight: float = 0.4  # 海马体记忆能力 40%
    base_capability_weight: float = 0.2  # 基础能力对标 20%
    reasoning_weight: float = 0.2  # 逻辑推理能力 20%
    edge_performance_weight: float = 0.1  # 端侧性能 10%
    self_loop_weight: float = 0.1  # 自闭环优化 10%
    
    # 海马体测评指标
    episodic_recall_accuracy: float = 0.95  # 情景记忆召回≥95%
    episodic_recall_completeness: float = 0.90  # 完整度≥90%
    pattern_separation_confusion: float = 0.03  # 混淆率≤3%
    long_sequence_retention: float = 0.90  # 长序列保持率≥90%
    temporal_logic_accuracy: float = 0.95  # 时序逻辑≥95%
    pattern_completion_rate: float = 0.85  # 模式补全≥85%
    anti_forgetting_rate: float = 0.95  # 抗遗忘≥95%
    cross_session_adaptation: float = 0.90  # 跨会话适配≥90%
    
    # 基础能力对标
    base_capability_retention: float = 0.95  # 不低于原生 95%
    
    # 逻辑推理提升
    reasoning_improvement: float = 0.60  # 超过原生 60%
    
    # 自闭环优化
    self_correction_accuracy: float = 0.90  # 自纠错≥90%
    hallucination_reduction: float = 0.70  # 幻觉下降≥70%


# ==================== 设备部署配置 ====================

@dataclass
class DeploymentConfig:
    """端侧部署配置"""
    # 目标设备
    target_devices: List[str] = field(default_factory=lambda: [
        "Android_Phone",
        "Raspberry_Pi_4B",
        "Raspberry_Pi_5"
    ])
    
    # 推理框架
    inference_framework: str = "MNN"  # 优先 MNN
    
    # 量化配置
    quantization_type: str = "INT4"
    quantization_method: str = "AWQ"  # Activation-aware Weight Quantization
    
    # 性能要求
    max_latency_ms: int = 10  # 单 token 延迟≤10ms
    max_memory_mb: int = 420
    stable_long_sequence: bool = True


# ==================== 全局配置 ====================

@dataclass
class BrainAIConfig:
    """类人脑双系统全闭环 AI架构全局配置"""
    model_name: str = "Qwen3.5-0.8B"
    model_path: str = "./models/Qwen3.5-0.8B"
    
    hard_constraints: HardConstraints = field(default_factory=HardConstraints)
    stdp: STDPConfig = field(default_factory=STDPConfig)
    hippocampus: HippocampusConfig = field(default_factory=HippocampusConfig)
    self_loop: SelfLoopConfig = field(default_factory=SelfLoopConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    
    # 随机种子
    seed: int = 42
    
    def set_seed(self):
        """设置全局随机种子"""
        import numpy as np
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)


# 默认配置实例
default_config = BrainAIConfig()
