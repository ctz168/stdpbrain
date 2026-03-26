"""
类人脑双系统全闭环 AI架构 - 核心配置文件
基于Qwen3.5-2B 底座模型
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch


# ==================== 核心刚性约束配置 ====================

@dataclass
class HardConstraints:
    """不可突破的硬性约束配置"""
    # 权重安全约束 - 优化版本：增强学习能力
    STATIC_WEIGHT_RATIO: float = 0.85  # 85% 静态基础权重比例
    DYNAMIC_WEIGHT_RATIO: float = 0.15  # 15% STDP动态增量权重比例（平衡稳定性和学习能力）
    
    # 端侧算力约束 - 针对 Qwen3.5-2B 优化
    MAX_MEMORY_MB: int = 1200  # INT8 量化后最大内存占用 (MB) - 2B模型需要更多内存
    REFRESH_PERIOD_MS: int = 10  # 10ms 刷新周期
    MAX_COMPUTE_OVERHEAD: float = 0.15  # 单周期算力开销不超过原生 15%
    
    # 窄窗口约束（类人脑注意力机制）
    NARROW_WINDOW_SIZE: int = 32  # 窄窗口大小：保留最近32个token（优化：从64降低，让KV压缩更早触发）
    ATTENTION_COMPLEXITY: str = "O(n×(W+K))"  # O(n×(W+K)) 复杂度，W=窗口大小，K=记忆锚点数
    NARROW_BAND_ENABLED: bool = True  # 启用窄带宽注意力
    MAX_CONTEXT_LENGTH: int = 512  # 最大上下文长度（超过后使用滑动窗口）
    NUM_MEMORY_ANCHORS: int = 5  # 记忆锚点数量（对应海马体容量）
    KV_CACHE_WARMUP: bool = True  # 启用KV cache预热，加速首token
    
    # ========== KV Cache 滑动窗口管理（新增）==========
    ENABLE_KV_SLIDING_WINDOW: bool = True  # 启用KV cache滑动窗口（实现无限上下文）
    KV_CACHE_WINDOW_SIZE: int = 32  # KV cache窗口大小（只保留最近32个token的KV）
    ENABLE_KV_HIPPOCAMPUS_INTEGRATION: bool = True  # 启用海马体KV存储（长期记忆）
    KV_EVICT_TO_HIPPOCAMPUS: bool = True  # 被释放的KV自动存储到海马体
    MAX_MEMORY_KV: int = 5  # 最大记忆KV数量（用于组合注意力）
    
    # 海马体内存约束 - 针对 2B 模型扩展
    HIPPOCAMPUS_MAX_MEMORY_MB: int = 5  # 情景记忆库最大 5MB（hidden_size 增加到 2048）


# ==================== STDP超参数配置 ====================

@dataclass
class STDPConfig:
    """STDP 时序可塑性权重更新配置 - 优化版本"""
    enabled: bool = True  # 是否启用 STDP 学习机制
    # 学习率 - 提升学习能力
    alpha_LTP: float = 0.025  # 权重增强学习率 (Long-Term Potentiation) - 从0.01提升到0.025
    beta_LTD: float = 0.02   # 权重减弱学习率 (Long-Term Depression) - 从0.008提升到0.02
    
    # 阈值
    update_threshold: float = 0.0005  # 最小更新阈值 - 降低以增强灵敏度
    weight_min: float = -1.0  # 权重下界
    weight_max: float = 1.0  # 权重上界
    
    # 时间窗口
    time_window_ms: int = 20  # STDP 时间窗口 (毫秒)
    decay_rate: float = 0.95  # 权重衰减率 - 从0.99降低到0.95，保留更多学习成果
    
    # 更新节点
    update_attention: bool = True  # 注意力层 STDP 更新
    update_ffn: bool = True  # FFN 层 STDP 更新
    update_self_eval: bool = True  # 自评判 STDP 更新
    update_hippocampus_gate: bool = True  # 海马体门控 STDP 更新
    
    # 噪声控制
    noise_ratio: float = 0.1  # 噪声注入比例 - 新增参数，控制随机性


# ==================== 海马体系统配置 ====================

@dataclass
class HippocampusConfig:
    """海马体记忆系统配置 - 优化版本"""
    # 特征编码 - 提升编码容量
    EC_feature_dim: int = 256  # 内嗅皮层编码维度 - 从64提升到256，增强特征表达能力
    DG_sparsity: float = 0.85  # 齿状回稀疏度 - 降低以提升记忆容量
    DG_orthogonalization: bool = True  # 模式分离正交化
    
    # 情景记忆存储
    CA3_memory_format: str = "ID+timestamp+skeleton+semantic_pointer+causal"
    CA3_max_capacity: int = 10000  # 最大记忆容量
    CA3_timestamp_precision_ms: int = 10  # 时间戳精度 10ms
    
    # 时序编码与门控 - 优化召回质量
    CA1_temporal_encoding: bool = True
    CA1_attention_gate: bool = True
    recall_topk: int = 3  # 每个周期召回 3 个记忆锚点 - 从2提升到3
    
    # 离线回放巩固
    SWR_enabled: bool = True
    SWR_idle_threshold_s: int = 300  # 空闲 5 分钟触发
    SWR_replay_frequency: float = 0.1  # 回放频率
    
    # 内存约束
    max_memory_bytes: int = 5 * 1024 * 1024  # 5MB (2B模型需要更大内存)
    use_cycle_buffer: bool = True  # 循环缓存
    
    # 召回阈值优化
    recall_threshold: float = 0.75  # 提升召回阈值，减少噪声记忆


# ==================== 自闭环优化系统配置 ====================

@dataclass
class SelfLoopConfig:
    """单智体自闭环优化系统配置"""
    # 模式 1: 自生成组合输出
    mode1_temperature_range: tuple = (0.8, 1.2)  # 官方推荐1.0，使用范围覆盖
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
        "方案", "决策", "建议", "专业", "准确", "精确",
        "制定", "计划", "规划", "分析", "评估", "策略"
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
    pretrain_quantization: str = "INT8"
    
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
    model_name: str = "Qwen3.5-2B"
    model_path: str = "./models/Qwen3.5-2B"
    
    # 量化配置（从 config.py 读取）
    quantization: str = "AUTO"  # 默认 AUTO 模式（GPU用INT8，CPU用FP32）
    QUANTIZATION: str = "AUTO"  # 兼容大写属性名
    
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
