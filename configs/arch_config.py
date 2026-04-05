"""
类人脑双系统全闭环 AI架构 - 核心配置文件
基于 Qwen3.5-0.8B 底座模型（动态读取 hidden_size）
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
    
    # 端侧算力约束 - 针对 0.8B 模型优化
    MAX_MEMORY_MB: int = 800  # INT8 量化后最大内存占用 (MB)
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
    ENABLE_KV_SLIDING_WINDOW: bool = False  # 暂时禁用KV cache滑动窗口（避免裁剪上下文）
    KV_CACHE_WINDOW_SIZE: int = 512  # KV cache窗口大小（增大到512，保持长对话稳定性）
    ENABLE_KV_HIPPOCAMPUS_INTEGRATION: bool = True  # 启用海马体KV存储（长期记忆）
    KV_EVICT_TO_HIPPOCAMPUS: bool = True  # 被释放的KV自动存储到海马体
    MAX_MEMORY_KV: int = 5  # 最大记忆KV数量（用于组合注意力）
    
    # 海马体内存约束 - 针对 0.8B 模型
    HIPPOCAMPUS_MAX_MEMORY_MB: int = 5  # 情景记忆库最大 5MB


# ==================== STDP超参数配置 ====================

@dataclass
class STDPConfig:
    """STDP 时序可塑性权重更新配置 - 优化版本"""
    enabled: bool = True  # 重新启用 STDP（降低学习率）
    # 学习率 - 大幅降低以避免干扰
    alpha_LTP: float = 0.001  # 权重增强学习率 (从0.05降低到0.001)
    beta_LTD: float = 0.001   # 权重减弱学习率 (从0.04降低到0.001)
    
    # 阈值
    update_threshold: float = 0.0001  # 最小更新阈值 - 从0.0005降低到0.0001，增强灵敏度
    weight_min: float = -1.0  # 权重下界
    weight_max: float = 1.0  # 权重上界
    
    # 时间窗口
    # 致命BUG修复：原 time_window_ms=20（毫秒），但 CPU 上每个 token 生成需要 200-500ms，
    # 导致所有 token 对的时间差远大于 20ms，STDP 时间窗口掩码永远为 False，
    # 权重永远不会更新，dynamic_weight 永远为 0.000000。
    # 修复：增大到 30000ms（30秒），覆盖 CPU 和 GPU 环境。
    # STDP 语义：最近 30 秒内共同激活的 token 对才能产生权重更新，
    # 这意味着当前正在处理的对话上下文中，相关词汇的连接会被增强。
    time_window_ms: int = 30000  # STDP 时间窗口 (毫秒) - 从20增加到30000
    time_constant: float = 10000.0  # STDP 时间常数 (毫秒) - 随窗口增大
    decay_rate: float = 0.99  # 权重衰减率 - 恢复到0.99，避免学习成果过快流失
    
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
    
    # 情景记忆存储 - 增强容量
    CA3_memory_format: str = "ID+timestamp+skeleton+semantic_pointer+causal"
    CA3_max_capacity: int = 50000  # 最大记忆容量 - 从10000提升到50000（支持10个来回x1000字）
    CA3_timestamp_precision_ms: int = 10  # 时间戳精度 10ms
    CA3_semantic_pointer_max_len: int = 200  # 语义指针最大长度 - 新增，支持更长记忆摘要
    
    # 时序编码与门控 - 优化召回质量
    CA1_temporal_encoding: bool = True
    CA1_attention_gate: bool = True
    recall_topk: int = 5  # 每个周期召回 5 个记忆锚点 - 从3提升到5
    
    # 模糊记忆召回 - 新增
    fuzzy_recall_enabled: bool = True  # 启用模糊召回（记不清时可以找回）
    fuzzy_recall_threshold: float = 0.5  # 模糊召回阈值（低于此值尝试更广泛搜索）
    
    # 离线回放巩固
    SWR_enabled: bool = True
    SWR_idle_threshold_s: int = 300  # 空闲 5 分钟触发
    SWR_replay_frequency: float = 0.1  # 回放频率
    
    # 内存约束
    max_memory_bytes: int = 10 * 1024 * 1024  # 10MB (提升以支持更多记忆)
    use_cycle_buffer: bool = True  # 循环缓存
    
    # 召回阈值优化
    # BUG FIX: 从0.65降到0.30 - embedding相似度天然偏低，0.65会过滤掉大量有效记忆
    # 特别是在用户用不同措辞查询同一信息时（如"记得我名字吗" vs "我叫张三"），
    # 余弦相似度通常在0.3-0.5之间，0.65的阈值导致几乎所有语义召回都失败
    recall_threshold: float = 0.30
    
    # ========== 人类记忆增强参数 ==========
    # 艾宾浩斯遗忘曲线
    ebbinghaus_enabled: bool = True               # 启用艾宾浩斯遗忘曲线
    ebbinghaus_initial_strength: float = 0.5      # 初始记忆强度
    ebbinghaus_initial_stability: float = 60.0     # 初始稳定度（秒）
    
    # 情绪记忆
    emotional_memory_enabled: bool = True           # 启用情绪记忆增强
    emotional_memory_min_intensity: float = 0.3    # 最低情绪强度阈值
    
    # 语境依赖记忆
    context_dependent_enabled: bool = True        # 启用语境依赖记忆
    context_boost_max: float = 0.3                 # 语境加成上限
    
    # 间隔效应
    spacing_effect_enabled: bool = True            # 启用间隔效应
    spacing_max_memories: int = 10000              # 间隔效应管理容量
    
    # 记忆干扰
    interference_enabled: bool = True              # 启用记忆干扰
    interference_threshold: float = 0.6             # 干扰触发阈值
    
    # 记忆来源监控
    source_monitoring_enabled: bool = True         # 启用记忆来源监控
    source_confidence_decay: float = 0.001         # 来源置信度衰减速率

    # ========== 新增模块参数 ==========
    # 联想记忆网络
    associative_enabled: bool = True               # 启用联想记忆网络
    associative_max_per_memory: int = 500          # 每条记忆的最大关联数
    associative_semantic_threshold: float = 0.6     # 语义关联的余弦相似度阈值

    # 记忆重构引擎
    reconstruction_enabled: bool = True             # 启用记忆重构引擎

    # 梦境巩固系统
    dream_enabled: bool = True                     # 启用梦境巩固系统
    dream_idle_threshold_min: float = 30.0          # 空闲触发阈值（分钟）

    # 创造性洞察引擎
    creative_enabled: bool = True                  # 启用创造性洞察引擎

    # 情绪驱动思维整合
    emotional_thinking_enabled: bool = True        # 启用情绪驱动思维整合


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
    model_name: str = "Qwen3.5-0.8B"
    model_path: str = "./models/Qwen3.5-0.8B"
    model_hidden_size: int = 1024  # Qwen3.5-0.8B hidden_size（动态读取，仅在模型加载前作为默认值）
    vocab_size: int = 248320  # Qwen3.5-0.8B vocab_size（供 STDP engine 等模块读取）
    
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
        import random
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)


# 默认配置实例
default_config = BrainAIConfig()
