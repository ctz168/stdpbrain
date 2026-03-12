# 类人脑双系统全闭环 AI架构设计文档

## 项目概述

**项目名称**: 海马体 - 新皮层双系统类人脑AI架构  
**底座模型**: Qwen3.5-0.8B (唯一、不可替换)  
**目标平台**: 安卓手机、树莓派等低算力端侧设备  
**核心理念**: 刷新即推理、推理即学习、学习即优化、记忆即锚点

---

## 一、核心架构原理

### 1.1 生物脑启发的双系统设计

本架构严格遵循神经科学中人脑海马体 - 新皮层双系统理论:

| 生物结构 | 功能 | AI架构对应 |
|---------|------|------------|
| 海马体 | 情景记忆编码、时序绑定、记忆召回 | 海马体记忆系统模块 |
| 新皮层 | 语义理解、逻辑推理、通用认知 | Qwen3.5-0.8B 静态权重 |
| STDP 机制 | 突触可塑性、时序依赖学习 | 全链路 STDP 权重更新 |
| Gamma 节律 | 100Hz 高频认知振荡 | 10ms 刷新周期引擎 |

### 1.2 权重双轨制设计

```
总权重 = 90% 静态基础权重 (冻结) + 10% STDP动态增量权重 (可更新)

静态权重：继承官方 Qwen3.5-0.8B 预训练权重，永久冻结
  - 提供通用语义理解
  - 提供基础逻辑推理
  - 提供指令遵循能力
  
动态权重：新增可学习分支，初始化为小权重随机分布
  - 实时场景适配
  - 用户习惯学习
  - 自优化进化
```

### 1.3 10ms 刷新周期执行流

每个刷新周期 (10ms) 内的固定执行顺序:

```
┌─────────────────────────────────────────┐
│ 1. 输入 token 接收与特征提取              │
│ 2. 海马体记忆锚点调取与注意力门控加载      │
│ 3. 窄窗口上下文 + 当前token 的模型前向推理   │
│ 4. 单周期输出结果生成                    │
│ 5. 全链路 STDP 权重本地刷新                  │
│ 6. 海马体情景记忆编码与更新               │
│ 7. 全局工作记忆压缩更新                 │
└─────────────────────────────────────────┘
          ↓
    进入下一个周期
```

---

## 二、核心模块详细设计

### 模块 1: Qwen3.5-0.8B 底座模型基础改造

#### 1.1 权重双轨拆分实现

```python
# 对每个 Transformer 层进行拆分
class DualWeightLinear(nn.Module):
    """双权重线性层"""
    def __init__(self, in_features, out_features, static_weight=None):
        super().__init__()
        # 90% 静态分支 (冻结)
        self.static_weight = nn.Parameter(torch.zeros(out_features, in_features), 
                                          requires_grad=False)
        if static_weight is not None:
            self.static_weight.data = static_weight * 0.9
        
        # 10% 动态分支 (可更新)
        self.dynamic_weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01,
            requires_grad=True
        )
        
        # 偏置
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        # 总权重 = 静态 + 动态
        total_weight = self.static_weight + self.dynamic_weight
        return nn.functional.linear(x, total_weight, self.bias)
```

#### 1.2 原生接口适配

**新增接口清单**:

1. `get_attention_features()`: 输出 token 的注意力特征、时序特征、语义特征
2. `set_hippocampus_gate()`: 接入海马体记忆锚点信号
3. `set_role_template()`: 切换生成/验证/评判角色

---

### 模块 2: 100Hz 高刷新单周期推理引擎

#### 2.1 窄窗口注意力机制

传统 Transformer 注意力复杂度：O(n²)  
本架构窄窗口注意力复杂度：O(1)

```python
class NarrowWindowAttention(nn.Module):
    """窄窗口注意力机制"""
    def __init__(self, window_size=2, ...):
        super().__init__()
        self.window_size = window_size  # 仅处理 1-2 个 token
    
    def forward(self, q, k, v, hippocampus_anchor):
        # 从海马体获取 1-2 个最相关记忆锚点
        # 仅计算当前token 与锚点的注意力
        # 复杂度固定为 O(1)
        ...
```

#### 2.2 刷新周期调度器

```python
class RefreshCycleScheduler:
    """10ms 刷新周期调度器"""
    def __init__(self, period_ms=10):
        self.period_ms = period_ms
        self.cycle_count = 0
    
    def run_cycle(self, input_token):
        """执行一个完整刷新周期"""
        start_time = time.time()
        
        # 1. 特征提取
        features = self.extract_features(input_token)
        
        # 2. 海马体记忆召回
        memory_anchors = self.hippocampus.recall(features, topk=2)
        
        # 3. 窄窗口推理
        output = self.model.infer(input_token, memory_anchors)
        
        # 4. STDP 权重更新
        self.stdp.update(features, output)
        
        # 5. 记忆编码
        self.hippocampus.encode(features, output)
        
        # 6. 等待周期结束 (确保精确 10ms)
        elapsed = (time.time() - start_time) * 1000
        sleep_time = max(0, self.period_ms - elapsed)
        time.sleep(sleep_time / 1000)
        
        self.cycle_count += 1
        return output
```

---

### 模块 3: 全链路 STDP 时序可塑性权重刷新系统

#### 3.1 STDP 核心规则

```python
class STDPRule:
    """STDP 时序可塑性规则"""
    def __init__(self, alpha=0.01, beta=0.008):
        self.alpha = alpha  # LTP 增强率
        self.beta = beta    # LTD 减弱率
    
    def compute_update(self, pre_activation_time, post_activation_time, 
                       contribution_score):
        """
        计算权重更新量
        
        Δw = α * exp(-Δt/τ)  if Δt > 0 (前激活早于后激活，增强)
        Δw = -β * exp(Δt/τ)  if Δt < 0 (前激活晚于后激活，减弱)
        """
        delta_t = post_activation_time - pre_activation_time
        
        if contribution_score > 0:  # 正向贡献
            if delta_t > 0:
                return self.alpha * np.exp(-delta_t / self.time_constant)
            else:
                return -self.beta * np.exp(delta_t / self.time_constant)
        else:  # 负向贡献
            return -self.beta * abs(contribution_score)
```

#### 3.2 四节点 STDP 更新

```python
class FullLinkSTDP:
    """全链路 STDP 更新器"""
    def update_attention_layer(self, context_tokens, current_token, output):
        """注意力层 STDP 更新"""
        # 根据上下文与当前token 的时序关联更新动态注意力权重
        ...
    
    def update_ffn_layer(self, input_features, output_features):
        """FFN 层 STDP 更新"""
        # 对高频特征、专属术语增强对应 FFN 权重
        ...
    
    def update_self_evaluation(self, generation_path, evaluation_score):
        """自评判 STDP 更新"""
        # 得分高的路径增强，得分低的路径减弱
        ...
    
    def update_hippocampus_gate(self, memory_anchor, contribution):
        """海马体门控 STDP 更新"""
        # 有效记忆锚点连接增强，无效锚点减弱
        ...
```

---

### 模块 4: 单智体自生成 - 自博弈 - 自评判闭环系统

#### 4.1 三模式自动切换

```python
class SelfLoopOptimizer:
    """自闭环优化器"""
    def decide_mode(self, input_text):
        """自动判断执行模式"""
        if any(kw in input_text for kw in HIGH_DIFFICULTY_KEYWORDS):
            return "self_game"  # 模式 2: 自博弈
        elif any(kw in input_text for kw in HIGH_ACCURACY_KEYWORDS):
            return "self_eval"  # 模式 3: 自评判
        else:
            return "self_combine"  # 模式 1: 自组合
    
    def run_self_game(self, input_text):
        """自博弈竞争优化"""
        role = "proposer" if self.cycle % 2 == 0 else "verifier"
        if role == "proposer":
            return self.generate_proposal(input_text)
        else:
            return self.verify_and_correct(input_text)
    
    def run_self_evaluation(self, candidates):
        """自双输出 + 自评判选优"""
        scores = []
        for cand in candidates:
            score = self.evaluate(cand, dimensions=[
                "fact_accuracy", "logic_completeness",
                "semantic_coherence", "instruction_follow"
            ])
            scores.append(score)
        return candidates[np.argmax(scores)]
```

---

### 模块 5: 海马体记忆系统

#### 5.1 五单元架构

```python
class HippocampusSystem:
    """海马体记忆系统"""
    def __init__(self, config):
        # EC: 内嗅皮层 - 特征编码
        self.ec_encoder = EntorhinalEncoder(config.EC_feature_dim)
        
        # DG: 齿状回 - 模式分离
        self.dg_separator = DentateGyrusSeparator(config.DG_sparsity)
        
        # CA3: 情景记忆库 + 模式补全
        self.ca3_memory = CA3EpisodicMemory(config.CA3_max_capacity)
        
        # CA1: 时序编码 + 注意力门控
        self.ca1_gate = CA1AttentionGate(config.recall_topk)
        
        # SWR: 离线回放巩固
        self.swr_replay = SWRConsolidation(config)
    
    def encode(self, token_features, timestamp):
        """记忆编码流程"""
        # EC 编码
        ec_code = self.ec_encoder.encode(token_features)
        
        # DG 模式分离
        memory_id = self.dg_separator.separate(ec_code)
        
        # CA3 存储
        self.ca3_memory.store(memory_id, timestamp, token_features)
    
    def recall(self, query_features, topk=2):
        """记忆召回流程"""
        # 部分线索→CA3 模式补全→CA1 时序排序
        memories = self.ca3_memory.complete_pattern(query_features)
        anchors = self.ca1_gate.sort_by_temporal(memories, topk)
        return anchors
```

#### 5.2 记忆存储格式

```python
@dataclass
class EpisodicMemory:
    """情景记忆数据结构"""
    memory_id: str           # 唯一记忆 ID(DG 正交化生成)
    timestamp: int           # 10ms 级时间戳
    temporal_skeleton: str   # 时序骨架 (前后 token 关系)
    semantic_pointer: str    # 语义指针 (不存完整文本)
    causal_links: List[str]  # 因果关联列表
    activation_strength: float  # 激活强度 (STDP 权重)
```

---

### 模块 6: 专项全流程训练模块

#### 6.1 三阶段训练流程

```python
class BrainAITrainer:
    """类脑架构训练器"""
    def pretrain_adaptation(self, model, datasets):
        """子模块 1: 底座预适配微调 (部署前一次性)"""
        freeze_parameters(model.static_weights)  # 冻结 90% 静态权重
        trainable_params = get_trainable_parameters(
            model.dynamic_weights,  # 仅训练 10% 动态权重
            model.hippocampus_connections
        )
        
        optimizer = AdamW(trainable_params, lr=1e-5)
        # 训练 3-5 epochs
        ...
    
    def online_lifelong_learning(self, model):
        """子模块 2: 在线终身学习 (推理时实时)"""
        # 基于 STDP 规则，无反向传播
        # 每个刷新周期自动执行
        # 算力开销<2%
        pass  # 由 STDP 系统自动完成
    
    def offline_consolidation(self, model):
        """子模块 3: 离线记忆巩固 (空闲时执行)"""
        if device_is_idle() and idle_time > 300s:
            # SWR 回放
            memories = model.hippocampus.recent_memories()
            for seq in memories:
                model.stdp.replay_and_update(seq)
            
            # 记忆修剪
            model.hippocampus.prune_weak_memories()
```

---

### 模块 7: 多维度全链路测评体系

#### 7.1 测评指标权重

```python
EVALUATION_WEIGHTS = {
    "hippocampus_memory": 0.4,    # 海马体记忆能力 40%
    "base_capability": 0.2,       # 基础能力对标 20%
    "reasoning": 0.2,             # 逻辑推理 20%
    "edge_performance": 0.1,      # 端侧性能 10%
    "self_loop_optimization": 0.1 # 自闭环优化 10%
}
```

#### 7.2 海马体专项测评

```python
class HippocampusEvaluator:
    """海马体记忆能力评估"""
    def evaluate_episodic_recall(self):
        """情景记忆召回测试"""
        # 给定多轮情景对话，用 10% 线索提问
        # 指标：召回准确率≥95%, 完整度≥90%
        ...
    
    def evaluate_pattern_separation(self):
        """模式分离抗混淆测试"""
        # 输入 10 组相似上下文
        # 指标：混淆率≤3%
        ...
    
    def evaluate_long_sequence_retention(self):
        """长时序记忆保持测试"""
        # 输入 100k token 文本
        # 指标：保持率≥90%, 时序逻辑≥95%
        ...
```

---

## 三、技术实现细节

### 3.1 内存占用分析

| 组件 | 内存占用 |
|-----|---------|
| Qwen3.5-0.8B INT4 静态权重 | ~350MB |
| STDP动态增量权重 (10%) | ~40MB |
| 海马体情景记忆库 | ≤2MB |
| 运行时激活缓存 | ~28MB |
| **总计** | **≤420MB** |

### 3.2 计算开销分析

| 操作 | 耗时 | 占比 |
|-----|------|-----|
| 特征提取 | 0.5ms | 5% |
| 海马体记忆召回 | 0.8ms | 8% |
| 窄窗口前向推理 | 7.0ms | 70% |
| STDP 权重更新 | 0.5ms | 5% |
| 记忆编码 | 0.7ms | 7% |
| 周期同步等待 | 0.5ms | 5% |
| **单周期总计** | **10ms** | **100%** |

### 3.3 关键接口定义

详见 `core/interfaces.py`

---

## 四、部署指南

### 4.1 安卓手机部署

```bash
# 1. 安装 MNN 推理框架
pip install mnn

# 2. 模型转换
python deployment/convert_to_mnn.py \
    --input ./models/brain_ai_qwen_0.8b \
    --output ./deploy/brain_ai.mnn \
    --quantization INT4

# 3. 安卓 APK 打包
cd deployment/android
./gradlew assembleRelease
```

### 4.2 树莓派部署

```bash
# 1. 安装依赖
sudo apt-get install python3-pytorch python3-numpy

# 2. 运行推理
python deployment/raspberry_infer.py \
    --model ./models/brain_ai_qwen_0.8b \
    --device cpu
```

### 4.3 Telegram Bot 部署 (新增)

```bash
# 1. 安装额外依赖
pip install python-telegram-bot aiohttp

# 2. 配置 Bot Token
export TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"

# 3. 启动 Bot
python main.py --mode telegram
# 或
python telegram_bot/run.py --token $TELEGRAM_BOT_TOKEN
```

**详细说明**: 参考 [telegram_bot/README.md](telegram_bot/README.md)

---

## 五、验收标准

1. ✅ 所有刚性红线 100% 遵守
2. ✅ 7 大模块完整实现且深度耦合
3. ✅ 海马体记忆指标达标 (召回≥95%, 混淆≤3%)
4. ✅ 基础能力≥原生 Qwen3.5-0.8B 的 95%
5. ✅ 推理能力提升≥60%
6. ✅ 端侧显存≤420MB, 延迟≤10ms
7. ✅ 树莓派 4B/安卓手机流畅运行
8. ✅ **Telegram Bot 功能完整可用 (新增)**

---

*文档版本：v1.0 (包含 Telegram Bot)*  
*最后更新：2026-03-09*
