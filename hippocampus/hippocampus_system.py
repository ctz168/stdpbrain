"""
海马体记忆系统 - 完整集成

将 EC、DG、CA3、CA1、SWR 五个子模块集成为完整的类脑海马体系统
提供统一的编码、召回、门控接口
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import time

from .ec_encoder import EntorhinalEncoder
from .dg_separator import DentateGyrusSeparator
from .ca3_memory import CA3EpisodicMemory
from .ca1_gate import CA1AttentionGate
from .swr_consolidation import SWRConsolidation


class HippocampusSystem(nn.Module):
    """
    完整的类人脑海马体系统
    
    整合五个核心子模块:
    - EC (内嗅皮层): 特征编码
    - DG (齿状回): 模式分离
    - CA3: 情景记忆存储 + 模式补全
    - CA1: 时序编码 + 注意力门控
    - SWR: 离线回放巩固
    
    输入：模型注意力层输出的 token 特征
    输出：1-2 个记忆锚点 + 注意力门控信号
    """
    
    def __init__(self, config, device: Optional[str] = None):
        super().__init__()
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 从配置加载参数
        hc_config = config.hippocampus
        
        # ========== 1. EC 内嗅皮层 - 特征编码 ==========
        self.ec_encoder = EntorhinalEncoder(
            input_dim=1024,              # Qwen3.5-0.8B hidden size (1024 for this variant)
            output_dim=hc_config.EC_feature_dim,
            sparsity=hc_config.DG_sparsity,
            freeze_encoder=True
        )
        
        # ========== 2. DG 齿状回 - 模式分离 ==========
        self.dg_separator = DentateGyrusSeparator(
            input_dim=hc_config.EC_feature_dim,
            output_dim=hc_config.EC_feature_dim * 2,  # 扩展到 128 维
            sparsity=hc_config.DG_sparsity,
            orthogonalization=hc_config.DG_orthogonalization
        )
        
        # ========== 3. CA3 情景记忆库 ==========
        self.ca3_memory = CA3EpisodicMemory(
            max_capacity=hc_config.CA3_max_capacity,
            feature_dim=hc_config.EC_feature_dim * 2,
            timestamp_precision_ms=hc_config.CA3_timestamp_precision_ms,
            recall_threshold=0.7,
            decay_rate=0.999
        )
        
        # ========== 4. CA1 注意力 gate ==========
        self.ca1_gate = CA1AttentionGate(
            feature_dim=hc_config.EC_feature_dim * 2,
            hidden_size=1024,             # Qwen3.5-0.8B hidden size
            recall_topk=hc_config.recall_topk,
            temporal_encoding=hc_config.CA1_temporal_encoding,
            gate_type="additive"
        )
        
        # ========== 5. SWR 离线回放巩固 ==========
        self.swr_consolidation = SWRConsolidation(
            config=config,
            hippocampus_module=self,
            idle_threshold_s=hc_config.SWR_idle_threshold_s,
            replay_frequency=hc_config.SWR_replay_frequency
        )
        
        # ========== 内存监控 ==========
        self.max_memory_bytes = hc_config.max_memory_bytes
        self.memory_usage_bytes = 0
        
        # ========== 性能优化: 缓存 ==========
        self._query_encoding_cache = {}  # 查询特征编码缓存
        self._memory_size_cache = 0  # 增量内存统计
        
        # ========== 周期计数器 ==========
        self.cycle_count = 0
    
    def encode(
        self,
        features: torch.Tensor,
        token_id: int,
        timestamp: int,
        context: Optional[List[dict]] = None,
        kv_features: Optional[Dict[str, torch.Tensor]] = None  # 新增: 用于窄带宽注意力的 KV 特征
    ) -> str:
        """
        记忆编码流程
        
        完整流程：Token 特征 → EC 编码 → DG 分离 → CA3 存储
        
        Args:
            features: Token 特征 [hidden_size] 或 [1, hidden_size]
            token_id: Token ID
            timestamp: 时间戳 (ms)
            context: 上下文信息
            kv_features: KV 特征字典 {'key': tensor, 'value': tensor}，用于窄带宽注意力
        
        Returns:
            memory_id: 生成的唯一记忆 ID
        """
        # ========== 1. 特征预处理 ==========
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        # ========== 2. EC 编码 ==========
        ec_code = self.ec_encoder.encode_single(features.squeeze(0))
        
        # ========== 3. DG 模式分离 ==========
        dg_output, memory_id = self.dg_separator.separate_and_id(ec_code)
        
        # ========== 4. 构建时序骨架 ==========
        temporal_skeleton = ""
        causal_links = []
        is_core = False
        content = ""
        
        if context and len(context) > 0:
            # 从前序上下文提取时序关系
            prev_tokens = [ctx.get('token_id', -1) for ctx in context]
            temporal_skeleton = f"{prev_tokens} -> {token_id}"
            
            # 简单因果推断 (实际应更复杂)
            causal_links.append(f"preceded_by_{prev_tokens[-1]}" if prev_tokens else "start")
            
            # 提取 is_core 和 content
            for ctx in context:
                if 'is_core' in ctx:
                    is_core = ctx['is_core']
                if 'content' in ctx:
                    content = ctx['content']
        
        # ========== 5. CA3 存储 ==========
        # 优先使用context中的semantic_pointer，否则使用token_id
        semantic_pointer = f"token_{token_id}"
        if context and len(context) > 0:
            # 从context中提取semantic_pointer
            for ctx in context:
                if 'semantic_pointer' in ctx and ctx['semantic_pointer']:
                    semantic_pointer = ctx['semantic_pointer']
                    break
        
        # ========== 5.5 提取 KV 特征（如果提供）==========
        key_features = None
        value_features = None
        if kv_features:
            key_features = kv_features.get('key', None)
            value_features = kv_features.get('value', None)
            if key_features is not None:
                key_features = key_features.detach().cpu()
            if value_features is not None:
                value_features = value_features.detach().cpu()
        
        self.ca3_memory.store(
            memory_id=memory_id,
            timestamp=timestamp,
            semantic_pointer=semantic_pointer,
            temporal_skeleton=temporal_skeleton,
            causal_links=causal_links,
            dg_features=dg_output.detach().cpu(),
            is_core=is_core,
            content=content,
            key_features=key_features,    # 新增
            value_features=value_features  # 新增
        )
        
        # ========== 6. 更新内存使用 ==========
        self._update_memory_usage()
        
        self.cycle_count += 1
        
        return memory_id
    
    def recall(
        self,
        query_features: torch.Tensor,
        topk: int = 2,
        query_semantic: Optional[str] = None,
        query_timestamp: Optional[int] = None
    ) -> List[dict]:
        """
        记忆召回流程（性能优化版）
        
        优化点:
        - 缓存查询特征编码结果
        - 减少不必要的拷贝和转换
        """
        # ========== 1. 特征预处理 ==========
        if query_features.dim() == 1:
            query_features = query_features.unsqueeze(0)
        
        # ========== 优化: 检查编码缓存 ==========
        # 使用特征哈希作为缓存键
        cache_key = hash(query_features.tobytes() if hasattr(query_features, 'tobytes') else str(query_features.mean().item()))
        
        if cache_key in self._query_encoding_cache:
            ec_code, dg_features = self._query_encoding_cache[cache_key]
        else:
            # EC 编码
            ec_code = self.ec_encoder.encode_single(query_features.squeeze(0))
            # DG 分离
            dg_features = self.dg_separator.forward(ec_code)
            # 缓存结果（限制缓存大小）
            if len(self._query_encoding_cache) < 100:
                self._query_encoding_cache[cache_key] = (ec_code, dg_features)
        
        # ========== 3. CA3 模式补全召回 ==========
        recall_threshold = getattr(self.config.hippocampus, 'recall_threshold', 0.75)
        
        memories = self.ca3_memory.complete_pattern(
            partial_cue={
                'features': dg_features,
                'semantic': query_semantic,
                'timestamp': query_timestamp
            },
            topk=topk * 3  # 多召回一些，后续过滤
        )
        
        # ========== 4. 过滤低质量记忆 ==========
        if hasattr(self.config.hippocampus, 'recall_threshold'):
            filtered_memories = []
            for mem in memories:
                if hasattr(mem, 'activation_strength') and mem.activation_strength < recall_threshold:
                    continue
                filtered_memories.append(mem)
            memories = filtered_memories[:topk * 2]
        
        # ========== 5. CA1 时序排序 ==========
        current_timestamp = int(time.time() * 1000)
        sorted_memories = self.ca1_gate.sort_by_temporal(
            memories=[mem.to_dict() for mem in memories],
            current_timestamp=current_timestamp,
            topk=topk
        )
        
        # ========== 6. 添加记忆本身的 DG 和 KV 特征到返回结果 ==========
        memory_dg_features = {}
        memory_kv_features = {}
        for mem in memories:
            if hasattr(mem, 'dg_features') and mem.dg_features is not None:
                memory_dg_features[mem.memory_id] = mem.dg_features
            # 提取 KV 特征（用于窄带宽注意力）
            if hasattr(mem, 'key_features') and mem.key_features is not None:
                memory_kv_features[mem.memory_id] = {
                    'key': mem.key_features,
                    'value': mem.value_features
                }
        
        for mem_dict in sorted_memories:
            mem_id = mem_dict.get('memory_id', '')
            
            # DG 特征
            if mem_id in memory_dg_features:
                try:
                    mem_dict['dg_features'] = memory_dg_features[mem_id].detach().cpu().numpy().tolist()
                except:
                    mem_dict['dg_features'] = None
            else:
                try:
                    mem_dict['dg_features'] = dg_features.detach().cpu().numpy().tolist()
                except:
                    mem_dict['dg_features'] = None
            
            # KV 特征（用于窄带宽注意力）
            if mem_id in memory_kv_features:
                try:
                    kv = memory_kv_features[mem_id]
                    mem_dict['key_features'] = kv['key'].detach().cpu().numpy().tolist() if kv['key'] is not None else None
                    mem_dict['value_features'] = kv['value'].detach().cpu().numpy().tolist() if kv['value'] is not None else None
                except:
                    mem_dict['key_features'] = None
                    mem_dict['value_features'] = None
            
            if 'content' not in mem_dict and mem_id in self.ca3_memory.memories:
                mem_dict['content'] = self.ca3_memory.memories[mem_id].content
            if 'is_core' not in mem_dict and mem_id in self.ca3_memory.memories:
                mem_dict['is_core'] = self.ca3_memory.memories[mem_id].is_core
        
        return sorted_memories

    def get_state(self) -> dict:
        """获取海马体系统的完整状态"""
        return {
            'ca3_state': self.ca3_memory.get_state(),
            'ec_encoder_state': self.ec_encoder.state_dict(),
            'dg_separator_state': self.dg_separator.state_dict(),
            'ca1_gate_state': self.ca1_gate.state_dict(),
            'cycle_count': self.cycle_count
        }

    def set_state(self, state: dict):
        """从状态字典恢复海马体系统"""
        if 'ca3_state' in state:
            self.ca3_memory.set_state(state['ca3_state'])
        if 'ec_encoder_state' in state:
            self.ec_encoder.load_state_dict(state['ec_encoder_state'])
        if 'dg_separator_state' in state:
            self.dg_separator.load_state_dict(state['dg_separator_state'])
        if 'ca1_gate_state' in state:
            self.ca1_gate.load_state_dict(state['ca1_gate_state'])
        self.cycle_count = state.get('cycle_count', 0)

    def generate_attention_gate(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        memory_anchors: List[dict]
    ) -> torch.Tensor:
        """
        生成注意力门控信号
        
        Args:
            query: Query 向量 [batch, seq, hidden]
            key: Key 向量 [batch, seq, hidden]
            memory_anchors: 记忆锚点列表
        
        Returns:
            gate_mask: 注意力门控掩码
        """
        return self.ca1_gate.forward(query, key, memory_anchors)
    
    def update_memory_strength(self, memory_id: str, delta: float):
        """
        更新记忆激活强度 (STDP 更新接口)
        
        Args:
            memory_id: 记忆 ID
            delta: 强度变化量
        """
        self.ca3_memory.update_memory_strength(memory_id, delta)
    
    def prune_weak_memories(self, threshold: float = 0.3) -> int:
        """
        修剪弱记忆
        
        Args:
            threshold: 强度阈值
        
        Returns:
            pruned_count: 修剪的记忆数量
        """
        count = self.ca3_memory.prune_weak_memories(threshold)
        return count
    
    def _calculate_current_usage(self) -> int:
        """计算当前内存使用的辅助方法"""
        import sys
        total_size = 0
        
        # 计算CA3记忆的实际内存占用
        for memory_id, memory in self.ca3_memory.memories.items():
            # 基础数据结构大小
            memory_size = sys.getsizeof(memory_id)
            
            # 特征向量大小
            if hasattr(memory, 'features') and memory.features is not None:
                memory_size += memory.features.element_size() * memory.features.nelement()
            
            # 文本数据大小
            if hasattr(memory, 'semantic_pointer'):
                memory_size += sys.getsizeof(memory.semantic_pointer)
            
            if hasattr(memory, 'context'):
                memory_size += sys.getsizeof(str(memory.context))
            
            total_size += memory_size
        
        # 加上EC和DG组件的内存
        if hasattr(self, 'ec_encoder') and self.ec_encoder is not None:
            for param in self.ec_encoder.parameters():
                total_size += param.element_size() * param.nelement()
        
        if hasattr(self, 'dg_separator') and self.dg_separator is not None:
            for param in self.dg_separator.parameters():
                total_size += param.element_size() * param.nelement()
        
        return total_size

    def _update_memory_usage(self):
        """增量更新内存使用统计"""
        # 只在有新记忆时才更新
        current_count = len(self.ca3_memory.memories)
        if not hasattr(self, '_last_memory_count'):
            self._last_memory_count = 0
        
        if current_count > self._last_memory_count:
            # 增量计算新增记忆的大小
            delta = current_count - self._last_memory_count
            avg_size = 1000  # 估算每个记忆的平均大小（字节）
            self.memory_usage_bytes += delta * avg_size
            self._last_memory_count = current_count
        
        # 每 100 次操作后做一次精确统计
        if self.cycle_count % 100 == 0:
            self.memory_usage_bytes = self._calculate_current_usage()
        
        # 检查是否超出限制
        if self.memory_usage_bytes > self.max_memory_bytes:
            excess_ratio = 1 - (self.max_memory_bytes / self.memory_usage_bytes)
            threshold = 0.3 + excess_ratio * 0.5
            self.prune_weak_memories(threshold)
            self.memory_usage_bytes = self._calculate_current_usage()
            self._last_memory_count = len(self.ca3_memory.memories)
    
    def start_swr_monitoring(self):
        """启动 SWR 离线回放监控"""
        self.swr_consolidation.start_monitoring()
    
    def stop_swr_monitoring(self):
        """停止 SWR 监控"""
        self.swr_consolidation.stop_monitoring()
    
    def trigger_swr_consolidation(self):
        """手动触发海马体 SWR 记忆巩固"""
        if hasattr(self, 'swr_consolidation'):
            self.swr_consolidation.trigger_manual_consolidation()

    def record_activity(self):
        """记录用户活动 (重置空闲计时器)"""
        self.swr_consolidation.record_activity()
    
    def add_replay_sequence(
        self,
        sequence_id: str,
        memories: List[dict],
        reward_signal: float
    ):
        """添加回放序列"""
        self.swr_consolidation.add_replay_sequence(
            sequence_id=sequence_id,
            memories=memories,
            reward_signal=reward_signal
        )
    
    def forward(
        self,
        features: torch.Tensor,
        mode: str = "recall",
        **kwargs
    ):
        """
        前向传播
        
        Args:
            features: 输入特征
            mode: 执行模式 ("encode" | "recall" | "gate")
            **kwargs: 额外参数
        
        Returns:
            根据模式返回不同结果
        """
        if mode == "encode":
            return self.encode(
                features=features,
                token_id=kwargs.get('token_id', 0),
                timestamp=kwargs.get('timestamp', int(time.time() * 1000)),
                context=kwargs.get('context')
            )
        
        elif mode == "recall":
            return self.recall(
                query_features=features,
                topk=kwargs.get('topk', 2),
                query_semantic=kwargs.get('query_semantic'),
                query_timestamp=kwargs.get('query_timestamp')
            )
        
        elif mode == "gate":
            return self.generate_attention_gate(
                query=kwargs.get('query'),
                key=kwargs.get('key'),
                memory_anchors=features  # 这里 features 是记忆锚点
            )
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def get_stats(self) -> dict:
        """获取系统统计信息"""
        return {
            'cycle_count': self.cycle_count,
            'num_memories': len(self.ca3_memory.memories),
            'memory_usage_mb': self.memory_usage_bytes / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'ca3_stats': self.ca3_memory.get_stats(),
            'swr_stats': self.swr_consolidation.get_stats(),
            'device': self.device
        }
    
    def reset(self):
        """重置系统状态"""
        self.cycle_count = 0
        self.ca3_memory.memories.clear()
        self.ca3_memory.time_index.clear()
        self.ca3_memory.semantic_index.clear()
        self._update_memory_usage()
    
    # ========== KV Cache 专用方法 (用于滑动窗口管理) ==========
    
    def store_kv_as_memory(
        self,
        kv_features: Dict[str, Any],
        context_text: str
    ) -> str:
        """
        将KV cache作为记忆存储
        
        用于滑动窗口管理器，将被释放的KV存储到海马体
        
        Args:
            kv_features: KV特征字典，包含:
                - key_features: Key特征向量
                - value_features: Value特征向量
                - num_layers: 层数
                - seq_len: 序列长度
                - timestamp: 时间戳
            context_text: 上下文文本
        
        Returns:
            memory_id: 生成的记忆ID
        """
        try:
            # 提取KV特征
            key_features = kv_features.get('key_features', [])
            value_features = kv_features.get('value_features', [])
            num_layers = kv_features.get('num_layers', 0)
            seq_len = kv_features.get('seq_len', 0)
            timestamp = kv_features.get('timestamp', int(time.time() * 1000))
            
            if not key_features or not value_features:
                logger.warning("[Hippocampus] KV特征为空，跳过存储")
                return ""
            
            # 使用KV特征作为记忆特征
            # 将特征向量转换为tensor
            key_tensor = torch.tensor(key_features, dtype=torch.float32, device=self.device)
            
            if key_tensor.dim() == 1:
                key_tensor = key_tensor.unsqueeze(0)
            
            # 使用key特征作为记忆编码的输入
            # EC编码
            ec_code = self.ec_encoder.encode_single(key_tensor.squeeze(0))
            
            # DG模式分离
            dg_output, memory_id = self.dg_separator.separate_and_id(ec_code)
            
            # 构建记忆上下文
            context = [{
                'content': context_text,
                'type': 'kv_memory',
                'num_layers': num_layers,
                'seq_len': seq_len,
                'is_core': False,
                'semantic_pointer': f"kv_{timestamp}"
            }]
            
            # 存储到CA3
            self.ca3_memory.store(
                memory_id=memory_id,
                timestamp=timestamp,
                semantic_pointer=f"kv_{timestamp}",
                temporal_skeleton="",
                causal_links=[],
                dg_features=dg_output.detach().cpu(),
                is_core=False,
                content=context_text,
                key_features=key_tensor.detach().cpu() if isinstance(key_tensor, torch.Tensor) else key_features,
                value_features=torch.tensor(value_features, dtype=torch.float32).detach().cpu() if isinstance(value_features, list) else value_features
            )
            
            logger.debug(
                f"[Hippocampus] KV已存储为记忆: "
                f"memory_id={memory_id}, "
                f"seq_len={seq_len}, "
                f"layers={num_layers}"
            )
            
            return memory_id
            
        except Exception as e:
            logger.warning(f"[Hippocampus] 存储KV记忆失败: {e}")
            return ""
    
    def recall_kv(
        self,
        query_features: torch.Tensor,
        topk: int = 2
    ) -> List[Dict[str, Any]]:
        """
        召回相关的KV记忆
        
        用于滑动窗口管理器，从海马体检索相关的KV特征
        
        Args:
            query_features: 查询特征
            topk: 返回的记忆数量
        
        Returns:
            kv_memories: KV记忆列表，每个元素包含:
                - memory_id: 记忆ID
                - kv_features: KV特征字典
                - content: 上下文文本
                - similarity: 相似度
        """
        try:
            # 使用常规召回方法
            memories = self.recall(
                query_features=query_features,
                topk=topk * 2  # 多召回一些，后续过滤
            )
            
            # 过滤出包含KV特征的记忆
            kv_memories = []
            for mem in memories:
                # 检查是否是KV记忆
                if mem.get('type') == 'kv_memory' or 'kv_features' in mem:
                    kv_memories.append({
                        'memory_id': mem.get('memory_id', ''),
                        'kv_features': {
                            'key_features': mem.get('key_features'),
                            'value_features': mem.get('value_features'),
                            'num_layers': mem.get('num_layers', 0),
                            'seq_len': mem.get('seq_len', 0)
                        },
                        'content': mem.get('content', ''),
                        'similarity': mem.get('activation_strength', 0.0)
                    })
            
            # 返回topk个
            return kv_memories[:topk]
            
        except Exception as e:
            logger.warning(f"[Hippocampus] 召回KV记忆失败: {e}")
            return []
