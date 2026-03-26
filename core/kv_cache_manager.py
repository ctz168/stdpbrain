"""
KV Cache 滑动窗口管理器

功能:
- 实现KV cache的滑动窗口机制，只保留最近N个token
- 自动释放超出窗口的KV cache
- 支持将被释放的KV存储到海马体
- 实现真正的"工作记忆(窗口KV) + 长期记忆(海马体)"架构

类人脑对应:
- 工作记忆: 前额叶皮层，容量有限（7±2个信息块）
- 长期记忆: 海马体，无限容量
- 注意力窗口: 窄带宽，只关注最近32个token
"""

import torch
from typing import List, Tuple, Optional, Dict, Any
import logging
import time

logger = logging.getLogger(__name__)


class KVCacheManager:
    """
    KV Cache滑动窗口管理器
    
    核心功能:
    1. 滑动窗口：只保留最近window_size个token的KV
    2. KV释放：自动释放超出窗口的KV
    3. 记忆存储：支持将释放的KV存储到海马体
    4. 组合注意力：窗口KV + 海马体召回的KV
    
    实现真正的无限上下文：
    - 内存占用: O(1) 固定（只保存窗口大小）
    - 上下文长度: 无限（海马体存储历史）
    - 生成速度: O(1) 恒定
    """
    
    def __init__(
        self,
        window_size: int = 32,
        enable_hippocampus: bool = True,
        max_memory_kv: int = 5
    ):
        """
        初始化KV Cache管理器
        
        Args:
            window_size: 窗口大小（默认32，类人脑工作记忆容量）
            enable_hippocampus: 是否启用海马体存储
            max_memory_kv: 最大记忆KV数量（用于组合注意力）
        """
        self.window_size = window_size
        self.enable_hippocampus = enable_hippocampus
        self.max_memory_kv = max_memory_kv
        
        # 统计信息
        self.total_tokens_processed = 0
        self.total_kv_evicted = 0
        self.total_kv_stored_to_hippocampus = 0
        
        # 被释放的KV缓存（临时存储，用于后续存储到海马体）
        self._evicted_kv_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        
        logger.info(
            f"[KVCacheManager] 初始化完成: "
            f"window_size={window_size}, "
            f"enable_hippocampus={enable_hippocampus}, "
            f"max_memory_kv={max_memory_kv}"
        )
    
    def trim_kv_cache(
        self,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
        current_token_text: Optional[str] = None,
        hippocampus: Optional[Any] = None
    ) -> Tuple[
        Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
        Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    ]:
        """
        修剪KV cache，只保留窗口大小
        
        Args:
            past_key_values: KV cache元组 ((K, V), (K, V), ...)
            current_token_text: 当前token文本（用于海马体存储）
            hippocampus: 海马体实例（用于存储释放的KV）
        
        Returns:
            trimmed_kv: 修剪后的KV cache
            evicted_kv: 被释放的KV列表
        """
        if past_key_values is None:
            return None, None
        
        # 获取当前KV长度
        first_layer_k = past_key_values[0][0]
        current_len = first_layer_k.shape[2]
        
        self.total_tokens_processed += 1
        
        # 如果长度小于窗口大小，不修剪
        if current_len <= self.window_size:
            return past_key_values, None
        
        logger.debug(
            f"[KVCacheManager] KV长度={current_len}, "
            f"窗口大小={self.window_size}, 触发滑动窗口"
        )
        
        # 提取被释放的KV
        evicted_kv = self._extract_evicted_kv(past_key_values, self.window_size)
        
        # 只保留窗口内的KV
        trimmed_kv = self._keep_window_kv(past_key_values, self.window_size)
        
        self.total_kv_evicted += current_len - self.window_size
        
        # 存储到海马体（如果启用）
        if self.enable_hippocampus and hippocampus is not None and evicted_kv:
            self._store_evicted_kv_to_hippocampus(
                evicted_kv, 
                current_token_text or "", 
                hippocampus
            )
        
        logger.info(
            f"[KVCacheManager] KV滑动窗口完成: "
            f"{current_len} -> {self.window_size} tokens, "
            f"释放{current_len - self.window_size}个KV"
        )
        
        return trimmed_kv, evicted_kv
    
    def _extract_evicted_kv(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        window_size: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        提取被释放的KV（窗口外的部分）
        
        Args:
            past_key_values: KV cache
            window_size: 窗口大小
        
        Returns:
            evicted_kv: 被释放的KV列表 [(K, V), (K, V), ...]
        """
        evicted = []
        
        for layer_idx, (k, v) in enumerate(past_key_values):
            # 提取窗口外的KV
            # k, v: [batch, num_heads, seq_len, head_dim]
            evicted_k = k[:, :, :-window_size, :].clone()
            evicted_v = v[:, :, :-window_size, :].clone()
            
            evicted.append((evicted_k, evicted_v))
        
        return evicted
    
    def _keep_window_kv(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        window_size: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        只保留窗口内的KV
        
        Args:
            past_key_values: KV cache
            window_size: 窗口大小
        
        Returns:
            trimmed_kv: 修剪后的KV cache
        """
        trimmed = []
        
        for k, v in past_key_values:
            # 只保留最近window_size个token
            trimmed_k = k[:, :, -window_size:, :].clone()
            trimmed_v = v[:, :, -window_size:, :].clone()
            
            trimmed.append((trimmed_k, trimmed_v))
        
        return tuple(trimmed)
    
    def _store_evicted_kv_to_hippocampus(
        self,
        evicted_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        context_text: str,
        hippocampus: Any
    ):
        """
        将被释放的KV存储到海马体
        
        Args:
            evicted_kv: 被释放的KV列表
            context_text: 上下文文本
            hippocampus: 海马体实例
        """
        try:
            # 提取KV特征（使用均值池化）
            all_k_features = []
            all_v_features = []
            
            for k, v in evicted_kv:
                # k, v: [batch, num_heads, seq_len, head_dim]
                # 沿序列维度池化
                k_mean = k.mean(dim=2)  # [batch, num_heads, head_dim]
                v_mean = v.mean(dim=2)
                
                all_k_features.append(k_mean)
                all_v_features.append(v_mean)
            
            # 拼接所有层的特征
            combined_k = torch.cat(all_k_features, dim=-1)  # [batch, num_heads, all_dims]
            combined_v = torch.cat(all_v_features, dim=-1)
            
            # 取第一层作为代表特征
            key_features = combined_k[0, 0, :].cpu().tolist()  # [all_dims]
            value_features = combined_v[0, 0, :].cpu().tolist()
            
            # 存储到海马体
            if hasattr(hippocampus, 'store_kv_as_memory'):
                memory_id = hippocampus.store_kv_as_memory(
                    kv_features={
                        'key_features': key_features,
                        'value_features': value_features,
                        'num_layers': len(evicted_kv),
                        'seq_len': evicted_kv[0][0].shape[2],
                        'timestamp': int(time.time() * 1000)
                    },
                    context_text=context_text
                )
                
                self.total_kv_stored_to_hippocampus += 1
                
                logger.debug(
                    f"[KVCacheManager] KV已存储到海马体: "
                    f"memory_id={memory_id}, "
                    f"seq_len={evicted_kv[0][0].shape[2]}"
                )
        except Exception as e:
            logger.warning(f"[KVCacheManager] 存储KV到海马体失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            stats: 统计信息字典
        """
        return {
            'total_tokens_processed': self.total_tokens_processed,
            'total_kv_evicted': self.total_kv_evicted,
            'total_kv_stored_to_hippocampus': self.total_kv_stored_to_hippocampus,
            'window_size': self.window_size,
            'enable_hippocampus': self.enable_hippocampus,
            'max_memory_kv': self.max_memory_kv
        }


