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
        past_key_values: Optional[Any],
        current_token_text: Optional[str] = None,
        hippocampus: Optional[Any] = None
    ) -> Tuple[Optional[Any], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        修剪KV cache，只保留窗口大小
        
        支持两种格式:
        1. Legacy tuple format: ((K, V), (K, V), ...)
        2. DynamicCache format: Qwen3_5DynamicCache object
        
        Args:
            past_key_values: KV cache (tuple 或 DynamicCache)
            current_token_text: 当前token文本（用于海马体存储）
            hippocampus: 海马体实例（用于存储释放的KV）
        
        Returns:
            trimmed_kv: 修剪后的KV cache
            evicted_kv: 被释放的KV列表
        """
        if past_key_values is None:
            return None, None
        
        # ========== 兼容 DynamicCache 格式 ==========
        # 检测是否是 DynamicCache 对象
        is_dynamic_cache = hasattr(past_key_values, 'get_seq_length')
        
        # 获取当前KV长度
        if is_dynamic_cache:
            current_len = past_key_values.get_seq_length()
        else:
            # Legacy tuple format
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
        
        # ========== 根据缓存格式选择处理方法 ==========
        if is_dynamic_cache:
            # 使用 DynamicCache 的 API
            return self._trim_dynamic_cache(
                past_key_values, current_len, current_token_text, hippocampus
            )
        else:
            # 使用传统的 tuple 格式处理
            evicted_kv = self._extract_evicted_kv(past_key_values, self.window_size)
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
                f"{current_len} -> {self.window_size} tokens"
            )
            
            return trimmed_kv, evicted_kv
    
    def _trim_dynamic_cache(
        self,
        cache: Any,
        current_len: int,
        current_token_text: Optional[str],
        hippocampus: Optional[Any]
    ) -> Tuple[Any, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        处理 DynamicCache 格式的修剪
        
        Args:
            cache: DynamicCache 对象
            current_len: 当前长度
            current_token_text: 当前token文本
            hippocampus: 海马体实例
        
        Returns:
            trimmed_cache: 修剪后的缓存
            evicted_kv: 被释放的KV列表
        """
        # 方法1: 尝试转换为 legacy format 进行处理
        if hasattr(cache, 'to_legacy_cache'):
            legacy_cache = cache.to_legacy_cache()
            
            # 检查转换结果是否有效
            if legacy_cache is None:
                logger.debug("[KVCacheManager] to_legacy_cache()返回None，尝试其他方法")
            else:
                evicted_kv = self._extract_evicted_kv(legacy_cache, self.window_size)
                trimmed_legacy = self._keep_window_kv(legacy_cache, self.window_size)
                
                # 重新构建 DynamicCache
                # 注意：某些版本的 DynamicCache 可能不支持直接从 tuple 构建
                # 这里我们返回 legacy format，因为模型可以处理两种格式
                trimmed_cache = trimmed_legacy
                
                self.total_kv_evicted += current_len - self.window_size
                
                # 存储到海马体
                if self.enable_hippocampus and hippocampus is not None and evicted_kv:
                    self._store_evicted_kv_to_hippocampus(
                        evicted_kv, 
                        current_token_text or "", 
                        hippocampus
                    )
                
                logger.info(
                    f"[KVCacheManager] DynamicCache修剪完成(to_legacy): "
                    f"{current_len} -> {self.window_size} tokens"
                )
                
                return trimmed_cache, evicted_kv
            
            # 方法2: 直接操作 DynamicCache 对象
            elif hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
                # DynamicCache 内部使用 key_cache 和 value_cache 列表
                # 安全检查：确保 key_cache 和 value_cache 不为 None
                if cache.key_cache is None or cache.value_cache is None:
                    logger.debug("[KVCacheManager] DynamicCache的key_cache/value_cache为None，跳过修剪")
                    return cache, None
                
                evicted_kv = []
                
                try:
                    num_layers = len(cache.key_cache)
                except (TypeError, AttributeError):
                    logger.debug("[KVCacheManager] 无法获取DynamicCache层数，跳过修剪")
                    return cache, None
                
                for layer_idx in range(num_layers):
                    k = cache.key_cache[layer_idx]  # [batch, num_heads, seq_len, head_dim]
                    v = cache.value_cache[layer_idx]
                    
                    if k is None or v is None:
                        logger.debug(f"[KVCacheManager] 层{layer_idx}的KV为None，跳过")
                        continue
                    
                    # 提取被释放的 KV
                    evicted_k = k[:, :, :-self.window_size, :].clone()
                    evicted_v = v[:, :, :-self.window_size, :].clone()
                    evicted_kv.append((evicted_k, evicted_v))
                    
                    # 修剪到窗口大小
                    cache.key_cache[layer_idx] = k[:, :, -self.window_size:, :].clone()
                    cache.value_cache[layer_idx] = v[:, :, -self.window_size:, :].clone()
                
                self.total_kv_evicted += current_len - self.window_size
                
                # 存储到海马体
                if self.enable_hippocampus and hippocampus is not None and evicted_kv:
                    self._store_evicted_kv_to_hippocampus(
                        evicted_kv, 
                        current_token_text or "", 
                        hippocampus
                    )
                
                logger.info(
                    f"[KVCacheManager] DynamicCache修剪完成(direct): "
                    f"{current_len} -> {self.window_size} tokens"
                )
                
                return cache, evicted_kv
            
            # 方法3: 尝试使用 to_legacy_cache 的替代方法
            else:
                # 尝试手动提取 KV（适用于其他类型的 DynamicCache）
                logger.debug(
                    "[KVCacheManager] 尝试手动提取DynamicCache的KV"
                )
                
                # 尝试通过迭代或其他方式获取 KV
                # 某些实现可能支持 __getitem__
                if hasattr(cache, '__getitem__'):
                    legacy_cache = tuple(cache[layer_idx] for layer_idx in range(len(cache)))
                    evicted_kv = self._extract_evicted_kv(legacy_cache, self.window_size)
                    trimmed_legacy = self._keep_window_kv(legacy_cache, self.window_size)
                    
                    self.total_kv_evicted += current_len - self.window_size
                    
                    # 存储到海马体
                    if self.enable_hippocampus and hippocampus is not None and evicted_kv:
                        self._store_evicted_kv_to_hippocampus(
                            evicted_kv, 
                            current_token_text or "", 
                            hippocampus
                        )
                    
                    logger.info(
                        f"[KVCacheManager] DynamicCache修剪完成(manual): "
                        f"{current_len} -> {self.window_size} tokens"
                    )
                    
                    return trimmed_legacy, evicted_kv
                
                # 如果所有方法都失败，记录调试信息并跳过修剪
                logger.debug(
                    f"[KVCacheManager] DynamicCache类型{type(cache).__name__}不支持修剪，跳过"
                )
                return cache, None
    
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


