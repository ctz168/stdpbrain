"""
真实 Qwen 模型集成接口

功能:
- 加载 Qwen 系列模型（支持 Qwen3.5, Qwen2.5 等）
- 将双权重层集成到真实模型中
- 提供完整的生成和对话接口
- 支持窄带宽注意力（记忆锚点注入）
"""

import torch
import torch.nn as nn
import os as _os
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import asyncio
import sys
import logging

# 配置日志
logger = logging.getLogger(__name__)

# ========== 窄带宽注意力补丁 ==========
# 在导入时应用补丁，修改 Qwen 内部注意力层（条件性：支持 Qwen3.5 和 Qwen2）
from core.qwen_narrow_band_patch import patch_qwen_attention, get_memory_anchor_store
NARROW_BAND_PATCHED = patch_qwen_attention()


# ========== Qwen3.5 / Qwen2.5 配置兼容辅助函数 ==========
def _get_qwen_config_attr(model_config, attr_name, default=None):
    """
    从模型配置中读取属性，兼容 Qwen3.5 和 Qwen2.5。
    
    Qwen3.5 (qwen3_5): hidden_size, vocab_size 等参数在 config.text_config 里
    Qwen2.5 (qwen2): 参数直接在 config 顶层
    
    优先级: config.text_config.{attr} > config.{attr} > default
    """
    if hasattr(model_config, 'text_config') and hasattr(model_config.text_config, attr_name):
        return getattr(model_config.text_config, attr_name)
    if hasattr(model_config, attr_name):
        return getattr(model_config, attr_name)
    return default


def _get_im_end_token_id(tokenizer):
    """
    动态获取 <|im_end|> 的 token id。
    
    Qwen2.5: im_end_token_id = 151645
    Qwen3.5: im_end_token_id 可能不同，需要从 tokenizer 获取
    
    策略:
    1. 尝试从 tokenizer 的 special_tokens 中获取
    2. 尝试将 '<|im_end|>' 编码
    3. 回退到 eos_token_id
    """
    if tokenizer is None:
        return 151645
    
    # 方法1: 从 additional_special_tokens 获取
    for token_str, token_id in getattr(tokenizer, 'added_tokens_encoder', {}).items():
        if token_str == '<|im_end|>':
            return token_id
    
    # 方法2: 尝试 convert_tokens_to_ids
    try:
        im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
        if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
            return im_end_id
    except Exception:
        pass
    
    # 方法3: 回退到 eos_token_id
    eos = tokenizer.eos_token_id
    if eos is not None:
        return eos
    
    # 最终回退
    return 151645

# ========== KV Cache 滑动窗口管理器 ==========
from core.kv_cache_manager import KVCacheManager
KV_CACHE_MANAGER_AVAILABLE = True


class QwenModelWrapper(nn.Module):
    """
    Qwen 模型包装器（兼容 Qwen3.5, Qwen2.5 等架构）
    
    将双权重层集成到真实 Qwen 模型中
    """
    
    def __init__(
        self, 
       model_path: str,
        config,
        device: str = "cpu",
        quantization: str = "INT8"
    ):
        super().__init__()
        import threading
        import os
        self._tokenizer_lock = threading.Lock() # 线程锁，防止 Already borrowed
        # 将相对路径转为绝对路径，兼容新版 transformers 不识别 ./ 前缀的问题
        self.model_path = os.path.abspath(model_path)
        self.config = config
        self.device = device
        
        # 统一读取优先级：1.全局 config.QUANTIZATION 2. 全局 config.quantization 3. 传参配置
        self.quantization = getattr(config, 'QUANTIZATION', getattr(config, 'quantization', quantization))
        
        print(f"[QwenWrapper] 正在加载真实 Qwen 模型...")
        print(f"  路径：{model_path}")
        print(f"  设备：{device}")
        print(f"  量化：{self.quantization}")
        
        # ========== 1. 加载 Tokenizer ==========
        # 使用 local_files_only=True 避免新版 transformers 递归调用 cached_file/cached_files
        self._is_local_model = _os.path.isdir(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
           self.model_path,
            trust_remote_code=True,
           padding_side="left",
           local_files_only=self._is_local_model
        )
        print(f"[OK] Tokenizer 加载成功，词表大小：{len(self.tokenizer)}")
        
        # ========== 2. 加载模型 ==========
        self.base_model = self._load_model_with_quantization()
        
        # ========== 3. 集成双权重层 ==========
        self._integrate_dual_weights()
        
        # ========== 4. 设置填充 token ==========
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        param_count = sum(p.numel() for p in self.parameters())
        print(f"[OK] Qwen 模型加载完成")
        print(f"  - 参数量：{param_count:,} ({param_count/1e6:.2f}M)")
        print(f"  - 设备：{device}")
    
    def _load_model_with_quantization(self):
        """根据量化类型加载模型 (针对 macOS/CPU 优化)"""
        # 自动检测 Apple Silicon (M1/M2/M3)
        is_mac = torch.backends.mps.is_available()
        
        # AUTO 模式：根据设备自动选择量化方式
        if self.quantization == "AUTO":
            if self.device == "cuda":
                self.quantization = "INT8"  # GPU 使用 INT8
            elif is_mac:
                self.quantization = "FP16"  # macOS 使用 FP16
            else:
                # CPU优化：使用FP16（比FP32快2x，内存减半）
                self.quantization = "FP16"
            print(f"  [AUTO] 自动选择量化方式: {self.quantization}")
        
        if self.quantization in ["INT4", "INT8"]:
            if self.device == "cuda":
                # CUDA 环境下的量化
                from transformers import BitsAndBytesConfig
                load_in_4bit = (self.quantization == "INT4")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=not load_in_4bit,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    device_map={"": self.device},
                    trust_remote_code=True,
                    local_files_only=self._is_local_model
                )
                print(f"  [OK] [CUDA] {self.quantization} 量化加载成功")
            elif is_mac:
                # macOS 下使用 FP16
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    dtype=torch.float16,
                    device_map={"": "mps"},
                    trust_remote_code=True,
                    local_files_only=self._is_local_model
                )
                print("  [OK] [macOS/MPS] 使用 FP16 加载成功")
            else:
                # CPU 环境：使用 FP16 加载 + 动态量化（避免内存溢出）
                print(f"  [!] [CPU] bitsandbytes 的 {self.quantization} 量化完全依赖 CUDA (Nvidia GPU)。")
                print(f"  [!] [CPU] 使用 FP16 加载 + 动态量化方式（内存占用减半）...")
                
                # Step 1: 以 FP16 加载（比 FP32 节省一半内存）
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    local_files_only=self._is_local_model
                )
                print("  [OK] FP16 加载完成，开始动态量化...")
                
                # Step 2: 动态量化线性层
                model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},  # 只量化线性层
                    dtype=torch.qint8
                )
                print(f"  [OK] [CPU] INT8 动态量化完成")
            
        else:  # FP16 或 FP32
            # 尊重用户的量化设置，而非自动判断
            if self.quantization == "FP16":
                dtype = torch.float16
            elif self.quantization == "FP32":
                dtype = torch.float32
            else:
                # AUTO 或其他：根据设备自动选择
                dtype = torch.float16 if (self.device == "cuda" or is_mac) else torch.float32
            target_device = "mps" if (is_mac and self.device != "cuda") else self.device
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map={"": target_device} if target_device != "cpu" else None,
                trust_remote_code=True,
                local_files_only=self._is_local_model
            )
            print(f"  [OK] {'FP16' if dtype == torch.float16 else 'FP32'} 加载成功")
        
        return model

    def tokenize_safe(self, text, **kwargs):
        """线程安全的 Tokenization 封装"""
        with self._tokenizer_lock:
            return self.tokenizer(text, **kwargs)

    def decode_safe(self, token_ids, **kwargs):
        """线程安全的 Decoding 封装"""
        with self._tokenizer_lock:
            return self.tokenizer.decode(token_ids, **kwargs)
    
    def encode_safe(self, text, **kwargs):
        """线程安全的 Encoding 封装"""
        with self._tokenizer_lock:
            return self.tokenizer.encode(text, **kwargs)

    def apply_chat_template_safe(self, messages, **kwargs):
        """线程安全的 Chat Template 应用"""
        with self._tokenizer_lock:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
    
    def _integrate_dual_weights(self):
        """
        将双权重层集成到 Qwen 模型中
        
        替换每个 Transformer 层的线性层为 DualWeightLinear
        """
        from core.dual_weight_layers import DualWeightLinear
        
        print("\n[集成] 开始集成双权重层...")
        
        # 遍历所有命名模块，寻找 Linear 层进行替换
        replaced_count = 0
        
        # 记录需要替换的目标属性名映射
        # 注意：我们只替换 Q, K, V, O 投影以及 MLP 的 gate, up, down 投影
        target_names = {
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj'
        }
        
        # 使用 list 存储层，避免遍历时修改字典
        for name, module in list(self.base_model.named_modules()):
            # ========== 修复：排除 GatedDeltaNet（线性注意力层）==========
            # Qwen3.5 混合使用 Qwen3_5Attention（全注意力）和 Qwen3_5GatedDeltaNet（线性注意力）
            # GatedDeltaNet 的内部投影（in_proj_qkv, out_proj 等）使用不同的计算路径
            # 被 DualWeightLinear 包裹后会破坏线性注意力的特殊计算（如 recurrent state）
            module_type = type(module).__name__
            if 'GatedDeltaNet' in module_type or 'DeltaNet' in module_type:
                continue
            
            # 检查是否包含目标投影
            for attr_name in target_names:
                if hasattr(module, attr_name):
                    target_layer = getattr(module, attr_name)
                    
                    # 确保是量化黑盒层或普通线性层，且尚未被包裹 (兼容 Linear4bit, Linear8bitLt 等)
                    if not isinstance(target_layer, DualWeightLinear) and hasattr(target_layer, 'weight'):
                        try:
                            # 创建双权重线性层 (安全包裹，兼容 4-bit)
                            dual_layer = DualWeightLinear(base_layer=target_layer)
                            
                            # 替换
                            setattr(module, attr_name, dual_layer)
                            replaced_count += 1
                        except Exception as e:
                            print(f"  [!] 替换层失败 {name}.{attr_name}: {e}")
        
        print(f"[OK] 已包裹 {replaced_count} 个量化/线性投影为双权重版本")
        
        # ========== 后置优化：如果是 CPU，执行动态量化 ==========
        # 注意：2B 模型的 CPU 动态量化非常慢（5-10分钟）且内存占用大
        # 建议：使用 AUTO 量化模式（GPU用INT8，CPU用FP32）
        if self.device == "cpu" and self.quantization == "INT8":
            import time
            start_time = time.time()
            print(f"  [*] 正在对基础模型执行动态量化 (INT8)...")
            print(f"  [!] 警告：CPU 动态量化可能需要 5-10 分钟，请耐心等待...")
            print(f"  [!] 提示：可以在 config.py 中设置 QUANTIZATION='FP32' 跳过此步骤")
            
            try:
                # 动态量化会寻找 nn.Linear。注意：DualWeightLinear 默认不被识别，这很好，
                # 因为我们需要动态权重保持浮点高精度以进行 STDP 学习。
                self.base_model = torch.quantization.quantize_dynamic(
                    self.base_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                elapsed = time.time() - start_time
                print(f"  [OK] [CPU] 后置动态量化完成 (耗时: {elapsed:.1f}秒)")
            except Exception as e:
                print(f"  [!] [CPU] 动态量化失败: {e}")
                print(f"  [!] [CPU] 继续使用 FP32 模式")
    
    def set_hippocampus_gate(self, gate_fn):
        """
        设置海马体注意力门控函数
        
        Args:
            gate_fn: function(query, key, memory_anchor) -> gate_mask
        """
        for name, module in self.base_model.named_modules():
            if hasattr(module, 'attn') and hasattr(module.attn, 'set_hippocampus_gate'):
                module.attn.set_hippocampus_gate(gate_fn)
        print(f"[OK] 已设置海马体门控函数")
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        memory_anchor: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """前向传播 (支持 KV-cache、记忆锚点和 inputs_embeds)"""
        # 确保默认值
        out_attentions = output_attentions if output_attentions is not None else False
        ret_dict = return_dict if return_dict is not None else True
        use_cache = use_cache if use_cache is not None else False
        
        # 存储记忆锚点，供注意力层使用
        self._current_memory_anchor = memory_anchor
        
        # 注意力层记忆锚点之前由 DualWeightAttention 管理，现已交由底层注意力层处理
        # (在此预留基座模型可能的 hippocampus_gate 设置钩子)
        
        # 清理 kwargs 避免重复冲突
        exclude = {'return_dict', 'output_attentions', 'output_hidden_states', 'use_cache', 'past_key_values', 'memory_anchor', 'input_ids', 'inputs_embeds'}
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in exclude}
        
        # 构建 base_model 的参数（input_ids 和 inputs_embeds 互斥）
        model_kwargs = {
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'output_attentions': out_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': ret_dict,
            **clean_kwargs
        }
        if input_ids is not None:
            model_kwargs['input_ids'] = input_ids
        if inputs_embeds is not None:
            model_kwargs['inputs_embeds'] = inputs_embeds
        
        # 使用 base_model 进行推理
        outputs = self.base_model(**model_kwargs)
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.6,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 40,
        presence_penalty: float = 0.8,
        **kwargs
    ):
        """
        生成文本
        
        Args:
            input_ids: 输入 token IDs
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            do_sample: 是否采样
            top_p: Top-p 采样参数
            top_k: Top-k 采样参数
        
        Returns:
            generated_ids: 生成的 token IDs
        """
        self.base_model.eval()
        
        with torch.no_grad():
            generated_ids = self.base_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id,
                presence_penalty=presence_penalty,
                **kwargs
            )
        
        return generated_ids
    
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = True
    ) -> Tuple[torch.Tensor, ...]:
        """
        获取隐藏层状态
        
        Args:
            input_ids: 输入 token IDs
            output_hidden_states: 是否输出隐藏状态
        
        Returns:
            hidden_states: 各层隐藏状态元组
        """
        outputs = self.base_model(
            input_ids=input_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        return outputs.hidden_states
    
    def get_all_dynamic_weights(self) -> Dict[str, torch.Tensor]:
        """获取所有动态权重"""
        dynamic_weights = {}
        from core.dual_weight_layers import DualWeightLinear
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, DualWeightLinear):
                dynamic_weights[name] = module.get_dynamic_weight()
        
        return dynamic_weights

    def save_checkpoint(self, path: str):
        """保存模型检查点 (仅保存双权重层的动态部分)"""
        dynamic_weights = self.get_all_dynamic_weights()
        torch.save(dynamic_weights, path)
        print(f"[QwenWrapper] 动态权重已保存到: {path}")

    def load_checkpoint(self, path: str):
        """加载模型检查点 (仅加载双权重层的动态部分)"""
        dynamic_weights = torch.load(path, map_location=self.device, weights_only=False)
        for layer_name, tensor_weight in dynamic_weights.items():
            self.apply_stdp_to_layer(layer_name, tensor_weight, lr=1.0) # 使用 lr=1.0 直接推入更新
        print(f"[QwenWrapper] 动态权重已从 {path} 加载")
    
    def apply_stdp_to_layer(
        self,
        layer_name: str,
        grad: torch.Tensor,
        lr: float = 0.01
    ):
        """对指定双权重层直接应用 STDP 更新"""
        for name, module in self.base_model.named_modules():
            if name == layer_name:
                if hasattr(module, 'apply_stdp_update'):
                    module.apply_stdp_update(grad, lr)
                break
    
    def apply_stdp_to_all(self, grad_dict: Dict[str, torch.Tensor], lr: float = 0.01):
        """对所有双权重层广播 STDP 更新"""
        for name, module in self.base_model.named_modules():
            if hasattr(module, 'apply_stdp_update'):
                # 精确匹配全名称分发
                if name in grad_dict:
                    module.apply_stdp_update(grad_dict[name], lr)


class QwenInterface:
    """
    Qwen 模型统一接口
    
    提供对 Qwen 模型的统一访问接口，包括量化、推理和优化功能。
    
    """

    def __init__(
        self,
       model_path: str,
        config,
        device: str = "cpu",
        quantization: str = "INT4"
    ):
        self.config = config
        self.device = device
        self.model = QwenModelWrapper(
            model_path=model_path,
            config=config,
            device=device,
            quantization=quantization
        )
        # 同步设备（模型可能回退到 CPU）
        self.device = self.model.device
        # 优化2: 模型加载后只调用一次 eval()，不在每次 forward_step 重复调用
        self.model.eval()
        
        # 优化1: CPU 线程控制 + MKLDNN 加速
        if self.device == "cpu":
            import multiprocessing
            num_cores = multiprocessing.cpu_count()
            # 使用全部逻辑线程（2C/4T 用 4 线程）
            torch.set_num_threads(num_cores)
            print(f"[QwenInterface] CPU 优化：设置线程数为 {torch.get_num_threads()}")
            
            # MKLDNN 加速：启用 oneDNN 后端优化
            if torch.backends.mkldnn.is_available():
                torch.backends.mkldnn.enabled = True
                print(f"[QwenInterface] MKLDNN 加速已启用")

        # 优化4: STDP 后台线程池，不阻塞生成循环
        import concurrent.futures
        self._stdp_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='stdp_worker'
        )
        # 每 N 步才提取一次 features 用于 STDP
        self._step_counter = 0
        self._stdp_every_n = getattr(getattr(config, 'stdp', None), 'every_n_tokens', 5)  # 默认每5步更新
        self._last_reward = 1.0  # 存储最近一次对话的评判得分
        
        # ========== 新增: 性能优化缓存 ==========
        # 增量 token 计数器（避免每步 torch.unique 全量扫描）
        self._token_counts = {}  # {token_id: count} - 仅记录当前会话
        
        # 模块索引（避免每次扫描 named_modules）
        self._stdp_layers_cache = None
        
        # 预分配惩罚张量（复用内存）
        # 兼容 Qwen3.5: vocab_size 在 text_config 里
        vocab_size = _get_qwen_config_attr(self.model.base_model.config, 'vocab_size', 248320)
        # 安全检查：确保从 logits 维度获取实际 vocab_size
        logger.info(f"[QwenInterface] 配置 vocab_size={vocab_size}")
        model_dtype = next(self.model.base_model.parameters()).dtype
        self._penalty_buffer = torch.zeros(vocab_size, device=self.device, dtype=model_dtype)
        
        # ========== 新增: 窄带宽注意力支持 ==========
        self._memory_anchors_cache = []  # 记忆锚点缓存
        self._narrow_band_enabled = True  # 默认启用窄带宽注意力
        
        # 统计信息
        self.total_generation_time = 0.0
        self.total_tokens_generated = 0
    
    def set_hippocampus_gate(self, gate_fn):
        """
        设置海马体注意力门控函数
        
        Args:
            gate_fn: function(query, key, memory_anchor) -> gate_mask
        """
        # 委托给内部的 QwenModelWrapper
        self.model.set_hippocampus_gate(gate_fn)
    
    # [FIX] 合并两个重复的 set_memory_anchors 方法定义。
    # 原代码有两个同名方法：第583行（简单版）和第628行（带参数版），
    # Python 中后者会覆盖前者，导致简单调用签名丢失。
    # 现统一为带完整参数的版本，同时保持向后兼容。
    def set_memory_anchors(self, anchors, max_anchors: int = 5, strength: float = 1.0):
        """
        设置记忆锚点（用于窄带宽注意力）
        
        对应大脑机制: 海马体向前额叶/新皮层传递记忆锚点
        
        Args:
            anchors: 记忆锚点列表，每个锚点包含:
                - key_features: 注意力 Key 特征
                - value_features: 注意力 Value 特征
                - activation_strength: 记忆强度
                - dg_features: DG 特征（如果没有 KV 特征，会从此生成）
            max_anchors: 最大锚点数量（工作记忆容量限制）
            strength: 锚点强度
        """
        # 使用全局存储
        if NARROW_BAND_PATCHED:
            anchor_store = get_memory_anchor_store()
            anchor_store.set_anchors(anchors, max_anchors, strength)
            logger.debug(f"[QwenInterface] 设置 {len(anchors)} 个记忆锚点")
        else:
            # 回退到本地缓存
            self._memory_anchors_cache = anchors[:max_anchors]
    
    @property
    def _tokenizer_lock(self):
        """将内部锁暴露给外部接口 (如 InnerThoughtEngine)"""
        return self.model._tokenizer_lock

    @property
    def tokenizer(self):
        return self.model.tokenizer
        
    @property
    def embeddings(self):
        return self.model.base_model.get_input_embeddings()

    def apply_stdp_to_all(self, grad_dict: Dict[str, torch.Tensor], lr: float = 0.01):
        """统一 STDP 更新接口"""
        self.model.apply_stdp_to_all(grad_dict, lr)

    def tokenize_safe(self, text, **kwargs):
        """传递调用到模型包装器"""
        return self.model.tokenize_safe(text, **kwargs)

    def decode_safe(self, token_ids, **kwargs):
        """传递调用到模型包装器"""
        return self.model.decode_safe(token_ids, **kwargs)
    
    def encode_safe(self, text, **kwargs):
        """传递调用到模型包装器"""
        return self.model.encode_safe(text, **kwargs)

    def apply_chat_template_safe(self, messages, **kwargs):
        """传递调用到模型包装器"""
        return self.model.apply_chat_template_safe(messages, **kwargs)
    
    def enable_narrow_band(self, enabled: bool = True):
        """启用/禁用窄带宽注意力"""
        self._narrow_band_enabled = enabled
        if NARROW_BAND_PATCHED:
            anchor_store = get_memory_anchor_store()
            anchor_store.enabled = enabled
    
    def clear_memory_anchors(self):
        """清除记忆锚点（在生成完成后调用）"""
        if NARROW_BAND_PATCHED:
            anchor_store = get_memory_anchor_store()
            anchor_store.clear()
        else:
            self._memory_anchors_cache = []
    

    def set_reward(self, reward: float):
        """设置反馈奖励 (来自优化器) - 写入日志文件，避免干扰终端输出"""
        self._last_reward = max(0.1, min(2.0, reward))
        try:
            with open("brain_debug.log", "a", encoding="utf-8") as f:
                f.write(f"[STDP] 接收新奖励反馈: {self._last_reward:.2f}\n")
        except Exception:
            pass
        sys.stderr.flush()

    def forward_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
        memory_anchor_id: Optional[str] = None,
        memory_anchor_gate: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,  # 新增：支持自定义embedding
        **kwargs
    ) -> Dict[str, Any]:
        """单步前向推理 (KV-cache + inference_mode + 条件特征提取)
        
        新增：支持inputs_embeds参数，用于目标向量注入
        """
        # 工具提示检测：简单的启发式
        is_tool_call = False
        if input_ids.shape[-1] > 0:
             last_token = input_ids[0, -1].item()
             # 占位：检测 '!' (33) 或 '{' (123) 或 'tool' 等 token ID
             # 实际项目中应根据 tokenizer 确定
             if last_token in [33, 123, 151644]: 
                 is_tool_call = True
        
        # 记录推理进度 (仅针对 CPU 缓慢环境)
        # if self.device == "cpu":
        #      print(".", end="", flush=True)
        # 优化2: eval() 已在 __init__ 中调用，不再每步重复
        outputs = {}
        start_time = time.time()
        
        # 优化3: 每 N 步才提取 hidden_states (STDP 所需)，其余步骤跳过
        self._step_counter += 1
        need_features = (self._step_counter % self._stdp_every_n == 0)
        
        # 优化2: inference_mode 比 no_grad 快 10-15%（跳过版本跟踪）
        with torch.inference_mode():
            exclude_keys = {'return_dict', 'output_attentions', 'memory_anchor'}
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in exclude_keys}
            
            # 构建forward参数：优先使用inputs_embeds
            forward_params = {
                'attention_mask': attention_mask,
                'past_key_values': past_key_values,
                'use_cache': use_cache,
                'memory_anchor': memory_anchor_gate,
                'output_attentions': False,
                'output_hidden_states': need_features,  # 优化3: 按需提取
                'return_dict': True,
                **clean_kwargs
            }
            
            # 如果提供了inputs_embeds，使用它而不是input_ids
            if inputs_embeds is not None:
                forward_params['inputs_embeds'] = inputs_embeds
                # 注意：使用inputs_embeds时，不能传递input_ids
            else:
                forward_params['input_ids'] = input_ids
            
            output_tensors = self.model(**forward_params)
            
            next_token_logits = output_tensors.logits[:, -1, :].clone()
            
            # ========== 优化: 增量 token 计数，避免每步 torch.unique 全量扫描 ==========
            repetition_penalty = kwargs.get('repetition_penalty', 1.0)
            presence_penalty = kwargs.get('presence_penalty', 0.0)
            
            if (repetition_penalty != 1.0 or presence_penalty > 0) and input_ids is not None:
                if input_ids.shape[-1] > 0:
                    # 增量更新：仅处理最新的 token（KV-cache 模式下只有 1 个新 token）
                    new_tokens = input_ids[0, -1:] if past_key_values is not None else input_ids[0]
                    
                    # 更新计数（增量）
                    for t in new_tokens.tolist():
                        self._token_counts[t] = self._token_counts.get(t, 0) + 1
                    
                    # 当是预填充阶段时，需要重建完整计数
                    if past_key_values is None and input_ids.shape[-1] > 1:
                        self._token_counts.clear()
                        for t in input_ids[0].tolist():
                            self._token_counts[t] = self._token_counts.get(t, 0) + 1
                    
                    # 转换为张量进行向量化惩罚
                    if self._token_counts:
                        seen_ids = torch.tensor(list(self._token_counts.keys()), dtype=torch.long, device=self.device)
                        counts_t = torch.tensor(list(self._token_counts.values()), dtype=torch.float32, device=self.device)
                        
                        # Repetition Penalty (向量化 scatter)
                        if repetition_penalty != 1.0:
                            seen_logits = next_token_logits[:, seen_ids]
                            penalized_logits = torch.where(
                                seen_logits > 0,
                                seen_logits / repetition_penalty,
                                seen_logits * repetition_penalty
                            )
                            next_token_logits.scatter_(1, seen_ids.unsqueeze(0), penalized_logits)
                        
                        # Presence Penalty
                        if presence_penalty > 0:
                            penalty = self._penalty_buffer.zero_()
                            penalty[seen_ids] = counts_t * presence_penalty
                            next_token_logits -= penalty.unsqueeze(0)

            # 优化: 合并温度缩放和采样
            temp = kwargs.get('temperature', 1.0)
            top_k = kwargs.get('top_k', 20)
            
            if temp > 0:
                # 温度缩放
                scaled_logits = next_token_logits / temp
                
                # Top-k 过滤
                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')
                
                # Softmax + Multinomial (合并为一次操作)
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪心解码
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            outputs['token_id'] = next_token.item()
            outputs['logits'] = next_token_logits
            
            # 优化3: 按需提取特征，不需要时用零向量占位 (STDP 可接受)
            if need_features and output_tensors.hidden_states:
                outputs['features'] = output_tensors.hidden_states[-1][:, -1, :].clone()
            else:
                # hidden_size 仅用于日志/调试，features 已标记为 None
                _hs = _get_qwen_config_attr(self.model.base_model.config, 'hidden_size', 1024)
                outputs['features'] = None  # 标记为不可用，STDP engine 跳过
                
            outputs['attention_output'] = torch.zeros(1)
            outputs['ffn_output'] = outputs['features'] if outputs['features'] is not None else torch.zeros(1)
            outputs['generation_path'] = str(next_token.item())
            outputs['evaluation_score'] = 35.0
            
        elapsed = time.time() - start_time
        outputs['cycle_time_ms'] = elapsed * 1000.0
        outputs['past_key_values'] = output_tensors.past_key_values if hasattr(output_tensors, 'past_key_values') else None
        self.total_tokens_generated += 1
        
        # ========== 优化: 使用缓存的模块索引，批量提交 ==========
        stdp_config = getattr(self.config, 'stdp', None)
        if stdp_config and getattr(stdp_config, 'enabled', False) and need_features:
            # 延迟初始化模块索引
            if self._stdp_layers_cache is None:
                self._stdp_layers_cache = [
                    (name, layer) for name, layer in self.model.base_model.named_modules()
                    if hasattr(layer, 'apply_stdp_update')
                ]
            
            current_reward = self._last_reward
            
            # 批量提交：用一个任务更新所有层（避免线程调度开销）
            cached_layers = self._stdp_layers_cache
            cached_input = input_ids.flatten()
            cached_token = outputs['token_id']
            cached_features = outputs.get('features', torch.zeros(1))
            cached_time = time.time() * 1000
            cached_tool = is_tool_call
            
            stdp_engine = getattr(getattr(self.config, 'stdp', None), 'stdp_engine', None)
            # BUG FIX: interfaces.py 将 stdp_engine 设置在 config.stdp_engine (直接属性)，
            # 而非 config.stdp.stdp_engine (嵌套属性)。上面的 getattr 查找嵌套路径，
            # 导致 stdp_engine 永远为 None，STDP 更新从未执行。
            # 兼容两种设置路径：先检查嵌套路径，再检查直接路径。
            if stdp_engine is None:
                stdp_engine = getattr(self.config, 'stdp_engine', None)
            if stdp_engine is None:
                return outputs

            def _batch_stdp_update():
                """批量更新所有 STDP 层（单次线程调度）"""
                for _, layer in cached_layers:
                    stdp_engine.update_attention_layer(
                        layer, cached_input, cached_token, cached_features,
                        cached_time, reward=current_reward, is_tool_call=cached_tool
                    )
            
            self._stdp_executor.submit(_batch_stdp_update)
        
        return outputs



    def generate_stream_sync(
        self,
        input_text: str,
        max_tokens: int = 150,
        temperature: float = 0.45,
        repetition_penalty: float = 1.1,  # 降低重复惩罚
        **kwargs
    ):
        """
        同步流式生成 (KV-cache + 预分配 input_ids 缓冲区)
        
        优化：
        - 滑动窗口上下文限制（类人脑工作记忆容量限制）
        - 记忆锚点辅助注意力（如果启用窄带宽）
        
        Yields:
            str: 每个生成的字符/词
        """
        # ========== 0. 滑动窗口配置 ==========
        _hc = getattr(getattr(self.model, 'config', None), 'hard_constraints', None)
        max_context = getattr(_hc, 'MAX_CONTEXT_LENGTH', 512)
        narrow_window = getattr(_hc, 'NARROW_WINDOW_SIZE', 5)
        
        # ========== 1. Tokenize 输入 (线程安全) ==========
        inputs = self.tokenize_safe(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_context
        )
        
        prompt_ids = inputs.input_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]
        
        # 预分配 input_ids 缓冲区
        max_total = prompt_len + max_tokens
        with self._tokenizer_lock:
            pad_id = self.tokenizer.pad_token_id or 0
            eos_id = self.tokenizer.eos_token_id
            
        input_ids_buf = torch.full(
            (1, max_total), pad_id,
            dtype=torch.long, device=self.device
        )
        input_ids_buf[:, :prompt_len] = prompt_ids
        cur_len = prompt_len
        
        # 重置增量 token 计数器（每次生成开始时）
        self._token_counts.clear()
        
        attention_mask = inputs.attention_mask.to(self.device)
        past_key_values = None
        
        # 定义停止token
        with self._tokenizer_lock:
            eos_token_id = self.tokenizer.eos_token_id
            im_end_token_id = _get_im_end_token_id(self.tokenizer)
        stop_token_ids = {eos_token_id, im_end_token_id}
        
        # 重复检测变量
        recent_tokens = []  # 最近生成的token
        ngram_repeat_count = {}  # n-gram重复计数
        max_repeat_allowed = 10  # 从6提升到10，避免过早截断0.8B模型思维
        
        for step in range(max_tokens):
            # KV-cache 模式: 只传最后一个 token
            if past_key_values is not None:
                model_input_ids = input_ids_buf[:, cur_len-1:cur_len]
            else:
                model_input_ids = input_ids_buf[:, :cur_len]
            
            step_outputs = self.forward_step(
                input_ids=model_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                repetition_penalty=repetition_penalty,  # 传递重复惩罚
                **kwargs
            )
            
            next_token_id = step_outputs['token_id']
            past_key_values = step_outputs['past_key_values']
            
            # ========== KV Cache 滑动窗口管理（已禁用）==========
            # 致命BUG修复：原代码在生成过程中修剪KV cache，会删除prompt中的用户输入tokens。
            # 根因：prompt可能100-200 tokens，KV_CACHE_WINDOW_SIZE=128，trim_threshold=256，
            # 当prompt_len+generated>256时修剪保留最后128个tokens，prompt开头的用户输入被删！
            # 模型从此只能看到自己生成的tokens，输出联想式的垃圾。
            # 修复：默认禁用KV cache滑动窗口。0.8B模型的max_tokens=200不会超出context window。
            enable_kv_sliding = False  # 直接禁用，不再从config读取
            
            token_text = self.decode_safe([next_token_id], skip_special_tokens=True)
            
            # 温和的阻断词拦截（仅拦截会导致格式破坏的严重标签）
            unsafe_keywords = ["<|im_start|>", "\nuser\n", "\nUser\n"]
            full_decode = self.decode_safe(input_ids_buf[0, prompt_len:cur_len+1], skip_special_tokens=True)
            if any(kw in full_decode for kw in unsafe_keywords):
                break
            
            # ========== N-gram 重复检测（温和版：不截断，只惩罚） ==========
            recent_tokens.append(next_token_id)
            if len(recent_tokens) > 40:  # 扩大到40个token
                recent_tokens.pop(0)
            
            # 检测2-gram和3-gram重复 - 但不截断，仅用于惩罚
            detected_loop = False
            if len(recent_tokens) >= 3:
                ngram3 = tuple(recent_tokens[-3:])
                ngram_repeat_count[ngram3] = ngram_repeat_count.get(ngram3, 0) + 1
                if ngram_repeat_count[ngram3] > max_repeat_allowed:
                    detected_loop = True  # 标记但不截断
            
            if len(recent_tokens) >= 4:
                ngram4 = tuple(recent_tokens[-4:])
                ngram_repeat_count[ngram4] = ngram_repeat_count.get(ngram4, 0) + 1
                if ngram_repeat_count[ngram4] > max_repeat_allowed * 3:
                    break  # 4-gram重复超过30次才真正截断（从*2放宽到*3）
            
            yield token_text
            
            if next_token_id in stop_token_ids:
                self.clear_memory_anchors()
                break
            
            # 写入缓冲区
            if cur_len < max_total:
                input_ids_buf[:, cur_len] = next_token_id
                cur_len += 1
            
            # 更新 attention mask
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=self.device, dtype=torch.long)], dim=-1
            )

    async def generate_stream(
        self,
        input_text: str,
        max_tokens: int = 150,
        temperature: float = 0.35,
        repetition_penalty: float = 1.1,  # 降低重复惩罚
        **kwargs
    ):
        """
        异步流式生成 (KV-cache + 预分配 input_ids 缓冲区 + 条件 STDP)
        """
        # ========== 1. Tokenize 输入 (使用线程安全包装) ==========
        inputs = self.tokenize_safe(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        prompt_ids = inputs.input_ids.to(self.device)       # [1, prompt_len]
        prompt_len = prompt_ids.shape[1]
        
        # 优化6: 预分配 input_ids 缓冲区，避免每步 torch.cat 分配内存
        max_total = prompt_len + max_tokens
        with self._tokenizer_lock:
            pad_id = self.tokenizer.pad_token_id or 0
            
        input_ids_buf = torch.full(
            (1, max_total), pad_id,
            dtype=torch.long, device=self.device
        )
        input_ids_buf[:, :prompt_len] = prompt_ids
        cur_len = prompt_len  # 指针追踪实际长度
        
        attention_mask = inputs.attention_mask.to(self.device)
        past_key_values = None
        
        # 定义停止token
        with self._tokenizer_lock:
            eos_token_id = self.tokenizer.eos_token_id
            im_end_token_id = _get_im_end_token_id(self.tokenizer)
        stop_token_ids = {eos_token_id, im_end_token_id}
        
        # 重复检测变量
        recent_tokens = []
        ngram_repeat_count = {}
        max_repeat_allowed = 10  # 从6提升到10，避免过早截断0.8B模型思维
        step_outputs = None  # 初始化，用于循环后提取隐藏状态
        hidden_state_capture_interval = 10  # 每10步捕获一次隐藏状态
        
        for step in range(max_tokens):
            # KV-cache 模式: 只传最后一个 token
            if past_key_values is not None:
                model_input_ids = input_ids_buf[:, cur_len-1:cur_len]
            else:
                model_input_ids = input_ids_buf[:, :cur_len]
            
            step_outputs = self.forward_step(
                input_ids=model_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                repetition_penalty=repetition_penalty,  # 传递重复惩罚
                **kwargs
            )
            
            next_token_id = step_outputs['token_id']
            past_key_values = step_outputs['past_key_values']
            
            # ========== KV Cache 滑动窗口管理（已禁用）==========
            # 同上：KV cache滑动窗口是输入丢失的根本原因，在此方法中也禁用
            enable_kv_sliding = False
            
            token_text = self.decode_safe([next_token_id], skip_special_tokens=True)
            
            # 温和的阻断词拦截（仅拦截严重格式破坏标签）
            unsafe_keywords = ["<|im_start|>", "\nuser\n", "\nUser\n"]
            full_decode = self.decode_safe(input_ids_buf[0, prompt_len:cur_len+1], skip_special_tokens=True)
            if any(kw in full_decode for kw in unsafe_keywords):
                break
            
            # ========== N-gram 重复检测（温和版） ==========
            recent_tokens.append(next_token_id)
            if len(recent_tokens) > 40:
                recent_tokens.pop(0)
            
            detected_loop = False
            if len(recent_tokens) >= 3:
                ngram3 = tuple(recent_tokens[-3:])
                ngram_repeat_count[ngram3] = ngram_repeat_count.get(ngram3, 0) + 1
                if ngram_repeat_count[ngram3] > max_repeat_allowed:
                    detected_loop = True
            
            if len(recent_tokens) >= 4:
                ngram4 = tuple(recent_tokens[-4:])
                ngram_repeat_count[ngram4] = ngram_repeat_count.get(ngram4, 0) + 1
                if ngram_repeat_count[ngram4] > max_repeat_allowed * 3:
                    break  # 4-gram超过30次才截断（从*2放宽到*3）
            
            yield token_text
            
            # ========== 周期性捕获并 yield 隐藏状态（维持意识连续性）==========
            if step > 0 and step % hidden_state_capture_interval == 0:
                hs = step_outputs.get('features')
                if hs is not None:
                    yield {"type": "hidden_state", "hidden_state": hs.clone()}
            
            if next_token_id in stop_token_ids:
                self.clear_memory_anchors()
                break
            
            # 优化6: 写入预分配缓冲区，无内存分配
            if cur_len < max_total:
                input_ids_buf[:, cur_len] = next_token_id
                cur_len += 1
            
            # 更新 attention mask (仍需 cat，但只是 1 维 bool tensor，开销极小)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=self.device, dtype=torch.long)], dim=-1
            )
        
        # ========== 生成结束后返回隐藏状态 ==========
        if step_outputs is not None:
            last_hidden_state = step_outputs.get('features')
            if last_hidden_state is not None:
                # features 已是最后一层最后一个token的隐藏状态 [1, hidden_size]
                yield {"type": "hidden_state", "hidden_state": last_hidden_state.clone()}


    def generate(
        self,
        input_text: str,
        max_tokens: int = 150,
        temperature: float = 0.35,
        use_self_loop: bool = False,
        memory_anchor: Optional[torch.Tensor] = None,
        goal_vector: Optional[torch.Tensor] = None,  # 新增：目标向量
        repetition_penalty: float = 1.1,  # 降低重复惩罚
        **kwargs
    ):
        """
        STDP Strict Generate with KV-cache optimization
        
        新增功能：
        - goal_vector: 目标向量，注入到输入embedding中
        """
        start_time = time.time()
        
        # ========== 1. Tokenize 输入 (线程安全) ==========
        inputs = self.tokenize_safe(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # ========== 0.5. 窄带宽注意力安全检查 ==========
        # 修复：仅当有记忆锚点时才启用KV压缩，否则禁用（避免破坏正常注意力）
        # Qwen3.5 的窄带宽补丁会在 seq > window_size 时压缩KV，
        # 但在没有锚点时这会丢失上下文，导致输出乱码
        _narrow_was_disabled = False
        if NARROW_BAND_PATCHED:
            anchor_store = get_memory_anchor_store()
            current_anchors = anchor_store.get_enabled_anchors()
            if not current_anchors or len(current_anchors) == 0:
                # 无锚点时禁用KV压缩，保持完整注意力
                anchor_store.enabled = False
                _narrow_was_disabled = True
        
        # ========== 1.5. 目标向量注入（新增）==========
        # 使用局部变量而非实例属性，避免线程安全问题
        modified_embeddings = None
        if goal_vector is not None:
            # 将目标向量转换到与模型相同的dtype和device
            goal_vector = goal_vector.to(device=self.device, dtype=next(self.model.base_model.parameters()).dtype)
            
            # 检查goal_vector维度（兼容 Qwen3.5: hidden_size 在 text_config 里）
            expected_dim = _get_qwen_config_attr(self.model.base_model.config, 'hidden_size', 1024)
            if goal_vector.dim() != 1 or goal_vector.shape[0] != expected_dim:
                logger.warning(f"[目标向量] 维度不匹配: 期望{expected_dim}, 实际{goal_vector.shape}")
                logger.warning(f"[目标向量] 尝试调整维度...")
                
                # 尝试调整维度
                if goal_vector.dim() == 1:
                    if goal_vector.shape[0] < expected_dim:
                        # 填充到正确维度
                        padding = torch.zeros(expected_dim - goal_vector.shape[0], device=self.device)
                        goal_vector = torch.cat([goal_vector, padding])
                    elif goal_vector.shape[0] > expected_dim:
                        # 裁剪到正确维度
                        goal_vector = goal_vector[:expected_dim]
                else:
                    logger.error(f"[目标向量] 无法处理多维goal_vector，跳过注入")
                    goal_vector = None
            
            if goal_vector is not None:
                # 获取输入embedding
                with torch.no_grad():
                    input_embeddings = self.model.base_model.get_input_embeddings()(input_ids)
                    # input_embeddings: [batch, seq_len, hidden_size]
                    
                    # 将目标向量扩展到序列长度
                    # goal_vector: [hidden_size] -> [1, seq_len, hidden_size]
                    goal_expanded = goal_vector.unsqueeze(0).unsqueeze(0).expand(
                        input_embeddings.shape[0], 
                        input_embeddings.shape[1], 
                        -1
                    )
                    
                    # 注入目标向量（加权融合）
                    goal_injection_weight = 0.03  # 目标向量权重（大幅降低，避免破坏embedding）
                    modified_embeddings = input_embeddings * (1 - goal_injection_weight) + goal_expanded * goal_injection_weight
                    
                    logger.info(f"[目标注入] 已将目标向量注入到输入embedding (权重={goal_injection_weight})")
        
        # 定义停止token
        eos_token_id = self.model.tokenizer.eos_token_id
        im_end_token_id = _get_im_end_token_id(self.model.tokenizer)
        stop_token_ids = {eos_token_id, im_end_token_id}
        
        generated_tokens = []
        past_key_values = None
        step_outputs = None
        
        # 从 kwargs 中移除 use_cache，避免重复传递
        forward_kwargs = {k: v for k, v in kwargs.items() if k != 'use_cache'}
        # 关键修复：降低惩罚参数默认值，避免过度抑制正常词汇
        if 'presence_penalty' not in forward_kwargs:
            forward_kwargs['presence_penalty'] = 0.0
        if 'repetition_penalty' not in forward_kwargs:
            forward_kwargs['repetition_penalty'] = 1.1
        
        # 预分配 input_ids 缓冲区（避免每步 torch.cat 拼接）
        prompt_len = input_ids.shape[1]
        max_total = prompt_len + max_tokens
        with self._tokenizer_lock:
            pad_id = self.model.tokenizer.pad_token_id or 0
        input_ids_buf = torch.full((1, max_total), pad_id, dtype=torch.long, device=self.device)
        input_ids_buf[:, :prompt_len] = input_ids
        cur_len = prompt_len
        
        # N-gram 重复检测（大幅放宽阈值，避免截断正常输出）
        # BUG修复：原阈值过于激进（2-gram>8、3-gram>6、4-gram>3 就 break），
        # 0.8B 小模型语言多样性低，正常中文对话很容易触发这些阈值，
        # 导致回复被中途截断（出现 "▌" 符号）。
        # 新策略：只检测真正严重的回环（5-gram连续重复），对低阶n-gram仅惩罚不截断。
        recent_tokens = []
        ngram_repeat_count = {}
        # 只对 5-gram 及以上执行硬截断（真正的语言回环）
        max_5gram_repeat = 4  # 5-gram 连续出现4次才判定为回环
        
        for step in range(max_tokens):
            # Bug修复：在最后一步强制提取隐藏状态（维持意识连续性）
            # 之前只每 _stdp_every_n 步提取一次，最后一个token大概率hidden_state=None
            if step == max_tokens - 1:
                self._step_counter = (self._stdp_every_n - 1)
            if past_key_values is not None:
                model_input_ids = input_ids_buf[:, cur_len-1:cur_len]
                step_inputs_embeds = None
            else:
                model_input_ids = input_ids_buf[:, :cur_len]
                step_inputs_embeds = modified_embeddings
                
            step_outputs = self.forward_step(
                input_ids=model_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                memory_anchor_gate=memory_anchor,
                inputs_embeds=step_inputs_embeds,
                **forward_kwargs
            )
            
            next_token_id = step_outputs['token_id']
            past_key_values = step_outputs['past_key_values']
            
            # N-gram 重复检测（新策略：低阶仅追踪，高阶才截断）
            recent_tokens.append(next_token_id)
            if len(recent_tokens) > 60:  # 扩大窗口到60
                recent_tokens.pop(0)
            
            # 低阶n-gram（2-4）：仅记录，不截断（正常中文很容易触发）
            if len(recent_tokens) >= 2:
                ng2 = tuple(recent_tokens[-2:])
                ngram_repeat_count[ng2] = ngram_repeat_count.get(ng2, 0) + 1
            if len(recent_tokens) >= 3:
                ng3 = tuple(recent_tokens[-3:])
                ngram_repeat_count[ng3] = ngram_repeat_count.get(ng3, 0) + 1
            if len(recent_tokens) >= 4:
                ng4 = tuple(recent_tokens[-4:])
                ngram_repeat_count[ng4] = ngram_repeat_count.get(ng4, 0) + 1
                # 4-gram 重复超过 8 次才截断（之前是3次，太激进）
                if ngram_repeat_count[ng4] > 8:
                    break
            # 高阶n-gram（5-gram及以上）：严格截断（真正的回环）
            if len(recent_tokens) >= 5:
                ng5 = tuple(recent_tokens[-5:])
                ngram_repeat_count[ng5] = ngram_repeat_count.get(ng5, 0) + 1
                if ngram_repeat_count[ng5] > max_5gram_repeat:
                    break
            
            generated_tokens.append(next_token_id)
            
            if next_token_id in stop_token_ids:
                self.clear_memory_anchors()
                if NARROW_BAND_PATCHED:
                    get_memory_anchor_store().enabled = True
                break
                
            # 写入预分配缓冲区
            if cur_len < max_total:
                input_ids_buf[:, cur_len] = next_token_id
                cur_len += 1
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device, dtype=torch.long)], dim=-1)
        
        # ========== 3. 解码输出 ==========
        output_text = self.decode_safe(
            generated_tokens,
            skip_special_tokens=True
        )
        
        # ========== 4. 统计 ==========
        elapsed = time.time() - start_time
        self.total_generation_time += elapsed
        self.total_tokens_generated += len(generated_tokens)
        
        # ========== 5. 构建返回结果 ==========
        from core.interfaces import BrainAIOutput
        
        # 获取最后一个 token 的隐藏状态 (核心：维持意识连续性)
        last_hidden_state = None
        if step_outputs is not None:
            last_hidden_state = step_outputs.get('features')
            if last_hidden_state is not None:
                # features 已是最后一层最后一个token的隐藏状态 [1, hidden_size]
                pass  # 直接使用，无需进一步切片

        # ========== 清理临时变量 ==========
        # modified_embeddings 已改为局部变量，无需手动清理
        
        # ========== 重新启用窄带宽注意力 ==========
        # 为下次生成（可能带锚点）恢复启用状态
        if NARROW_BAND_PATCHED and _narrow_was_disabled:
            try:
                get_memory_anchor_store().enabled = True
            except Exception:
                pass

        return BrainAIOutput(
            text=output_text,
            tokens=generated_tokens,
            confidence=min(0.95, 0.7 + len(output_text) / 200.0),
            hidden_state=last_hidden_state,
            memory_anchors=[],  # 由海马体模块填充
            stdp_stats={},  # 由 STDP 模块填充
            cycle_stats={
                'total_cycles': self.total_tokens_generated,
                'avg_cycle_time_ms': (self.total_generation_time / self.total_tokens_generated * 1000)
                                    if self.total_tokens_generated > 0 else 0
            }
        )
    
    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        对话接口
        
        Args:
            message: 用户消息
            history: 对话历史
        
        Returns:
            response: 回复文本
        """
        # 构建带上下文的输入
        if history:
            context = "\n".join([
                f"{h['role']}: {h['content']}" 
                for h in history[-5:]
            ])
            full_input = f"{context}\nUser: {message}\nAssistant:"
        else:
            full_input = f"User: {message}\nAssistant:"
        
        # 生成回复
        output = self.generate(full_input, max_tokens=200, **kwargs)
        
        return output.text
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'system': {
                'total_tokens': self.total_tokens_generated,
                'total_time': self.total_generation_time,
                'avg_time_per_token': (
                    self.total_generation_time / self.total_tokens_generated * 1000
                    if self.total_tokens_generated > 0 else 0
                ),
                'device': self.device,
                'quantization': self.model.quantization
            }
        }
        
    def apply_stdp_to_layer(
        self,
        layer_name: str,
        grad_dict: Dict[str, torch.Tensor],
        lr: float = 0.01
    ):
        """Apply STDP update to a specific layer by name."""
        for name, module in self.model.base_model.named_modules():
            if name == layer_name and hasattr(module, 'apply_stdp_update'):
                if layer_name in grad_dict:
                    module.apply_stdp_update(grad_dict[layer_name], lr)
                break
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'dynamic_weights': self.model.get_all_dynamic_weights(),
            'config': self.config,
            'stats': self.get_stats()
        }
        torch.save(checkpoint, path)
        print(f"[QwenInterface] 检查点已保存：{path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # 恢复动态权重
        dynamic_weights = checkpoint.get('dynamic_weights', {})
        restored_count = 0
        
        for layer_name, weights in dynamic_weights.items():
            try:
                # 查找对应的层并恢复权重
                for name, module in self.model.base_model.named_modules():
                    if name == layer_name and hasattr(module, 'apply_stdp_update'):
                        # 使用 apply_stdp_update 方法恢复权重
                        module.apply_stdp_update(weights, lr=1.0)  # lr=1.0 直接设置权重
                        restored_count += 1
                        break
                    elif hasattr(module, 'q_proj') and hasattr(module, 'apply_stdp_update'):
                        # 对于注意力层，尝试匹配子层
                        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                            if f"{layer_name}.{proj_name}" in weights or proj_name in weights:
                                proj = getattr(module, proj_name, None)
                                if proj and hasattr(proj, 'dynamic_weight'):
                                    w = weights.get(proj_name, weights.get(f"{layer_name}.{proj_name}"))
                                    if w is not None:
                                        proj.dynamic_weight.data.copy_(w)
                                        restored_count += 1
            except Exception as e:
                print(f"[QwenInterface] 恢复权重失败 {layer_name}: {e}")
        
        # 恢复统计信息
        if 'stats' in checkpoint:
            stats = checkpoint['stats']
            if 'system' in stats:
                self.total_generation_time = stats['system'].get('total_time', 0)
                self.total_tokens_generated = stats['system'].get('total_tokens', 0)
        
        print(f"[QwenInterface] 检查点已加载：{path} (恢复 {restored_count} 个权重层)")

