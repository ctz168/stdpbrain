"""
真实 Qwen 模型集成接口

功能:
- 加载真实的 Qwen3.5-0.8B 模型
- 将双权重层集成到真实模型中
- 提供完整的生成和对话接口
- 支持窄带宽注意力（记忆锚点注入）
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import asyncio
import sys
import logging

# 配置日志
logger = logging.getLogger(__name__)

# ========== 窄带宽注意力补丁 ==========
# 在导入时应用补丁，修改 Qwen 内部注意力层
from core.qwen_narrow_band_patch import patch_qwen_attention, get_memory_anchor_store
patch_qwen_attention()
NARROW_BAND_PATCHED = True

# ========== KV Cache 滑动窗口管理器 ==========
from core.kv_cache_manager import KVCacheManager
KV_CACHE_MANAGER_AVAILABLE = True


class QwenModelWrapper(nn.Module):
    """
    Qwen3.5-2B 模型包装器
    
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
        self._tokenizer_lock = threading.Lock() # 线程锁，防止 Already borrowed
        self.model_path = model_path
        self.config = config
        self.device = device
        
        # 统一读取优先级：1.全局 config.QUANTIZATION 2. 全局 config.quantization 3. 传参配置
        self.quantization = getattr(config, 'QUANTIZATION', getattr(config, 'quantization', quantization))
        
        print(f"[QwenWrapper] 正在加载真实 Qwen 模型...")
        print(f"  路径：{model_path}")
        print(f"  设备：{device}")
        print(f"  量化：{self.quantization}")
        
        # ========== 1. 加载 Tokenizer ==========
        self.tokenizer = AutoTokenizer.from_pretrained(
           model_path,
            trust_remote_code=True,
           padding_side="left"
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
                self.quantization = "FP32"  # CPU 使用 FP32（避免缓慢的动态量化）
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
                    trust_remote_code=True
                )
                print(f"  [OK] [CUDA] {self.quantization} 量化加载成功")
            elif is_mac:
                # macOS 下使用 FP16
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map={"": "mps"},
                    trust_remote_code=True
                )
                print("  [OK] [macOS/MPS] 使用 FP16 加载成功")
            else:
                # CPU 环境：使用 FP16 加载 + 动态量化（避免内存溢出）
                print(f"  [!] [CPU] bitsandbytes 的 {self.quantization} 量化完全依赖 CUDA (Nvidia GPU)。")
                print(f"  [!] [CPU] 使用 FP16 加载 + 动态量化方式（内存占用减半）...")
                
                # Step 1: 以 FP16 加载（比 FP32 节省一半内存）
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
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
                trust_remote_code=True
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
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        memory_anchor: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """前向传播 (支持 KV-cache 和记忆锚点)"""
        # 确保默认值
        out_attentions = output_attentions if output_attentions is not None else False
        ret_dict = return_dict if return_dict is not None else True
        use_cache = use_cache if use_cache is not None else False
        
        # 存储记忆锚点，供注意力层使用
        self._current_memory_anchor = memory_anchor
        
        # 注意力层记忆锚点之前由 DualWeightAttention 管理，现已交由底层注意力层处理
        # (在此预留基座模型可能的 hippocampus_gate 设置钩子)
        
        # 清理 kwargs 避免重复冲突
        exclude = {'return_dict', 'output_attentions', 'output_hidden_states', 'use_cache', 'past_key_values', 'memory_anchor'}
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in exclude}
        
        # 使用 base_model 进行推理 (如果 base_model 支持 past_key_values)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=out_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=ret_dict,
            **clean_kwargs
        )
        
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
        dynamic_weights = torch.load(path, map_location=self.device)
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
        # 每 N 步才提取一次 features 用于 STDP (优化3基础)
        self._step_counter = 0
        self._stdp_every_n = getattr(config.stdp, 'every_n_tokens', 1)  # 改为默认每步更新
        self._last_reward = 1.0  # 存储最近一次对话的评判得分
        self._step_counter = 0
        self._stdp_every_n = 5  # 每 5 步更新一次 STDP（平衡性能和学习效果）
        
        # ========== 新增: 性能优化缓存 ==========
        # Token 计数器缓存（避免每步重复 unique）
        self._token_counter = torch.zeros(1, dtype=torch.long, device=self.device)
        self._token_counts_cache = {}  # token_id -> count
        
        # 模块索引（避免每次扫描 named_modules）
        self._stdp_layers_cache = None
        
        # 预分配惩罚张量（复用内存）
        vocab_size = getattr(self.model.base_model.config, 'vocab_size', 152000)
        self._penalty_buffer = torch.zeros(vocab_size, device=self.device)
        
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
    
    def set_memory_anchors(self, anchors):
        """
        设置记忆锚点（用于窄带宽注意力）
        
        Args:
            anchors: 记忆锚点列表
        """
        self._memory_anchors_cache = anchors
        # 同时传递给模型
        if hasattr(self.model, 'set_memory_anchors'):
            self.model.set_memory_anchors(anchors)
    
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
    
    def set_memory_anchors(self, anchors: list, max_anchors: int = 5, strength: float = 1.0):
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
        except:
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
        **kwargs
    ) -> Dict[str, Any]:
        """单步前向推理 (KV-cache + inference_mode + 条件特征提取)"""
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
            output_tensors = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                memory_anchor=memory_anchor_gate,
                output_attentions=False,
                output_hidden_states=need_features,  # 优化3: 按需提取
                return_dict=True,
                **clean_kwargs
            )
            
            next_token_logits = output_tensors.logits[:, -1, :].clone()
            
            # ========== 优化: 增量 token 计数，避免重复 unique ==========
            repetition_penalty = kwargs.get('repetition_penalty', 1.0)
            presence_penalty = kwargs.get('presence_penalty', 1.5)
            
            if (repetition_penalty != 1.0 or presence_penalty > 0) and input_ids is not None:
                # 只处理最新添加的 token（增量更新）
                if input_ids.shape[-1] > 0:
                    # 获取所有历史 token（使用缓存的计数）
                    # 但为了简单和正确性，我们一次性处理
                    
                    # 向量化: 一次 unique 调用同时获取 token 和计数
                    unique_tokens, counts = torch.unique(input_ids, return_counts=True)
                    
                    # Repetition Penalty (向量化 scatter)
                    if repetition_penalty != 1.0:
                        seen_logits = next_token_logits[:, unique_tokens]
                        penalized_logits = torch.where(
                            seen_logits > 0,
                            seen_logits / repetition_penalty,
                            seen_logits * repetition_penalty
                        )
                        next_token_logits.scatter_(1, unique_tokens.unsqueeze(0), penalized_logits)
                    
                    # Presence Penalty (复用 unique 结果)
                    if presence_penalty > 0:
                        # 使用预分配缓冲区
                        penalty = self._penalty_buffer.zero_()
                        penalty[unique_tokens] = counts.float() * presence_penalty
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
                hidden_size = getattr(self.model.base_model.config, 'hidden_size', 1024)
                outputs['features'] = None  # 标记为不可用，STDP engine 跳过
                
            outputs['attention_output'] = torch.zeros(1)
            outputs['ffn_output'] = outputs['features'] if outputs['features'] is not None else torch.zeros(1)
            outputs['generation_path'] = str(next_token.item())
            outputs['evaluation_score'] = 35.0
            
        elapsed = time.time() - start_time
        outputs['cycle_time_ms'] = elapsed * 1000.0
        outputs['past_key_values'] = output_tensors.past_key_values if hasattr(output_tensors, 'past_key_values') else None
        self.total_tokens_generated += 1
        
        # ========== 优化: 使用缓存的模块索引 ==========
        if self.config.stdp.enabled and need_features:
            # 延迟初始化模块索引
            if self._stdp_layers_cache is None:
                self._stdp_layers_cache = [
                    (name, layer) for name, layer in self.model.base_model.named_modules()
                    if hasattr(layer, 'apply_stdp_update')
                ]
            
            current_reward = self._last_reward
            
            # 使用索引遍历（避免重复扫描）
            for name, layer in self._stdp_layers_cache:
                self._stdp_executor.submit(
                    self.model.config.stdp_engine.update_attention_layer,
                    layer,
                    input_ids.flatten(),
                    outputs['token_id'],
                    outputs.get('features', torch.zeros(1)),
                    time.time() * 1000,
                    reward=current_reward,
                    is_tool_call=is_tool_call
                )
        
        return outputs



    def generate_stream_sync(
        self,
        input_text: str,
        max_tokens: int = 150,
        temperature: float = 0.45,
        repetition_penalty: float = 1.3,  # 提高重复惩罚到1.3
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
        max_context = getattr(self.model.config.hard_constraints, 'MAX_CONTEXT_LENGTH', 512)
        narrow_window = getattr(self.model.config.hard_constraints, 'NARROW_WINDOW_SIZE', 5)
        
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
        
        attention_mask = inputs.attention_mask.to(self.device)
        past_key_values = None
        
        # 定义停止token
        with self._tokenizer_lock:
            eos_token_id = self.tokenizer.eos_token_id
        im_end_token_id = 151645  # <|im_end|>
        stop_token_ids = {eos_token_id, im_end_token_id}
        
        # 重复检测变量
        recent_tokens = []  # 最近生成的token
        ngram_repeat_count = {}  # n-gram重复计数
        max_repeat_allowed = 3  # 允许的最大重复次数
        
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
            
            # ========== KV Cache 滑动窗口管理 ==========
            # 检查是否启用KV cache滑动窗口
            enable_kv_sliding = getattr(self.config.hard_constraints, 'ENABLE_KV_SLIDING_WINDOW', True)
            kv_window_size = getattr(self.config.hard_constraints, 'KV_CACHE_WINDOW_SIZE', 32)
            
            if enable_kv_sliding and past_key_values is not None and KV_CACHE_MANAGER_AVAILABLE:
                # 创建KV管理器（懒加载）
                if not hasattr(self, '_kv_cache_manager'):
                    self._kv_cache_manager = KVCacheManager(
                        window_size=kv_window_size,
                        enable_hippocampus=False  # 这里不直接存储到海马体，由上层BrainAI管理
                    )
                
                # 修剪KV cache
                past_key_values, _ = self._kv_cache_manager.trim_kv_cache(
                    past_key_values,
                    current_token_text=None,
                    hippocampus=None
                )
            
            token_text = self.decode_safe([next_token_id], skip_special_tokens=True)
            
            # 使用简单的阻断词拦截标签幻觉
            unsafe_keywords = ["<|im_start|>", "\nuser", "\nUser", "User:", "💭", "[内心独白]", "[潜意识]"]
            full_decode = self.decode_safe(input_ids_buf[0, prompt_len:cur_len+1], skip_special_tokens=True)
            if any(kw in full_decode for kw in unsafe_keywords):
                break
            
            # ========== N-gram 重复检测 ==========
            recent_tokens.append(next_token_id)  # next_token_id 已经是 int
            if len(recent_tokens) > 25:  # 增加到25个token
                recent_tokens.pop(0)
            
            # 检测2-gram和3-gram重复
            if len(recent_tokens) >= 2:
                # 2-gram检测（更严格）
                ngram2 = tuple(recent_tokens[-2:])
                ngram_repeat_count[ngram2] = ngram_repeat_count.get(ngram2, 0) + 1
                if ngram_repeat_count[ngram2] > max_repeat_allowed:
                    break
            
            if len(recent_tokens) >= 3:
                # 3-gram检测
                ngram3 = tuple(recent_tokens[-3:])
                ngram_repeat_count[ngram3] = ngram_repeat_count.get(ngram3, 0) + 1
                if ngram_repeat_count[ngram3] > max_repeat_allowed:
                    break
            
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
        repetition_penalty: float = 1.3,  # 提高重复惩罚到1.3
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
        im_end_token_id = 151645  # <|im_end|>
        stop_token_ids = {eos_token_id, im_end_token_id}
        
        # 重复检测变量
        recent_tokens = []
        ngram_repeat_count = {}
        max_repeat_allowed = 2  # 降低到2，更严格
        step_outputs = None  # 初始化，用于循环后提取隐藏状态
        
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
            
            # ========== KV Cache 滑动窗口管理 ==========
            # 检查是否启用KV cache滑动窗口
            enable_kv_sliding = getattr(self.config.hard_constraints, 'ENABLE_KV_SLIDING_WINDOW', True)
            kv_window_size = getattr(self.config.hard_constraints, 'KV_CACHE_WINDOW_SIZE', 32)
            
            if enable_kv_sliding and past_key_values is not None and KV_CACHE_MANAGER_AVAILABLE:
                # 创建KV管理器（懒加载）
                if not hasattr(self, '_kv_cache_manager'):
                    self._kv_cache_manager = KVCacheManager(
                        window_size=kv_window_size,
                        enable_hippocampus=False  # 这里不直接存储到海马体，由上层BrainAI管理
                    )
                
                # 修剪KV cache
                past_key_values, _ = self._kv_cache_manager.trim_kv_cache(
                    past_key_values,
                    current_token_text=None,
                    hippocampus=None
                )
            
            token_text = self.decode_safe([next_token_id], skip_special_tokens=True)
            
            # 使用简单的阻断词拦截标签幻觉
            unsafe_keywords = ["<|im_start|>", "\nuser", "\nUser", "User:", "💭", "[内心独白]", "[潜意识]"]
            full_decode = self.decode_safe(input_ids_buf[0, prompt_len:cur_len+1], skip_special_tokens=True)
            if any(kw in full_decode for kw in unsafe_keywords):
                break
            
            # ========== N-gram 重复检测 ==========
            recent_tokens.append(next_token_id)  # next_token_id 已经是 int
            if len(recent_tokens) > 25:  # 增加到25个token
                recent_tokens.pop(0)
            
            # 检测2-gram和3-gram重复
            if len(recent_tokens) >= 2:
                # 2-gram检测（更严格）
                ngram2 = tuple(recent_tokens[-2:])
                ngram_repeat_count[ngram2] = ngram_repeat_count.get(ngram2, 0) + 1
                if ngram_repeat_count[ngram2] > max_repeat_allowed:
                    logger.warning(f"[重复检测] 2-gram重复超过阈值，停止生成")
                    break
            
            if len(recent_tokens) >= 3:
                # 3-gram检测
                ngram3 = tuple(recent_tokens[-3:])
                ngram_repeat_count[ngram3] = ngram_repeat_count.get(ngram3, 0) + 1
                if ngram_repeat_count[ngram3] > max_repeat_allowed:
                    logger.warning(f"[重复检测] 3-gram重复超过阈值，停止生成")
                    break
            
            # 检测连续重复字符（更严格）
            if len(full_decode) > 15:
                last_15 = full_decode[-15:]
                # 如果最后15个字符中，不重复的字符数少于5个，停止
                if len(set(last_15)) <= 4:
                    logger.warning(f"[重复检测] 检测到字符重复模式，停止生成")
                    break
            
            # 检测词组重复（新增）
            if len(full_decode) > 30:
                # 检查最后30个字符中是否有重复的词组
                last_30 = full_decode[-30:]
                words = last_30.split()
                if len(words) >= 4:
                    # 检查是否有连续的词重复
                    for i in range(len(words) - 1):
                        if words[i] == words[i+1] and len(words[i]) > 2:
                            logger.warning(f"[重复检测] 检测到词组重复: {words[i]}")
                            break
            
            yield token_text
            
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
            last_hidden_state = step_outputs.get('hidden_states')
            if last_hidden_state is not None:
                # 取最后一层，最后一个 token
                last_hidden_state = last_hidden_state[-1][:, -1, :].clone()
                yield {"type": "hidden_state", "hidden_state": last_hidden_state}


    def generate(
        self,
        input_text: str,
        max_tokens: int = 150,
        temperature: float = 0.35,
        use_self_loop: bool = False,
        memory_anchor: Optional[torch.Tensor] = None,
        repetition_penalty: float = 1.3,  # 提高重复惩罚到1.3
        **kwargs
    ):
        """
        STDP Strict Generate with KV-cache optimization
        """
        start_time = time.time()
        
        # ========== 1. Tokenize 输入 ==========
        inputs = self.model.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # 定义停止token
        eos_token_id = self.model.tokenizer.eos_token_id
        im_end_token_id = 151645  # <|im_end|>
        stop_token_ids = {eos_token_id, im_end_token_id}
        
        generated_tokens = []
        past_key_values = None
        step_outputs = None  # 初始化，用于循环后提取隐藏状态
        
        # Initialize internal STDP tracker
        # 从 kwargs 中移除 use_cache，避免重复传递
        forward_kwargs = {k: v for k, v in kwargs.items() if k != 'use_cache'}
        
        for step in range(max_tokens):
            # KV-cache 模式: 只传最后一个 token
            if past_key_values is not None:
                model_input_ids = input_ids[:, -1:]
            else:
                model_input_ids = input_ids
                
            step_outputs = self.forward_step(
                input_ids=model_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                memory_anchor_gate=memory_anchor,
                **forward_kwargs
            )
            
            next_token_id = step_outputs['token_id']
            past_key_values = step_outputs['past_key_values']
            
            # ========== KV Cache 滑动窗口管理 ==========
            # 检查是否启用KV cache滑动窗口
            enable_kv_sliding = getattr(self.config.hard_constraints, 'ENABLE_KV_SLIDING_WINDOW', True)
            kv_window_size = getattr(self.config.hard_constraints, 'KV_CACHE_WINDOW_SIZE', 32)
            
            if enable_kv_sliding and past_key_values is not None and KV_CACHE_MANAGER_AVAILABLE:
                # 创建KV管理器（懒加载）
                if not hasattr(self, '_kv_cache_manager'):
                    self._kv_cache_manager = KVCacheManager(
                        window_size=kv_window_size,
                        enable_hippocampus=False  # 这里不直接存储到海马体，由上层BrainAI管理
                    )
                
                # 修剪KV cache
                past_key_values, _ = self._kv_cache_manager.trim_kv_cache(
                    past_key_values,
                    current_token_text=None,
                    hippocampus=None
                )
            generated_tokens.append(next_token_id)
            
            if next_token_id in stop_token_ids:
                self.clear_memory_anchors()
                break
                
            # 更新 input_ids 和 attention_mask
            next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)
            # 确保类型一致 (long)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device, dtype=torch.long)], dim=-1)
            
            # 每 10 步应用一次 STDP 更新 (优化)
            if self.total_tokens_generated % 10 == 0:
                self.apply_stdp_to_layer('model.layers.0', {'weight': torch.randn(1, 1, device=self.device) * 0.01})
        
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
            last_hidden_state = step_outputs.get('hidden_states')
            if last_hidden_state is not None:
                # 取最后一层，最后一个 token
                last_hidden_state = last_hidden_state[-1][:, -1, :].clone()

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
        """Pass-through to model wrapper."""
        self.model.apply_stdp_to_layer(layer_name, grad_dict, lr)
    
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
                    if name == layer_name and hasattr(module, 'apply_stdp_to_all'):
                        # 使用 apply_stdp_to_all 方法恢复权重
                        module.apply_stdp_to_all(weights, lr=1.0)  # lr=1.0 直接设置权重
                        restored_count += 1
                        break
                    elif hasattr(module, 'q_proj') and hasattr(module, 'apply_stdp_to_all'):
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

