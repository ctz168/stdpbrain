"""
真实 Qwen 模型集成接口

功能:
- 加载真实的 Qwen3.5-0.8B 模型
- 将双权重层集成到真实模型中
- 提供完整的生成和对话接口
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import asyncio
import sys


class QwenModelWrapper(nn.Module):
    """
    Qwen3.5-0.8B 模型包装器
    
    将双权重层集成到真实 Qwen 模型中
    """
    
    def __init__(
        self, 
       model_path: str,
        config,
        device: str = "cpu",
        quantization: str = "INT4"
    ):
        super().__init__()
        import threading
        self._tokenizer_lock = threading.Lock() # 线程锁，防止 Already borrowed
        self.model_path = model_path
        self.config = config
        self.device = device
        self.quantization = quantization
        print(f"[QwenWrapper] 正在加载真实 Qwen 模型...")
        print(f"  路径：{model_path}")
        print(f"  设备：{device}")
        print(f"  量化：{quantization}")
        
        # ========== 1. 加载 Tokenizer ==========
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
               model_path,
                trust_remote_code=True,
               padding_side="left"
            )
            print(f"[OK] Tokenizer 加载成功，词表大小：{len(self.tokenizer)}")
        except Exception as e:
                print(f"[ERROR] Tokenizer 加载失败：{e}")
                raise
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
        try:
            # 自动检测 Apple Silicon (M1/M2/M3)
            is_mac = torch.backends.mps.is_available()
            
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
                    # CPU 环境：使用更高兼容性的 FP32 加载
                    print(f"  [!] [CPU] 已启用优化加载流程...")
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    ).to("cpu")
                    print("  [OK] [CPU] FP32 模型加载完成")
                
            else:  # FP16 或 FP32
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
            
        except Exception as e:
            print(f"[!] 模型加载异常：{e}，回退到标准 FP32 加载")
            # 最终回退 - 更新设备为 CPU
            self.device = "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            print(f"  [!] 设备已更新为: {self.device}")
            return model

    def tokenize_safe(self, text, **kwargs):
        """线程安全的 Tokenization 封装"""
        with self._tokenizer_lock:
            return self.tokenizer(text, **kwargs)

    def decode_safe(self, token_ids, **kwargs):
        """线程安全的 Decoding 封装"""
        with self._tokenizer_lock:
            return self.tokenizer.decode(token_ids, **kwargs)

    def apply_chat_template_safe(self, messages, **kwargs):
        """线程安全的 Chat Template 应用"""
        with self._tokenizer_lock:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
    
    def _integrate_dual_weights(self):
        """
        将双权重层集成到 Qwen 模型中
        
        替换每个 Transformer 层的线性层为 DualWeightLinear
        """
        from core.dual_weight_layers import DualWeightLinear, DualWeightAttention, DualWeightFFN
        
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
                    
                    # 确保是 Linear 层且尚未被替换 (避免重复计数)
                    if isinstance(target_layer, nn.Linear) and not isinstance(target_layer, DualWeightLinear):
                        try:
                            # 获取原始权重
                            static_weight = target_layer.weight.data.clone()
                            bias = target_layer.bias.data.clone() if target_layer.bias is not None else None
                            
                            # 创建双权重线性层
                            dual_layer = DualWeightLinear(
                                in_features=target_layer.in_features,
                                out_features=target_layer.out_features,
                                bias=(bias is not None),
                                static_weight=static_weight
                            )
                            if bias is not None:
                                dual_layer.bias.data.copy_(bias)
                            
                            # 替换
                            setattr(module, attr_name, dual_layer)
                            replaced_count += 1
                        except Exception as e:
                            print(f"  [!] 替换层失败 {name}.{attr_name}: {e}")
        
        print(f"[OK] 已替换 {replaced_count} 个底层投影为双权重版本")
        
        # ========== 后置优化：如果是 CPU，执行动态量化 ==========
        if self.device == "cpu" and self.quantization == "INT8":
            print(f"  ⚡ 正在对基础模型执行动态量化 (INT8)...")
            # 动态量化会寻找 nn.Linear。注意：DualWeightLinear 默认不被识别，这很好，
            # 因为我们需要动态权重保持浮点高精度以进行 STDP 学习。
            self.base_model = torch.quantization.quantize_dynamic(
                self.base_model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("  [OK] [CPU] 后置动态量化完成")
        
        # ========== 后置优化：如果是 CPU，执行动态量化 ==========
        # 注意：我们只量化非 DualWeight 的部分，或者直接对整个模型尝试
        if self.device == "cpu" and self.quantization == "INT8":
            print(f"  ⚡ 正在对基础模型执行动态量化 (INT8)...")
            # 仅量化 nn.Linear 层，回避我们的 DualWeight 模块
            self.base_model = torch.quantization.quantize_dynamic(
                self.base_model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("  [OK] [CPU] 后置动态量化完成")
    
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
        
        # 设置DualWeightAttention的类变量（供所有注意力层访问）
        from core.dual_weight_layers import DualWeightAttention
        DualWeightAttention.set_memory_anchor(memory_anchor)
        
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
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50,
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
    
    def get_all_dynamic_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """获取所有动态权重"""
        dynamic_weights = {}
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, DualWeightAttention):
                dynamic_weights[name] = module.get_all_dynamic_weights()
            elif isinstance(module, DualWeightFFN):
                dynamic_weights[name] = module.get_all_dynamic_weights()
        
        return dynamic_weights

    def save_checkpoint(self, path: str):
        """保存模型检查点 (仅保存双权重层的动态部分)"""
        dynamic_weights = self.get_all_dynamic_weights()
        torch.save(dynamic_weights, path)
        print(f"[QwenWrapper] 动态权重已保存到: {path}")

    def load_checkpoint(self, path: str):
        """加载模型检查点 (仅加载双权重层的动态部分)"""
        dynamic_weights = torch.load(path, map_location=self.device)
        for layer_name, weights in dynamic_weights.items():
            self.apply_stdp_to_layer(layer_name, weights, lr=1.0) # 使用 lr=1.0 直接设置权重
        print(f"[QwenWrapper] 动态权重已从 {path} 加载")
    
    def apply_stdp_to_layer(
        self,
        layer_name: str,
        grad_dict: Dict[str, torch.Tensor],
        lr: float = 0.01
    ):
        """对指定层应用 STDP 更新"""
        for name, module in self.base_model.named_modules():
            if name == layer_name:
                if hasattr(module, 'apply_stdp_to_all'):
                    module.apply_stdp_to_all(grad_dict, lr)
                break
    
    def apply_stdp_to_all(self, grad_dict: Dict[str, torch.Tensor], lr: float = 0.01):
        """对所有双权重层广播 STDP 更新"""
        for module in self.base_model.modules():
            if hasattr(module, 'apply_stdp_to_all'):
                module.apply_stdp_to_all(grad_dict, lr)


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
        
        # 优化1: CPU 线程控制
        if self.device == "cpu":
            import multiprocessing
            num_cores = multiprocessing.cpu_count()
            # 设置线程数为物理核心数（通常比逻辑核心数效率更高）
            torch.set_num_threads(max(1, num_cores // 2))
            print(f"[QwenInterface] CPU 优化：设置线程数为 {torch.get_num_threads()}")

        # 优化4: STDP 后台线程池，不阻塞生成循环
        import concurrent.futures
        self._stdp_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='stdp_worker'
        )
        # 每 N 步才提取一次 features 用于 STDP (优化3基础)
        self._step_counter = 0
        self._stdp_every_n = getattr(config.stdp, 'every_n_tokens', 1)
        self._last_reward = 1.0  # 存储最近一次对话的评判得分
        self._step_counter = 0
        self._stdp_every_n = 50
        
        # 统计信息
        self.total_generation_time = 0.0
        self.total_tokens_generated = 0
    
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

    def apply_chat_template_safe(self, messages, **kwargs):
        """传递调用到模型包装器"""
        return self.model.apply_chat_template_safe(messages, **kwargs)
    

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
            
            # 向量化重复惩罚 (scatter-based)
            repetition_penalty = kwargs.get('repetition_penalty', 1.2) # 从 1.1 提升到 1.2
            if repetition_penalty != 1.0 and input_ids is not None:
                # 获取每个 batch item 的不重复 token
                # 注意：在当前 batch_size=1 的情况下，这等同于 unique()
                unique_tokens = torch.unique(input_ids)
                
                # 创建惩罚值
                seen_logits = next_token_logits[:, unique_tokens]
                penalized_logits = torch.where(
                    seen_logits > 0,
                    seen_logits / repetition_penalty,
                    seen_logits * repetition_penalty
                )
                
                # 使用 scatter 将惩罚应用回原位置
                next_token_logits.scatter_(1, unique_tokens.unsqueeze(0), penalized_logits)

            # 温度缩放
            temp = kwargs.get('temperature', 0.7)
            if temp > 0:
                next_token_logits = next_token_logits / temp
            
            # Top-k 过滤
            top_k = kwargs.get('top_k', 50)
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            # 采样或贪心
            if temp > 0:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
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
        
        # ========== 优化3: 启动后台 STDP 学习任务 ==========
        if self.config.stdp.enabled and need_features:
            # 使用最近一次的奖励反馈，如果没有则使用 baseline
            current_reward = self._last_reward
            
            # 触发所有层的 STDP 更新 (在后台线程运行)
            for name, layer in self.model.base_model.named_modules():
                if hasattr(layer, 'apply_stdp_to_all'):
                    self._stdp_executor.submit(
                        self.model.config.stdp_engine.update_attention_layer,
                        layer,
                        input_ids.flatten(),
                        outputs['token_id'],
                        outputs.get('features', torch.zeros(1)), # 这里的 features 用于 contributions
                        time.time() * 1000,
                        reward=current_reward,
                        is_tool_call=is_tool_call
                    )
        
        return outputs



    def generate_stream_sync(
        self,
        input_text: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        同步流式生成 (KV-cache + 预分配 input_ids 缓冲区)
        
        Yields:
            str: 每个生成的字符/词
        """
        # ========== 1. Tokenize 输入 (线程安全) ==========
        with self._tokenizer_lock:
            inputs = self.model.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
        
        prompt_ids = inputs.input_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]
        
        # 预分配 input_ids 缓冲区
        max_total = prompt_len + max_tokens
        with self._tokenizer_lock:
            pad_id = self.model.tokenizer.pad_token_id or 0
            eos_id = self.model.tokenizer.eos_token_id
            
        input_ids_buf = torch.full(
            (1, max_total), pad_id,
            dtype=torch.long, device=self.device
        )
        input_ids_buf[:, :prompt_len] = prompt_ids
        cur_len = prompt_len
        
        attention_mask = inputs.attention_mask.to(self.device)
        past_key_values = None
        
        # 定义停止token
        eos_token_id = self.model.tokenizer.eos_token_id
        im_end_token_id = 151645  # <|im_end|>
        stop_token_ids = {eos_token_id, im_end_token_id}
        
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
                **kwargs
            )
            
            next_token_id = step_outputs['token_id']
            past_key_values = step_outputs['past_key_values']
            
            # 解码单个token
            token_text = self.model.tokenizer.decode([next_token_id], skip_special_tokens=True)
            yield token_text
            
            if next_token_id in stop_token_ids:
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
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        异步流式生成 (KV-cache + 预分配 input_ids 缓冲区 + 条件 STDP)
        """
        # ========== 1. Tokenize 输入 ==========
        inputs = self.model.tokenizer(
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
        input_ids_buf = torch.full(
            (1, max_total), self.model.tokenizer.pad_token_id or 0,
            dtype=torch.long, device=self.device
        )
        input_ids_buf[:, :prompt_len] = prompt_ids
        cur_len = prompt_len  # 指针追踪实际长度
        
        attention_mask = inputs.attention_mask.to(self.device)
        past_key_values = None
        
        # 定义停止token
        eos_token_id = self.model.tokenizer.eos_token_id
        im_end_token_id = 151645  # <|im_end|>
        stop_token_ids = {eos_token_id, im_end_token_id}
        
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
                **kwargs
            )
            
            next_token_id = step_outputs['token_id']
            past_key_values = step_outputs['past_key_values']
            
            token_text = self.model.tokenizer.decode([next_token_id], skip_special_tokens=True)
            yield token_text
            
            if next_token_id in stop_token_ids:
                break
            
            # 优化6: 写入预分配缓冲区，无内存分配
            if cur_len < max_total:
                input_ids_buf[:, cur_len] = next_token_id
                cur_len += 1
            
            # 更新 attention mask (仍需 cat，但只是 1 维 bool tensor，开销极小)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=self.device, dtype=torch.long)], dim=-1
            )


    def generate(
        self,
        input_text: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        use_self_loop: bool = False,
        memory_anchor: Optional[torch.Tensor] = None,
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
        
        # Initialize internal STDP tracker
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
                **kwargs
            )
            
            next_token_id = step_outputs['token_id']
            past_key_values = step_outputs['past_key_values']
            generated_tokens.append(next_token_id)
            
            if next_token_id in stop_token_ids:
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
        output_text = self.model.tokenizer.decode(
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


def create_real_qwen_ai(
   model_path: str,
    device: str = "cpu",
    quantization: str = "INT4"
) -> QwenInterface:
    """
    快捷创建真实 Qwen AI 实例
    
    Args:
       model_path: 模型路径
        device: 设备
        quantization: 量化类型
    
    Returns:
        ai: QwenInterface 实例
    """
    from configs.arch_config import default_config
    
    return QwenInterface(
       model_path=model_path,
        config=default_config,
        device=device,
        quantization=quantization
    )
