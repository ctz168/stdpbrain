"""
GLM 模型集成接口

功能:
- 支持智谱 AI 的 GLM 系列模型 (ChatGLM-6B, GLM-4 等)
- 将双权重层集成到 GLM 模型中
- 提供完整的生成和对话接口
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time


class GLMModelWrapper(nn.Module):
    """
    GLM 模型包装器
    
    支持智谱 AI 的 GLM 系列模型，将双权重层集成到模型中
    """
    
    def __init__(
        self, 
        model_path: str,
        config,
        device: str = "cpu",
        quantization: str = "FP16"
    ):
        super().__init__()
        self.model_path = model_path
        self.config = config
        self.device = device
        self.quantization = quantization
        
        print(f"[GLMWrapper] 正在加载 GLM 模型...")
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
        
        # ========== 2. 加载模型 ==========
        self.base_model = self._load_model_with_quantization()
        
        # ========== 3. 集成双权重层 ==========
        self._integrate_dual_weights()
        
        # ========== 4. 设置填充 token ==========
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        param_count = sum(p.numel() for p in self.parameters())
        print(f"✓ GLM 模型加载完成")
        print(f"  - 参数量：{param_count:,} ({param_count/1e6:.2f}M)")
        print(f"  - 设备：{device}")
    
    def _load_model_with_quantization(self):
        """根据量化类型加载模型"""
        try:
            is_mac = torch.backends.mps.is_available()
            
            if self.quantization in ["INT4", "INT8"]:
                if self.device == "cuda":
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
                    print(f"  ✓ [CUDA] {self.quantization} 量化加载成功")
                elif is_mac:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        device_map={"": "mps"},
                        trust_remote_code=True
                    )
                    print("  ✓ [macOS/MPS] 使用 FP16 加载成功")
                else:
                    print(f"  ⚠️ [CPU] 已启用优化加载流程...")
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    ).to("cpu")
                    print("  ✓ [CPU] FP32 模型加载完成")
            else:
                dtype = torch.float16 if (self.device == "cuda" or is_mac) else torch.float32
                target_device = "mps" if (is_mac and self.device != "cuda") else self.device
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    device_map={"": target_device} if target_device != "cpu" else None,
                    trust_remote_code=True
                )
                print(f"  ✓ {'FP16' if dtype == torch.float16 else 'FP32'} 加载成功")
            
            return model
            
        except Exception as e:
            print(f"⚠️ 模型加载异常：{e}，回退到标准 FP32 加载")
            self.device = "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            print(f"  ⚠️ 设备已更新为: {self.device}")
            return model
    
    def _integrate_dual_weights(self):
        """将双权重层集成到 GLM 模型中"""
        from core.dual_weight_layers import DualWeightLinear
        
        print("\n[集成] 开始集成双权重层到 GLM...")
        
        replaced_count = 0
        
        # GLM 模型的目标层名称
        # GLM 使用 query_key_value 和 dense 作为注意力层
        # MLP 使用 dense_h_to_4h 和 dense_4h_to_h
        target_names = {
            'query_key_value', 'dense',  # 注意力层
            'dense_h_to_4h', 'dense_4h_to_h'  # MLP 层
        }
        
        for name, module in list(self.base_model.named_modules()):
            for attr_name in target_names:
                if hasattr(module, attr_name):
                    target_layer = getattr(module, attr_name)
                    
                    if isinstance(target_layer, nn.Linear) and not isinstance(target_layer, DualWeightLinear):
                        try:
                            static_weight = target_layer.weight.data.clone()
                            bias = target_layer.bias.data.clone() if target_layer.bias is not None else None
                            
                            dual_layer = DualWeightLinear(
                                in_features=target_layer.in_features,
                                out_features=target_layer.out_features,
                                bias=(bias is not None),
                                static_weight=static_weight
                            )
                            if bias is not None:
                                dual_layer.bias.data.copy_(bias)
                            
                            setattr(module, attr_name, dual_layer)
                            replaced_count += 1
                        except Exception as e:
                            print(f"  ⚠️ 替换层失败 {name}.{attr_name}: {e}")
        
        print(f"✓ 已替换 {replaced_count} 个底层投影为双权重版本")
    
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
        """前向传播"""
        out_attentions = output_attentions if output_attentions is not None else False
        ret_dict = return_dict if return_dict is not None else True
        use_cache = use_cache if use_cache is not None else False
        
        # 设置记忆锚点
        self._current_memory_anchor = memory_anchor
        
        exclude = {'return_dict', 'output_attentions', 'output_hidden_states', 'use_cache', 'past_key_values', 'memory_anchor'}
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in exclude}
        
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
        """生成文本"""
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
        """获取隐藏层状态"""
        outputs = self.base_model(
            input_ids=input_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        return outputs.hidden_states
    
    def get_all_dynamic_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """获取所有动态权重"""
        from core.dual_weight_layers import DualWeightLinear, DualWeightAttention, DualWeightFFN
        
        dynamic_weights = {}
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, DualWeightAttention):
                dynamic_weights[name] = module.get_all_dynamic_weights()
            elif isinstance(module, DualWeightFFN):
                dynamic_weights[name] = module.get_all_dynamic_weights()
            elif isinstance(module, DualWeightLinear):
                dynamic_weights[name] = {'dynamic_weight': module.dynamic_weight.clone()}
        
        return dynamic_weights
    
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


class GLMInterface:
    """
    GLM 模型统一接口
    
    提供与 QwenInterface 相同的接口，便于切换
    """
    
    def __init__(
        self,
        model_path: str,
        config,
        device: str = "cpu",
        quantization: str = "FP16"
    ):
        self.config = config
        self.device = device
        self.model = GLMModelWrapper(
            model_path=model_path,
            config=config,
            device=device,
            quantization=quantization
        )
        self.device = self.model.device
        self.model.eval()
        
        # CPU 线程控制
        if self.device == "cpu":
            import multiprocessing
            num_cores = multiprocessing.cpu_count()
            torch.set_num_threads(max(1, num_cores // 2))
            print(f"[GLMInterface] CPU 优化：设置线程数为 {torch.get_num_threads()}")
        
        # STDP 后台线程池
        import concurrent.futures
        self._stdp_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='stdp_worker'
        )
        self._step_counter = 0
        self._stdp_every_n = 50
        self._last_reward = 1.0
        
        # 统计信息
        self.total_generation_time = 0.0
        self.total_tokens_generated = 0
    
    @property
    def tokenizer(self):
        return self.model.tokenizer
    
    @property
    def embeddings(self):
        return self.model.base_model.get_input_embeddings()
    
    def apply_stdp_to_all(self, grad_dict: Dict[str, torch.Tensor], lr: float = 0.01):
        """统一 STDP 更新接口"""
        self.model.apply_stdp_to_all(grad_dict, lr)
    
    def set_reward(self, reward: float):
        """设置反馈奖励"""
        self._last_reward = max(0.1, min(2.0, reward))
        print(f"[GLMInterface] 已接收新奖励反馈: {self._last_reward:.2f}")
    
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
        """单步前向推理"""
        if self.device == "cpu":
            print(".", end="", flush=True)
        
        outputs = {}
        start_time = time.time()
        
        self._step_counter += 1
        need_features = (self._step_counter % self._stdp_every_n == 0)
        
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
                output_hidden_states=need_features,
                return_dict=True,
                **clean_kwargs
            )
            
            next_token_logits = output_tensors.logits[:, -1, :].clone()
            
            # 重复惩罚
            repetition_penalty = kwargs.get('repetition_penalty', 1.1)
            if repetition_penalty != 1.0 and input_ids is not None:
                unique_tokens = torch.unique(input_ids)
                seen_logits = next_token_logits[:, unique_tokens]
                penalized_logits = torch.where(
                    seen_logits > 0,
                    seen_logits / repetition_penalty,
                    seen_logits * repetition_penalty
                )
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
            
            # 采样
            if temp > 0:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            outputs['token_id'] = next_token.item()
            outputs['logits'] = next_token_logits
            
            if need_features and output_tensors.hidden_states:
                outputs['features'] = output_tensors.hidden_states[-1][:, -1, :].clone()
            else:
                outputs['features'] = None
            
            outputs['attention_output'] = torch.zeros(1)
            outputs['ffn_output'] = outputs['features'] if outputs['features'] is not None else torch.zeros(1)
            outputs['generation_path'] = str(next_token.item())
            outputs['evaluation_score'] = 35.0
        
        elapsed = time.time() - start_time
        outputs['cycle_time_ms'] = elapsed * 1000.0
        outputs['past_key_values'] = output_tensors.past_key_values if hasattr(output_tensors, 'past_key_values') else None
        self.total_tokens_generated += 1
        
        return outputs
    
    async def generate_stream(
        self,
        input_text: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ):
        """流式生成"""
        inputs = self.model.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        prompt_ids = inputs.input_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]
        
        max_total = prompt_len + max_tokens
        input_ids_buf = torch.full(
            (1, max_total), self.model.tokenizer.pad_token_id or 0,
            dtype=torch.long, device=self.device
        )
        input_ids_buf[:, :prompt_len] = prompt_ids
        cur_len = prompt_len
        
        attention_mask = inputs.attention_mask.to(self.device)
        past_key_values = None
        
        eos_token_id = self.model.tokenizer.eos_token_id
        stop_token_ids = {eos_token_id}
        
        for step in range(max_tokens):
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
            
            if cur_len < max_total:
                input_ids_buf[:, cur_len] = next_token_id
                cur_len += 1
            
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
        """生成文本"""
        start_time = time.time()
        
        inputs = self.model.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        eos_token_id = self.model.tokenizer.eos_token_id
        stop_token_ids = {eos_token_id}
        
        generated_tokens = []
        past_key_values = None
        
        for step in range(max_tokens):
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
            
            next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=-1)
        
        output_text = self.model.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        elapsed = time.time() - start_time
        self.total_generation_time += elapsed
        self.total_tokens_generated += len(generated_tokens)
        
        from core.interfaces_working import BrainAIOutput
        
        return BrainAIOutput(
            text=output_text,
            tokens=generated_tokens,
            confidence=min(0.95, 0.7 + len(output_text) / 200.0),
            memory_anchors=[],
            stdp_stats={},
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
        """对话接口"""
        if history:
            context = "\n".join([
                f"{h['role']}: {h['content']}" 
                for h in history[-5:]
            ])
            full_input = f"{context}\nUser: {message}\nAssistant:"
        else:
            full_input = f"User: {message}\nAssistant:"
        
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
        """对指定层应用 STDP 更新"""
        self.model.apply_stdp_to_layer(layer_name, grad_dict, lr)
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'dynamic_weights': self.model.get_all_dynamic_weights(),
            'config': self.config,
            'stats': self.get_stats()
        }
        torch.save(checkpoint, path)
        print(f"[GLMInterface] 检查点已保存：{path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        dynamic_weights = checkpoint.get('dynamic_weights', {})
        restored_count = 0
        
        for layer_name, weights in dynamic_weights.items():
            try:
                for name, module in self.model.base_model.named_modules():
                    if name == layer_name and hasattr(module, 'apply_stdp_to_all'):
                        module.apply_stdp_to_all(weights, lr=1.0)
                        restored_count += 1
                        break
            except Exception as e:
                print(f"[GLMInterface] 恢复权重失败 {layer_name}: {e}")
        
        if 'stats' in checkpoint:
            stats = checkpoint['stats']
            if 'system' in stats:
                self.total_generation_time = stats['system'].get('total_time', 0)
                self.total_tokens_generated = stats['system'].get('total_tokens', 0)
        
        print(f"[GLMInterface] 检查点已加载：{path} (恢复 {restored_count} 个权重层)")


def create_glm_ai(
    model_path: str,
    device: str = "cpu",
    quantization: str = "FP16"
) -> GLMInterface:
    """
    快捷创建 GLM AI 实例
    
    Args:
        model_path: 模型路径
        device: 设备
        quantization: 量化类型
    
    Returns:
        ai: GLMInterface 实例
    """
    from configs.arch_config import default_config
    
    return GLMInterface(
        model_path=model_path,
        config=default_config,
        device=device,
        quantization=quantization
    )
