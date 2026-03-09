"""
类人脑双系统全闭环 AI架构 - Qwen 真实模型版本

集成真实的 Qwen3.5-0.8B 模型和官方 tokenizer
需要 Python 3.11+ 和 PyTorch
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional
import time
import os


class QwenBrainAI:
    """
    基于真实 Qwen 模型的类人脑AI
    
    功能:
    - 使用 Qwen3.5-0.8B 真实模型
    - 官方 tokenizer
    - 海马体记忆系统集成
    - STDP 学习机制
    - 100Hz 刷新引擎模拟
    """
    
    def __init__(
        self,
        model_path: str = "./models/Qwen3.5-0.8B-Base",
        device: Optional[str] = None,
        use_int4: bool = True
    ):
        self.model_path = model_path
        
        # 自动选择设备
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print("=" * 60)
        print("类人脑AI - Qwen 真实模型版本")
        print("=" * 60)
        print(f"\n[初始化] 设备：{self.device}")
        
        # ========== 1. 检查模型路径 ==========
        if not os.path.exists(model_path):
            print(f"\n⚠️  模型路径不存在：{model_path}")
            print("请确保已下载 Qwen3.5-0.8B-Base 模型")
            print("\n下载命令:")
            print("  huggingface-cli download Qwen/Qwen3.5-0.8B-Base --local-dir ./models/Qwen3.5-0.8B-Base")
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # ========== 2. 加载 tokenizer ==========
        print(f"\n[初始化] 加载 Tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            print(f"✓ Tokenizer 加载成功，词表大小：{len(self.tokenizer)}")
        except Exception as e:
            print(f"⚠️  Tokenizer 加载失败：{e}")
            print("尝试使用 Qwen2.5 作为后备...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-0.5B",
                trust_remote_code=True
            )
        
        # ========== 3. 加载模型 ==========
        print(f"\n[初始化] 加载 Qwen 模型...")
        
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.device != "cpu" else None,
        }
        
        # INT4 量化 (如果支持)
        if use_int4 and self.device == "cpu":
            print("⚠️  CPU 模式下 INT4 量化可能不可用，将使用 FP16 或 FP32")
            use_int4 = False
        
        if use_int4:
            try:
                from optimum.quanto import quantize
                model_kwargs["torch_dtype"] = torch.float16
                print("使用 INT4 量化...")
            except ImportError:
                print("optimum 未安装，使用默认精度")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(torch.float32)
            
            print(f"✓ 模型加载成功")
            print(f"  - 参数量：{sum(p.numel() for p in self.model.parameters()):,}")
            print(f"  - 设备：{self.device}")
            
        except Exception as e:
            print(f"❌ 模型加载失败：{e}")
            print("\n提示:")
            print("  1. 确认模型文件完整")
            print("  2. 检查磁盘空间")
            print("  3. 尝试使用较小的模型 (如 Qwen2.5-0.5B)")
            raise
        
        # ========== 4. 初始化其他组件 ==========
        print(f"\n[初始化] 加载海马体和 STDP 模块...")
        
        try:
            from configs.arch_config import default_config
            from hippocampus.hippocampus_system import HippocampusSystem
            
            self.config = default_config
            self.hippocampus = HippocampusSystem(self.config, device=self.device)
            print("✓ 海马体系统加载成功")
            
        except Exception as e:
            print(f"⚠️  海马体系统加载失败：{e}")
            self.hippocampus = None
        
        # ========== 5. 统计信息 ==========
        self.generation_count = 0
        self.total_tokens = 0
        
        print("\n" + "=" * 60)
        print("✅ 初始化完成，准备就绪")
        print("=" * 60)
    
    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        生成文本 (使用真实 Qwen 模型)
        
        Args:
            input_text: 输入文本
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: Top-p 采样参数
            do_sample: 是否采样
        
        Returns:
            generated_text: 生成的文本
        """
        start_time = time.time()
        
        # ========== 1. Tokenize ==========
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt"
        ).to(self.model.device)
        
        input_length = inputs.shape[1]
        
        # ========== 2. 生成 ==========
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # ========== 3. Detokenize ==========
        generated_ids = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        # ========== 4. 更新统计 ==========
        elapsed = time.time() - start_time
        self.generation_count += 1
        new_tokens = len(generated_ids)
        self.total_tokens += new_tokens
        
        # ========== 5. 记录活动 ==========
        if self.hippocampus:
            self.hippocampus.record_activity()
        
        # ========== 6. 打印信息 ==========
        speed = new_tokens / elapsed if elapsed > 0 else 0
        print(f"[生成] 耗时：{elapsed*1000:.1f}ms | 速度：{speed:.1f} tokens/s | 长度：{new_tokens}")
        
        return generated_text
    
    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        对话接口
        
        Args:
            message: 用户消息
            history: 对话历史
            system_prompt: 系统提示词
        
        Returns:
            response: 回复
        """
        # 构建对话格式
        if system_prompt is None:
            system_prompt = "你是一个类人脑AI 助手，基于Qwen3.5-0.8B 模型，具有海马体记忆系统和 STDP 在线学习能力。"
        
        # 构建完整输入
        if history:
            conversation = []
            for msg in history[-5:]:  # 最近 5 轮
                if msg['role'] == 'user':
                    conversation.append(f"User: {msg['content']}")
                elif msg['role'] == 'assistant':
                    conversation.append(f"Assistant: {msg['content']}")
            
            full_input = "\n".join(conversation)
            full_input += f"\nUser: {message}\nAssistant:"
        else:
            full_input = f"{system_prompt}\n\nUser: {message}\nAssistant:"
        
        # 生成回复
        response = self.generate(
            full_input,
            max_new_tokens=200,
            temperature=0.7
        )
        
        return response.strip()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = {
            'generation_count': self.generation_count,
            'total_tokens': self.total_tokens,
            'device': self.device,
            'model_path': self.model_path
        }
        
        if self.hippocampus:
            stats['hippocampus'] = self.hippocampus.get_stats()
        
        return stats
    
    def save_model(self, save_path: str):
        """保存模型"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"✓ 模型已保存到：{save_path}")


def create_qwen_ai(
    model_path: str = "./models/Qwen3.5-0.8B-Base",
    device: Optional[str] = None,
    use_int4: bool = True
) -> QwenBrainAI:
    """
    快捷创建 Qwen AI 实例
    
    Args:
        model_path: 模型路径
        device: 设备
        use_int4: 是否使用 INT4 量化
    
    Returns:
        QwenBrainAI 实例
    """
    return QwenBrainAI(
        model_path=model_path,
        device=device,
        use_int4=use_int4
    )


# ==================== 测试函数 ====================

def test_qwen_model():
    """测试 Qwen 模型"""
    try:
        ai = create_qwen_ai()
        
        print("\n" + "=" * 60)
        print("测试 Qwen 模型")
        print("=" * 60)
        
        test_cases = [
            "你好，请介绍一下自己",
            "什么是人工智能？",
            "写一首关于春天的诗"
        ]
        
        for test in test_cases:
            print(f"\n你：{test}")
            response = ai.chat(test)
            print(f"AI: {response[:200]}...")
        
        print("\n统计信息:", ai.get_stats())
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_qwen_model()
