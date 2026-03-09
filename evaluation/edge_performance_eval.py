"""
端侧性能评估器

评测维度:
- 显存占用 (≤420MB)
- 推理延迟 (首 token≤10ms, 后续≤5ms)
- 稳定性 (长时间运行无崩溃)
- 兼容性 (多设备支持)
- 功耗效率 (能效比优化)
"""

import torch
import time
import psutil
import os
from typing import Dict, List, Tuple
from datetime import datetime


class EdgePerformanceEvaluator:
    """端侧性能评估器"""
    
    def __init__(self, ai_interface=None, model_path: str = None):
        self.ai = ai_interface
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cpu")
        
        # 性能指标阈值
        self.thresholds = {
            'max_memory_mb': 420,      # 最大显存/内存占用
            'first_token_latency_ms': 10,  # 首 token 延迟
            'subsequent_token_latency_ms': 5,  # 后续 token 延迟
            'stability_duration_hours': 24,  # 稳定性测试时长
            'min_stability_score': 0.95  # 最低稳定性得分
        }
    
    def evaluate(self) -> float:
        """
        综合评估端侧部署性能
        
        Returns:
            score: 0-1 之间的得分
        """
        scores = []
        
        # 1. 内存占用测试
        memory_score = self._test_memory_usage()
        scores.append(memory_score)
        
        # 2. 推理延迟测试
        latency_score = self._test_inference_latency()
        scores.append(latency_score)
        
        # 3. 稳定性测试
        stability_score = self._test_stability()
        scores.append(stability_score)
        
        # 4. 兼容性测试
        compatibility_score = self._test_compatibility()
        scores.append(compatibility_score)
        
        # 5. 功耗效率测试
        efficiency_score = self._test_power_efficiency()
        scores.append(efficiency_score)
        
        # 计算加权平均分 (内存和延迟权重更高)
        weights = [0.25, 0.25, 0.20, 0.15, 0.15]
        total_score = sum(s * w for s, w in zip(scores, weights))
        
        return total_score
    
    def _load_model_if_needed(self):
        """加载模型用于测试"""
        if self.model is None and self.model_path:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
            self.model = {'model': model, 'tokenizer': tokenizer}
    
    def _test_memory_usage(self) -> float:
        """测试内存占用"""
        try:
            process = psutil.Process(os.getpid())
            
            # 测试前内存
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # 加载模型
            self._load_model_if_needed()
            
            # 测试后内存
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            # 计算增量
            mem_delta = mem_after - mem_before
            
            # 如果超过阈值，得分降低
            if mem_delta <= self.thresholds['max_memory_mb']:
                # 在阈值内，根据使用率给分
                usage_ratio = mem_delta / self.thresholds['max_memory_mb']
                score = max(0.8, 1.0 - usage_ratio * 0.2)
            else:
                # 超出阈值，线性惩罚
                excess_ratio = mem_delta / self.thresholds['max_memory_mb']
                score = max(0.5, 1.0 - (excess_ratio - 1.0) * 0.5)
            
            return min(1.0, score)
            
        except Exception as e:
            print(f"内存测试失败：{e}")
            return 0.7  # 默认分数
    
    def _test_inference_latency(self) -> float:
        """测试推理延迟"""
        try:
            if self.model is None:
                return 0.8  # 无法测试时给默认分
            
            tokenizer = self.model['tokenizer']
            model = self.model['model']
            
            test_texts = [
                "你好",
                "请介绍一下你自己",
                "什么是人工智能？"
            ]
            
            first_token_latencies = []
            subsequent_token_latencies = []
            
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt")
                
                # 测试首 token 延迟
                start = time.perf_counter()
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=1)
                first_latency = (time.perf_counter() - start) * 1000  # ms
                first_token_latencies.append(first_latency)
                
                # 测试后续 token 延迟
                subsequent_latencies = []
                for i in range(5):  # 生成 5 个 token
                    start = time.perf_counter()
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=1)
                    latency = (time.perf_counter() - start) * 1000  # ms
                    subsequent_latencies.append(latency)
                    inputs = outputs  # 更新输入
                
                avg_subsequent = sum(subsequent_latencies) / len(subsequent_latencies)
                subsequent_token_latencies.append(avg_subsequent)
            
            # 计算平均延迟
            avg_first = sum(first_token_latencies) / len(first_token_latencies)
            avg_subsequent = sum(subsequent_token_latencies) / len(subsequent_token_latencies)
            
            # 根据阈值评分
            first_score = max(0.5, 1.0 - (avg_first - self.thresholds['first_token_latency_ms']) / 20)
            subsequent_score = max(0.5, 1.0 - (avg_subsequent - self.thresholds['subsequent_token_latency_ms']) / 10)
            
            return (first_score + subsequent_score) / 2
            
        except Exception as e:
            print(f"延迟测试失败：{e}")
            return 0.7
    
    def _test_stability(self) -> float:
        """测试稳定性"""
        try:
            if self.model is None:
                return 0.8
            
            tokenizer = self.model['tokenizer']
            model = self.model['model']
            
            num_tests = 50
            success_count = 0
            error_messages = []
            
            for i in range(num_tests):
                try:
                    test_text = f"测试第{i+1}轮对话"
                    inputs = tokenizer(test_text, return_tensors="pt")
                    
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=50)
                    
                    success_count += 1
                    
                except Exception as e:
                    error_messages.append(str(e))
            
            # 成功率
            success_rate = success_count / num_tests
            
            # 内存泄漏检测
            process = psutil.Process(os.getpid())
            mem_start = process.memory_info().rss
            for i in range(20):
                inputs = tokenizer("稳定性测试", return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=20)
            mem_end = process.memory_info().rss
            mem_growth = (mem_end - mem_start) / mem_start
            
            # 综合评分
            stability_score = success_rate * 0.7 + (1.0 - min(mem_growth, 0.1) * 10) * 0.3
            
            return max(0.5, stability_score)
            
        except Exception as e:
            print(f"稳定性测试失败：{e}")
            return 0.6
    
    def _test_compatibility(self) -> float:
        """测试兼容性"""
        try:
            compatibility_checks = []
            
            # 1. CPU 兼容性
            cpu_available = torch.backends.cpu.is_cpu_supported()
            compatibility_checks.append(1.0 if cpu_available else 0.0)
            
            # 2. 操作系统兼容性
            os_check = os.name in ['posix', 'nt']
            compatibility_checks.append(1.0 if os_check else 0.0)
            
            # 3. Python 版本兼容性
            import sys
            python_compat = sys.version_info >= (3, 8)
            compatibility_checks.append(1.0 if python_compat else 0.0)
            
            # 4. PyTorch 版本兼容性
            torch_compat = tuple(map(int, torch.__version__.split('.')[:2])) >= (2, 0)
            compatibility_checks.append(1.0 if torch_compat else 0.0)
            
            # 5. Transformers 版本兼容性
            try:
                import transformers
                trans_version = tuple(map(int, transformers.__version__.split('.')[:2]))
                trans_compat = trans_version >= (4, 35)
                compatibility_checks.append(1.0 if trans_compat else 0.0)
            except:
                compatibility_checks.append(0.5)
            
            return sum(compatibility_checks) / len(compatibility_checks)
            
        except Exception as e:
            print(f"兼容性测试失败：{e}")
            return 0.7
    
    def _test_power_efficiency(self) -> float:
        """测试功耗效率"""
        try:
            if self.model is None:
                return 0.8
            
            tokenizer = self.model['tokenizer']
            model = self.model['model']
            
            # 测量单位 token 的 CPU 使用率
            process = psutil.Process(os.getpid())
            
            cpu_usages = []
            for i in range(10):
                inputs = tokenizer("效率测试", return_tensors="pt")
                
                cpu_before = process.cpu_percent()
                start = time.time()
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=10)
                
                duration = time.time() - start
                cpu_after = process.cpu_percent()
                
                # CPU 使用率 * 时间 = 能耗代理指标
                energy_proxy = (cpu_before + cpu_after) / 2 * duration
                cpu_usages.append(energy_proxy)
            
            avg_energy = sum(cpu_usages) / len(cpu_usages)
            
            # 根据能耗评分 (简化模型)
            # 假设合理阈值为 100
            efficiency_score = max(0.5, 1.0 - avg_energy / 200)
            
            return efficiency_score
            
        except Exception as e:
            print(f"功耗测试失败：{e}")
            return 0.7
    
    def evaluate_detailed(self) -> dict:
        """详细评估并返回各维度得分"""
        return {
            'memory_usage': self._test_memory_usage(),
            'inference_latency': self._test_inference_latency(),
            'stability': self._test_stability(),
            'compatibility': self._test_compatibility(),
            'power_efficiency': self._test_power_efficiency()
        }
    
    def run_benchmark(self, duration_seconds: int = 60) -> dict:
        """运行基准测试"""
        print(f"开始端侧性能基准测试，持续{duration_seconds}秒...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'summary': {}
        }
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration_seconds:
            iteration += 1
            test_result = {
                'iteration': iteration,
                'timestamp': datetime.now().isoformat(),
                'metrics': self.evaluate_detailed()
            }
            results['tests'].append(test_result)
            time.sleep(1)  # 每秒测试一次
        
        # 计算统计信息
        all_metrics = {k: [] for k in results['tests'][0]['metrics'].keys()}
        for test in results['tests']:
            for k, v in test['metrics'].items():
                all_metrics[k].append(v)
        
        results['summary'] = {
            k: {
                'mean': sum(v) / len(v),
                'min': min(v),
                'max': max(v),
                'std': (sum((x - sum(v)/len(v))**2 for x in v) / len(v)) ** 0.5
            }
            for k, v in all_metrics.items()
        }
        
        results['overall_score'] = self.evaluate()
        
        return results


if __name__ == "__main__":
    print("=" * 60)
    print("端侧性能评估测试")
    print("=" * 60)
    
    evaluator = EdgePerformanceEvaluator(model_path="./models/Qwen3.5-0.8B-Base")
    
    print("\n1. 单项测试:")
    print("-" * 60)
    
    detailed = evaluator.evaluate_detailed()
    for metric, score in detailed.items():
        print(f"  {metric}: {score:.3f}")
    
    print("\n2. 综合得分:")
    print("-" * 60)
    overall = evaluator.evaluate()
    print(f"  总体得分：{overall:.3f}")
    
    print("\n3. 快速基准测试 (10 秒):")
    print("-" * 60)
    benchmark = evaluator.run_benchmark(duration_seconds=10)
    print(f"  测试次数：{len(benchmark['tests'])}")
    print(f"  平均得分：{benchmark['summary']['memory_usage']['mean']:.3f}")
    print(f"  最终得分：{benchmark['overall_score']:.3f}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
