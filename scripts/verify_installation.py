#!/usr/bin/env python3
"""
项目安装验证脚本

检查所有模块是否正确安装和配置
"""

import sys
import importlib
from pathlib import Path


def check_python_version():
    """检查 Python 版本"""
    print("[1/8] 检查 Python 版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python 版本过低，需要 3.8+, 当前 {version.major}.{version.minor}")
        return False


def check_dependencies():
    """检查依赖包"""
    print("\n[2/8] 检查依赖包...")
    
    required_packages = [
        'torch',
        'numpy',
        'transformers',
        'yaml',
        'tqdm'
    ]
    
    all_ok = True
    for package in required_packages:
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {package} ({version})")
        except ImportError:
            print(f"  ✗ {package} (未安装)")
            all_ok = False
    
    return all_ok


def check_project_structure():
    """检查项目结构"""
    print("\n[3/8] 检查项目结构...")
    
    root_dir = Path(__file__).parent.parent
    required_dirs = [
        'configs',
        'core',
        'hippocampus',
        'self_loop',
        'training',
        'evaluation',
        'tests'
    ]
    
    all_ok = True
    for dir_name in required_dirs:
        dir_path = root_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (缺失)")
            all_ok = False
    
    return all_ok


def check_core_modules():
    """检查核心模块"""
    print("\n[4/8] 检查核心模块...")
    
    modules = [
        ('configs.arch_config', 'BrainAIConfig'),
        ('core.dual_weight_layers', 'DualWeightLinear'),
        ('core.stdp_engine', 'STDPEngine'),
        ('core.refresh_engine', 'RefreshCycleEngine'),
        ('core.interfaces', 'BrainAIInterface'),
        ('hippocampus.hippocampus_system', 'HippocampusSystem'),
        ('self_loop.self_loop_optimizer', 'SelfLoopOptimizer'),
        ('evaluation.evaluator', 'BrainAIEvaluator'),
        ('training.trainer', 'BrainAITrainer'),
    ]
    
    all_ok = True
    for module_name, class_name in modules:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            print(f"  ✓ {module_name}.{class_name}")
        except (ImportError, AttributeError) as e:
            print(f"  ✗ {module_name}.{class_name} ({e})")
            all_ok = False
    
    return all_ok


def check_hippocampus_modules():
    """检查海马体子模块"""
    print("\n[5/8] 检查海马体子模块...")
    
    modules = [
        ('hippocampus.ec_encoder', 'EntorhinalEncoder'),
        ('hippocampus.dg_separator', 'DentateGyrusSeparator'),
        ('hippocampus.ca3_memory', 'CA3EpisodicMemory'),
        ('hippocampus.ca1_gate', 'CA1AttentionGate'),
        ('hippocampus.swr_consolidation', 'SWRConsolidation'),
    ]
    
    all_ok = True
    for module_name, class_name in modules:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            print(f"  ✓ {module_name}.{class_name}")
        except (ImportError, AttributeError) as e:
            print(f"  ✗ {module_name}.{class_name} ({e})")
            all_ok = False
    
    return all_ok


def check_model_files():
    """检查模型文件"""
    print("\n[6/8] 检查模型文件...")
    
    root_dir = Path(__file__).parent.parent
    model_path = root_dir / 'models' / 'Qwen3.5-0.8B-Base'
    
    if model_path.exists():
        print(f"  ✓ 模型目录存在：{model_path}")
        
        # 检查关键文件
        key_files = [
            'config.json',
            'pytorch_model.bin',
            'tokenizer.json'
        ]
        
        all_ok = True
        for file_name in key_files:
            file_path = model_path / file_name
            if file_path.exists():
                print(f"    ✓ {file_name}")
            else:
                print(f"    ✗ {file_name} (缺失)")
                all_ok = False
        
        return all_ok
    else:
        print(f"  ⚠ 模型目录不存在：{model_path}")
        print(f"  提示：运行以下命令下载模型:")
        print(f"    huggingface-cli download Qwen/Qwen3.5-0.8B-Base --local-dir ./models/Qwen3.5-0.8B-Base")
        return True  # 不阻断验证


def run_basic_tests():
    """运行基础测试"""
    print("\n[7/8] 运行基础测试...")
    
    try:
        # 测试配置加载
        from configs.arch_config import default_config
        print(f"  ✓ 配置加载成功")
        
        # 测试 STDP 规则
        from core.stdp_engine import STDPRule
        rule = STDPRule()
        delta_w = rule.compute_update(0, 10, 0.8)
        assert delta_w > 0, "STDP LTP 规则失败"
        print(f"  ✓ STDP 规则测试通过")
        
        # 测试双权重层
        import torch
        from core.dual_weight_layers import DualWeightLinear
        layer = DualWeightLinear(100, 100)
        x = torch.randn(2, 100)
        y = layer(x)
        assert y.shape == (2, 100), "双权重层前向传播失败"
        print(f"  ✓ 双权重层测试通过")
        
        return True
    except Exception as e:
        print(f"  ✗ 测试失败：{e}")
        return False


def check_documentation():
    """检查文档"""
    print("\n[8/8] 检查文档...")
    
    root_dir = Path(__file__).parent.parent
    docs = [
        'README.md',
        'ARCHITECTURE.md',
        'PROJECT_SUMMARY.md',
        'QUICKSTART.md',
        'requirements.txt'
    ]
    
    all_ok = True
    for doc in docs:
        doc_path = root_dir / doc
        if doc_path.exists():
            print(f"  ✓ {doc}")
        else:
            print(f"  ✗ {doc} (缺失)")
            all_ok = False
    
    return all_ok


def main():
    """主函数"""
    print("=" * 60)
    print("类人脑双系统全闭环 AI架构 - 安装验证")
    print("=" * 60)
    
    checks = [
        ("Python 版本", check_python_version()),
        ("依赖包", check_dependencies()),
        ("项目结构", check_project_structure()),
        ("核心模块", check_core_modules()),
        ("海马体模块", check_hippocampus_modules()),
        ("模型文件", check_model_files()),
        ("基础测试", run_basic_tests()),
        ("文档", check_documentation())
    ]
    
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    
    for name, ok in checks:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
    
    print(f"\n总计：{passed}/{total} 项检查通过")
    
    if passed == total:
        print("\n🎉 所有检查通过！系统已就绪。")
        print("\n下一步:")
        print("  1. 如果还未下载模型，请运行:")
        print("     huggingface-cli download Qwen/Qwen3.5-0.8B-Base --local-dir ./models/Qwen3.5-0.8B-Base")
        print("  2. 运行对话示例：python main.py --mode chat")
        return 0
    else:
        print(f"\n⚠ 有 {total - passed} 项检查未通过，请检查上述错误。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
