#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化效果验证 - 配置文件直接读取版本
"""

import re

def read_config_values():
    """直接从配置文件读取关键参数"""
    
    config_path = "configs/arch_config.py"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取关键参数
    params = {}
    
    # 权重比例
    static_match = re.search(r'STATIC_WEIGHT_RATIO:\s*float\s*=\s*([\d.]+)', content)
    dynamic_match = re.search(r'DYNAMIC_WEIGHT_RATIO:\s*float\s*=\s*([\d.]+)', content)
    
    if static_match and dynamic_match:
        params['static_ratio'] = float(static_match.group(1))
        params['dynamic_ratio'] = float(dynamic_match.group(1))
    
    # STDP参数
    ltp_match = re.search(r'alpha_LTP:\s*float\s*=\s*([\d.]+)', content)
    ltd_match = re.search(r'beta_LTD:\s*float\s*=\s*([\d.]+)', content)
    threshold_match = re.search(r'update_threshold:\s*float\s*=\s*([\d.]+)', content)
    decay_match = re.search(r'decay_rate:\s*float\s*=\s*([\d.]+)', content)
    
    if ltp_match:
        params['ltp'] = float(ltp_match.group(1))
    if ltd_match:
        params['ltd'] = float(ltd_match.group(1))
    if threshold_match:
        params['threshold'] = float(threshold_match.group(1))
    if decay_match:
        params['decay'] = float(decay_match.group(1))
    
    # 海马体参数
    ec_dim_match = re.search(r'EC_feature_dim:\s*int\s*=\s*(\d+)', content)
    dg_sparse_match = re.search(r'DG_sparsity:\s*float\s*=\s*([\d.]+)', content)
    recall_topk_match = re.search(r'recall_topk:\s*int\s*=\s*(\d+)', content)
    
    if ec_dim_match:
        params['ec_dim'] = int(ec_dim_match.group(1))
    if dg_sparse_match:
        params['dg_sparse'] = float(dg_sparse_match.group(1))
    if recall_topk_match:
        params['recall_topk'] = int(recall_topk_match.group(1))
    
    return params


def main():
    print("\n" + "=" * 60)
    print("  类人脑AI优化效果验证")
    print("=" * 60)
    
    try:
        params = read_config_values()
        
        print("\n[配置参数验证]")
        print("-" * 60)
        
        # 1. 权重比例
        if 'static_ratio' in params and 'dynamic_ratio' in params:
            static = params['static_ratio']
            dynamic = params['dynamic_ratio']
            print(f"权重比例:")
            print(f"  静态权重: {static:.2f} (目标: 0.85)")
            print(f"  动态权重: {dynamic:.2f} (目标: 0.15)")
            print(f"  权重总和: {static + dynamic:.2f} (应为1.0)")
            
            if static == 0.85 and dynamic == 0.15:
                print("  [OK] 权重比例优化成功")
            else:
                print("  [!] 权重比例与目标不符")
        
        # 2. STDP学习率
        if 'ltp' in params and 'ltd' in params:
            ltp = params['ltp']
            ltd = params['ltd']
            print(f"\nSTDP学习率:")
            print(f"  LTP学习率: {ltp:.4f} (目标: 0.025)")
            print(f"  LTD学习率: {ltd:.4f} (目标: 0.020)")
            
            if ltp >= 0.025:
                print("  [OK] LTP学习率已提升")
            else:
                print("  [!] LTP学习率未达标")
            
            if ltd >= 0.02:
                print("  [OK] LTD学习率已提升")
            else:
                print("  [!] LTD学习率未达标")
        
        # 3. 更新阈值
        if 'threshold' in params:
            threshold = params['threshold']
            print(f"\n更新阈值:")
            print(f"  更新阈值: {threshold:.4f} (目标: 0.0005)")
            
            if threshold <= 0.0005:
                print("  [OK] 更新阈值已降低，灵敏度提升")
            else:
                print("  [!] 更新阈值未达标")
        
        # 4. 衰减率
        if 'decay' in params:
            decay = params['decay']
            print(f"\n衰减率:")
            print(f"  衰减率: {decay:.2f} (目标: 0.95)")
            
            if decay <= 0.95:
                print("  [OK] 衰减率已优化，保留更多学习成果")
            else:
                print("  [!] 衰减率未达标")
        
        # 5. 海马体编码维度
        if 'ec_dim' in params:
            ec_dim = params['ec_dim']
            print(f"\n海马体编码维度:")
            print(f"  EC编码维度: {ec_dim} (目标: 256)")
            
            if ec_dim >= 256:
                print("  [OK] 编码维度已提升，特征表达能力增强")
            else:
                print("  [!] 编码维度未达标")
        
        # 6. DG稀疏度
        if 'dg_sparse' in params:
            dg_sparse = params['dg_sparse']
            print(f"\nDG稀疏度:")
            print(f"  DG稀疏度: {dg_sparse:.2f} (目标: 0.85)")
            
            if dg_sparse <= 0.85:
                print("  [OK] DG稀疏度已降低，记忆容量提升")
            else:
                print("  [!] DG稀疏度未达标")
        
        # 7. 召回topk
        if 'recall_topk' in params:
            recall_topk = params['recall_topk']
            print(f"\n召回topk:")
            print(f"  召回topk: {recall_topk} (目标: 3)")
            
            if recall_topk >= 3:
                print("  [OK] 召回topk已提升，召回质量提升")
            else:
                print("  [!] 召回topk未达标")
        
        # 总结
        print("\n" + "=" * 60)
        print("  优化总结")
        print("=" * 60)
        
        optimizations = []
        
        if params.get('static_ratio') == 0.85:
            optimizations.append("权重比例优化")
        
        if params.get('ltp', 0) >= 0.025:
            optimizations.append("LTP学习率提升")
        
        if params.get('ltd', 0) >= 0.02:
            optimizations.append("LTD学习率提升")
        
        if params.get('threshold', 1) <= 0.0005:
            optimizations.append("更新阈值降低")
        
        if params.get('decay', 1) <= 0.95:
            optimizations.append("衰减率优化")
        
        if params.get('ec_dim', 0) >= 256:
            optimizations.append("编码维度提升")
        
        if params.get('dg_sparse', 1) <= 0.85:
            optimizations.append("DG稀疏度优化")
        
        if params.get('recall_topk', 0) >= 3:
            optimizations.append("召回topk提升")
        
        print(f"\n已完成 {len(optimizations)} 项优化:")
        for i, opt in enumerate(optimizations, 1):
            print(f"  {i}. {opt}")
        
        print("\n" + "=" * 60)
        print("  下一步建议")
        print("=" * 60)
        print("\n1. 运行实际对话测试:")
        print("   python main.py --mode chat")
        print("\n2. 观察以下改进:")
        print("   - 独白质量和稳定性提升")
        print("   - 记忆召回更准确")
        print("   - 学习速度加快")
        print("\n3. 测试学习效果:")
        print("   - 进行多次对话")
        print("   - 测试AI是否能记住用户信息")
        print("   - 观察推理能力是否提升")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] 配置读取失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()