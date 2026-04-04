#!/usr/bin/env python3
"""分步记忆测试 - 每步保存结果到文件，不怕超时"""
import os, sys, time, json
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')

OUT = '/tmp/mem_result.json'

def save(data):
    with open(OUT, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def step(name):
    print(f'\n{"="*50}\n  {name}\n{"="*50}', flush=True)

result = {'steps': [], 'final_score': None}

try:
    step('1. 初始化')
    from configs.arch_config import BrainAIConfig
    from core.interfaces import BrainAIInterface
    c = BrainAIConfig()
    c.model_path = os.path.abspath('models/Qwen3.5-0.8B')
    t0 = time.time()
    b = BrainAIInterface(c, device='cpu')
    result['steps'].append({'step': 'init', 'time': round(time.time()-t0, 1), 'status': 'ok'})
    print(f'  初始化完成: {time.time()-t0:.1f}s', flush=True)
    save(result)

    # ===== 存入信息 =====
    step('2. 存入个人信息 (4轮)')
    info = [
        ('我叫小明，请记住我的名字', '名字'),
        ('我来自深圳南山区', '城市'),
        ('我的手机号是13812345678', '手机号'),
        ('我在腾讯公司做程序员', '工作'),
    ]
    for q, tag in info:
        t0 = time.time()
        r = b.chat(q)
        elapsed = round(time.time()-t0, 1)
        result['steps'].append({'step': f'store_{tag}', 'question': q, 'answer': r[:100], 'time': elapsed, 'status': 'ok'})
        print(f'  [{tag}] {elapsed}s | Q: {q}', flush=True)
        print(f'         A: {r[:80]}', flush=True)
        save(result)

    # ===== 干扰 =====
    step('3. 干扰对话 (2轮)')
    distracts = ['今天天气怎么样', '给我讲个笑话']
    for d in distracts:
        t0 = time.time()
        r = b.chat(d)
        elapsed = round(time.time()-t0, 1)
        result['steps'].append({'step': f'distract', 'question': d, 'answer': r[:80], 'time': elapsed})
        print(f'  [干扰] {elapsed}s | Q: {d}', flush=True)
        print(f'          A: {r[:60]}', flush=True)
        save(result)

    # ===== 回忆 =====
    step('4. 记忆召回测试 (6题)')
    recall = [
        ('你记得我叫什么名字吗？', '小明', '名字'),
        ('我来自哪个城市？', '深圳', '城市'),
        ('我住在哪里？', '南山', '区域'),
        ('我的手机号是多少？', '13812345678', '手机号'),
        ('我在哪家公司工作？', '腾讯', '公司'),
        ('我的职业是什么？', '程序员', '职业'),
    ]
    hits = 0
    for q, expected, tag in recall:
        t0 = time.time()
        r = b.chat(q)
        elapsed = round(time.time()-t0, 1)
        hit = expected in r
        if hit: hits += 1
        result['steps'].append({
            'step': f'recall_{tag}', 'question': q, 'expected': expected,
            'answer': r[:150], 'time': elapsed, 'hit': hit
        })
        mark = '✅' if hit else '❌'
        print(f'  {mark} [{tag}] {elapsed}s | 期望: {expected}', flush=True)
        print(f'         A: {r[:100]}', flush=True)
        save(result)

    # ===== 系统状态 =====
    step('5. 记忆系统状态')
    try:
        stats = b.get_stats()
        hc = stats.get('hippocampus', {})
        mem_info = {
            'num_memories': hc.get('num_memories', 'N/A'),
            'memory_mb': round(hc.get('memory_usage_mb', 0), 3),
            'cycle_count': stats.get('cycle_count', 'N/A'),
            'stdp_updates': stats.get('total_stdp_updates', 'N/A'),
        }
        result['steps'].append({'step': 'stats', **mem_info})
        print(f'  记忆数量: {mem_info["num_memories"]}', flush=True)
        print(f'  内存占用: {mem_info["memory_mb"]} MB', flush=True)
        print(f'  STDP更新: {mem_info["stdp_updates"]}', flush=True)
        
        # 召回记忆详情
        recalled = b._last_recalled_memories
        if recalled:
            print(f'  最后召回数量: {len(recalled)}', flush=True)
            for i, m in enumerate(recalled[:3]):
                if isinstance(m, dict):
                    ct = m.get('content', '')[:60]
                    sp = m.get('semantic_pointer', '')[:60]
                    act = m.get('activation_strength', 0)
                    print(f'    [{i+1}] act={act:.2f} content={ct}', flush=True)
                    if sp: print(f'        pointer={sp}', flush=True)
        else:
            print('  最后召回: (空)', flush=True)
    except Exception as e:
        result['steps'].append({'step': 'stats', 'error': str(e)})
        print(f'  获取状态失败: {e}', flush=True)

    save(result)

    # ===== 最终结果 =====
    step('6. 最终结果')
    total = len(recall)
    result['final_score'] = {'hits': hits, 'total': total, 'rate': f'{hits}/{total} = {hits/total*100:.1f}%'}
    print(f'\n  📊 记忆召回准确率: {hits}/{total} = {hits/total*100:.1f}%', flush=True)
    print(f'\n{"="*50}', flush=True)
    save(result)
    print('结果已保存到 /tmp/mem_result.json', flush=True)

except Exception as e:
    result['error'] = str(e)
    import traceback; result['traceback'] = traceback.format_exc()
    save(result)
    print(f'\n❌ 错误: {e}', flush=True)
    print('部分结果已保存到 /tmp/mem_result.json', flush=True)
