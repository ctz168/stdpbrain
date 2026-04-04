#!/usr/bin/env python3
"""完整记忆测试 - 后台运行版"""
import os,sys,time,json
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')

LOG='/tmp/mem_bg.log'
def log(s):
    with open(LOG,'a') as f: f.write(f'{time.strftime("%H:%M:%S")} {s}\n')
    print(s,flush=True)

log('=== START ===')
from configs.arch_config import BrainAIConfig
from core.interfaces import BrainAIInterface
c=BrainAIConfig(); c.model_path=os.path.abspath('models/Qwen3.5-0.8B')
t0=time.time(); b=BrainAIInterface(c,device='cpu'); log(f'init {time.time()-t0:.0f}s')

# 存入（一次存4条）
store_q='我叫小明，来自深圳南山区，手机号13812345678，在腾讯做程序员，请记住这些信息'
t1=time.time(); r1=b.chat(store_q); log(f'store {time.time()-t1:.0f}s: {r1[:80]}')

# 干扰
t2=time.time(); r2=b.chat('今天天气怎么样'); log(f'dist1 {time.time()-t2:.0f}s: {r2[:40]}')
t3=time.time(); r3=b.chat('给我讲个笑话'); log(f'dist2 {time.time()-t3:.0f}s: {r3[:40]}')

# 回忆
tests=[
    ('你记得我叫什么名字吗','小明','名字'),
    ('我来自哪个城市','深圳','城市'),
    ('我的手机号是多少','13812345678','手机号'),
    ('我在哪里工作','腾讯','公司'),
    ('我的职业是什么','程序员','职业'),
]
results=[]; hits=0
for q,exp,tag in tests:
    t=time.time(); r=b.chat(q); hit=exp in r
    if hit: hits+=1
    m='✅' if hit else '❌'
    results.append({'tag':tag,'q':q,'exp':exp,'r':r[:200],'hit':hit,'t':round(time.time()-t,1)})
    log(f'{m} [{tag}] {time.time()-t:.0f}s exp={exp} r={r[:80]}')

# 状态
try:
    stats=b.get_stats(); hc=stats.get('hippocampus',{})
    log(f'memories={hc.get("num_memories")} stdp={stats.get("total_stdp_updates")}')
    rec=b._last_recalled_memories
    log(f'recalled={len(rec) if rec else 0}')
    if rec:
        for i,m in enumerate(rec[:5]):
            if isinstance(m,dict):
                log(f'  [{i+1}] act={m.get("activation_strength",0):.3f} content={m.get("content","")[:60]}')
except Exception as e:
    log(f'stats_err: {e}')

score=f'{hits}/{len(tests)}={hits/len(tests)*100:.0f}%'
log(f'\n=== FINAL: {score} ===')
with open('/tmp/mem_bg_result.json','w') as f:
    json.dump({'score':score,'hits':hits,'total':len(tests),'results':results},f,ensure_ascii=False,indent=2)
log('DONE')
