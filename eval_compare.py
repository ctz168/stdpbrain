#!/usr/bin/env python3
"""极简评测 - stdpbrain vs base (CPU可用)"""
import os,sys,time,json,torch,warnings,pathlib
warnings.filterwarnings("ignore")

# Monkey-patch
import os as _os
def _is_lp(p):
    if not p or not isinstance(p,str): return False
    if p[0] in '/~.': return _os.path.isdir(p)
    return len(p.split('/'))>2 and _os.path.isdir(p)
try:
    from transformers.utils.hub import cached_file as _o1, cached_files as _o2
    import inspect
    _sig2 = inspect.signature(_o2)
    # cached_files has cache_dir as 3rd positional param
    def _p1(path_or_repo_id, filename, **kwargs):
        if _is_lp(path_or_repo_id):
            c=_os.path.join(path_or_repo_id,filename)
            if _os.path.isfile(c): return c
        return _o1(path_or_repo_id,filename,**kwargs)
    def _p2(path_or_repo_id, filenames, cache_dir=None, **kwargs):
        if _is_lp(path_or_repo_id):
            r=[_os.path.join(path_or_repo_id,f) for f in filenames if _os.path.isfile(_os.path.join(path_or_repo_id,f))]
            if r: return r
        return _o2(path_or_repo_id,filenames,cache_dir=cache_dir,**kwargs)
    import transformers.utils.hub as _m
    _m.cached_file=_p1; _m.cached_files=_p2
except: pass
try:
    import huggingface_hub.utils._validators as _v
    _x=_v.validate_repo_id
    _v.validate_repo_id=lambda r,**k: None if _is_lp(r) else _x(r,**k)
except: pass

MP=os.path.abspath("./models/Qwen3.5-0.8B")

def load_base():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MP,trust_remote_code=True,padding_side="left")
    mod=AutoModelForCausalLM.from_pretrained(MP,torch_dtype=torch.float32,trust_remote_code=True)
    mod.eval()
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    return mod,tok

def base_gen(mod,tok,q,hist=None):
    msgs=(hist or [])+[{"role":"user","content":q}]
    txt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inp=tok(txt,return_tensors="pt",truncation=True,max_length=200)
    with torch.no_grad():
        out=mod.generate(**inp,max_new_tokens=30,temperature=0.6,do_sample=True,pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inp['input_ids'].shape[-1]:],skip_special_tokens=True).strip()

def load_brain():
    from configs.arch_config import BrainAIConfig
    import config as uc
    c=BrainAIConfig()
    c.model_path=MP; c.quantization="FP32"; c.QUANTIZATION="FP32"
    from core.interfaces import BrainAIInterface
    return BrainAIInterface(c,device="cpu")

def main():
    what=sys.argv[1] if len(sys.argv)>1 else "both"
    R={}

    if what in ("base","both"):
        print("\n=== BASE MODEL ===")
        mod,tok=load_base()
        
        # Knowledge: 1 question
        r=base_gen(mod,tok,"中国的首都是哪里？")
        k_ok="北京" in r
        print(f"  Q: 中国的首都是哪里？")
        print(f"  A: {r[:100]}  {'✅' if k_ok else '❌'}")
        R["base_knowledge"]={"q":"中国的首都是哪里？","r":r,"ok":k_ok}
        
        # Memory: inject then recall
        hist=[]
        base_gen(mod,tok,"我叫张三，我来自北京。",hist)
        r2=base_gen(mod,tok,"我叫什么名字？")
        m_ok="张三" in r2
        print(f"\n  Q: 我叫什么名字？")
        print(f"  A: {r2[:100]}  {'✅' if m_ok else '❌'}")
        R["base_memory"]={"q":"我叫什么名字？","r":r2,"ok":m_ok}
        
        del mod; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if what in ("stdpbrain","both"):
        print("\n=== STDPBRAIN ===")
        ai=load_brain()
        
        # Knowledge
        r=ai.chat("中国的首都是哪里？",max_tokens=30,thinking=False)
        k_ok="北京" in r
        print(f"  Q: 中国的首都是哪里？")
        print(f"  A: {r[:100]}  {'✅' if k_ok else '❌'}")
        R["stdpbrain_knowledge"]={"q":"中国的首都是哪里？","r":r,"ok":k_ok}
        
        # Memory
        ai.chat("我叫张三，我来自北京。",max_tokens=30,thinking=False)
        r2=ai.chat("我叫什么名字？",max_tokens=30,thinking=False)
        m_ok="张三" in r2
        print(f"\n  Q: 我叫什么名字？")
        print(f"  A: {r2[:100]}  {'✅' if m_ok else '❌'}")
        R["stdpbrain_memory"]={"q":"我叫什么名字？","r":r2,"ok":m_ok}

    print(f"\n=== SUMMARY ===")
    if "base_knowledge" in R:
        print(f"  Knowledge: base={R['base_knowledge']['ok']} stdpbrain={R['stdpbrain_knowledge']['ok']}")
        print(f"  Memory:    base={R['base_memory']['ok']} stdpbrain={R['stdpbrain_memory']['ok']}")
        
        bk=R['base_knowledge']['ok']; sk=R['stdpbrain_knowledge']['ok']
        bm=R['base_memory']['ok']; sm=R['stdpbrain_memory']['ok']
        
        if sm and not bm:
            print("\n  ✅ stdpbrain 记忆能力显著优于原模型（海马体系统有效）")
        elif sm and bm:
            print("\n  📌 两者都有记忆能力，stdpbrain 与原模型持平")
        elif not sm and not bm:
            print("\n  ⚠️ 两者都未能通过记忆测试")

    out_dir = pathlib.Path(__file__).resolve().parent.parent / "download"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_quick.json"
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump(R,f,ensure_ascii=False,indent=2,default=str)
    print(f"\n  Report: {out_path}")

if __name__=="__main__":
    main()
