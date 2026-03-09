# 类人脑 AI 架构 - 快速参考卡

## 🚀 一键启动

```bash
# 1. 激活环境
conda activate stdpbrain

# 2. 验证环境
python check_env.py

# 3. 开始使用
python simple_test.py         # 简单测试
python main.py --mode chat    # 对话模式
```

## 📦 核心文件

| 文件 | 功能 | 行数 |
|------|------|------|
| `core/qwen_interface.py` | Qwen 真实模型接口 | ~300 |
| `core/dual_weight_layers.py` | 双权重层实现 | ~350 |
| `core/stdp_engine.py` | STDP 学习引擎 | ~400 |
| `hippocampus/hippocampus_system.py` | 海马体系统 | ~350 |
| `main.py` | 主入口 | ~200 |

**总计**: ~4,800 行代码

## 🧠 架构组件

```
Qwen3.5-0.8B 底座
├── 90% 静态权重 (冻结)
└── 10% 动态权重 (STDP 学习)

海马体系统
├── EC 编码器
├── DG 分离器
├── CA3 记忆库 (≤2MB)
├── CA1 门控
└── SWR 巩固

自闭环优化
├── 自生成组合
├── 自博弈竞争
└── 自评判选优
```

## ⚡ 性能指标

| 项目 | CPU 模式 | GPU 模式 (预期) |
|------|---------|---------------|
| 生成速度 | 3-4 tokens/s | 30-50 tokens/s |
| 内存占用 | ~3GB | ~1.5GB |
| 启动时间 | ~10s | ~5s |

## 🔧 常用命令

### 测试
```bash
python check_env.py          # 环境验证
python simple_test.py        # 简单测试
python final_test.py         # 完整测试
```

### 运行
```bash
python main.py --mode chat              # 对话
python main.py --mode generate --input "你好"  # 生成
python main.py --mode stats              # 统计
```

### Telegram Bot
```bash
python main.py --mode telegram --telegram-token TOKEN
```

## 📊 测试结果

✅ **通过率：100%**

- ✅ 模型加载正常
- ✅ Tokenizer 工作正常
- ✅ 文本生成正常
- ✅ 对话功能正常
- ✅ 海马体系统正常
- ✅ 统计数据正确

## 🛠️ 故障排除

### Python 版本错误
```bash
conda create -n stdpbrain python=3.11
conda activate stdpbrain
```

### 缺少依赖
```bash
./setup_conda_env.sh
```

### 模型缺失
```bash
huggingface-cli download Qwen/Qwen3.5-0.8B-Base --local-dir ./models/Qwen3.5-0.8B-Base
```

## 📚 文档导航

| 文档 | 用途 |
|------|------|
| [README.md](README.md) | 项目说明 |
| [RUN_GUIDE.md](RUN_GUIDE.md) | 运行指南 ⭐ |
| [TEST_REPORT.md](TEST_REPORT.md) | 测试报告 ⭐ |
| [PROJECT_COMPLETION.md](PROJECT_COMPLETION.md) | 完成总结 ⭐ |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 架构详解 |

## 💡 代码示例

### 基础使用
```python
from core.qwen_interface import create_qwen_ai

ai = create_qwen_ai(model_path="./models/Qwen3.5-0.8B-Base")
response = ai.chat("你好")
print(response)
```

### 高级用法
```python
# 文本生成
text = ai.generate("人工智能的核心是", max_new_tokens=100)

# 查看统计
stats = ai.get_stats()
print(f"生成{stats['generation_count']}次，{stats['total_tokens']} tokens")

# 海马体状态
if 'hippocampus' in stats:
    hp = stats['hippocampus']
   print(f"记忆数：{hp['num_memories']}")
   print(f"内存：{hp['memory_usage_mb']:.2f} MB")
```

## 🎯 关键特性

- ✅ **真实模型**: Qwen3.5-0.8B 真实权重
- ✅ **双系统**: 海马体 - 新皮层架构
- ✅ **STDP 学习**: 时序依赖可塑性
- ✅ **100Hz 刷新**: 10ms 周期推理
- ✅ **在线学习**: 无需反向传播
- ✅ **边缘友好**: INT4 量化支持

## 📞 支持

- GitHub: https://github.com/ctz168/stdpbrain
- 问题反馈：提交 Issue
- 文档：查看项目根目录 *.md 文件

---

**版本**: 1.0  
**更新日期**: 2026-03-09  
**状态**: ✅ 生产就绪
