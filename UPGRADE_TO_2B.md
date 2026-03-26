# Qwen3.5-2B 模型升级说明

## 🎯 升级概览

已将项目从 **Qwen3.5-0.8B** 升级到 **Qwen3.5-2B**，模型参数量提升 **2.5 倍**，预期性能显著提升。

---

## 📊 关键参数对比

| 参数 | 0.8B 版本 | 2B 版本 | 提升 |
|------|----------|---------|------|
| **参数量** | 0.8B | 2B | **2.5x** |
| **Hidden Size** | 1024 | **2048** | **2x** |
| **层数** | 24 | 24 | - |
| **注意力头** | 16 | 8Q+2KV | 架构优化 |
| **上下文长度** | 32K | **262K** | **8x** |
| **文件大小** | ~1.6GB | **~4-5GB** | 3x |

---

## ✅ 已完成的代码修改

### 1. **核心配置文件**
- ✅ `config.py` - 模型路径更新
- ✅ `configs/arch_config.py` - 模型名称和路径
- ✅ `download_model.py` - 下载脚本

### 2. **模型接口层**
- ✅ `core/interfaces.py` - hidden_size 1024 → 2048
- ✅ `core/qwen_interface.py` - 模型描述
- ✅ `core/dual_weight_layers.py` - 模块说明
- ✅ `core/self_encoder.py` - 维度参数

### 3. **海马体系统**
- ✅ `hippocampus/hippocampus_system.py` - input_dim/hidden_size
- ✅ `hippocampus/ec_encoder.py` - input_dim
- ✅ `hippocampus/ca1_gate.py` - hidden_size

### 4. **用户界面**
- ✅ `telegram_bot/bot.py` - 欢迎信息
- ✅ `telegram_bot/stream_handler.py` - 测试回复
- ✅ `main.py` - 帮助信息和提示

---

## 🚀 下一步操作

### 步骤 1: 下载新模型

```bash
# 方法 1: 使用下载脚本（推荐）
python download_model.py

# 方法 2: 使用 huggingface-cli
huggingface-cli download Qwen/Qwen3.5-2B --local-dir ./models/Qwen3.5-2B

# 方法 3: 使用 modelscope-cli（国内更快）
modelscope download --model Qwen/Qwen3.5-2B --local_dir ./models/Qwen3.5-2B
```

**注意**：2B 模型约 4-5GB，下载时间约 10-30 分钟（取决于网络速度）

---

### 步骤 2: 清理旧模型（可选）

如果磁盘空间紧张，可以删除旧模型：

```bash
# Windows
rmdir /s /q models\Qwen3.5-0.8B

# Linux/Mac
rm -rf models/Qwen3.5-0.8B
```

---

### 步骤 3: 重启服务

```bash
# Telegram Bot
启动telegram_bot.bat

# 对话模式
对话模式.bat

# 持续独白
持续独白观察.bat
```

---

## 💡 性能预期

### 回答质量提升
- ✅ **逻辑推理能力增强** - 2.5倍参数量带来更强的推理能力
- ✅ **回答更详细** - 不再过度简短
- ✅ **上下文理解更强** - 支持 262K 超长上下文
- ✅ **知识覆盖更广** - 更大的知识库

### 内存需求
- **CPU 模式**: 建议 16GB+ 内存（INT8 量化）
- **GPU 模式**: 建议 8GB+ 显存（INT8 量化）
- **FP16 模式**: 建议 16GB+ 显存

---

## ⚠️ 注意事项

### 1. **首次加载较慢**
2B 模型加载时间约 30-60 秒（比 0.8B 慢 2-3 倍）

### 2. **内存占用增加**
- 0.8B: ~2GB 内存
- **2B**: ~5-6GB 内存（INT8 量化）

### 3. **生成速度**
- CPU: 约 5-10 tokens/s（比 0.8B 慢 50%）
- GPU: 约 20-30 tokens/s（与 0.8B 接近）

### 4. **量化建议**
- **INT8**: 推荐用于 CPU / 低显存 GPU
- **FP16**: 推荐用于 8GB+ 显存的 GPU
- **INT4**: 不推荐（会损失太多质量）

---

## 🔧 故障排除

### 问题 1: 内存不足
```bash
# 解决方案: 使用 INT8 量化
# config.py 中设置
QUANTIZATION = "INT8"
```

### 问题 2: 模型加载失败
```bash
# 检查模型文件完整性
ls models/Qwen3.5-2B/
# 应该看到 config.json, model.safetensors, tokenizer.json 等文件

# 重新下载
python download_model.py
```

### 问题 3: 生成速度太慢
```bash
# 解决方案 1: 启用 KV Cache
# 已默认启用

# 解决方案 2: 降低 max_tokens
# 在调用时设置 max_tokens=50

# 解决方案 3: 使用 GPU
DEVICE = "cuda"  # config.py
```

---

## 📈 技术细节

### Qwen3.5-2B 架构特点

1. **混合注意力机制**
   - 门控 DeltaNet（线性注意力）
   - 门控注意力（传统注意力）
   - 兼顾长距离依赖和计算效率

2. **旋转位置编码**
   - 支持 262K 超长上下文
   - 无需额外训练即可扩展

3. **SwiGLU 激活**
   - 比 GELU 更平滑
   - 提升训练稳定性

---

## 🎓 参考资源

- [Qwen3.5-2B ModelScope](https://modelscope.cn/models/Qwen/Qwen3.5-2B)
- [Qwen3.5-2B Hugging Face](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Qwen3 技术报告](https://arxiv.org/abs/2505.09388)

---

## 📝 更新日志

**2026-03-27**
- ✅ 升级模型从 0.8B 到 2B
- ✅ 更新所有 hidden_size 参数 (1024 → 2048)
- ✅ 更新所有文档和提示信息
- ✅ 优化生成参数（temperature, presence_penalty）
- ✅ 改进系统提示词以提升回答质量

---

## 🤝 需要帮助？

如果遇到问题：
1. 检查本文档的故障排除部分
2. 查看 `main.py` 的错误提示
3. 运行 `python download_model.py` 重新下载模型

祝使用愉快！🎉
