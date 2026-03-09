# 端侧部署指南

## 安卓手机部署

### 前置要求

- Android Studio Arctic Fox 或更高版本
- NDK 21+
- MNN 推理框架 (或使用 PyTorch Mobile)

### 步骤 1: 模型转换

```bash
cd deployment/scripts
python convert_to_mnn.py \
    --input ../../models/Qwen3.5-0.8B-Base \
    --output ../android/app/src/main/assets/brain_ai.mnn \
    --quantization INT4
```

### 步骤 2: 构建 APK

```bash
cd deployment/android
./gradlew assembleRelease
```

生成的 APK 位于：`app/build/outputs/apk/release/app-release.apk`

### 步骤 3: 安装与运行

```bash
adb install app-release.apk
adb shell am start -n com.brainai.app/.MainActivity
```

## 树莓派部署

### 前置要求

- Raspberry Pi 4B 或更高版本 (推荐 Pi 5)
- Raspberry Pi OS 64-bit
- Python 3.8+

### 步骤 1: 安装依赖

```bash
cd deployment/raspberry
sudo apt-get update
sudo apt-get install -y python3-pip python3-numpy libopenblas-dev
pip3 install -r requirements.txt
```

### 步骤 2: 模型优化

```bash
python optimize_model.py \
    --input ../../models/Qwen3.5-0.8B-Base \
    --output ./optimized_model \
    --quantization INT4
```

### 步骤 3: 运行推理

```bash
python infer.py \
    --model ./optimized_model \
    --input "你好"
```

### 性能基准

在树莓派 4B (4GB RAM) 上:

- 显存占用：~398MB (INT4)
- 单 token 延迟：~8.5ms
- 刷新周期：10ms (符合 100Hz 要求)

## 模型量化

### INT4 量化 (推荐)

```bash
python scripts/quantize.py \
    --model Qwen/Qwen3.5-0.8B-Base \
    --output quantized_model_int4 \
    --bits 4 \
    --method awq
```

### INT8 量化

```bash
python scripts/quantize.py \
    --model Qwen/Qwen3.5-0.8B-Base \
    --output quantized_model_int8 \
    --bits 8
```

## 性能优化建议

1. **使用 MNN 推理框架**: 比原生 PyTorch 快 2-3x
2. **启用 CPU 绑核**: 将推理线程绑定到大核
3. **预分配内存**: 避免运行时动态分配
4. **批量处理**: 小批量 (batch_size=1-2) 推理更高效

## 故障排查

### 问题：显存溢出

解决:
- 确认使用 INT4 量化
- 减小海马体记忆库容量 (`CA3_max_capacity`)
- 关闭不必要的后台应用

### 问题：推理延迟过高

解决:
- 检查是否启用了窄窗口注意力
- 确认刷新周期配置为 10ms
- 降低温度参数 (`temperature`)

##  benchmark 脚本

```bash
python scripts/benchmark.py \
    --model ./optimized_model \
    --duration 60 \
    --output benchmark_results.json
```

---

*详细文档请参考项目根目录的 README.md*
