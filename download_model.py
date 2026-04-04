from huggingface_hub import snapshot_download
import os
import sys

# 模型配置 - Qwen3.5-0.8B
model_id = "Qwen/Qwen3.5-0.8B"
local_dir = "models/Qwen3.5-0.8B"

print(f"开始下载 {model_id} 到 {local_dir}...")
print("提示：0.8B 模型文件约 1.5-2GB")
os.makedirs(local_dir, exist_ok=True)

# 启用高速下载
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"✓ 下载完成！模型已保存到 {local_dir}")
except Exception as e:
    print(f"\n[错误] 模型下载失败: {e}")
    print("请检查网络连接或代理设置，然后重试。")
    sys.exit(1)
