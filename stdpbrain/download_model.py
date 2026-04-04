from huggingface_hub import snapshot_download
import os

# 模型配置 - Qwen2.5-0.5B-Instruct (Qwen2ForCausalLM)
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
local_dir = "models/Qwen2.5-0.5B-Instruct"

print(f"开始下载 {model_id} 到 {local_dir}...")
print("提示：0.5B 模型文件约 1-2GB")
os.makedirs(local_dir, exist_ok=True)

# 启用高速下载
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print(f"✓ 下载完成！模型已保存到 {local_dir}")
