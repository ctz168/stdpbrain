from huggingface_hub import snapshot_download
import os

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
local_dir = "weights/DeepSeek-R1-Distill-Qwen-1.5B"

print(f"Downloading {model_id} to {local_dir}...")
os.makedirs(local_dir, exist_ok=True)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print("Download complete.")
