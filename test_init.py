
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(os.getcwd())))

from configs.arch_config import default_config
from core.interfaces import BrainAIInterface

config = default_config
config.model_path = "./models/Qwen3.5-0.8B-Base"
print(f"Testing with model_path: {config.model_path}")

try:
    ai = BrainAIInterface(config, device="cpu")
    print(f"Success! is_real_model: {ai.is_real_model}")
except Exception as e:
    print(f"Failed initialization: {e}")
    import traceback
    traceback.print_exc()
