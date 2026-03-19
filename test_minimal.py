import torch
from core.interfaces import BrainAIInterface
from configs.arch_config import BrainAIConfig
import time

def test_inference():
    config = BrainAIConfig()
    config.model_path = "./models/Qwen3.5-0.8B"
    config.device = "cpu"
    
    print("Loading AI interface...")
    ai = BrainAIInterface(config, device="cpu")
    
    history = []
    user_input = "你好，请自我介绍。"
    
    print(f"Testing chat with input: {user_input}")
    start_time = time.time()
    try:
        response = ai.chat(user_input, history)
        elapsed = time.time() - start_time
        print(f"Response: {response}")
        print(f"Time taken: {elapsed:.2f}s")
    except Exception as e:
        print(f"Error during chat: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
