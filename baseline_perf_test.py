import time
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from configs.arch_config import BrainAIConfig
from core.interfaces import BrainAIInterface

def run_performance_test():
    print("Initializing AI...")
    config = BrainAIConfig()
    ai = BrainAIInterface(config)
    
    user_input = "你好，请自我介绍一下并告诉我今天的天气怎么样？"
    history = []
    
    print(f"Testing serial performance with input: {user_input}")
    
    # Run once to warm up
    ai.chat(user_input, history)
    
    results = []
    for i in range(3):
        start_time = time.time()
        response = ai.chat(user_input, history)
        elapsed = time.time() - start_time
        results.append(elapsed)
        print(f"Iteration {i+1}: {elapsed*1000:.2f}ms")
    
    avg_time = sum(results) / len(results)
    print(f"\nAverage time: {avg_time*1000:.2f}ms")
    
    # Save baseline to a file
    with open("baseline_perf.txt", "w") as f:
        f.write(f"Average time: {avg_time*1000:.2f}ms\n")
        f.write(f"Results: {results}\n")

if __name__ == "__main__":
    run_performance_test()
