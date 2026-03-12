
import sys
import os

print("Starting test...")
try:
    from configs.arch_config import BrainAIConfig
    print("BrainAIConfig imported")
    from core.interfaces import BrainAIInterface
    print("BrainAIInterface imported")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

print("All imports successful")
