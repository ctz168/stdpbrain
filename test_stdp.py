import torch
import torch.nn as nn
from core.stdp_engine import FullLinkSTDP
from configs.arch_config import default_config

def test_stdp():
    config = default_config
    stdp = FullLinkSTDP(config, device="cpu")
    
    # Mock parameters
    context_tokens = torch.tensor([1, 2, 3])
    current_token = 4
    timestamp = 1000.0
    
    # Mock layer
    class SubLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.dynamic_weight = nn.Parameter(torch.randn(10, 10))

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = SubLayer()
            self.k_proj = SubLayer()
            self.v_proj = SubLayer()
            self.o_proj = SubLayer()
            
        def apply_stdp_to_all(self, grad_dict, lr):
            print(f"STDP applied with lr={lr}, grad keys={grad_dict.keys()}")

    layer = MockLayer()
    
    # Record some activations
    stdp.record_activation('tokens', context_tokens, 990.0)
    
    # Update
    print("Running update_attention_layer...")
    stdp.update_attention_layer(
        layer, 
        context_tokens, 
        current_token, 
        torch.zeros(1), 
        timestamp,
        reward=2.0,
        is_tool_call=True
    )
    print("Done.")

if __name__ == "__main__":
    test_stdp()
