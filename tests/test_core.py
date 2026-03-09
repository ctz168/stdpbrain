"""
核心模块测试
"""

import pytest
import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDualWeightLinear:
    """测试双权重线性层"""
    
    def test_initialization(self):
        """测试初始化"""
        from core.dual_weight_layers import DualWeightLinear
        
        layer = DualWeightLinear(1024, 1024)
        
        assert layer.static_weight.shape == (1024, 1024)
        assert layer.dynamic_weight.shape == (1024, 1024)
        assert not layer.static_weight.requires_grad
        assert layer.dynamic_weight.requires_grad
    
    def test_forward(self):
        """测试前向传播"""
        from core.dual_weight_layers import DualWeightLinear
        
        layer = DualWeightLinear(1024, 1024)
        x = torch.randn(2, 1024)
        
        output = layer(x)
        
        assert output.shape == (2, 1024)
    
    def test_stdp_update(self):
        """测试 STDP 权重更新"""
        from core.dual_weight_layers import DualWeightLinear
        
        layer = DualWeightLinear(1024, 1024)
        initial_weight = layer.dynamic_weight.clone()
        
        # 应用 STDP 更新
        delta_w = torch.randn_like(layer.dynamic_weight) * 0.01
        layer.apply_stdp_update(delta_w, lr=0.01)
        
        # 验证权重已更新
        assert not torch.equal(layer.dynamic_weight, initial_weight)


class TestSTDPEngine:
    """测试 STDP 引擎"""
    
    def test_stdp_rule(self):
        """测试 STDP 规则"""
        from core.stdp_engine import STDPRule
        
        rule = STDPRule(alpha_LTP=0.01, beta_LTD=0.008)
        
        # LTP: 前激活早于后激活
        delta_w_ltp = rule.compute_update(
            pre_time=0, post_time=10, contribution=0.8
        )
        assert delta_w_ltp > 0
        
        # LTD: 前激活晚于后激活
        delta_w_ltd = rule.compute_update(
            pre_time=10, post_time=0, contribution=0.8
        )
        assert delta_w_ltd < 0
        
        # 超出时间窗口
        delta_w_outside = rule.compute_update(
            pre_time=0, post_time=50, contribution=0.8
        )
        assert delta_w_outside == 0


class TestHippocampusSystem:
    """测试海马体系统"""
    
    def test_encode_recall(self):
        """测试记忆编码与召回"""
        from configs.arch_config import default_config
        from hippocampus.hippocampus_system import HippocampusSystem
        
        config = default_config
        hippocampus = HippocampusSystem(config)
        
        # 编码
        features = torch.randn(1024)
        memory_id = hippocampus.encode(
            features=features,
            token_id=123,
            timestamp=int(time.time() * 1000)
        )
        
        assert memory_id.startswith("mem_")
        
        # 召回
        anchors = hippocampus.recall(features, topk=2)
        assert len(anchors) <= 2


def run_tests():
    """运行所有测试"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
