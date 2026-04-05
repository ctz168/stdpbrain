"""
用户反馈处理器

核心功能:
- 检测用户的正面/负面反馈
- 触发 STDP 学习（LTP/LTD）
- 存储错误回复为负样本
- 调整目标系统的奖励权重

类人脑对应:
- 多巴胺系统: 正反馈 → 多巴胺释放 → LTP增强
- 杏仁核: 负反馈 → 压力激素 → LTD抑制
"""

import re
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeedbackResult:
    """反馈结果"""
    is_feedback: bool  # 是否是反馈
    is_positive: bool  # 正面还是负面
    intensity: float   # 强度 (0.0-1.0)
    keywords_matched: List[str]  # 匹配的关键词
    suggested_action: str  # 建议的动作


class UserFeedbackHandler:
    """
    用户反馈处理器
    
    检测用户消息中的反馈信号，并触发相应的学习机制
    """
    
    def __init__(self):
        # ========== 负面反馈关键词 ==========
        # 强负面（极度不满）
        self.strong_negative = [
            "狗屁", "什么玩意", "胡说八道", "乱说", "完全错了",
            "太蠢了", "智障", "垃圾", "废物", "没用",
            "算错了", "根本不对", "完全错误", "一派胡言",
            "什么狗屁", "脑子有病", "傻", "白痴"
        ]
        
        # 中等负面（明确纠正）
        self.medium_negative = [
            "不对", "错了", "不是这样", "理解错了", "搞错了",
            "不对吧", "应该是", "其实", "实际上", "你弄错了",
            "算的不对", "搞错了", "不对啊", "错了啦", "不是",
            "重新算", "再想想", "不对不对", "搞什么"
        ]
        
        # 弱负面（质疑/困惑）
        self.weak_negative = [
            "？", "??", "？？？", "是吗", "真的吗", "确定吗",
            "怎么可能", "不会吧", "不太对", "好像不对",
            "有问题", "这不对", "不对劲", "奇怪"
        ]
        
        # ========== 正面反馈关键词 ==========
        # 强正面（非常满意）
        self.strong_positive = [
            "太棒了", "非常好", "太好了", "完美", "厉害",
            "很棒", "很好", "赞", "优秀", "excellent", "perfect",
            "太对了", "完全正确", "说得太好了", "正解"
        ]
        
        # 中等正面（认可/感谢）
        self.medium_positive = [
            "谢谢", "感谢", "好的", "对", "是的", "没错",
            "了解了", "明白了", "懂了", "好的呢", "嗯嗯",
            "可以", "行", "ok", "OK", "好的呢"
        ]
        
        # 弱正面（继续互动）
        self.weak_positive = [
            "嗯", "哦", "啊", "呢", "继续", "还有吗",
            "然后呢", "接下来", "说说", "讲讲"
        ]
    
    def detect_feedback(self, user_message: str) -> FeedbackResult:
        """
        检测用户消息中的反馈信号
        
        Args:
            user_message: 用户消息
            
        Returns:
            FeedbackResult: 反馈结果
        """
        msg = user_message.strip().lower()
        matched_keywords = []
        
        # ========== 检测负面反馈 ==========
        # 强负面
        for kw in self.strong_negative:
            if kw in msg:
                matched_keywords.append(kw)
        
        if matched_keywords:
            return FeedbackResult(
                is_feedback=True,
                is_positive=False,
                intensity=1.0,  # 最强
                keywords_matched=matched_keywords,
                suggested_action="strong_penalty"
            )
        
        # 中等负面
        matched_keywords = []
        for kw in self.medium_negative:
            if kw in msg:
                matched_keywords.append(kw)
        
        if matched_keywords:
            return FeedbackResult(
                is_feedback=True,
                is_positive=False,
                intensity=0.7,
                keywords_matched=matched_keywords,
                suggested_action="medium_penalty"
            )
        
        # 弱负面（需要结合上下文判断）
        # 只有消息很短且包含多个问号时才判断为负面
        # [FIX] 原代码: len(msg) < 20 and msg.count("？") >= 2 or msg.count("?") >= 2
        # 由于 and 优先级高于 or，导致长消息中只要含有2个 ? 就被误判为负面反馈。
        # 修正: 添加括号确保只有短消息才触发弱负面检测。
        if len(msg) < 20 and (msg.count("？") >= 2 or msg.count("?") >= 2):
            return FeedbackResult(
                is_feedback=True,
                is_positive=False,
                intensity=0.4,
                keywords_matched=["multiple_questions"],
                suggested_action="weak_penalty"
            )
        
        # ========== 检测正面反馈 ==========
        # 强正面
        matched_keywords = []
        for kw in self.strong_positive:
            if kw in msg:
                matched_keywords.append(kw)
        
        if matched_keywords:
            return FeedbackResult(
                is_feedback=True,
                is_positive=True,
                intensity=1.0,
                keywords_matched=matched_keywords,
                suggested_action="strong_reward"
            )
        
        # 中等正面
        matched_keywords = []
        for kw in self.medium_positive:
            if kw in msg:
                matched_keywords.append(kw)
        
        if matched_keywords:
            return FeedbackResult(
                is_feedback=True,
                is_positive=True,
                intensity=0.6,
                keywords_matched=matched_keywords,
                suggested_action="medium_reward"
            )
        
        # 弱正面（继续互动）
        # 只有消息很短且包含这些词时才判断为正面
        if len(msg) < 10:
            for kw in self.weak_positive:
                if kw in msg:
                    return FeedbackResult(
                        is_feedback=True,
                        is_positive=True,
                        intensity=0.3,
                        keywords_matched=[kw],
                        suggested_action="weak_reward"
                    )
        
        # 不是明显的反馈
        return FeedbackResult(
            is_feedback=False,
            is_positive=True,  # 默认中性偏正
            intensity=0.5,
            keywords_matched=[],
            suggested_action="none"
        )
    
    def compute_stdp_reward(self, feedback: FeedbackResult) -> float:
        """
        根据反馈计算 STDP 奖励值
        
        Args:
            feedback: 反馈结果
            
        Returns:
            float: STDP 奖励值 (0.0-2.0)
                - 0.0-0.3: 强惩罚（LTD）
                - 0.3-0.7: 弱惩罚
                - 0.7-1.3: 中性
                - 1.3-2.0: 强奖励（LTP）
        """
        if not feedback.is_feedback:
            return 1.0  # 中性
        
        if feedback.is_positive:
            # 正面反馈 → 奖励
            # intensity=1.0 → reward=2.0 (最强)
            # intensity=0.6 → reward=1.4
            # intensity=0.3 → reward=1.1
            return 1.0 + feedback.intensity
        else:
            # 负面反馈 → 惩罚
            # intensity=1.0 → reward=0.1 (最强惩罚)
            # intensity=0.7 → reward=0.3
            # intensity=0.4 → reward=0.6
            return 1.0 - feedback.intensity * 0.9
    
    def should_store_as_negative_sample(self, feedback: FeedbackResult) -> bool:
        """
        判断是否应该将上一个回复存储为负样本
        
        Args:
            feedback: 反馈结果
            
        Returns:
            bool: 是否存储为负样本
        """
        return (
            feedback.is_feedback and 
            not feedback.is_positive and 
            feedback.intensity >= 0.7
        )


# 全局实例 + 线程安全锁
_user_feedback_handler = None
_feedback_handler_lock = __import__('threading').Lock()

def get_feedback_handler() -> UserFeedbackHandler:
    """获取全局反馈处理器（线程安全单例）"""
    global _user_feedback_handler
    if _user_feedback_handler is None:
        with _feedback_handler_lock:
            if _user_feedback_handler is None:
                _user_feedback_handler = UserFeedbackHandler()
    return _user_feedback_handler
