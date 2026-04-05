"""
人类思维增强模块 (Human Thinking Enhancements)

设计理念:
基于认知心理学和神经科学，为 AI 内心思维引擎注入类人思维机制。
每个模块可独立使用，也可组合协作，模拟真实人类思维中的多种认知过程。

核心模块:
1. DualProcessThinking     - 双系统思维 (Kahneman 系统1/系统2)
2. CognitiveBiasEngine     - 认知偏差引擎 (锚定、可得性、确认偏差等)
3. EnhancedMetacognition   - 增强元认知 (自信校准、知识缺口检测、错误识别)
4. AnalogicalReasoningEngine - 类比推理引擎 (表面/结构相似性)
5. WorkingMemoryManager    - 工作记忆管理 (Miller 定律 7±2)
6. TemporalDiscounting     - 时间折扣 (双曲折扣函数)

参考理论:
- Kahneman (2011) "Thinking, Fast and Slow"
- Miller (1956) "The Magical Number Seven"
- Tversky & Kahneman (1974) 认知偏差理论
- Holyoak & Thagard (1995) 类比推理理论
- Ainslie (1975) 双曲时间折扣理论
"""

import torch
import torch.nn as nn
import numpy as np
import re
import math
import time
import hashlib
from typing import (
    Dict, List, Any, Tuple, Optional, Set, Union,
    Callable, DefaultDict
)
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict


# ==================== 1. 双系统思维 (Dual-Process Thinking) ====================

class ThinkingSystem(Enum):
    """思维系统类型"""
    SYSTEM1 = "system1"  # 系统1: 快速直觉
    SYSTEM2 = "system2"  # 系统2: 缓慢分析


@dataclass
class DualProcessResult:
    """双系统思维结果"""
    system: ThinkingSystem          # 选中的系统
    confidence: float               # 置信度 (0-1)
    reason: str                     # 选择原因
    generation_config: Dict[str, Any]  # 生成参数配置
    should_trigger_system2: bool = False  # 系统1是否应触发系统2


class DualProcessThinking:
    """
    双系统思维引擎 - 实现 Kahneman 的双过程理论
    
    系统1 (直觉/快速):
    - 模式识别、启发式、自动响应
    - 适用于熟悉模式、简单问题、情感话题
    - 低温度、高置信度、快速响应
    - 使用联想记忆召回
    
    系统2 (分析/缓慢):
    - 深度推理、逐步逻辑
    - 适用于复杂问题、数学、逻辑推理
    - 高温度、仔细输出
    - 使用序列推理
    
    平滑过渡: 系统1在不确定时可触发系统2进行二次确认
    """

    def __init__(self, system1_trigger_threshold: float = 0.6):
        """
        初始化双系统思维引擎
        
        Args:
            system1_trigger_threshold: 系统1自动触发的置信度阈值 (0-1)
                越高越倾向于使用系统2
        """
        self.system1_threshold = system1_trigger_threshold

        # ========== 系统1 关键词库 (直觉/快速/情感) ==========
        self.system1_keywords: Dict[str, List[str]] = {
            "emotional": [  # 情感话题 → 直觉
                "感觉", "心情", "喜欢", "讨厌", "开心", "难过", "生气",
                "害怕", "感动", "幸福", "痛苦", "焦虑", "紧张", "放松",
                "爱", "恨", "羡慕", "嫉妒", "同情", "温暖", "孤独",
                "兴奋", "失望", "满足", "后悔", "怀念", "期待",
                "comfortable", "happy", "sad", "angry", "love", "hate",
                "fear", "hope", "joy", "grief", "lonely", "excited",
            ],
            "familiar": [  # 熟悉话题 → 直觉
                "你好", "谢谢", "再见", "好的", "是", "不是", "知道",
                "明白", "了解", "简单", "容易", "当然", "没错",
                "hello", "thanks", "yes", "no", "okay", "sure",
            ],
            "creative": [  # 创意话题 → 直觉
                "想象", "创意", "灵感", "幻想", "故事", "诗歌",
                "音乐", "艺术", "设计", "dream", "imagine", "story",
                "creative", "art", "music", "poetry", "design",
            ],
            "opinion": [  # 观点表达 → 直觉
                "觉得", "认为", "看来", "应该", "也许", "大概",
                "可能", "似乎", "I think", "I believe", "maybe",
                "probably", "seems", "opinion",
            ],
        }

        # ========== 系统2 关键词库 (分析/逻辑/数学) ==========
        self.system2_keywords: Dict[str, List[str]] = {
            "math": [  # 数学 → 分析
                "计算", "多少", "等于", "加", "减", "乘", "除",
                "平方", "立方", "根", "方程", "函数", "积分", "导数",
                "calculate", "compute", "equal", "plus", "minus",
                "multiply", "divide", "equation", "formula", "math",
                "number", "percent", "ratio", "probability",
            ],
            "logic": [  # 逻辑 → 分析
                "推理", "推导", "证明", "因为", "所以", "如果...那么",
                "前提", "结论", "逻辑", "矛盾", "蕴含", "等价",
                "reason", "logic", "prove", "because", "therefore",
                "if...then", "premise", "conclusion", "contradiction",
                "deduction", "induction",
            ],
            "complex": [  # 复杂问题 → 分析
                "分析", "比较", "对比", "区别", "联系", "关系",
                "原因", "结果", "影响", "机制", "原理", "本质",
                "详细", "深入", "系统", "框架", "结构", "层次",
                "analyze", "compare", "explain", "why", "how",
                "mechanism", "principle", "structure", "detail",
                "comprehensive", "systematic", "framework",
            ],
            "factual": [  # 事实核查 → 分析
                "正确", "错误", "真假", "准确", "数据", "统计",
                "证据", "来源", "验证", "考证", "研究", "论文",
                "correct", "wrong", "true", "false", "data",
                "evidence", "source", "verify", "research", "paper",
            ],
        }

        # ========== 数学表达式正则 (最高优先级) ==========
        self._re_math_expr = re.compile(
            r'\d+\s*[+\-*/=<>]\s*\d+|'      # 基础算术
            r'log|ln|sin|cos|tan|sqrt|'       # 数学函数
            r'x\s*\^|x\*\*|\d+\s*%',         # 幂运算和百分比
            re.IGNORECASE
        )
        self._re_chinese_number = re.compile(
            r'第[一二三四五六七八九十]+|'
            r'[一二三四五六七八九十]+[个条项件]|'
            r'百分之\d+|'
            r'几[个条项件次]'
        )

        # ========== 系统1→系统2 触发历史 ==========
        self._system1_to_system2_triggers: deque = deque(maxlen=50)

        # ========== 预编译正则 ==========
        self._re_non_chinese = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9]')

    def classify_input(self, text: str) -> Tuple[str, float]:
        """
        分类输入文本，决定使用哪个思维系统
        
        Args:
            text: 输入文本
            
        Returns:
            (system_name, confidence): 系统名称 ("system1" 或 "system2") 和置信度
        """
        if not text or len(text.strip()) == 0:
            return ("system1", 0.3)

        text_lower = text.lower()
        scores = {"system1": 0.0, "system2": 0.0}
        reasons = {"system1": [], "system2": []}

        # 1. 数学表达式检测 (最高优先级 → 强制系统2)
        if self._re_math_expr.search(text) or self._re_chinese_number.search(text):
            scores["system2"] += 3.0
            reasons["system2"].append("检测到数学表达式或数字运算")

        # 2. 系统2 关键词匹配 (分析/逻辑/复杂)
        for category, keywords in self.system2_keywords.items():
            match_count = sum(1 for kw in keywords if kw in text_lower)
            if match_count > 0:
                # 数学/逻辑类权重更高
                weight = 1.5 if category in ("math", "logic") else 1.0
                scores["system2"] += match_count * weight
                reasons["system2"].append(f"{category}: 命中{match_count}个关键词")

        # 3. 系统1 关键词匹配 (直觉/情感/熟悉)
        for category, keywords in self.system1_keywords.items():
            match_count = sum(1 for kw in keywords if kw in text_lower)
            if match_count > 0:
                # 情感类权重稍高
                weight = 1.2 if category == "emotional" else 1.0
                scores["system1"] += match_count * weight
                reasons["system1"].append(f"{category}: 命中{match_count}个关键词")

        # 4. 文本长度分析 (长文本倾向系统2)
        text_len = len(text.strip())
        if text_len > 100:
            scores["system2"] += 0.5
            reasons["system2"].append(f"文本较长({text_len}字符)，需要仔细分析")
        elif text_len < 20:
            scores["system1"] += 0.3
            reasons["system1"].append("文本简短，适合直觉响应")

        # 5. 问题类型分析 (疑问句复杂度)
        question_indicators = ["?", "？", "吗", "呢", "怎么", "如何", "为什么",
                               "what", "how", "why", "which", "whether"]
        question_count = sum(1 for q in question_indicators if q in text_lower)
        if question_count > 0:
            # 多重问题 → 系统2
            if question_count >= 2:
                scores["system2"] += 1.0
                reasons["system2"].append(f"检测到{question_count}个疑问词，多重问题")
            # "为什么"类深层问题 → 系统2
            elif "为什么" in text or "why" in text_lower:
                scores["system2"] += 1.5
                reasons["system2"].append("深层因果关系问题")
            # 简单问题 → 系统1
            elif any(q in text for q in ["吗", "呢", "?"]):
                scores["system1"] += 0.3
                reasons["system1"].append("简单是非/选择问题")

        # 6. 标点符号分析 (列表、编号 → 系统2)
        list_indicators = ["1.", "2.", "3.", "一、", "二、", "首先", "其次",
                           "finally", "firstly", "secondly"]
        list_count = sum(1 for li in list_indicators if li in text)
        if list_count >= 2:
            scores["system2"] += 0.8
            reasons["system2"].append(f"检测到{list_count}个列表标记，结构化问题")

        # 7. 计算总分和置信度
        total_score = scores["system1"] + scores["system2"]
        if total_score == 0:
            # 无法判断时默认系统1 (保守策略)
            return ("system1", 0.4)

        # 归一化置信度
        winner = max(scores, key=scores.get)
        confidence = min(0.95, scores[winner] / (total_score + 0.1))
        confidence = max(0.3, confidence)  # 最低置信度

        return (winner, round(confidence, 3))

    def configure_generation(self, system: str) -> Dict[str, Any]:
        """
        根据思维系统配置生成参数
        
        Args:
            system: "system1" 或 "system2"
            
        Returns:
            生成参数字典 (temperature, top_k, top_p, max_tokens, repetition_penalty)
        """
        if system == "system1":
            # 系统1: 快速直觉 — 低温度、高确定性、短输出
            return {
                "temperature": 0.3,           # 低温度 → 更确定的输出
                "top_k": 20,                  # 窄采样
                "top_p": 0.85,                # 保守核采样
                "max_tokens": 256,            # 较短输出
                "repetition_penalty": 1.0,    # 无惩罚 (允许自然重复)
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "do_sample": True,
            }
        else:
            # 系统2: 缓慢分析 — 高温度、广泛探索、长输出
            return {
                "temperature": 0.7,           # 适中温度 → 平衡创造与准确
                "top_k": 50,                  # 更广采样
                "top_p": 0.95,                # 广泛核采样
                "max_tokens": 2048,           # 允许长输出
                "repetition_penalty": 1.15,   # 适度惩罚避免重复
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1,
                "do_sample": True,
            }

    def should_trigger_system2(self, text: str, system1_confidence: float) -> bool:
        """
        判断系统1是否应触发系统2进行二次确认
        
        平滑过渡机制: 当系统1置信度低于阈值时，自动触发系统2
        
        Args:
            text: 输入文本
            system1_confidence: 系统1的分类置信度
            
        Returns:
            是否应触发系统2
        """
        # 1. 置信度过低 → 触发系统2
        if system1_confidence < 0.45:
            self._system1_to_system2_triggers.append({
                "reason": "low_confidence",
                "confidence": system1_confidence,
                "timestamp": time.time()
            })
            return True

        # 2. 涉及重要决策 → 触发系统2
        important_keywords = [
            "决定", "决策", "建议", "投资", "医疗", "法律", "合同",
            "重要", "关键", "serious", "important", "decision",
            "invest", "medical", "legal", "contract",
        ]
        if any(kw in text for kw in important_keywords):
            self._system1_to_system2_triggers.append({
                "reason": "important_decision",
                "timestamp": time.time()
            })
            return True

        # 3. 矛盾检测 → 系统1产生矛盾时触发系统2
        contradiction_pairs = [
            ("但是", "也"), ("虽然", "但是"), ("一方面", "另一方面"),
            ("虽然", "but also"), ("however", "also"),
        ]
        if any(a in text and b in text for a, b in contradiction_pairs):
            self._system1_to_system2_triggers.append({
                "reason": "contradiction_detected",
                "timestamp": time.time()
            })
            return True

        return False

    def process(self, text: str) -> DualProcessResult:
        """
        完整的双系统思维处理流程
        
        1. 分类输入
        2. 判断是否需要系统2介入
        3. 配置生成参数
        
        Args:
            text: 输入文本
            
        Returns:
            DualProcessResult 包含完整的处理结果
        """
        # 步骤1: 分类
        system, confidence = self.classify_input(text)

        # 步骤2: 如果是系统1，检查是否需要触发系统2
        should_trigger_s2 = False
        if system == "system1":
            should_trigger_s2 = self.should_trigger_system2(text, confidence)
            if should_trigger_s2:
                system = "system2"
                confidence = max(confidence, 0.6)  # 提升置信度

        # 步骤3: 配置生成参数
        gen_config = self.configure_generation(system)

        # 选择原因说明
        reason_map = {
            "system1": "直觉快速响应 — 模式匹配/情感/熟悉话题",
            "system2": "深度分析推理 — 复杂问题/数学/逻辑",
        }
        reason = reason_map.get(system, "未知")

        return DualProcessResult(
            system=ThinkingSystem(system),
            confidence=confidence,
            reason=reason,
            generation_config=gen_config,
            should_trigger_system2=should_trigger_s2,
        )

    def get_transition_stats(self) -> Dict[str, Any]:
        """获取系统1→系统2触发统计"""
        triggers = list(self._system1_to_system2_triggers)
        if not triggers:
            return {"total_triggers": 0}

        reason_counts: DefaultDict[str, int] = defaultdict(int)
        for t in triggers:
            reason_counts[t["reason"]] += 1

        return {
            "total_triggers": len(triggers),
            "reason_breakdown": dict(reason_counts),
            "recent_triggers": triggers[-5:],
        }


# ==================== 2. 认知偏差引擎 (Cognitive Bias Engine) ====================

class BiasType(Enum):
    """认知偏差类型"""
    ANCHORING = "anchoring"              # 锚定偏差
    AVAILABILITY = "availability"        # 可得性启发
    CONFIRMATION = "confirmation"        # 确认偏差
    RECENCY = "recency"                  # 近因偏差
    HALO = "halo"                        # 光环效应
    FRAMING = "framing"                  # 框架效应


@dataclass
class Anchor:
    """锚定点"""
    text: str                # 锚定内容
    weight: float            # 锚定强度 (0-1)
    timestamp: float         # 创建时间
    source: str = ""         # 来源上下文
    decay_rate: float = 0.01  # 衰减速率


@dataclass
class InformationItem:
    """信息条目 (用于近因偏差和可得性计算)"""
    content: str
    timestamp: float
    importance: float = 0.5
    salience: float = 0.5    # 显著性/突出程度
    emotional_intensity: float = 0.0  # 情感强度


class CognitiveBiasEngine:
    """
    认知偏差引擎 - 模拟真实人类认知偏差
    
    目的: 让 AI 的行为更自然、更像人类，而非追求绝对理性。
    人类思维本身就不是完全理性的，适度的认知偏差使交互更真实。
    
    支持的偏差:
    1. 锚定偏差 (Anchoring Bias): 首个信息对判断的过度影响
    2. 可得性启发 (Availability Heuristic): 近期/显著记忆感觉更可能
    3. 确认偏差 (Confirmation Bias): 倾向于支持已有信念的信息
    4. 近因偏差 (Recency Bias): 近期信息权重更大
    5. 光环效应 (Halo Effect): 整体印象影响具体判断
    6. 框架效应 (Framing Effect): 问题描述方式影响决策
    """

    def __init__(
        self,
        anchor_decay_rate: float = 0.01,
        availability_time_window: float = 3600.0,  # 1小时
        recency_half_life: float = 300.0,           # 5分钟半衰期
    ):
        """
        初始化认知偏差引擎
        
        Args:
            anchor_decay_rate: 锚定权重衰减速率
            availability_time_window: 可得性计算的时间窗口 (秒)
            recency_half_life: 近因偏差的半衰期 (秒)
        """
        # ========== 锚定偏差 ==========
        self._anchors: List[Anchor] = []
        self.anchor_decay_rate = anchor_decay_rate

        # ========== 可得性启发 ==========
        self.availability_time_window = availability_time_window

        # ========== 近因偏差 ==========
        self.recency_half_life = recency_half_life

        # ========== 确认偏差 ==========
        self._existing_beliefs: Dict[str, float] = {}  # 信念 → 强度

        # ========== 光环效应 ==========
        self._global_impression: Dict[str, float] = {}  # 主题 → 整体印象分

        # ========== 框架效应 ==========
        self._frame_keywords: Dict[str, List[str]] = {
            "gain": [  # 收益框架 → 倾向冒险
                "获得", "收益", "赢", "成功", "机会", "好处", "优势",
                "gain", "win", "profit", "opportunity", "advantage", "success",
            ],
            "loss": [  # 损失框架 → 倾向保守
                "损失", "失败", "风险", "危险", "威胁", "坏处", "代价",
                "lose", "loss", "risk", "danger", "threat", "cost", "failure",
            ],
        }

        # ========== 预编译正则 ==========
        self._re_whitespace = re.compile(r'\s+')

    # ---------- 锚定偏差 (Anchoring Bias) ----------

    def set_anchor(self, anchor_text: str, weight: float = 0.7, source: str = ""):
        """
        设置锚定点
        
        第一个进入对话的信息会成为"锚点"，后续判断会被拉向它。
        
        Args:
            anchor_text: 锚定内容
            weight: 锚定强度 (0-1)，越强对后续影响越大
            source: 锚定来源上下文
        """
        anchor = Anchor(
            text=anchor_text,
            weight=max(0.0, min(1.0, weight)),
            timestamp=time.time(),
            source=source,
            decay_rate=self.anchor_decay_rate,
        )
        self._anchors.append(anchor)

    def get_anchor_bias(self) -> Dict[str, Any]:
        """
        获取当前锚定偏差
        
        锚定权重随时间衰减: w(t) = w0 * exp(-decay * elapsed)
        
        Returns:
            锚定偏差信息: 当前活跃锚点及其有效权重
        """
        now = time.time()
        active_anchors = []

        for anchor in self._anchors:
            elapsed = now - anchor.timestamp
            # 指数衰减
            effective_weight = anchor.weight * math.exp(-anchor.decay_rate * elapsed)
            
            if effective_weight > 0.05:  # 只保留有效锚点
                active_anchors.append({
                    "text": anchor.text,
                    "original_weight": anchor.weight,
                    "effective_weight": round(effective_weight, 4),
                    "elapsed_seconds": round(elapsed, 1),
                    "source": anchor.source,
                })

        # 按有效权重排序
        active_anchors.sort(key=lambda x: x["effective_weight"], reverse=True)

        # 计算总体锚定强度
        total_bias = sum(a["effective_weight"] for a in active_anchors)
        total_bias = min(1.0, total_bias)  # 饱和上限

        return {
            "total_anchor_bias": round(total_bias, 4),
            "active_anchors": active_anchors[:5],  # 最多返回5个
            "total_stored": len(self._anchors),
        }

    def _apply_anchor_to_value(self, original_value: float, direction: str = "neutral") -> float:
        """
        将锚定偏差应用于数值判断
        
        Args:
            original_value: 原始值
            direction: 锚定方向 ("high"/"low"/"neutral")
            
        Returns:
            偏差后的值
        """
        bias_info = self.get_anchor_bias()
        total_bias = bias_info["total_anchor_bias"]

        if total_bias < 0.05:
            return original_value

        # 锚定效应: 将判断拉向锚点方向 (±20% 范围内)
        if direction == "high":
            return original_value * (1.0 + total_bias * 0.2)
        elif direction == "low":
            return original_value * (1.0 - total_bias * 0.2)
        return original_value

    # ---------- 可得性启发 (Availability Heuristic) ----------

    def compute_availability_bias(
        self,
        query: str,
        memories: List[InformationItem],
    ) -> Dict[str, float]:
        """
        计算可得性偏差
        
        越近期、越显著、情感越强烈的记忆，在判断中越容易被召回，
        导致高估此类事件的发生概率。
        
        Args:
            query: 查询/判断主题
            memories: 记忆列表
            
        Returns:
            各记忆的可用性加权概率 (按偏差调整后)
        """
        if not memories:
            return {}

        now = time.time()
        availability_scores: Dict[str, float] = {}

        for item in memories:
            # 时间衰减 (越近越高)
            elapsed = now - item.timestamp
            if elapsed > self.availability_time_window:
                time_factor = 0.01  # 窗口外极低
            else:
                time_factor = 1.0 - (elapsed / self.availability_time_window)

            # 显著性加权
            salience_factor = item.salience

            # 情感强度加权 (强烈情感更容易被回忆)
            emotion_factor = 0.5 + 0.5 * item.emotional_intensity

            # 综合可用性得分
            score = time_factor * salience_factor * emotion_factor
            # Add query-relevance factor
            if query:
                query_relevance = self._compute_text_similarity(query, item.content)
                score = score * (0.5 + 0.5 * query_relevance)
            availability_scores[item.content] = round(score, 4)

        # 归一化为概率分布
        total = sum(availability_scores.values())
        if total > 0:
            availability_scores = {
                k: round(v / total, 4)
                for k, v in availability_scores.items()
            }

        return availability_scores

    # ---------- 确认偏差 (Confirmation Bias) ----------

    def set_belief(self, belief: str, strength: float = 0.5):
        """
        设置已有信念
        
        Args:
            belief: 信念内容
            strength: 信念强度 (0-1)
        """
        self._existing_beliefs[belief] = max(0.0, min(1.0, strength))

    def apply_confirmation_bias(
        self,
        options: List[str],
        existing_beliefs: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        应用确认偏差 —— 倾向于支持已有信念的选项
        
        与已有信念一致的选项会被提升排名，不一致的会被降低。
        
        Args:
            options: 待排序的选项列表
            existing_beliefs: 已有信念 (若为None使用内部存储)
            
        Returns:
            排序后的 (选项, 偏差分数) 列表
        """
        beliefs = existing_beliefs or self._existing_beliefs
        if not beliefs or not options:
            return [(opt, 0.5) for opt in options]

        scored_options = []
        for option in options:
            base_score = 0.5  # 基础分数

            # 计算与已有信念的一致性
            max_consistency = 0.0
            for belief, strength in beliefs.items():
                # 简单关键词重叠作为一致性度量
                overlap = self._compute_text_similarity(option, belief)
                consistency = overlap * strength
                max_consistency = max(max_consistency, consistency)

            # 确认偏差: 一致选项加分 (最多+0.3)
            biased_score = base_score + max_consistency * 0.3
            biased_score = min(1.0, biased_score)

            scored_options.append((option, round(biased_score, 4)))

        # 按偏差分数降序排列
        scored_options.sort(key=lambda x: x[1], reverse=True)
        return scored_options

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度计算 (字符级 Jaccard)"""
        set1 = set(text1)
        set2 = set(text2)
        if not set1 or not set2:
            return 0.0
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union)

    # ---------- 近因偏差 (Recency Bias) ----------

    def apply_recency_weight(
        self,
        information_items: List[InformationItem],
    ) -> List[Tuple[InformationItem, float]]:
        """
        应用近因偏差 — 越近期的信息权重越高
        
        使用指数衰减: w(t) = exp(-ln(2) * elapsed / half_life)
        
        Args:
            information_items: 信息条目列表
            
        Returns:
            加权后的 (信息条目, 权重) 列表
        """
        if not information_items:
            return []

        now = time.time()
        weighted_items = []

        for item in information_items:
            elapsed = now - item.timestamp
            # 指数衰减 (半衰期模型)
            time_weight = math.exp(-math.log(2) * elapsed / self.recency_half_life)
            # 结合重要性
            combined_weight = time_weight * item.importance
            weighted_items.append((item, round(combined_weight, 4)))

        # 按权重降序排列
        weighted_items.sort(key=lambda x: x[1], reverse=True)
        return weighted_items

    # ---------- 光环效应 (Halo Effect) ----------

    def set_impression(self, topic: str, score: float):
        """
        设置整体印象 (光环效应的来源)
        
        Args:
            topic: 主题
            score: 印象分 (0-1, 1=极好)
        """
        self._global_impression[topic] = max(0.0, min(1.0, score))

    def apply_halo_effect(self, topic: str, specific_judgment: float) -> float:
        """
        应用光环效应 — 整体印象影响具体判断
        
        Args:
            topic: 主题
            specific_judgment: 基于具体证据的独立判断
            
        Returns:
            受光环效应影响后的判断
        """
        halo = self._global_impression.get(topic, 0.5)
        # 光环效应: 整体印象向具体判断"渗透"约15%
        halo_biased = specific_judgment * 0.85 + halo * 0.15
        return round(halo_biased, 4)

    # ---------- 框架效应 (Framing Effect) ----------

    def detect_frame(self, text: str) -> Optional[str]:
        """
        检测文本的框架 (收益/损失)
        
        Args:
            text: 输入文本
            
        Returns:
            "gain" 或 "loss" 或 None
        """
        text_lower = text.lower()
        gain_count = sum(1 for kw in self._frame_keywords["gain"] if kw in text_lower)
        loss_count = sum(1 for kw in self._frame_keywords["loss"] if kw in text_lower)

        if gain_count > loss_count + 1:
            return "gain"
        elif loss_count > gain_count + 1:
            return "loss"
        return None

    def apply_framing_effect(self, value: float, frame: str) -> float:
        """
        应用框架效应
        
        收益框架 → 倾向乐观估计 (值+10%)
        损失框架 → 倾向悲观估计 (值-10%)
        
        Args:
            value: 原始值
            frame: "gain" 或 "loss"
            
        Returns:
            框架调整后的值
        """
        if frame == "gain":
            return round(value * 1.10, 4)  # 乐观偏差
        elif frame == "loss":
            return round(value * 0.90, 4)  # 悲观偏差
        return value

    # ---------- 综合偏差检测 ----------

    def detect_bias_susceptibility(self, text: str) -> Dict[str, Any]:
        """
        检测文本可能激活哪些认知偏差
        
        Args:
            text: 输入文本
            
        Returns:
            各偏差类型的易感性评分 (0-1)
        """
        susceptibility: Dict[str, float] = {}
        reasons: Dict[str, List[str]] = {}

        text_lower = text.lower()

        # 1. 锚定偏差: 包含数字/比较词 → 易受锚定
        anchor_score = 0.0
        anchor_reasons = []
        if re.search(r'\d+', text):
            anchor_score += 0.4
            anchor_reasons.append("包含数字信息")
        if any(w in text for w in ["大概", "估计", "约", "接近", "approximately"]):
            anchor_score += 0.3
            anchor_reasons.append("包含估计性描述")
        if self._anchors:
            anchor_score += 0.3
            anchor_reasons.append(f"已有{len(self._anchors)}个锚定点")
        susceptibility["anchoring"] = min(1.0, anchor_score)
        reasons["anchoring"] = anchor_reasons

        # 2. 可得性偏差: 情感/生动描述 → 易受可得性影响
        availability_score = 0.0
        avail_reasons = []
        emotion_words = ["震惊", "惊讶", "可怕", "太", "非常", "特别",
                         "shocking", "terrible", "amazing", "very", "extremely"]
        if any(w in text for w in emotion_words):
            availability_score += 0.4
            avail_reasons.append("包含强烈情感词汇")
        if len(text) > 200:
            availability_score += 0.2
            avail_reasons.append("描述详细，容易形成鲜明记忆")
        susceptibility["availability"] = min(1.0, availability_score)
        reasons["availability"] = avail_reasons

        # 3. 确认偏差: 已有信念相关话题
        confirmation_score = 0.0
        confirm_reasons = []
        if self._existing_beliefs:
            for belief in self._existing_beliefs:
                if self._compute_text_similarity(text, belief) > 0.3:
                    confirmation_score += 0.4
                    confirm_reasons.append(f"与已有信念'{belief}'相关")
                    break
        susceptibility["confirmation"] = min(1.0, confirmation_score)
        reasons["confirmation"] = confirm_reasons

        # 4. 近因偏差: 短期事件评估
        recency_score = 0.3  # 基础: 人类普遍有近因偏差
        recency_reasons = ["人类普遍存在的近因倾向"]
        if any(w in text for w in ["最近", "刚才", "今天", "昨天", "recently"]):
            recency_score += 0.4
            recency_reasons.append("涉及近期事件")
        susceptibility["recency"] = min(1.0, recency_score)
        reasons["recency"] = recency_reasons

        # 5. 光环效应
        halo_score = 0.0
        halo_reasons = []
        if any(w in text for w in ["专家", "权威", "名人", "教授", "博士",
                                     "expert", "authority", "professor", "doctor"]):
            halo_score += 0.5
            halo_reasons.append("涉及权威/专家，可能触发光环效应")
        susceptibility["halo"] = min(1.0, halo_score)
        reasons["halo"] = halo_reasons

        # 6. 框架效应
        frame = self.detect_frame(text)
        framing_score = 0.0
        framing_reasons = []
        if frame:
            framing_score = 0.6
            framing_reasons.append(f"检测到{frame}框架")
        susceptibility["framing"] = framing_score
        reasons["framing"] = framing_reasons

        return {
            "susceptibility": {k: round(v, 3) for k, v in susceptibility.items()},
            "reasons": reasons,
            "dominant_bias": max(susceptibility, key=susceptibility.get) if susceptibility else None,
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取认知偏差引擎统计信息"""
        return {
            "anchor_count": len(self._anchors),
            "belief_count": len(self._existing_beliefs),
            "impression_count": len(self._global_impression),
            "anchor_bias": self.get_anchor_bias(),
        }


# ==================== 3. 增强元认知 (Enhanced Metacognition) ====================

@dataclass
class ConfidenceRecord:
    """置信度记录 (用于校准)"""
    task_id: str
    task_description: str
    predicted_confidence: float
    actual_success: Optional[float] = None  # 0-1 或 None(未记录)
    timestamp: float = 0.0


@dataclass
class KnowledgeGap:
    """知识缺口"""
    query: str
    gap_description: str
    severity: float  # 0-1, 1=完全不知道
    suggested_action: str = ""
    timestamp: float = 0.0


@dataclass
class PotentialError:
    """潜在错误"""
    error_type: str       # 错误类型
    description: str      # 错误描述
    location: str = ""    # 在文本中的位置
    severity: float = 0.5 # 严重程度 (0-1)
    suggestion: str = ""  # 修正建议


class EnhancedMetacognition:
    """
    增强元认知系统 - 自我监控与自信校准
    
    核心能力:
    1. 自信校准: 追踪预测置信度与实际准确率的偏差，持续校准
    2. 知识缺口识别: 监控自身知识盲区
    3. 错误检测: 在输出前捕捉潜在错误
    4. 澄清判断: 判断何时应主动向用户请求澄清
    """

    def __init__(
        self,
        calibration_window_size: int = 100,
        overconfidence_penalty: float = 0.1,
    ):
        """
        初始化元认知系统
        
        Args:
            calibration_window_size: 校准窗口大小 (最近N条记录)
            overconfidence_penalty: 过度自信惩罚系数
        """
        # ========== 置信度校准 ==========
        self._confidence_records: deque = deque(maxlen=calibration_window_size)
        self.calibration_window = calibration_window_size
        self.overconfidence_penalty = overconfidence_penalty

        # 校准参数: calibrated = raw * scaling_factor + offset
        self._calibration_scale: float = 1.0
        self._calibration_offset: float = 0.0

        # ========== 知识缺口追踪 ==========
        self._knowledge_gaps: List[KnowledgeGap] = []
        self._covered_topics: Set[str] = set()  # 已知覆盖主题

        # ========== 错误模式库 ==========
        self._error_patterns: Dict[str, List[str]] = {
            "contradiction": [
                "但是", "然而", "不过", "可是", "although", "however",
                "but", "nevertheless", "on the other hand",
            ],
            "uncertainty": [
                "可能", "也许", "大概", "似乎", "好像", "不确定",
                "maybe", "perhaps", "probably", "might", "seem",
            ],
            "absolute": [
                "绝对", "一定", "肯定", "毫无疑问", "必然", "毫无疑问",
                "absolutely", "definitely", "certainly", "must", "always",
                "never", "impossible",
            ],
            "logical_fallacy": [
                "因为...所以", "显然", "众所周知", "大家都知道",
                "everyone knows", "obviously", "clearly", "naturally",
            ],
        }

        # ========== 预编译正则 ==========
        self._re_number_mismatch = re.compile(
            r'(\d+[\.,]?\d*)\s*[约大概]\s*(\d+[\.,]?\d*)'
        )
        self._re_self_contradiction = re.compile(
            r'(是|对|正确|对|yes|true).{0,20}(不是|错|错误|不对|no|false)|'
            r'(不是|错|错误|不对|no|false).{0,20}(是|对|正确|对|yes|true)',
            re.IGNORECASE
        )
        self._re_chinese_chars = re.compile(r'[\u4e00-\u9fa5]')

        # ========== 统计 ==========
        self._total_predictions = 0
        self._clarification_requests = 0

    # ---------- 自信校准 ----------

    def predict_confidence(self, task_description: str) -> float:
        """
        预测任务置信度
        
        基于任务特征和校准历史，预测对当前任务的把握程度。
        
        Args:
            task_description: 任务描述
            
        Returns:
            校准后的置信度 (0-1)
        """
        self._total_predictions += 1
        raw_confidence = self._estimate_raw_confidence(task_description)

        # 应用校准参数
        calibrated = raw_confidence * self._calibration_scale + self._calibration_offset
        calibrated = max(0.05, min(0.98, calibrated))

        # 记录预测 (actual_success 后续通过 record_outcome 填充)
        task_id = hashlib.md5(
            f"{task_description}_{time.time()}".encode()
        ).hexdigest()[:12]

        record = ConfidenceRecord(
            task_id=task_id,
            task_description=task_description,
            predicted_confidence=calibrated,
            timestamp=time.time(),
        )
        self._confidence_records.append(record)

        return round(calibrated, 4)

    def _estimate_raw_confidence(self, task_description: str) -> float:
        """
        估算原始置信度 (基于任务特征)
        
        因素:
        - 任务长度 (短任务通常更确定)
        - 熟悉度 (关键词匹配已知主题)
        - 复杂度 (数学、逻辑 → 降低置信度)
        - 模糊性 (包含不确定词汇 → 降低)
        """
        confidence = 0.6  # 基础置信度

        text = task_description.lower()
        text_len = len(task_description)

        # 长度调整 (适中长度 → 最高置信度)
        if text_len < 10:
            confidence += 0.1  # 简短 → 更确定
        elif text_len > 200:
            confidence -= 0.1  # 过长 → 不确定性增加

        # 主题覆盖度 (已知主题 → 提升置信度)
        topic_keywords = self._extract_topic_keywords(task_description)
        covered = sum(1 for kw in topic_keywords if kw in self._covered_topics)
        if topic_keywords:
            coverage_ratio = covered / len(topic_keywords)
            confidence += coverage_ratio * 0.2

        # 复杂度惩罚 (数学/逻辑 → 降低)
        complex_keywords = ["计算", "推导", "证明", "数学", "方程", "逻辑",
                            "calculate", "prove", "math", "equation", "logic"]
        if any(kw in text for kw in complex_keywords):
            confidence -= 0.15

        # 模糊性惩罚 (不确定词汇 → 降低)
        uncertain_keywords = ["可能", "也许", "不确定", "不清楚", "好像",
                              "maybe", "uncertain", "unsure", "not sure"]
        uncertain_count = sum(1 for kw in uncertain_keywords if kw in text)
        confidence -= uncertain_count * 0.05

        # 知识缺口惩罚
        gap_count = sum(1 for gap in self._knowledge_gaps
                        if self._compute_text_similarity(task_description, gap.query) > 0.3)
        if gap_count > 0:
            confidence -= gap_count * 0.1

        return max(0.1, min(0.95, confidence))

    def record_outcome(self, task_id: str, predicted_confidence: float, actual_success: float):
        """
        记录任务实际结果，用于校准
        
        Args:
            task_id: 任务ID (由 predict_confidence 返回的记录)
            predicted_confidence: 预测置信度
            actual_success: 实际成功率 (0-1)
        """
        # 查找并更新记录
        for record in self._confidence_records:
            if record.task_id == task_id:
                record.actual_success = max(0.0, min(1.0, actual_success))
                break

        # 定期更新校准参数
        self._update_calibration()

    def _update_calibration(self):
        """
        更新校准参数
        
        使用线性回归在预测置信度和实际成功率之间拟合:
        calibrated = raw * scale + offset
        
        目标: 让校准后的置信度尽量接近实际成功率。
        """
        completed = [r for r in self._confidence_records if r.actual_success is not None]
        if len(completed) < 5:
            return  # 样本不足，不更新

        predicted = np.array([r.predicted_confidence for r in completed])
        actual = np.array([r.actual_success for r in completed])

        # 简单线性回归
        if np.std(predicted) > 1e-6:
            correlation = np.corrcoef(predicted, actual)[0, 1]
            # 缩放因子: 让预测标准差接近实际标准差
            scale = np.std(actual) / (np.std(predicted) + 1e-8)
            scale = max(0.5, min(2.0, scale))  # 限制范围

            # 偏移: 让预测均值接近实际均值
            offset = np.mean(actual) - np.mean(predicted) * scale
            offset = max(-0.2, min(0.2, offset))  # 限制范围

            # 平滑更新 (防止剧烈变化)
            alpha = 0.1
            self._calibration_scale = (1 - alpha) * self._calibration_scale + alpha * scale
            self._calibration_offset = (1 - alpha) * self._calibration_offset + alpha * offset

    def get_calibration_stats(self) -> Dict[str, Any]:
        """
        获取校准统计信息
        
        Returns:
            包含校准指标、ECE (Expected Calibration Error) 等的字典
        """
        completed = [r for r in self._confidence_records if r.actual_success is not None]
        total_records = len(self._confidence_records)

        if len(completed) < 2:
            return {
                "total_records": total_records,
                "completed_records": len(completed),
                "status": "需要更多数据 (至少5条)",
                "calibration_scale": round(self._calibration_scale, 4),
                "calibration_offset": round(self._calibration_offset, 4),
            }

        predicted = np.array([r.predicted_confidence for r in completed])
        actual = np.array([r.actual_success for r in completed])

        # ECE (Expected Calibration Error) - 分箱计算
        n_bins = 10
        ece = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (predicted >= bin_boundaries[i]) & (predicted <= bin_boundaries[i + 1])
            else:
                mask = (predicted >= bin_boundaries[i]) & (predicted < bin_boundaries[i + 1])
            if mask.sum() > 0:
                avg_confidence = predicted[mask].mean()
                avg_accuracy = actual[mask].mean()
                bin_weight = mask.sum() / len(predicted)
                ece += bin_weight * abs(avg_confidence - avg_accuracy)

        # 相关性
        if len(predicted) > 2 and np.std(predicted) > 1e-6 and np.std(actual) > 1e-6:
            correlation = float(np.corrcoef(predicted, actual)[0, 1])
        else:
            correlation = 0.0

        # 平均偏差
        mean_error = float(np.mean(predicted - actual))

        return {
            "total_records": total_records,
            "completed_records": len(completed),
            "calibration_scale": round(self._calibration_scale, 4),
            "calibration_offset": round(self._calibration_offset, 4),
            "expected_calibration_error": round(ece, 4),
            "confidence_accuracy_correlation": round(correlation, 4),
            "mean_confidence_error": round(mean_error, 4),
            "avg_predicted_confidence": round(float(predicted.mean()), 4),
            "avg_actual_success": round(float(actual.mean()), 4),
            "overconfidence_ratio": round(
                float((predicted > actual + 0.1).sum() / len(predicted)), 4
            ) if len(predicted) > 0 else 0.0,
        }

    # ---------- 知识缺口识别 ----------

    def identify_knowledge_gap(
        self,
        query: str,
        memory_coverage: float = 0.0,
    ) -> Optional[KnowledgeGap]:
        """
        识别知识缺口
        
        当查询涉及未覆盖的主题或超出记忆范围时，标记为知识缺口。
        
        Args:
            query: 查询文本
            memory_coverage: 记忆系统返回的覆盖率 (0-1)
            
        Returns:
            KnowledgeGap 对象或 None
        """
        if memory_coverage > 0.7:
            return None  # 覆盖率足够，无缺口

        # 提取主题关键词
        topic_keywords = self._extract_topic_keywords(query)
        uncovered = [kw for kw in topic_keywords if kw not in self._covered_topics]

        if not uncovered:
            return None

        # 严重程度: 覆盖率越低、未覆盖关键词越多 → 越严重
        severity = (1.0 - memory_coverage) * 0.6 + (len(uncovered) / max(len(topic_keywords), 1)) * 0.4

        if severity < 0.2:
            return None  # 轻微缺口不报告

        gap = KnowledgeGap(
            query=query,
            gap_description=f"对主题 '{'、'.join(uncovered[:3])}' 的了解不足",
            severity=round(min(1.0, severity), 4),
            suggested_action=f"需要补充关于 {'、'.join(uncovered[:3])} 的知识",
            timestamp=time.time(),
        )
        self._knowledge_gaps.append(gap)

        # 保留最近的20个缺口
        if len(self._knowledge_gaps) > 20:
            self._knowledge_gaps = self._knowledge_gaps[-20:]

        return gap

    def mark_topic_covered(self, topic: str):
        """标记主题为已覆盖"""
        keywords = self._extract_topic_keywords(topic)
        for kw in keywords:
            self._covered_topics.add(kw)

    def _extract_topic_keywords(self, text: str) -> List[str]:
        """提取主题关键词"""
        # 提取中文词组 (2-4字)
        chinese_matches = self._re_chinese_chars.findall(text)
        keywords = []
        for i in range(len(chinese_matches) - 1):
            word = chinese_matches[i] + chinese_matches[i + 1]
            if len(word) >= 2:
                keywords.append(word)

        # 提取英文单词
        english_matches = re.findall(r'[a-zA-Z]{3,}', text)
        keywords.extend([w.lower() for w in english_matches])

        # 去重
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)

        return unique[:10]

    # ---------- 错误检测 ----------

    def check_for_errors(
        self,
        output_text: str,
        context: str = "",
    ) -> List[PotentialError]:
        """
        在输出前检测潜在错误
        
        检查项:
        1. 自相矛盾
        2. 过度绝对化表述
        3. 逻辑跳跃
        4. 不一致的时间/数量信息
        5. 冗余重复
        
        Args:
            output_text: 待检查的输出文本
            context: 上下文信息 (用于一致性检查)
            
        Returns:
            潜在错误列表
        """
        errors: List[PotentialError] = []

        if not output_text or len(output_text) < 5:
            return errors

        # 1. 自相矛盾检测
        if self._re_self_contradiction.search(output_text):
            errors.append(PotentialError(
                error_type="contradiction",
                description="文本中可能存在自相矛盾的表述",
                severity=0.7,
                suggestion="建议重新审视前后一致性",
            ))

        # 2. 过度绝对化检测
        absolute_matches = []
        for kw in self._error_patterns["absolute"]:
            if kw in output_text:
                absolute_matches.append(kw)
        if len(absolute_matches) >= 2:
            errors.append(PotentialError(
                error_type="overconfidence",
                description=f"使用了{len(absolute_matches)}个绝对化词汇: {absolute_matches}",
                severity=0.5,
                suggestion="考虑使用更谨慎的表述",
            ))

        # 3. 不确定性与绝对性混用
        has_uncertain = any(kw in output_text for kw in self._error_patterns["uncertainty"])
        has_absolute = any(kw in output_text for kw in self._error_patterns["absolute"])
        if has_uncertain and has_absolute:
            errors.append(PotentialError(
                error_type="tone_inconsistency",
                description="文本同时包含不确定和绝对化表述，语气不一致",
                severity=0.4,
                suggestion="统一语气风格，避免不确定与绝对混用",
            ))

        # 4. 数量不一致检测 (同一个上下文中的数字差异)
        numbers_in_text = re.findall(r'(\d+[\.,]?\d*)', output_text)
        if len(numbers_in_text) > 2:
            # 检查是否有近似但不相等的数字 (可能错误)
            unique_numbers = set(float(n.replace(',', '.')) for n in numbers_in_text)
            if len(unique_numbers) > 3:
                errors.append(PotentialError(
                    error_type="data_inconsistency",
                    description=f"文本中出现{len(unique_numbers)}个不同数值，需检查是否一致",
                    severity=0.3,
                    suggestion="核实所有数值是否准确",
                ))

        # 5. 冗余重复检测 (简单版: 长子串重复)
        if len(output_text) > 50:
            clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', output_text)
            for length in [8, 6, 4]:
                if len(clean) < length * 3:
                    continue
                found_repetition = False
                for i in range(len(clean) - length * 2):
                    segment = clean[i:i + length]
                    if clean.count(segment) >= 3:
                        errors.append(PotentialError(
                            error_type="repetition",
                            description=f"检测到重复片段: '{segment}' 出现多次",
                            severity=0.6,
                            suggestion="精简重复内容",
                        ))
                        found_repetition = True
                        break
                if found_repetition:
                    break

        # 6. 上下文一致性检查
        if context:
            context_consistency = self._check_context_consistency(output_text, context)
            if context_consistency < 0.3:
                errors.append(PotentialError(
                    error_type="context_mismatch",
                    description="输出与上下文信息不一致",
                    severity=0.6,
                    suggestion="确保回答与用户提供的上下文一致",
                ))

        return errors

    def _check_context_consistency(self, output: str, context: str) -> float:
        """检查输出与上下文的一致性"""
        # 提取上下文中的关键实体 (简单版: 数字和专有名词)
        context_numbers = set(re.findall(r'\d+[\.,]?\d*', context))
        output_numbers = set(re.findall(r'\d+[\.,]?\d*', output))

        if not context_numbers:
            return 0.8  # 无可比较的数字，默认一致

        # 检查关键数字是否被保留
        preserved = context_numbers & output_numbers
        return len(preserved) / len(context_numbers) if context_numbers else 0.8

    # ---------- 澄清判断 ----------

    def should_ask_for_clarification(self, query: str, confidence: float) -> bool:
        """
        判断是否应主动向用户请求澄清
        
        以下情况应请求澄清:
        1. 置信度低 (< 0.4)
        2. 查询模糊 (太短或缺乏具体信息)
        3. 查询包含歧义 (多义词/多义句)
        4. 涉及关键决策但信息不足
        
        Args:
            query: 用户查询
            confidence: 当前置信度
            
        Returns:
            是否应请求澄清
        """
        # 置信度过低 → 直接请求
        if confidence < 0.35:
            self._clarification_requests += 1
            return True

        # 查询模糊性检查
        ambiguity_score = self._assess_query_ambiguity(query)

        # 综合判断
        should_clarify = (
            (confidence < 0.45 and ambiguity_score > 0.5) or
            (ambiguity_score > 0.7) or
            (len(query.strip()) < 5 and confidence < 0.6)
        )

        if should_clarify:
            self._clarification_requests += 1

        return should_clarify

    def _assess_query_ambiguity(self, query: str) -> float:
        """评估查询的模糊程度 (0-1)"""
        score = 0.0

        # 太短 → 模糊
        if len(query.strip()) < 5:
            score += 0.4

        # 包含歧义指示词
        ambiguous_indicators = ["还是", "或者", "哪个", "什么意思", "还是说",
                                "either", "or", "which one", "what do you mean"]
        if any(w in query for w in ambiguous_indicators):
            score += 0.3

        # 代词过多 (指代不明)
        pronouns = ["它", "他", "她", "这个", "那个", "it", "this", "that", "he", "she"]
        pronoun_count = sum(1 for p in pronouns if p in query)
        score += pronoun_count * 0.15

        return min(1.0, score)

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度 (字符级 Jaccard)"""
        set1 = set(text1)
        set2 = set(text2)
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    def get_stats(self) -> Dict[str, Any]:
        """获取元认知系统统计信息"""
        return {
            "total_predictions": self._total_predictions,
            "completed_records": sum(
                1 for r in self._confidence_records if r.actual_success is not None
            ),
            "knowledge_gaps": len(self._knowledge_gaps),
            "covered_topics": len(self._covered_topics),
            "clarification_requests": self._clarification_requests,
            "calibration": self.get_calibration_stats(),
        }


# ==================== 4. 类比推理引擎 (Analogical Reasoning Engine) ====================

@dataclass
class AnalogyMapping:
    """类比映射"""
    source_concept: str          # 源概念
    target_concept: str          # 目标概念
    surface_similarity: float    # 表面相似度 (0-1)
    structural_similarity: float # 结构相似度 (0-1)
    overall_score: float         # 综合分数
    mapping_details: Dict[str, str]  # 概念映射详情: 源属性 → 目标属性
    confidence: float = 0.5      # 映射置信度


class AnalogicalReasoningEngine:
    """
    类比推理引擎 - 在不同情境间建立平行关系
    
    基于结构映射理论 (Structure-Mapping Theory, Gentner 1983):
    - 表面相似性: 共享表面特征 (名称、外观)
    - 结构相似性: 共享关系结构 (因果关系、层级关系)
    - 好的类比更依赖结构相似性而非表面相似性
    
    核心方法:
    1. find_analogy: 在两个情境间寻找类比映射
    2. transfer_knowledge: 从源情境迁移知识到目标情境
    3. generate_analogical_explanation: 生成类比解释文本
    """

    def __init__(
        self,
        surface_weight: float = 0.3,
        structural_weight: float = 0.7,
        similarity_threshold: float = 0.2,
    ):
        """
        初始化类比推理引擎
        
        Args:
            surface_weight: 表面相似性权重
            structural_weight: 结构相似性权重 (通常更高)
            similarity_threshold: 最低相似度阈值
        """
        self.surface_weight = surface_weight
        self.structural_weight = structural_weight
        self.threshold = similarity_threshold

        # ========== 关系模板库 (用于结构相似性计算) ==========
        # 格式: (关系关键词, 权重)
        self._relation_patterns: List[Tuple[str, float]] = [
            ("导致", 0.9), ("引起", 0.9), ("造成", 0.8),
            ("因为", 0.85), ("所以", 0.85), ("因此", 0.8),
            ("促进", 0.7), ("阻碍", 0.7), ("推动", 0.7),
            ("属于", 0.6), ("包含", 0.6), ("部分", 0.6),
            ("类似", 0.8), ("像", 0.7), ("如同", 0.7),
            ("cause", 0.9), ("lead", 0.85), ("because", 0.85),
            ("result", 0.8), ("promote", 0.7), ("prevent", 0.7),
            ("similar", 0.8), ("like", 0.7), ("analogous", 0.7),
        ]

        # ========== 类比缓存 ==========
        self._analogy_cache: Dict[str, AnalogyMapping] = {}

        # ========== 预编译正则 ==========
        self._re_relation = re.compile(
            r'(导致|引起|造成|因为|所以|因此|促进|阻碍|推动|属于|包含|'
            r'部分|类似|像|如同|cause|lead|because|result|promote|prevent|'
            r'similar|like|analogous)'
        )

    def find_analogy(
        self,
        source_situation: str,
        target_situation: str,
    ) -> AnalogyMapping:
        """
        在两个情境间寻找类比映射
        
        Args:
            source_situation: 源情境 (已知的、熟悉的)
            target_situation: 目标情境 (待理解的、新的)
            
        Returns:
            AnalogyMapping 包含完整的类比映射信息
        """
        # 1. 表面相似性
        surface_sim = self._compute_surface_similarity(source_situation, target_situation)

        # 2. 结构相似性
        structural_sim, mapping_details = self._compute_structural_similarity(
            source_situation, target_situation
        )

        # 3. 综合分数 (结构权重更高)
        overall = (
            surface_sim * self.surface_weight +
            structural_sim * self.structural_weight
        )

        # 4. 映射置信度
        confidence = min(0.95, overall * 1.2) if overall > self.threshold else overall * 0.5

        mapping = AnalogyMapping(
            source_concept=source_situation,
            target_concept=target_situation,
            surface_similarity=round(surface_sim, 4),
            structural_similarity=round(structural_sim, 4),
            overall_score=round(overall, 4),
            mapping_details=mapping_details,
            confidence=round(confidence, 4),
        )

        # 缓存
        cache_key = f"{hash(source_situation)}_{hash(target_situation)}"
        self._analogy_cache[cache_key] = mapping

        return mapping

    def _compute_surface_similarity(self, text1: str, text2: str) -> float:
        """
        计算表面相似性
        
        基于共享词汇/字符的 Jaccard 系数
        """
        # 字符级
        chars1 = set(text1)
        chars2 = set(text2)
        char_sim = len(chars1 & chars2) / len(chars1 | chars2) if chars1 | chars2 else 0

        # 词汇级 (中文: 双字词)
        def get_ngrams(text, n=2):
            return [text[i:i+n] for i in range(len(text) - n + 1)]

        ngrams1 = set(get_ngrams(text1))
        ngrams2 = set(get_ngrams(text2))
        ngram_sim = len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2) if ngrams1 | ngrams2 else 0

        return (char_sim * 0.4 + ngram_sim * 0.6)

    def _compute_structural_similarity(
        self,
        text1: str,
        text2: str,
    ) -> Tuple[float, Dict[str, str]]:
        """
        计算结构相似性
        
        基于关系模式的匹配: 两个情境是否描述了相似的关系结构？
        返回相似度和概念映射细节。
        """
        # 提取关系模式
        relations1 = self._extract_relations(text1)
        relations2 = self._extract_relations(text2)

        if not relations1 and not relations2:
            return 0.0, {}

        # 计算关系集合的 Jaccard 相似度
        set1 = set(relations1)
        set2 = set(relations2)
        relation_overlap = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

        # 生成概念映射 (基于共享上下文中的实体)
        mapping_details = self._generate_concept_mapping(text1, text2)

        # 映射覆盖率 (映射了多少概念)
        coverage = len(mapping_details) / max(len(set1 | set2), 1)

        structural_sim = relation_overlap * 0.6 + coverage * 0.4
        return structural_sim, mapping_details

    def _extract_relations(self, text: str) -> List[str]:
        """从文本中提取关系模式"""
        relations = []
        for pattern, weight in self._relation_patterns:
            if pattern in text:
                relations.append(pattern)
        return relations

    def _generate_concept_mapping(self, text1: str, text2: str) -> Dict[str, str]:
        """
        生成概念映射
        
        找出两个文本中语义上对应的概念对。
        使用简单的上下文窗口共现作为映射线索。
        """
        mapping = {}

        # 提取名词性短语 (简单版: 2-4字中文词组)
        def extract_phrases(text):
            phrases = []
            # 中文词组
            chinese = re.findall(r'[\u4e00-\u9fa5]{2,4}', text)
            phrases.extend(chinese)
            # 英文单词
            english = re.findall(r'[a-zA-Z]{3,}', text)
            phrases.extend([w.lower() for w in english])
            return list(set(phrases))

        phrases1 = extract_phrases(text1)
        phrases2 = extract_phrases(text2)

        # 对每个源短语，找最相似的目标短语
        for p1 in phrases1:
            best_match = ""
            best_score = 0.3  # 最低匹配阈值
            for p2 in phrases2:
                score = self._phrase_similarity(p1, p2)
                if score > best_score:
                    best_score = score
                    best_match = p2
            if best_match:
                mapping[p1] = best_match

        return mapping

    def _phrase_similarity(self, phrase1: str, phrase2: str) -> float:
        """计算短语相似度 (字符重叠)"""
        if phrase1 == phrase2:
            return 1.0
        set1 = set(phrase1)
        set2 = set(phrase2)
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    def transfer_knowledge(
        self,
        source_memory: str,
        target_query: str,
    ) -> Dict[str, Any]:
        """
        知识迁移 —— 从源情境迁移洞察到目标情境
        
        Args:
            source_memory: 源情境的记忆/知识
            target_query: 目标情境的查询
            
        Returns:
            迁移结果: 包括类比映射和迁移的洞察
        """
        # 1. 建立类比映射
        analogy = self.find_analogy(source_memory, target_query)

        # 2. 提取源情境中的关键洞察
        source_insights = self._extract_insights(source_memory)

        # 3. 将洞察迁移到目标情境
        transferred = []
        for insight in source_insights:
            # 用类比映射替换概念
            migrated = insight
            for src_concept, tgt_concept in analogy.mapping_details.items():
                migrated = migrated.replace(src_concept, tgt_concept)

            if migrated != insight:  # 只有真正发生迁移才算
                transferred.append(migrated)

        # 4. 如果直接映射效果不佳，使用结构性迁移
        if not transferred and analogy.structural_similarity > 0.3:
            transferred = self._structural_transfer(
                source_memory, target_query, analogy.mapping_details
            )

        return {
            "source": source_memory,
            "target": target_query,
            "analogy": analogy,
            "source_insights": source_insights,
            "transferred_insights": transferred,
            "transfer_confidence": round(
                analogy.overall_score * len(transferred) / max(len(source_insights), 1), 4
            ),
            "is_useful": analogy.overall_score > self.threshold and len(transferred) > 0,
        }

    def _extract_insights(self, text: str) -> List[str]:
        """从文本中提取关键洞察 (因果句、总结句)"""
        insights = []

        # 按句分割
        sentences = re.split(r'[。！？!?.;；\n]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        for sentence in sentences:
            # 包含因果关系的句子 → 洞察
            if any(kw in sentence for kw in ["因为", "所以", "导致", "因此", "原因",
                                              "because", "therefore", "result", "cause"]):
                insights.append(sentence)

            # 包含总结性词汇的句子 → 洞察
            if any(kw in sentence for kw in ["总之", "综上", "关键", "核心", "本质",
                                              "总之", "in summary", "key", "essential"]):
                insights.append(sentence)

        return insights[:5]  # 最多5条洞察

    def _structural_transfer(
        self,
        source: str,
        target: str,
        mapping: Dict[str, str],
    ) -> List[str]:
        """结构性知识迁移 (基于关系模式而非表面词汇)"""
        transferred = []

        # 提取源的关系模式
        source_relations = self._extract_relations(source)
        target_relations = self._extract_relations(target)

        # 共享的关系模式是迁移的基础
        shared = set(source_relations) & set(target_relations)

        if shared:
            # 生成基于共享关系的迁移建议
            relation_str = "、".join(list(shared)[:3])
            transferred.append(
                f"源情境和目标情境共享关系模式: {relation_str}。"
                f"源情境中的因果/逻辑结构可能适用于目标情境。"
            )

        return transferred

    def generate_analogical_explanation(
        self,
        source: str,
        target: str,
        mapping: AnalogyMapping,
    ) -> str:
        """
        生成类比解释文本
        
        Args:
            source: 源情境
            target: 目标情境
            mapping: 类比映射
            
        Returns:
            自然的类比解释文本
        """
        if mapping.overall_score < self.threshold:
            return f"'{source[:20]}' 和 '{target[:20]}' 之间的类比关系较弱。"

        # 构建解释
        parts = []

        # 1. 总述
        parts.append(
            f"'{source[:30]}' 和 '{target[:30]}' 之间存在类比关系。"
        )

        # 2. 相似性说明
        if mapping.surface_similarity > 0.3:
            parts.append(
                f"它们在表面特征上有一定相似性 (相似度: {mapping.surface_similarity:.0%})。"
            )

        if mapping.structural_similarity > 0.2:
            parts.append(
                f"更重要的是，它们具有相似的关系结构 (结构相似度: {mapping.structural_similarity:.0%})。"
            )

        # 3. 具体映射
        if mapping.mapping_details:
            mapping_pairs = [
                f"'{k}' → '{v}'"
                for k, v in list(mapping.mapping_details.items())[:5]
            ]
            parts.append(f"概念对应关系: {', '.join(mapping_pairs)}。")

        # 4. 总结
        overall_pct = mapping.overall_score
        if overall_pct > 0.5:
            parts.append("这是一个较好的类比，可以用源情境的理解帮助理解目标情境。")
        else:
            parts.append("这个类比有一定的参考价值，但需谨慎使用。")

        return "".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """获取类比推理引擎统计信息"""
        cached = list(self._analogy_cache.values())
        return {
            "cached_analogies": len(cached),
            "avg_overall_score": round(
                sum(a.overall_score for a in cached) / len(cached), 4
            ) if cached else 0.0,
            "high_quality_analogies": sum(
                1 for a in cached if a.overall_score > 0.5
            ),
        }


# ==================== 5. 工作记忆管理 (Working Memory Manager) ====================

@dataclass
class WorkingMemoryItem:
    """工作记忆条目"""
    content: str                # 内容
    importance: float           # 重要性 (0-1)
    timestamp: float            # 加入时间
    access_count: int = 0       # 访问次数
    last_access_time: float = 0.0  # 最后访问时间
    category: str = "general"   # 类别
    emotional_weight: float = 0.0  # 情感权重


class WorkingMemoryManager:
    """
    工作记忆管理器 - 实现 Miller's Law (7±2 项)
    
    模拟人类工作记忆的有限容量:
    - 默认容量: 7 项 (Miller 1956)
    - 范围: 5-9 项 (个体差异)
    - 超出容量时，驱逐最不重要/最久未用的条目
    - 驱逐策略: 综合考虑重要性、近期性、访问频率
    """

    def __init__(
        self,
        capacity: int = 7,
        recency_weight: float = 0.3,
        importance_weight: float = 0.4,
        frequency_weight: float = 0.2,
        emotional_weight: float = 0.1,
    ):
        """
        初始化工作记忆管理器
        
        Args:
            capacity: 工作记忆容量 (默认7, Miller's Law)
            recency_weight: 时间新近性权重
            importance_weight: 重要性权重
            frequency_weight: 访问频率权重
            emotional_weight: 情感权重
        """
        self.min_capacity = 5
        self.max_capacity = 9
        if not isinstance(capacity, int) or capacity <= 0:
            capacity = max(self.min_capacity, 7)  # fallback to default
        self.capacity = max(self.min_capacity, min(self.max_capacity, capacity))

        # ========== 权重配置 ==========
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.frequency_weight = frequency_weight
        self.emotional_weight = emotional_weight

        # ========== 工作记忆存储 ==========
        self._items: List[WorkingMemoryItem] = []

        # ========== 统计 ==========
        self._total_add_attempts = 0
        self._total_evictions = 0
        self._total_accesses = 0

    def add_item(
        self,
        item: str,
        importance: float = 0.5,
        category: str = "general",
        emotional_weight: float = 0.0,
    ) -> bool:
        """
        添加条目到工作记忆
        
        Args:
            item: 条目内容
            importance: 重要性 (0-1)
            category: 类别标签
            emotional_weight: 情感权重 (0-1)
            
        Returns:
            是否成功添加 (True=成功, False=被拒绝或被驱逐)
        """
        if not item or not item.strip():
            return False

        self._total_add_attempts += 1
        now = time.time()

        # 检查是否已存在相同内容
        existing_idx = next(
            (i for i, existing in enumerate(self._items) if existing.content == item),
            None
        )

        if existing_idx is not None:
            # 已存在: 更新访问信息而非重复添加
            self._items[existing_idx].access_count += 1
            self._items[existing_idx].last_access_time = now
            return True

        # 容量检查
        if len(self._items) >= self.capacity:
            # 尝试驱逐最不重要的条目
            evicted = self.prioritize_and_evict()
            if not evicted:
                # 无法驱逐 (所有条目都很重要)，拒绝新条目
                return False

        # 创建新条目
        wm_item = WorkingMemoryItem(
            content=item.strip(),
            importance=max(0.0, min(1.0, importance)),
            timestamp=now,
            access_count=1,
            last_access_time=now,
            category=category,
            emotional_weight=max(0.0, min(1.0, emotional_weight)),
        )
        self._items.append(wm_item)
        return True

    def get_items(self) -> List[Dict[str, Any]]:
        """
        获取当前工作记忆内容
        
        Returns:
            条目列表 (按优先级排序)
        """
        now = time.time()
        result = []

        for item in self._items:
            result.append({
                "content": item.content,
                "importance": item.importance,
                "category": item.category,
                "access_count": item.access_count,
                "age_seconds": round(now - item.timestamp, 1),
                "retention_score": round(self._compute_retention_score(item, now), 4),
            })

        # 按保留分数降序排列
        result.sort(key=lambda x: x["retention_score"], reverse=True)
        return result

    def prioritize_and_evict(self) -> bool:
        """
        优先级排序并驱逐最低优先级的条目
        
        驱逐策略:
        retention_score = w_recency * recency + w_importance * importance
                        + w_frequency * frequency + w_emotional * emotional
        
        Returns:
            是否成功驱逐
        """
        if not self._items or len(self._items) == 0:
            return False

        now = time.time()

        # 计算每个条目的保留分数
        scores = []
        for item in self._items:
            score = self._compute_retention_score(item, now)
            scores.append((item, score))

        # 按分数升序排列 (驱逐分数最低的)
        scores.sort(key=lambda x: x[1])

        # 驱逐最低分的条目
        if scores[0][1] < 0.5:  # 只有低于阈值的才驱逐
            evicted_item = scores[0][0]
            self._items.remove(evicted_item)
            self._total_evictions += 1
            return True

        return False

    def _compute_retention_score(self, item: WorkingMemoryItem, now: float) -> float:
        """
        计算条目的保留分数
        
        综合考虑: 时间新近性、重要性、访问频率、情感权重
        """
        # 1. 时间新近性 (越新越高, 指数衰减)
        age = now - item.timestamp
        recency = math.exp(-age / 300.0)  # 5分钟半衰期

        # 2. 重要性 (直接使用)
        importance = item.importance

        # 3. 访问频率 (归一化)
        frequency = min(1.0, item.access_count / 5.0)  # 5次访问达到最大

        # 4. 情感权重 (高情感 → 更难被遗忘)
        emotional = item.emotional_weight

        # 加权组合
        score = (
            self.recency_weight * recency +
            self.importance_weight * importance +
            self.frequency_weight * frequency +
            self.emotional_weight * emotional
        )

        return score

    def get_load(self) -> float:
        """
        获取当前工作记忆负载 (0.0-1.0)
        
        0.0 = 空闲, 1.0 = 满载
        
        Returns:
            负载比例
        """
        if self.capacity <= 0:
            return 1.0
        return len(self._items) / self.capacity

    def access_item(self, item_content: str) -> bool:
        """
        访问工作记忆中的条目 (更新访问记录)
        
        Args:
            item_content: 条目内容
            
        Returns:
            是否找到并访问
        """
        for item in self._items:
            if item.content == item_content:
                item.access_count += 1
                item.last_access_time = time.time()
                self._total_accesses += 1
                return True
        return False

    def get_context_summary(self, max_length: int = 200) -> str:
        """
        获取工作记忆的压缩摘要
        
        将当前工作记忆内容压缩为一段连贯的文本，
        用于注入到思维上下文中。
        
        Args:
            max_length: 摘要最大长度
            
        Returns:
            工作记忆摘要文本
        """
        if not self._items:
            return "（工作记忆为空）"

        # 按重要性排序
        sorted_items = sorted(self._items, key=lambda x: x.importance, reverse=True)

        parts = []
        current_length = 0

        for item in sorted_items:
            # 截断过长条目
            content = item.content[:50]
            entry = f"[{item.category}] {content}"

            if current_length + len(entry) > max_length:
                remaining = len(sorted_items) - len(parts)
                if remaining > 0:
                    parts.append(f"...还有{remaining}项")
                break

            parts.append(entry)
            current_length += len(entry)

        return " | ".join(parts)

    def clear(self):
        """清空工作记忆"""
        self._items.clear()

    def set_capacity(self, new_capacity: int):
        """
        动态调整工作记忆容量
        
        模拟认知负荷变化: 困难任务时容量降低，简单任务时容量增大。
        
        Args:
            new_capacity: 新容量 (5-9)
        """
        new_capacity = max(self.min_capacity, min(self.max_capacity, new_capacity))
        self.capacity = new_capacity

        # 如果当前条目超过新容量，驱逐多余的
        while len(self._items) > self.capacity:
            evicted = self.prioritize_and_evict()
            if not evicted:
                # All remaining items scored >= 0.5; force-evict the lowest
                now = time.time()
                self._items.sort(key=lambda it: self._compute_retention_score(it, now))
                self._items.pop(0)
                self._total_evictions += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取工作记忆统计信息"""
        return {
            "capacity": self.capacity,
            "current_load": round(self.get_load(), 4),
            "item_count": len(self._items),
            "total_add_attempts": self._total_add_attempts,
            "total_evictions": self._total_evictions,
            "total_accesses": self._total_accesses,
            "items": self.get_items(),
        }


# ==================== 6. 时间折扣 (Temporal Discounting) ====================

class EmotionalState(Enum):
    """情绪状态 - 影响时间偏好"""
    POSITIVE = "positive"        # 积极 → 更耐心
    NEUTRAL = "neutral"          # 中性
    NEGATIVE = "negative"        # 消极 → 更短视
    STRESSED = "stressed"        # 压力 → 极度短视
    RELAXED = "relaxed"          # 放松 → 非常耐心


class TemporalDiscounting:
    """
    时间折扣引擎 - 模拟人类时间偏好
    
    核心理论: 双曲折扣 (Hyperbolic Discounting)
    V = A / (1 + k * D)
    
    其中:
    - V: 折扣后的主观价值
    - A: 客观价值 (未来收益)
    - k: 折扣率 (个体差异 + 情绪状态)
    - D: 延迟时间
    
    特点:
    - 短期延迟的折价远大于长期延迟 (双曲线 vs 指数曲线)
    - 情绪状态影响折扣率: 积极情绪 → 更耐心; 消极情绪 → 更短视
    - 可解释"即时满足"偏好 (present bias)
    """

    def __init__(
        self,
        base_discount_rate: float = 0.1,
        time_unit: str = "days",
    ):
        """
        初始化时间折扣引擎
        
        Args:
            base_discount_rate: 基础折扣率 k (越大越短视)
            time_unit: 时间单位 ("seconds", "minutes", "hours", "days", "months")
        """
        self.base_k = base_discount_rate
        self.time_unit = time_unit

        # ========== 情绪-折扣率映射 ==========
        self._emotion_k_map: Dict[str, float] = {
            "positive": 0.05,    # 积极 → 低折扣率 → 更耐心
            "neutral": 0.10,     # 中性 → 基准
            "negative": 0.20,    # 消极 → 高折扣率 → 更短视
            "stressed": 0.40,    # 压力 → 极高折扣率
            "relaxed": 0.03,     # 放松 → 极低折扣率 → 非常耐心
        }

        # ========== 时间单位换算 (统一为天) ==========
        self._time_unit_multipliers: Dict[str, float] = {
            "seconds": 1.0 / 86400,
            "minutes": 1.0 / 1440,
            "hours": 1.0 / 24,
            "days": 1.0,
            "months": 30.0,
            "years": 365.0,
        }

        # ========== 延迟关键词映射 ==========
        self._delay_keywords: Dict[str, float] = {
            # 中文
            "立刻": 0, "马上": 0, "现在": 0, "今天": 1,
            "明天": 1, "后天": 2, "下周": 7, "下个月": 30,
            "明年": 365, "以后": 90, "将来": 365,
            "几天": 3, "几周": 21, "几个月": 90,
            "一周": 7, "一个月": 30, "半年": 180, "一年": 365,
            # 英文
            "now": 0, "immediately": 0, "today": 1,
            "tomorrow": 1, "next week": 7, "next month": 30,
            "next year": 365, "later": 90, "future": 365,
            "few days": 3, "few weeks": 21, "few months": 90,
            "a week": 7, "a month": 30, "half year": 180, "a year": 365,
        }

    def discount_value(
        self,
        base_value: float,
        delay_description: str,
        emotional_state: str = "neutral",
    ) -> Dict[str, Any]:
        """
        计算折扣后的主观价值
        
        V = A / (1 + k * D)
        
        Args:
            base_value: 基础价值 (未来收益的客观价值)
            delay_description: 延迟描述 (如 "3天"、"下个月"、"tomorrow")
            emotional_state: 情绪状态 ("positive"/"neutral"/"negative"/"stressed"/"relaxed")
            
        Returns:
            包含折扣值、延迟天数、折扣率等信息的字典
        """
        # 1. 解析延迟时间
        delay_days = self._parse_delay(delay_description)

        # 2. 获取情绪调整后的折扣率
        k = self._get_adjusted_k(emotional_state)

        # 3. 计算双曲折扣
        discounted_value = base_value / (1 + k * delay_days) if delay_days > 0 else base_value

        # 4. 计算损失比例
        loss_ratio = 1.0 - (discounted_value / base_value) if base_value > 0 else 0

        return {
            "base_value": base_value,
            "discounted_value": round(discounted_value, 4),
            "delay_days": round(delay_days, 2),
            "discount_rate_k": round(k, 4),
            "value_loss_ratio": round(loss_ratio, 4),
            "emotional_state": emotional_state,
            "formula": f"V = {base_value:.2f} / (1 + {k:.3f} * {delay_days:.1f}) = {discounted_value:.4f}",
        }

    def compare_immediate_vs_delayed(
        self,
        immediate_value: float,
        delayed_value: float,
        delay_description: str,
        emotional_state: str = "neutral",
    ) -> Dict[str, Any]:
        """
        比较即时收益与延迟收益
        
        模拟 "现在拿小钱" vs "以后拿大钱" 的决策困境
        
        Args:
            immediate_value: 即时收益的价值
            delayed_value: 延迟收益的客观价值
            delay_description: 延迟时间描述
            emotional_state: 情绪状态
            
        Returns:
            比较结果和偏好建议
        """
        # 计算延迟收益的折扣值
        discount_result = self.discount_value(delayed_value, delay_description, emotional_state)
        subjective_delayed = discount_result["discounted_value"]

        # 比较
        if subjective_delayed > immediate_value:
            preference = "delayed"
            reasoning = (
                f"虽然延迟收益需要等待{discount_result['delay_days']:.1f}天，"
                f"但考虑到长期价值，延迟收益的主观价值 ({subjective_delayed:.2f}) "
                f"仍高于即时收益 ({immediate_value:.2f})。"
            )
        elif subjective_delayed < immediate_value:
            preference = "immediate"
            reasoning = (
                f"由于时间折扣效应，延迟收益的主观价值 ({subjective_delayed:.2f}) "
                f"低于即时收益 ({immediate_value:.2f})。"
                f"折扣导致了 {discount_result['value_loss_ratio']:.0%} 的价值损失。"
            )
        else:
            preference = "indifferent"
            reasoning = "即时收益与折扣后的延迟收益主观价值相同，决策无差异。"

        # 计算盈亏平衡点: 延迟收益需要多大才能抵消即时收益？
        # A / (1 + k * D) = immediate_value → A = immediate_value * (1 + k * D)
        k = discount_result["discount_rate_k"]
        delay_days = discount_result["delay_days"]
        breakeven_delayed = immediate_value * (1 + k * delay_days)

        return {
            "immediate_value": immediate_value,
            "delayed_value": delayed_value,
            "subjective_delayed_value": round(subjective_delayed, 4),
            "preference": preference,
            "reasoning": reasoning,
            "value_difference": round(subjective_delayed - immediate_value, 4),
            "discount_details": discount_result,
            "breakeven_delayed_value": round(breakeven_delayed, 4),
            "is_rational_choice": (
                preference == "delayed" and delayed_value > immediate_value
            ) or (
                preference == "immediate" and delayed_value <= immediate_value
            ),
        }

    def _parse_delay(self, delay_description: str) -> float:
        """
        解析延迟描述为天数
        
        支持格式:
        - 关键词: "明天" → 1天, "下个月" → 30天
        - 数字+单位: "3天" → 3天, "2周" → 14天
        
        Args:
            delay_description: 延迟描述文本
            
        Returns:
            延迟天数
        """
        desc_lower = delay_description.lower().strip()

        # 1. 直接关键词匹配
        best_days = 0.0
        for keyword, days in self._delay_keywords.items():
            if keyword in desc_lower:
                # 尝试提取前面的数字修饰
                num_match = re.search(r'(\d+)\s*' + re.escape(keyword), desc_lower)
                if num_match:
                    best_days = max(best_days, float(num_match.group(1)) * days)
                else:
                    best_days = max(best_days, float(days))
        if best_days > 0:
            return best_days

        # 2. 数字+单位模式
        unit_days = {
            "秒": 1/86400, "second": 1/86400,
            "分钟": 1/1440, "minute": 1/1440,
            "小时": 1/24, "hour": 1/24,
            "天": 1, "日": 1, "day": 1,
            "周": 7, "week": 7,
            "月": 30, "month": 30,
            "年": 365, "year": 365,
        }
        for unit, multiplier in unit_days.items():
            pattern = r'(\d+\.?\d*)\s*' + re.escape(unit)
            match = re.search(pattern, desc_lower)
            if match:
                return float(match.group(1)) * multiplier

        # 3. 默认: 无法解析时返回中等延迟
        return 30.0

    def _get_adjusted_k(self, emotional_state: str) -> float:
        """
        获取情绪调整后的折扣率
        
        情绪影响:
        - 积极/放松 → 低 k → 更耐心 (愿意等待更大收益)
        - 消极/压力 → 高 k → 更短视 (偏好即时满足)
        
        Args:
            emotional_state: 情绪状态
            
        Returns:
            调整后的折扣率 k
        """
        emotion_k = self._emotion_k_map.get(emotional_state, self.base_k)
        # 混合基础率和情绪率
        return (self.base_k + emotion_k) / 2.0

    def compute_discount_curve(
        self,
        base_value: float,
        max_delay_days: float = 365.0,
        emotional_state: str = "neutral",
        num_points: int = 50,
    ) -> Dict[str, Any]:
        """
        计算完整的折扣曲线
        
        用于可视化和分析时间偏好特征。
        
        Args:
            base_value: 基础价值
            max_delay_days: 最大延迟天数
            emotional_state: 情绪状态
            num_points: 采样点数
            
        Returns:
            包含延迟数组和折扣值数组的字典
        """
        k = self._get_adjusted_k(emotional_state)
        delays = np.linspace(0, max_delay_days, num_points)
        values = np.array([base_value / (1 + k * d) for d in delays])

        return {
            "delays": delays.tolist(),
            "values": values.tolist(),
            "discount_rate_k": k,
            "base_value": base_value,
            "half_life_days": round(1.0 / k, 2) if k > 0 else float('inf'),
            "emotional_state": emotional_state,
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取时间折扣引擎统计信息"""
        return {
            "base_discount_rate": self.base_k,
            "time_unit": self.time_unit,
            "supported_emotional_states": list(self._emotion_k_map.keys()),
            "emotion_discount_rates": dict(self._emotion_k_map),
        }


# ==================== 工具函数 ====================

def create_human_thinking_suite(
    working_memory_capacity: int = 7,
    discount_rate: float = 0.1,
    system1_threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    一键创建完整的人类思维增强套件
    
    Args:
        working_memory_capacity: 工作记忆容量
        discount_rate: 基础时间折扣率
        system1_threshold: 系统1触发阈值
        
    Returns:
        包含所有模块实例的字典
    """
    return {
        "dual_process": DualProcessThinking(
            system1_trigger_threshold=system1_threshold
        ),
        "cognitive_bias": CognitiveBiasEngine(),
        "metacognition": EnhancedMetacognition(),
        "analogical_reasoning": AnalogicalReasoningEngine(),
        "working_memory": WorkingMemoryManager(capacity=working_memory_capacity),
        "temporal_discounting": TemporalDiscounting(base_discount_rate=discount_rate),
    }
